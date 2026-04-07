import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os import makedirs
from os.path import join
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# PQuant imports
from pquant.core.torch.layers import PQDense
from pquant import get_model_losses, get_layer_keep_ratio, get_ebops
from pquant import train_model, apply_final_compression

class QuantizedAutoencoderModel(nn.Module):
    """Kompakter quantisierter Autoencoder."""
    def __init__(self, config, layers=[32, 16, 4, 16, 32], n_inputs=10, in_out_bits=(1, 4, 11)):
        super().__init__()
        self.pq_layers = nn.ModuleList()
        
        current_inputs = n_inputs
        for i, nodes in enumerate(layers):
            is_first = (i == 0)
            is_last = (i == len(layers) - 1)
            
            self.pq_layers.append(
                PQDense(
                    config, current_inputs, nodes,
                    in_quant_bits=in_out_bits if is_first else None,
                    out_quant_bits=in_out_bits if is_last else None,
                    quantize_output=is_last,
                    enable_pruning=True
                )
            )
            current_inputs = nodes

    def forward(self, x):
        for i, layer in enumerate(self.pq_layers):
            x = layer(x)
            if i < len(self.pq_layers) - 1:
                x = F.relu(x)
        return x

class Autoencoder(BaseEstimator):
    def __init__(
            self,
            config,
            save_path=None,
            load=False,
            n_inputs=8,
            layers=[8, 32, 16, 2, 16, 32, 8],
            lr=0.001,
            early_stopping=False,
            patience=10,
            no_gpu=False,
            val_split=0.2,
            batch_size=128,
            epochs=100,
            verbose=False):

        self.config = config
        self.save_path = save_path
        self.clsf_model_path = join(save_path, "AE_models/") if save_path else None
        
        self.layers = layers
        self.n_inputs = n_inputs
        self.lr = lr
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.early_stopping = early_stopping
        self.patience = patience

        # Device Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_gpu else "cpu")
        
        # Model & Optimizer
        self.model = QuantizedAutoencoderModel(config, layers=layers, n_inputs=n_inputs).to(self.device)
        
        # Optimizer Setup mit getrennten Parameter-Gruppen
        params = list(self.model.named_parameters())
        self.optimizer = optim.Adam([
            {"params": [v for n, v in params if "threshold" in n and v.requires_grad], "weight_decay": 0},
            {"params": [v for n, v in params if "threshold" not in n and v.requires_grad], "weight_decay": 1e-4}
        ], lr=self.lr)

        self.loss_fn = F.mse_loss
        self.history = {"keep_ratio": [], "val_loss": [], "ebops": []}

        if load:
            self.load_best_model()

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_data(self, X):
        """Hilfsmethode zur Bereinigung und Konvertierung."""
        X_clean = X[~np.isnan(X).any(axis=1)]
        return torch.tensor(X_clean, dtype=torch.float32).to(self.device)

    def fit(self, X, X_val=None, **kwargs):
        # Validation Split falls nötig
        if X_val is None:
            X_train_np, X_val_np = train_test_split(X, test_size=self.val_split, shuffle=True)
        else:
            X_train_np, X_val_np = X, X_val

        # Data Loading
        train_loader = DataLoader(TensorDataset(self._prepare_data(X_train_np)), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(self._prepare_data(X_val_np)), batch_size=self.batch_size)

        if self.clsf_model_path:
            makedirs(self.clsf_model_path, exist_ok=True)

        # Definition der PQuant Steps (Korrektur der Daten-Extraktion)
        def pquant_train_step(model, loader, device, loss_fn, optimizer, epoch):
            model.train()
            total_loss = 0
            for (batch_data,) in loader: # Entpackt das Tuple aus TensorDataset
                optimizer.zero_grad()
                outputs = model(batch_data)
                
                recon_loss = loss_fn(outputs, batch_data)
                pquant_penalty = get_model_losses(model, torch.tensor(0., device=device))
                
                loss = recon_loss + pquant_penalty
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        def pquant_val_step(model, loader, device, loss_fn, epoch):
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for (batch_data,) in loader:
                    outputs = model(batch_data)
                    total_val_loss += loss_fn(outputs, batch_data).item()
            
            avg_loss = total_val_loss / len(loader)
            keep_ratio = get_layer_keep_ratio(model).item()
            
            # History tracken
            self.history["val_loss"].append(avg_loss)
            self.history["keep_ratio"].append(keep_ratio)
            self.history["ebops"].append(get_ebops(model).cpu().numpy())

            if self.verbose:
                print(f"[Epoch {epoch}] Loss: {avg_loss:.6f} | Sparsity: {1-keep_ratio:.2%}")
            
            return avg_loss, avg_loss

        # PQuant Training Start
        print("Starting PQuant-managed training...")
        train_model(
            model=self.model,
            config=self.config,
            train_func=pquant_train_step,
            valid_func=pquant_val_step,
            trainloader=train_loader,
            testloader=val_loader,
            device=self.device,
            loss_function=self.loss_fn,
            optimizer=self.optimizer,
            input_shape=(self.n_inputs,),
            gather_ebops=True
        )

        print("Applying final compression...")
        apply_final_compression(self.model)
        self.model.eval()
        return self

    def transform(self, X):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(x_tensor).cpu().numpy()

    def predict_proba(self, X):
        reco = self.transform(X)
        return np.sum((X - reco)**2, axis=-1)

    def fit_transform(self, X, m=None, X_val=None, m_val=None):
        """Trains and then transforms the provided data to the latent space.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Not implemented for this model.
        X_val : numpy.ndarray, optional
            Validation input data.
        m_val : numpy.ndarray, optional
            Not implemented for this model.

        Returns
        -------
        Xt : numpy.ndarray
            Latent space representation of the input data.
        """
        return self.fit(X, m=m, X_val=X_val, m_val=m_val).transform(X, m=m)

    def inverse_transform(self, Xt, m=None):
        raise NotImplementedError(
            "inverse_transform not implemented")

    def log_jacobian_determinant(self, X, m=None):
        raise NotImplementedError(
            "log_jacobian_determinant not implemented")

    def jacobian_determinant(self, X, m=None):
        raise NotImplementedError(
            "jacobian_determinant not implemented")

    def inverse_jacobian_determinant(self, X, m=None):
        raise NotImplementedError(
            "inverse_jacobian_determinant not implemented")

    def inverse_log_jacobian_determinant(self, X, m=None):
        raise NotImplementedError(
            "inverse_log_jacobian_determinant not implemented")

    def sample(self, n_samples=1, m=None):
        raise NotImplementedError(
            "sample not implemented")

    def score_samples(self, X, m=None):
        raise NotImplementedError("score_samples not implemented")

    def score(self, X, m=None):
        raise NotImplementedError("score not implemented")

    def load_best_model(self):
        """Loads the best model state from the provided save_path.
        """
        val_losses = self.load_val_loss()
        best_epoch = np.argmin(val_losses)
        self.load_epoch_model(best_epoch)
        self.model.eval()

    def load_train_loss(self):
        """Loads the training loss from the provided save_path.

        Returns
        -------
        train_loss : numpy.ndarray
            Training loss.
        """
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load train loss")
        return np.load(self._train_loss_path())

    def load_val_loss(self):
        """Loads the validation loss from the provided save_path.

        Returns
        -------
        val_loss : numpy.ndarray
            Validation loss.
        """
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load val loss")
        return np.load(self._val_loss_path())

    def load_epoch_model(self, epoch):
        """Loads the model state from the provided save_path at the
        specified epoch.

        Parameters
        ----------
        epoch : int
            Epoch at which to load the model state.
        """
        self._load_model(self._model_path(epoch))

    def _load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path,
                                              map_location=self.device))

    def _save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def _train_loss_path(self):
        return join(self.save_path, "CLSF_train_losses.npy")

    def _val_loss_path(self):
        return join(self.save_path, "CLSF_val_losses.npy")

    def _model_path(self, epoch):
        return join(self.clsf_model_path, f"CLSF_epoch_{epoch}.par")