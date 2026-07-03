import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from os import makedirs
from os.path import join
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#from pquant.core.torch.layers import PQDense
from pquant.layers import PQDense
from pquant import get_model_losses, get_layer_keep_ratio
from pquant import get_ebops
from pquant import train_model, apply_final_compression

import hls4ml
from hls4ml.utils import config_from_pytorch_model
from hls4ml.converters import convert_from_pytorch_model

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler


class QuantizedAutoencoderModel(nn.Module):
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
    """An autoencoder based on torch but wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.

    Parameters
    ----------
    save_path : str, optional
        Path to save the model to. If None, no model is saved.
        If provided, the model will use the best checkpoint after training.
    load : bool, optional
        Whether to load the model from save_path.
    n_inputs : int, default=4
        Number of input features.
    layers : list, default=[8, 32, 16, 2, 16, 32, 8]
        List of integers, specifying the number of nodes in each layer.
    lr : float, default=0.001
        Learning rate during training.
    early_stopping : bool, default=False
        Whether to use early stopping. If set, the provided number of
        epochs will be treated as an upper limit.
    patience : int, default=10
        Number of epochs to wait for improvement before stopping, if early
        stopping is used.
    no_gpu : bool, default=False
        Turns off GPU usages. By default the GPU is used if available.
    val_split : float, default=0.2
        Fraction of the training set to use for validation. Only has an
        effect if no validation set is provided to the fit method.
    batch_size : int, default=128
        Batch size during training.
    epochs : int, default=100
        Number of epochs to train for. In case early stopping is used,
        this is treated as an upper limit. Then also None can be provided,
        in which case the training will continue until early stopping
        is triggered.
    verbose : bool, default=False
        Whether to print progress during training.
    """

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
        if save_path is not None:
            self.clsf_model_path = join(save_path, "AE_models")
        else:
            self.clsf_model_path = None

        self.layers = layers
        self.n_inputs = n_inputs
        self.lr = lr
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_gpu else "cpu")

        self.model = QuantizedAutoencoderModel(config, layers=layers, n_inputs=n_inputs)

        
        params = list(self.model.named_parameters())
        self.optimizer = optim.Adam([
            {"params": [v for n, v in params if "threshold" in n and v.requires_grad], "weight_decay": 0},
            {"params": [v for n, v in params if "threshold" not in n and v.requires_grad], "weight_decay": 1e-4}
        ], lr=self.lr)


        self.loss = F.mse_loss
        self.no_gpu = no_gpu
        
        self.history = {"train_loss": [],"keep_ratio": [], "val_loss": [], "val_accuracies": [],"ebops": []}

        self.model.to(self.device)
        self.model(torch.randn(1, n_inputs).to(self.device))
        # defaulting to eval mode, switching to train mode in fit()
        self.model.eval()
        self.load = load

        if load:
            self.load_best_model()

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
                  
    def predict_proba(self, X, m=None):
        """Runs input data through the model and computes reconstruction
        error (MSE loss) which
        can be used as an anomaly score

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray, optional
            Not implemented for this model.

        Returns
        -------
        reco_error : numpy.ndarray
            Reconstruction erorr (MSE loss) on each example
        """

        if m is not None:
            raise NotImplementedError(
                "Conditional data not implemented for Autoencoder")

        reco = self.transform(X, m=m)
        reco_error = np.mean((X - reco)**2, axis=-1)

        return reco_error

    def predict_log_proba(self, X, m=None):
        """Runs input data through the model and computes the logarithmic
        reconstruction error (MSE loss) which can be used as an anomaly score

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray, optional
            Not implemented for this model.

        Returns
        -------
        reco_error : numpy.ndarray
            Log reconstruction erorr (MSE loss) on each example
        """

        if m is not None:
            raise NotImplementedError(
                "Conditional data not implemented for Autoencoder")

        return np.log(self.predict_proba(X))

    def transform(self, X, m=None):
        """Compresses and decompresses the input data

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray, optional
            Not implemented for this model.

        Returns
        -------
        prediction : numpy.ndarray
            Output array
        """

        if m is not None:
            raise NotImplementedError(
                "Conditional data not implemented for Autoencoder")

        with torch.no_grad():
            self.model.eval()
            X = torch.from_numpy(X).type(torch.FloatTensor).to(self.device)
            prediction = self.model.forward(X).detach().cpu().numpy()
        return prediction

    def fit(self, X, m=None, X_val=None, m_val=None,
            sample_weight=None, sample_weight_val=None):
        """Fits (trains) the model to the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray, optional
            Not implemented for this model.
        X_val : numpy.ndarray, optional
            Validation input data.
        m_val : numpy.ndarray, optional
            Not implemented for this model.
        sample_weight : numpy.ndarray, optional (Not yet implemented!)
            Sample weights for the training data.
        sample_weight_val : numpy.ndarray, optional  (Not yet implemented!)
            Sample weights for the validation data.

        Returns
        -------
        self : object
            An instance of the classifier.
        """

        if m is not None or m_val is not None:
            raise NotImplementedError(
                "Conditional data not implemented for Autoencoder")

        assert not (self.epochs is None and not self.early_stopping), (
            "A finite number of epochs must be set if early stopping"
            " is not used!")

        if sample_weight is not None or sample_weight_val is not None:
            raise NotImplementedError(
                "Sample weights for autoencoder training not yet implemented!")

        # allowing not to provide validation set, just for compatibility with
        # the sklearn API
        if X_val is None:
            if self.val_split is None or not (self.val_split > 0.
                                              and self.val_split < 1.):
                raise ValueError("val_split is needs to be provided and lie "
                                 "between 0 and 1 in case X_val is "
                                 "not provided!")
            else:
                X_train, X_val = train_test_split(
                    X, test_size=self.val_split, shuffle=True)
        else:
            X_train = X.copy()

        if self.clsf_model_path is not None:
            makedirs(self.clsf_model_path, exist_ok=True)

        nan_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[nan_mask]

        nan_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[nan_mask]

        # build data loader out of numpy arrays

        X_train_torch = torch.from_numpy(X_train).type(torch.FloatTensor).to(self.device)
        X_train_dataset = torch.utils.data.TensorDataset(X_train_torch)
        train_loader = torch.utils.data.DataLoader(X_train_dataset, batch_size=self.batch_size, shuffle=True)

        X_val_torch = torch.from_numpy(X_val).type(torch.FloatTensor).to(self.device)
        X_val_dataset = torch.utils.data.TensorDataset(X_val_torch)
        val_loader = torch.utils.data.DataLoader(X_val_dataset, batch_size=self.batch_size, shuffle=True)

        
        # training loop
        def pquant_train_step(model, trainloader, device, loss_function, optimizer, epoch, **kwargs):
            model.train()
            total_loss = 0
            for data in trainloader:
                inputs = data[0].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Loss
                recon_loss = loss_function(outputs, inputs)
                pquant_loss = get_model_losses(model, torch.tensor(0.).to(device))
                
                loss = recon_loss + pquant_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(trainloader)
            self.history["train_loss"].append(avg_loss)

            if self.save_path is not None:
                np.save(self._train_loss_path(), np.array(self.history["train_loss"]))

            return avg_loss

        # validation loop
        def pquant_val_step(model, testloader, device, loss_function, epoch, **kwargs):
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in testloader:
                    inputs = data[0].to(device)
                    outputs = model(inputs)

                    recon_loss = loss_function(outputs, inputs)
                    pquant_loss = get_model_losses(model, torch.tensor(0.).to(device))
                    loss = recon_loss + pquant_loss
                    val_loss += loss.item()
                    
            avg_loss = val_loss / len(testloader)
            keep_ratio = get_layer_keep_ratio(model).item()
            #self.history["ebops"].append(get_ebops(model).detach().cpu().numpy())

            calc_heavy_metrics = (epoch % 10 == 0)
            if calc_heavy_metrics or epoch == self.epochs:
                current_ebops = get_ebops(model).item()
                self.history["ebops"].append(current_ebops)
            else:
                last_ebops = self.history["ebops"][-1] if self.history["ebops"] else 0
                self.history["ebops"].append(last_ebops)

            if self.save_path is not None:
                np.save(self._val_loss_path(), np.array(self.history["val_loss"]))
                self._save_model(self._model_path(epoch))

            if self.verbose:
                print(f"Epoch {epoch}: Val Loss {avg_loss:.6f} | Ratio {keep_ratio:.2%}")
            
            return avg_loss
        
        
        # Starting training
        train_model(
            model=self.model,
            config=self.config,
            train_func=pquant_train_step,
            valid_func=pquant_val_step,
            trainloader=train_loader,
            testloader=val_loader,
            device=self.device,
            loss_function=self.loss,
            optimizer=self.optimizer,
            input_shape=(self.n_inputs,),
            gather_ebops=True
        )

        # fixing final weights
        apply_final_compression(self.model)
        
        self.model.eval()
        if self.save_path is not None:
            print("Loading best model state...")
            self.load_best_model()

        return self



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
        #self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        #self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

        state_dict = torch.load(model_path, map_location=self.device)
        keys_to_delete = [k for k in state_dict.keys() if '.i' in k or '.f' in k or '.b' in k]
        for k in keys_to_delete:
            del state_dict[k] 
        self.model.load_state_dict(state_dict, strict=False)

    def _save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def _train_loss_path(self):
        return join(self.save_path, "CLSF_train_losses.npy")

    def _val_loss_path(self):
        return join(self.save_path, "CLSF_val_losses.npy")

    def _model_path(self, epoch):
        return join(self.clsf_model_path, f"CLSF_epoch_{epoch}.par")
    
    def plot_results (self, score_background, score_signal):
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Verlauf der verbleibenden Gewichte (Sparsity)
        ax[0, 0].plot(self.history["keep_ratio"], label="Remaining Weights %", color='blue')
        #ax[0, 0].set_title("Weight Pruning Progress")
        ax[0, 0].set_xlabel("Epoch", fontsize=15)
        ax[0, 0].set_ylabel("Ratio of Remaining Weights", fontsize=15)
        ax[0, 0].legend()

        # Validierungs-Loss (MSE)
        ax[0, 1].plot(self.history["val_loss"], label="Val Loss (MSE)", color='orange')
        #ax[0, 1].set_title("Reconstruction Error")
        ax[0, 1].set_xlabel("Epoch", fontsize=15)
        ax[0, 1].set_ylabel("Mean Squared Error (MSE)", fontsize=15)
        ax[0, 1].legend()

        # EBOPs (Hardware-Effizienz)
        ax[1, 1].plot(self.history["ebops"], label="EBOPs", color='green')
        #ax[1, 1].set_title("Effective Bit Operations")
        ax[1, 1].set_xlabel("Epoch", fontsize=15)
        ax[1, 1].set_ylabel("Effective Bit Operations", fontsize=15)
        ax[1, 1].legend()

        # Histogramm der Scores
        ax[1, 0].hist(score_background, bins=50, alpha=0.5, label='Bkg', density=True)
        ax[1, 0].hist(score_signal, bins=50, alpha=0.5, label='Sig', density=True)
        #ax[1, 0].set_title("Score Distribution")
        ax[1, 0].set_xlabel("Score", fontsize=15)
        ax[1, 0].set_ylabel("Probability Density", fontsize=15)
        ax[1, 0].set_xlim(0, 2)
        ax[1, 0].legend()



        plt.tight_layout()
        plt.savefig(join(self.save_path, 'plots.png'))




    def plot_loss_histogram(self, score_bg, score_sig):
        """Erstellt ein Histogramm der Rekonstruktionsverluste."""
        plt.figure(figsize=(9, 6))
        limit = np.percentile(score_sig, 98)
        
        plt.hist(score_bg, bins=100, density=True, alpha=0.5, label='Background (SM)', color='royalblue', range=(0, limit))
        plt.hist(score_sig, bins=100, density=True, alpha=0.6, label='Signal (BSM)', color='crimson', range=(0, limit))

        plt.xlabel('Reconstruction Loss (MSE)', fontsize=15)
        plt.ylabel('Probability Density', fontsize=15)
        plt.legend(frameon=True)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(join(self.save_path, 'loss_histogram.png'), dpi=300)
        plt.close()

    def plot_roc_curve(self, score_bg, score_sig):
        """Erstellt die ROC-Kurve und berechnet AUC."""
        y_true = np.concatenate([np.zeros(len(score_bg)), np.ones(len(score_sig))])
        y_scores = np.concatenate([score_bg, score_sig])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AE Model (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=15)
        plt.ylabel('True Positive Rate (TPR)', fontsize=15)
        #plt.title('Anomaly Detection Performance', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(join(self.save_path, 'ROC_Kurve.png'), dpi=300)
        plt.close()

    def plot_learning_curve(self, train_loss_path, val_loss_path):
        """Plottet die Trainings- und Validierungsverluste."""
        train_losses = np.load(train_loss_path)
        val_losses = np.load(val_loss_path)

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training', color='#1f77b4', lw=2)
        plt.plot(val_losses, label='Validation', color='#ff7f0e', lw=2)

        #plt.yscale('log')
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss (MSE)', fontsize=15)
        #plt.title('Autoencoder Training Progress')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(join(self.save_path, 'learning_curve.png'), bbox_inches='tight')
        plt.close()


    def export_to_hls(self, X_test_bkg, X_test_sig, backend='vitis', target='xcvu9p-flga2104-2L-e'):
        print("Starting hls4ml conversion...")
        
        self.model.eval()
        self.model.to("cpu")
        output_dir = join(self.save_path, "hls_project")
        os.makedirs(output_dir, exist_ok=True)
        
        
        # hls4ml config
        hls_config = config_from_pytorch_model(
            self.model,
            input_shape=(None, self.n_inputs), 
            granularity='name',
            backend=backend,
            transpose_outputs=True
        )
        

        # convert model
        hls_model = convert_from_pytorch_model(
            self.model,
            io_type='io_parallel',
            output_dir=output_dir,
            backend=backend,
            hls_config=hls_config,
            part=target,
        )
        
        hls_model.compile()
        
        # Background
        X_bkg_c = np.ascontiguousarray(X_test_bkg).astype(np.float32)
        p_hls_bkg = hls_model.predict(X_bkg_c)
        score_bkg_hls = np.mean((X_bkg_c - p_hls_bkg)**2, axis=-1)
        # Signal
        X_sig_c = np.ascontiguousarray(X_test_sig).astype(np.float32)
        p_hls_sig = hls_model.predict(X_sig_c)
        score_sig_hls = np.mean((X_sig_c - p_hls_sig)**2, axis=-1)

        # PyTorch Predictions
        with torch.no_grad():
            p_torch_bkg = self.model(torch.from_numpy(X_bkg_c)).numpy()
            p_torch_sig = self.model(torch.from_numpy(X_sig_c)).numpy()
            
        score_bkg_torch = np.mean((X_bkg_c - p_torch_bkg)**2, axis=-1)
        score_sig_torch = np.mean((X_sig_c - p_torch_sig)**2, axis=-1)

        # Result Print
        print(f"\nVergleich (Mean MSE):")
        print(f"Bkg -> PyTorch: {np.mean(score_bkg_torch):.6f} | HLS (C-Sim): {np.mean(score_bkg_hls):.6f}")
        print(f"Sig -> PyTorch: {np.mean(score_sig_torch):.6f} | HLS (C-Sim): {np.mean(score_sig_hls):.6f}")

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 5)) # Höhe auf 5 reduziert, da 1x2 Grid
        
        # Plot für die Hardware-Ergebnisse (HLS)
        ax[0].hist(score_bkg_hls, bins=50, alpha=0.5, label='Bkg', density=True)
        ax[0].hist(score_sig_hls, bins=50, alpha=0.5, label='Sig', density=True)
        ax[0].set_xlim(0, 6)
        ax[0].set_xlabel("Score", fontsize=12)
        ax[0].set_ylabel("Probability Density", fontsize=12)
        ax[0].legend()
        ax[0].set_title("HLS Hardware Performance (C-Sim)")

        # Plot für die Software-Ergebnisse (Torch)
        ax[1].hist(score_bkg_torch, bins=50, alpha=0.5, label='Bkg', density=True)
        ax[1].hist(score_sig_torch, bins=50, alpha=0.5, label='Sig', density=True)
        ax[1].set_xlim(0, 6)
        ax[1].set_xlabel("Score", fontsize=12)
        ax[1].set_ylabel("Probability Density", fontsize=12)
        ax[1].legend()
        ax[1].set_title("PyTorch Software Performance")

        plt.tight_layout()
        plt.savefig(join(self.save_path, 'hls_comparison.png'))
        plt.close() 

        self.model.to(self.device)
        return score_bkg_hls, score_sig_hls, score_bkg_torch, score_sig_torch