# training loop
        self.model.train()
        for epoch in range(self.epochs if self.epochs is not None else 10000):
            print('\nEpoch: {}'.format(epoch))
            pbar = tqdm(total=len(train_loader.dataset))
            epoch_train_loss = 0.
            epoch_val_loss = 0.

            for i, batch in enumerate(train_loader):

                batch_inputs = batch[0]
                batch_inputs = batch_inputs.to(self.device)

                self.optimizer.zero_grad()
                batch_outputs = self.model(batch_inputs)
                reconstruction_loss = self.loss(batch_outputs, batch_inputs)
                loss_2 = get_model_losses(self.model, torch.tensor(0.).to(self.device))
                total_loss = reconstruction_loss + 0.01 * loss_2
                total_loss.backward()
                self.optimizer.step()
                epoch_train_loss += total_loss.item()
                if self.verbose:
                    pbar.update(batch_inputs.size(0))
                    pbar.set_description(
                        "Train loss: {:.6f}".format(
                            epoch_train_loss / (i + 1)))

            epoch_train_loss /= (i + 1)
            if self.verbose:
                pbar.close()

            with torch.no_grad():
                self.model.eval()
                for i, batch in enumerate(val_loader):

                    batch_inputs = batch[0]
                    batch_inputs = batch_inputs.to(self.device)
                    batch_outputs = self.model(batch_inputs)

                    recon_loss_val = self.loss(batch_outputs, batch_inputs)
                    pquant_loss_val = get_model_losses(self.model, torch.tensor(0.).to(self.device))
                    total_val_loss = recon_loss_val + 0.01 * pquant_loss_val
            
                    epoch_val_loss += total_val_loss.item()
                epoch_val_loss /= (i + 1)
                current_ratio = get_layer_keep_ratio(self.model)
                self.history_keep_ratio.append(current_ratio.item())
            print(f"Validation loss: {epoch_val_loss:.6f} | Remaining Weights: {current_ratio:.2%}")

            if epoch == 0:
                train_losses = np.array([epoch_train_loss])
                val_losses = np.array([epoch_val_loss])
            else:
                train_losses = np.concatenate(
                    (train_losses, np.array([epoch_train_loss])))
                val_losses = np.concatenate(
                    (val_losses, np.array([epoch_val_loss])))

            if self.save_path is not None:
                np.save(self._train_loss_path(),
                        train_losses)
                np.save(self._val_loss_path(),
                        val_losses)
                self._save_model(self._model_path(epoch))

            if self.early_stopping:
                if epoch > self.patience:
                    if np.all(val_losses[-self.patience:] >
                              val_losses[-self.patience - 1]):
                        print("Early stopping at epoch", epoch)
                        break

        self.model.eval()
        if self.save_path is not None:
            print("Loading best model state...")
            self.load_best_model()

        return self
    