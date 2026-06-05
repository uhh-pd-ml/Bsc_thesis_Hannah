from datetime import datetime
from sklearn.preprocessing import StandardScaler
import h5py
import os
import numpy as np
from ae_basis import Autoencoder

##### Initialization & Paths #####
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_PATH = os.path.join("/beegfs/u/bbd1146/ae_data_output", f"run_{timestamp}")

N_INPUTS = 60
LAYERS = [60, 40, 20, 2, 20, 40, 60]
#LAYERS = [10, 8, 4, 2, 4, 8, 10]


###### Data Preparation #######
# Load background (b_br) training data
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    j1_train = f['jet1_PFCands'][:, :30] 
    j2_train = f['jet2_PFCands'][:, :30]

X_train_raw = np.concatenate([j1_train, j2_train], axis=1)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)

# Ensure reproducibility
Autoencoder.set_seed(42)


##### Model Definition ######
ae_model = Autoencoder(
    n_inputs=N_INPUTS,
    layers=LAYERS,
    lr=0.001,
    batch_size=256,
    epochs=50,
    early_stopping=True,
    patience=5,
    verbose=True,
    save_path=RUN_PATH
)


###### Training ######
ae_model.fit(X_train)


###### Testing #######
# Load Signal Region (SR) data for both Background (Normal) and Signal (Anomaly)
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    j1_b_sr = f['jet1_PFCands'][:, :30]
    j2_b_sr = f['jet2_PFCands'][:, :30]
X_b_sr_raw = np.concatenate([j1_b_sr, j2_b_sr], axis=1)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    j1_s_sr = f['jet1_PFCands'][:, :30]
    j2_s_sr = f['jet2_PFCands'][:, :30]
X_s_sr_raw = np.concatenate([j1_s_sr, j2_s_sr], axis=1)

X_b_sr_scaled = scaler.transform(X_b_sr_raw)
X_s_sr_scaled = scaler.transform(X_s_sr_raw)

# Calculate and save Reconstruction loss (MSE)
score_background = ae_model.predict_proba(X_b_sr_scaled)
score_signal = ae_model.predict_proba(X_s_sr_scaled)

np.save(os.path.join(RUN_PATH, 'scores_bg.npy'), score_background)
np.save(os.path.join(RUN_PATH, 'scores_sig.npy'), score_signal)

# Results
print(f"\n--- Results ---")
print(f"Number of Background Events (SR): {len(X_b_sr_scaled)}")
print(f"Number of Signal Events (SR):     {len(X_s_sr_scaled)}")
print(f"MSE Background: {np.mean(score_background):.6f}")
print(f"MSE Signal:     {np.mean(score_signal):.6f}")


###### Visualization & Analysis #######
# Generate reconstruction plots, error histograms, and ROC curves
ae_model.plot_loss_histogram(score_background, score_signal)
ae_model.plot_roc_curve(score_background, score_signal)
ae_model.plot_learning_curve(f"{RUN_PATH}/CLSF_train_losses.npy", f"{RUN_PATH}/CLSF_val_losses.npy")