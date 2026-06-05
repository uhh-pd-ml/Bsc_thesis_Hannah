from datetime import datetime
from sklearn.preprocessing import StandardScaler
import h5py
import os
import numpy as np
from ae_basis import Autoencoder

##### Initialization & Paths #####
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_PATH = os.path.join("/beegfs/u/bbd1146/ae_data_output", f"run_{timestamp}")

N_INPUTS = 26
LAYERS = [26, 18, 12, 2, 12, 18, 26]
#LAYERS = [10, 8, 4, 2, 4, 8, 10]

# Feature Selection
#useful_indices = [0,1,2,3,5,6,9,11,18,22]
useful_indices = list(range(26))


###### Data Preparation #######
# Load background (b_br) training data
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    kinematics = f['jet_kinematics'][:]
    jettiness = f['jettiness'][:]

X_train_raw = np.concatenate([kinematics, jettiness], axis=1)[:, useful_indices]

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
# Background in Signal Region
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    kin_b_sr = f['jet_kinematics'][:]
    jet_b_sr = f['jettiness'][:]
# Signal in Signal Region
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    kin_s_sr = f['jet_kinematics'][:]
    jet_s_sr = f['jettiness'][:]

# Apply the same scaling transformation used for training
X_b_sr_raw = np.concatenate([kin_b_sr, jet_b_sr], axis=1)[:, useful_indices]
X_s_sr_raw = np.concatenate([kin_s_sr, jet_s_sr], axis=1)[:, useful_indices]

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