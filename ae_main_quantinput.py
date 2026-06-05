import os
os.environ["KERAS_BACKEND"] = "torch" # Do this before importing Keras to use Torch PQLayers

import torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np
from ae_qubasis import Autoencoder
from pquant import dst_config

##### Initialization & Paths #####
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_PATH = os.path.join("/beegfs/u/bbd1146/ae_data_output", f"run_{timestamp}")

N_INPUTS = 26
LAYERS = [26, 18, 12, 2, 12, 18, 26]
#LAYERS = [10, 8, 4, 2, 4, 8, 10]

# Feature Selection
#useful_indices = [0,1,2,3,5,6,9,11,18,22]
useful_indices = list(range(26))

###### Quantization & Training Config #######
config = dst_config()
config.pruning_parameters.enable_pruning = False # Set to True for Decoupled Structured Tuning (DST)
config.training_parameters.epochs = 50
#config.training_parameters.fine_tuning_epochs = 50

config.quantization_parameters.default_data_integer_bits = 3.
config.quantization_parameters.default_data_fractional_bits = 5.
config.quantization_parameters.default_weight_integer_bits = 1.
config.quantization_parameters.default_weight_fractional_bits = 6.
config.quantization_parameters.overflow_mode_data = "SAT"
config.quantization_parameters.overflow_mode_parameters = "SAT_SYM"
# For HGQ
config.quantization_parameters.use_high_granularity_quantization = True
config.quantization_parameters.hgq_beta = 1e-8
# For DST
#config.pruning_parameters.alpha = 5e-5


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
    config=config,
    n_inputs=N_INPUTS,
    layers=LAYERS,
    lr=0.001,
    batch_size=256,
    epochs=config.training_parameters.epochs,
    early_stopping=True,
    patience=5,
    verbose=True,
    save_path=RUN_PATH
)

###### Training ######
ae_model.fit(X_train)



with h5py.File('/beegfs/u/bbd1146/daten_26F/test_daten_b_int16.h5', 'r') as f:
    kin_b_sr = f['jet_kinematics'][:]
    jet_b_sr = f['jettiness'][:]

# Signal in Signal Region
with h5py.File('/beegfs/u/bbd1146/daten_26F/test_daten_s_int16.h5', 'r') as f:
    kin_s_sr = f['jet_kinematics'][:]
    jet_s_sr = f['jettiness'][:]

# 1. Zusammenfügen der geladenen int16-Arrays
X_b_sr_int16 = np.concatenate([kin_b_sr, jet_b_sr], axis=1)[:, useful_indices]
X_s_sr_int16 = np.concatenate([kin_s_sr, jet_s_sr], axis=1)[:, useful_indices]

# WICHTIG: Rückkonvertierung von int16-Festkomma zu Float32 (durch 2^10 teilen)
X_b_sr_scaled = X_b_sr_int16.astype(np.float32) / 2048.0
X_s_sr_scaled = X_s_sr_int16.astype(np.float32) / 2048.0

# Calculate and save Reconstruction loss (MSE)
score_background = ae_model.predict_proba(X_b_sr_scaled)
score_signal = ae_model.predict_proba(X_s_sr_scaled)

np.save(os.path.join(RUN_PATH, 'scores_bg.npy'), score_background)
np.save(os.path.join(RUN_PATH, 'scores_sig.npy'), score_signal)

# Results
print(f"\n--- Results ---")
print(f"Number of Background Events (SR): {len(X_b_sr_scaled)}")
print(f"Number of Signal Events (SR):     {len(X_s_sr_scaled)}")
print(f"MSE Background (Mittelwert): {np.mean(score_background):.6f}")
print(f"MSE Signal (Mittelwert):     {np.mean(score_signal):.6f}")
print(f"MSE Background (Erstes Event): {score_background[0]:.6f}")
print(f"MSE Signal (Erstes Event):     {score_signal[0]:.6f}")


# transform() liefert die Werte nach dem Durchlauf durch den AE
reconstructed_bkg = ae_model.transform(X_b_sr_scaled)

# 2. Vergleich: Original vs. Rekonstruktion (Nur für das ERSTE Event [0])
print("\n--- Feature Vergleich (Erstes Background Event) ---")
print(f"{'Feature Index':<15} | {'Original (Scaled)':<20} | {'Rekonstruiert':<20}")
print("-" * 60)

for i in range(N_INPUTS):
    orig = X_b_sr_scaled[0, i]
    reco = reconstructed_bkg[0, i]
    print(f"{i:<15} | {orig:<20.6f} | {reco:<20.6f}")



###### Visualization & Analysis #######
# Generate reconstruction plots, error histograms, and ROC curves
ae_model.plot_results(score_background, score_signal)
ae_model.plot_loss_histogram(score_background, score_signal)
ae_model.plot_roc_curve(score_background, score_signal)
ae_model.plot_learning_curve(f"{RUN_PATH}/CLSF_train_losses.npy", f"{RUN_PATH}/CLSF_val_losses.npy")

