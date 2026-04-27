from datetime import datetime
from sklearn.preprocessing import StandardScaler
import h5py
import os
import numpy as np
import torch
from ae_basis import Autoencoder

##### Initialization & Paths #####
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_PATH = os.path.join("/beegfs/u/bbd1146/ae_data_output", f"run_{timestamp}")
os.makedirs(RUN_PATH, exist_ok=True)

N_INPUTS = 26
LAYERS = [26, 18, 12, 2, 12, 18, 26]

###### 1. Data Preparation (von Regressor-Outputs) #######

X_train_regressed = np.load('/beegfs/u/bbd1146/regression/run_20260423_143832/ae_train_bg.npy')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_regressed)

# Ensure reproducibility
Autoencoder.set_seed(42)


##### 2. Model Definition ######
ae_model = Autoencoder(
    n_inputs=N_INPUTS,
    layers=LAYERS,
    lr=0.001,
    batch_size=256,
    epochs=100, 
    early_stopping=True,
    patience=10,
    verbose=True,
    save_path=RUN_PATH
)


###### 3. Training ######

ae_model.fit(X_train)


###### 4. Testing #######

X_b_sr_regressed = np.load('/beegfs/u/bbd1146/regression/run_20260423_143832/ae_test_bg.npy') 
X_s_sr_regressed = np.load('/beegfs/u/bbd1146/regression/run_20260423_143832/ae_test_sig.npy') 


X_b_sr_scaled = scaler.transform(X_b_sr_regressed)
X_s_sr_scaled = scaler.transform(X_s_sr_regressed)


score_background = ae_model.predict_proba(X_b_sr_scaled)
score_signal = ae_model.predict_proba(X_s_sr_scaled)


np.save(os.path.join(RUN_PATH, 'scores_bg.npy'), score_background)
np.save(os.path.join(RUN_PATH, 'scores_sig.npy'), score_signal)


###### 5. Evaluation #######
print(f"\n--- Trigger Performance ---")
print(f"MSE Background (Regressor-Input): {np.mean(score_background):.6f}")
print(f"MSE Signal (Regressor-Input):     {np.mean(score_signal):.6f}")


ae_model.plot_loss_histogram(score_background, score_signal)
ae_model.plot_roc_curve(score_background, score_signal)