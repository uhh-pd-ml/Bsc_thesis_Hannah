from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import h5py
import numpy as np
from ae_basis import Autoencoder
import torch
import random

# Daten aus den H5-Files laden
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    kinematics = f['jet_kinematics'][:]
    jettiness = f['jettiness'][:]
    
# Jettiness und kinematics kombinieren ([:, useful_indices] einfügen 3x)
useful_indices = [0,1,2,5,6,9,11,18,22]
#useful_indices = [0,1,2,3,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
X_train_raw = np.concatenate([kinematics, jettiness], axis=1)[:, useful_indices]


# Delta phi berechnen und anhängen
dphi = np.abs(kinematics[:, 4] - kinematics[:, 8])
dphi[dphi > np.pi] = 2 * np.pi - dphi[dphi > np.pi]
dphi = dphi.reshape(-1, 1)
X_train_raw = np.concatenate([X_train_raw, dphi], axis=1)

# Skalieren
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)

# seed setzen
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# Architektur
#my_layers = [12, 6, 2, 6, 12, 26]
#my_layers=[32, 16, 2, 16, 32, 26]
my_layers=[16, 8, 2, 8, 16, 10]

ae_model = Autoencoder(
    n_inputs=10,
    layers=my_layers,
    lr=0.001,
    batch_size=256,
    epochs=50,
    early_stopping=True,
    patience=5,
    verbose=True,
    save_path="ae_data" 
)

# Training
ae_model.fit(X_train)

# Lade Test-Daten (b_sr und s_sr)
# Background in Signal Region
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    kin_b_sr = f['jet_kinematics'][:]
    jet_b_sr = f['jettiness'][:]

# Signal in Signal Region
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    kin_s_sr = f['jet_kinematics'][:]
    jet_s_sr = f['jettiness'][:]


X_b_sr_raw = np.concatenate([kin_b_sr, jet_b_sr], axis=1)[:, useful_indices]
X_s_sr_raw = np.concatenate([kin_s_sr, jet_s_sr], axis=1)[:, useful_indices]

X_b_sr_scaled = scaler.transform(X_b_sr_raw)
X_s_sr_scaled = scaler.transform(X_s_sr_raw)

# Berechne den Rekonstruktionsfehler (MSE)
score_background = ae_model.predict_proba(X_b_sr_scaled)
score_signal = ae_model.predict_proba(X_s_sr_scaled)

np.save('ae_data/scores_bg.npy', score_background)
np.save('ae_data/scores_sig.npy', score_signal)

print(f"\n--- Ergebnisse ---")
print(f"Number of Background Events (SR): {len(X_b_sr_scaled)}")
print(f"Number of Signal Events (SR):     {len(X_s_sr_scaled)}")
print(f"MSE Background: {np.mean(score_background):.6f}")
print(f"MSE Signal:     {np.mean(score_signal):.6f}")