from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import h5py
import numpy as np
from ae_basis import Autoencoder

# Daten aus den H5-Files laden
with h5py.File('/beegfs/u/bbd1146/daten_test2/events_b_br.h5', 'r') as f:
    kinematics = f['jet_kinematics'][:]
    jettiness = f['jettiness'][:]
    
# Jettiness und kinematics kombinieren
X_train_raw = np.concatenate([kinematics, jettiness], axis=1)

# Skalieren
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)

# Architektur
my_layers = [16, 8, 4, 8, 16, 22] 

ae_model = Autoencoder(
    n_inputs=22,
    layers=my_layers,
    lr=0.001,
    batch_size=256,
    epochs=50,
    early_stopping=True,
    patience=5,
    verbose=True,
    save_path="mein_erster_ae" # Hier werden die Checkpoints gespeichert
)

# Training
ae_model.fit(X_train)

# Lade Test-Daten (b_sr und s_sr)
# Background in Signal Region
with h5py.File('/beegfs/u/bbd1146/daten_test2/events_b_sr.h5', 'r') as f:
    kin_b_sr = f['jet_kinematics'][:]
    jet_b_sr = f['jettiness'][:]

# Signal in Signal Region
with h5py.File('/beegfs/u/bbd1146/daten_test2/events_s_sr.h5', 'r') as f:
    kin_s_sr = f['jet_kinematics'][:]
    jet_s_sr = f['jettiness'][:]


X_b_sr_scaled = scaler.transform(np.concatenate([kin_b_sr, jet_b_sr], axis=1))
X_s_sr_scaled = scaler.transform(np.concatenate([kin_s_sr, jet_s_sr], axis=1))

# Berechne den Rekonstruktionsfehler (MSE)
score_background = ae_model.predict_proba(X_b_sr_scaled)
score_signal = ae_model.predict_proba(X_s_sr_scaled)

print(f"\n--- Ergebnisse ---")
print(f"Anzahl Background Events (SR): {len(score_background)}")
print(f"Anzahl Signal Events (SR):     {len(score_signal)}")
print(f"Mittlerer Fehler Background: {np.mean(score_background):.6f}")
print(f"Mittlerer Fehler Signal:     {np.mean(score_signal):.6f}")

#ROC-Kurve
y_true = np.concatenate([np.zeros(len(score_background)), np.ones(len(score_signal))])
y_scores = np.concatenate([score_background, score_signal])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Zufallsklassifikator')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('ROC_Kurve.png')