import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
import os
from sklearn.preprocessing import StandardScaler
from ae_basis import Autoencoder

# 1. PARAMETER
feature_names = [
    "Mjj", "Delta_Eta_jj",                   # Event-Ebene (0, 1)
    "Jet1_pT", "Jet1_Eta", "Jet1_Phi", "Jet1_Mass",  # Jet 1 (2, 3, 4, 5)
    "Jet2_pT", "Jet2_Eta", "Jet2_Phi", "Jet2_Mass",  # Jet 2 (6, 7, 8, 9)
    "Jet3_pT", "Jet3_Eta", "Jet3_Phi", "Jet3_Mass",  # Jet 3 / Zusatz (10, 11, 12, 13)
    "Tau_Global_1", "Tau_Global_2", "Tau_Global_3","Tau_Global_4",  # Global Jettiness 
    "Tau1", "Tau2", "Tau3", "Tau4"  # Substruktur 
]

# 2. DATEN LADEN & SKALIEREN
# Wir laden kurz die Trainingsdaten, um den Scaler exakt gleich zu fitten
with h5py.File('/beegfs/u/bbd1146/daten/events_b_br.h5', 'r') as f:
    X_train_raw = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

scaler = StandardScaler()
scaler.fit(X_train_raw)

with h5py.File('/beegfs/u/bbd1146/daten/events_s_sr.h5', 'r') as f:
    X_s_raw = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

# Skalieren und Filtern
X_s_scaled = scaler.transform(X_s_raw)

# 3. MODELL LADEN
ae_model = Autoencoder(
    n_inputs=22,
    layers=[12, 6, 2, 6, 12, 22],
    save_path="AE_models"
)

ae_model._load_model("trash_ae/mein_erster_ae/AE_models/CLSF_epoch_49.par") 
ae_model.model.eval() 

# 4. SHAP FUNKTION
def map_model_to_mse(x_np):
    x_tensor = torch.tensor(x_np).float()
    with torch.no_grad():
        reco = ae_model.model(x_tensor)
        mse = torch.mean((x_tensor - reco)**2, dim=1)
    return mse.numpy()

# 5. SHAP AUSFÜHREN
# Hintergrund-Daten vorbereiten
X_train_ref = scaler.transform(X_train_raw[:1000])
# Den Explainer erstellen
explainer = shap.KernelExplainer(map_model_to_mse, X_train_ref)

# SHAP-Werte für ein paar Signal-Events berechnen
shap_values = explainer.shap_values(X_s_scaled[:50])

# 6. PLOT SPEICHERN
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_s_scaled[:50], feature_names=feature_names, plot_type="violin", show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')