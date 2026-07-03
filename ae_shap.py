import os
os.environ["KERAS_BACKEND"] = "torch" 

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from sklearn.preprocessing import StandardScaler
from ae_qubasis import Autoencoder
from pquant import dst_config

# 1. PARAMETER
feature_names = [
    # --- Event-Ebene ---
    r"$M_{jj}$", 
    r"$\Delta\eta_{jj}$",
    
    # --- Jet 1 Kinematik ---
    r"$\text{Jet 1}\ p_T$", 
    r"$\text{Jet 1}\ \eta$", 
    r"$\text{Jet 1}\ \phi$", 
    r"$\text{Jet 1}\ m$",
    
    # --- Jet 2 Kinematik ---
    r"$\text{Jet 2}\ p_T$", 
    r"$\text{Jet 2}\ \eta$", 
    r"$\text{Jet 2}\ \phi$", 
    r"$\text{Jet 2}\ m$",
    
    # --- Jet 3 Kinematik ---
    r"$\text{Jet 3}\ p_T$", 
    r"$\text{Jet 3}\ \eta$", 
    r"$\text{Jet 3}\ \phi$", 
    r"$\text{Jet 3}\ m$",
    
    # --- Global N-Jettiness ---
    r"$\text{Global}\ \tau_{1}$", 
    r"$\text{Global}\ \tau_{2}$", 
    r"$\text{Global}\ \tau_{3}$", 
    r"$\text{Global}\ \tau_{4}$",
    
    # --- Jet 1 Substruktur ---
    r"$\text{Jet 1}\ \tau_{1}$", 
    r"$\text{Jet 1}\ \tau_{2}$", 
    r"$\text{Jet 1}\ \tau_{3}$", 
    r"$\text{Jet 1}\ \tau_{4}$",
    
    # --- Jet 2 Substruktur ---
    r"$\text{Jet 2}\ \tau_{1}$", 
    r"$\text{Jet 2}\ \tau_{2}$", 
    r"$\text{Jet 2}\ \tau_{3}$", 
    r"$\text{Jet 2}\ \tau_{4}$"
]


# 2. DATEN LADEN & SKALIEREN
# Wir laden kurz die Trainingsdaten, um den Scaler exakt gleich zu fitten
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    X_train_raw = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

scaler = StandardScaler()
scaler.fit(X_train_raw)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    X_s_raw = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

# Skalieren und Filtern
X_s_scaled = scaler.transform(X_s_raw)

###### Quantization & Training Config #######
config = dst_config()
config.pruning_parameters.enable_pruning = False # Set to True for DST
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
config.quantization_parameters.hgq_beta = 1e-6
config.quantization_parameters.hgq_gamma = 1e-6


# 3. MODELL LADEN
ae_model = Autoencoder(
    config=config,
    n_inputs=26,
    layers=[26, 18, 12, 2, 12, 18, 26],
    save_path="AE_models"
)

ae_model._load_model("/beegfs/u/bbd1146/ae_data_output/run_20260608_144331/AE_models/CLSF_epoch_49.par") 
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
plt.savefig("/beegfs/u/bbd1146/shap_summary.png", bbox_inches='tight')