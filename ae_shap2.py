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

# 1. PARAMETER (Index 13 wurde entfernt: "Jet 3 m")
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
    # r"$\text{Jet 3}\ m$",  <-- Das war das ursprüngliche 13. Feature (Index 13)
    
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


# 2. DATEN LADEN, FILTERN & SKALIEREN
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    X_train_raw = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    X_s_raw = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

# --- NEU: Index 13 (das 14. Element) aus den Arrays löschen ---
# Da Python bei 0 anfängt zu zählen, ist das 13. Feature genau Index 13.
X_train_raw = np.delete(X_train_raw, 13, axis=1)
X_s_raw = np.delete(X_s_raw, 13, axis=1)

# Erst danach den Scaler fitten und transformieren
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_s_scaled = scaler.transform(X_s_raw)

###### Quantization & Training Config #######
config = dst_config()
config.pruning_parameters.enable_pruning = False 
config.training_parameters.epochs = 50

config.quantization_parameters.default_data_integer_bits = 3.
config.quantization_parameters.default_data_fractional_bits = 5.
config.quantization_parameters.default_weight_integer_bits = 1.
config.quantization_parameters.default_weight_fractional_bits = 6.
config.quantization_parameters.overflow_mode_data = "SAT"
config.quantization_parameters.overflow_mode_parameters = "SAT_SYM"
config.quantization_parameters.use_high_granularity_quantization = True
config.quantization_parameters.hgq_beta = 1e-6
config.quantization_parameters.hgq_gamma = 1e-6


# 3. MODELL LADEN (n_inputs und die äußeren Layer von 26 auf 25 reduziert)
ae_model = Autoencoder(
    config=config,
    n_inputs=25,                           # <-- Geändert auf 25
    layers=[25, 18, 12, 2, 12, 18, 25],    # <-- Äußere Schichten geändert auf 25
    save_path="AE_models"
)

# WICHTIG: Da dein Checkpoint feste Gewichte für 26 Inputs hat, 
# sorgt das in ae_qubasis.py genutzte strict=False dafür, dass das Modell trotz 
# des fehlenden Gewichtsvektors für das 13. Feature geladen werden kann.
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
X_train_ref = scaler.transform(X_train_raw[:1000])
explainer = shap.KernelExplainer(map_model_to_mse, X_train_ref)

# SHAP-Werte berechnen
shap_values = explainer.shap_values(X_s_scaled[:50])

# 6. PLOT SPEICHERN
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_s_scaled[:50], feature_names=feature_names, plot_type="violin", show=False)
plt.savefig("/beegfs/u/bbd1146/shap_summary.png", bbox_inches='tight')