import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import h5py
import torch
import shap
from sklearn.preprocessing import StandardScaler
from ae_basis import Autoencoder



score_bg = np.load('ae_data/scores_bg.npy')
score_sig = np.load('ae_data/scores_sig.npy')

#Loss Histogram
plt.figure(figsize=(8, 6))
plt.hist(score_bg, bins=100, density=True, alpha=0.5, 
         label='Background (QCD)', color='royalblue', range=(0, np.percentile(score_sig, 98)))

plt.hist(score_sig, bins=100, density=True, alpha=0.6, 
         label='Signal (BSM)', color='crimson', range=(0, np.percentile(score_sig, 98)))

#plt.yscale('log')
plt.xlabel('Autoencoder Reconstruction Loss (MSE)')
plt.ylabel('Probability Density (normalized)')
#plt.title('Anomaly Detection: Loss Distribution')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig('ae_data/loss_histogram.png')



# ROC-Kurve
y_true = np.concatenate([np.zeros(len(score_bg)), np.ones(len(score_sig))])
y_scores = np.concatenate([score_bg, score_sig])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('ae_data/ROC_Kurve.png')



# Learing curve
train_losses = np.load("ae_data/CLSF_train_losses.npy")
val_losses = np.load("ae_data/CLSF_val_losses.npy")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)

plt.title('Autoencoder Learning Curve', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig("ae_data/learning_curve.png", bbox_inches='tight')



# Shap Plot
# Parameter
feature_names = [
    "Mjj", "Delta_Eta_jj",                   # Event-Ebene (0, 1)
    "Jet1_pT", "Jet1_Eta", "Jet1_Phi", "Jet1_Mass",  # Jet 1 (2, 3, 4, 5)
    "Jet2_pT", "Jet2_Eta", "Jet2_Phi", "Jet2_Mass",  # Jet 2 (6, 7, 8, 9)
    "Jet3_pT", "Jet3_Eta", "Jet3_Phi", "Jet3_Mass",  # Jet 3 (10, 11, 12, 13)
    "Tau_Global_1", "Tau_Global_2", "Tau_Global_3","Tau_Global_4",  # Global Jettiness 
    "Tau1_jet1", "Tau2_jet1", "Tau3_jet1", "Tau4_jet1",  # Substruktur 
    "Tau1_jet2", "Tau2_jet2", "Tau3_jet2", "Tau4_jet2"  # Substruktur 
]

# 2. Load and skale data
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    X_train_raw = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

scaler = StandardScaler()
scaler.fit(X_train_raw)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    X_s_raw = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

X_s_scaled = scaler.transform(X_s_raw)

# load model
ae_model = Autoencoder(
    n_inputs=26,
    layers=[26, 18, 12, 2, 12, 18, 26],
    save_path="AE_models"
)

ae_model._load_model("ae_data/AE_models/CLSF_epoch_35.par") 
ae_model.model.eval() 

# shap function
def map_model_to_mse(x_np):
    x_tensor = torch.tensor(x_np).float()
    with torch.no_grad():
        reco = ae_model.model(x_tensor)
        mse = torch.mean((x_tensor - reco)**2, dim=1)
    return mse.numpy()

X_train_ref = scaler.transform(X_train_raw[:1000])
explainer = shap.KernelExplainer(map_model_to_mse, X_train_ref)
shap_values = explainer.shap_values(X_s_scaled[:50])

plt.figure(figsize=(12, 12))
shap.summary_plot(shap_values, X_s_scaled[:50], feature_names=feature_names, plot_type="violin", max_display=len(feature_names), show=False)
plt.savefig("ae_data/shap_summary.png", bbox_inches='tight')