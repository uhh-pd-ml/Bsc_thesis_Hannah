import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from ae_basis import Autoencoder

# --- KONFIGURATION & PFADE ---
BASE_PATH = "/beegfs/u/bbd1146/ae_data_output"
MODEL_PATH = "/beegfs/u/bbd1146/ae_data_output/AE_models/CLSF_epoch_35.par"
DATA_BG = "/beegfs/u/bbd1146/daten_26F/events_b_br.h5"
DATA_SIG = "/beegfs/u/bbd1146/daten_26F/events_s_sr.h5"

N_INPUTS = 26
LAYERS = [26, 18, 12, 2, 12, 18, 26]

os.makedirs(BASE_PATH, exist_ok=True)
plt.style.use('ggplot')

FEATURE_NAMES = [
    "Mjj", "Delta_Eta_jj",
    "Jet1_pT", "Jet1_Eta", "Jet1_Phi", "Jet1_Mass",
    "Jet2_pT", "Jet2_Eta", "Jet2_Phi", "Jet2_Mass",
    "Jet3_pT", "Jet3_Eta", "Jet3_Phi", "Jet3_Mass",
    "Tau_Global_1", "Tau_Global_2", "Tau_Global_3","Tau_Global_4",
    "Tau1_jet1", "Tau2_jet1", "Tau3_jet1", "Tau4_jet1",
    "Tau1_jet2", "Tau2_jet2", "Tau3_jet2", "Tau4_jet2"
]

# --- PLOT FUNKTIONEN ---

def plot_loss_histogram(score_bg, score_sig, filename="loss_histogram.png"):
    """Erstellt ein Histogramm der Rekonstruktionsverluste."""
    plt.figure(figsize=(9, 6))
    limit = np.percentile(score_sig, 98)
    
    plt.hist(score_bg, bins=100, density=True, alpha=0.5, 
             label='Background (QCD)', color='royalblue', range=(0, limit))
    plt.hist(score_sig, bins=100, density=True, alpha=0.6, 
             label='Signal (BSM)', color='crimson', range=(0, limit))

    plt.xlabel('Autoencoder Reconstruction Loss (MSE)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/{filename}", dpi=300)
    plt.close()

def plot_roc_curve(score_bg, score_sig, filename="ROC_Kurve.png"):
    """Erstellt die ROC-Kurve und berechnet AUC."""
    y_true = np.concatenate([np.zeros(len(score_bg)), np.ones(len(score_sig))])
    y_scores = np.concatenate([score_bg, score_sig])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AE Model (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Anomaly Detection Performance', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f"{BASE_PATH}/{filename}", dpi=300)
    plt.close()

def plot_learning_curve(train_loss_path, val_loss_path, filename="learning_curve.png"):
    """Plottet die Trainings- und Validierungsverluste."""
    train_losses = np.load(train_loss_path)
    val_losses = np.load(val_loss_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training', color='#1f77b4', lw=2)
    plt.plot(val_losses, label='Validation', color='#ff7f0e', lw=2)

    #plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoencoder Training Progress')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(f"{BASE_PATH}/{filename}", bbox_inches='tight')
    plt.close()

def run_shap_analysis(model, scaler, X_sig_raw, X_train_raw, filename="shap_summary.png"):
    """Führt die SHAP-Analyse durch."""
    # Referenzdaten für SHAP (Background-Verteilung)
    X_ref = scaler.transform(X_train_raw[:1000])
    X_test = scaler.transform(X_sig_raw[:50])

    def model_predict_mse(x_np):
        x_tensor = torch.tensor(x_np).float()
        with torch.no_grad():
            reco = model.model(x_tensor)
            # Berechne MSE pro Event für SHAP
            mse = torch.mean((x_tensor - reco)**2, dim=1)
        return mse.numpy()

    explainer = shap.KernelExplainer(model_predict_mse, X_ref)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_test, 
        feature_names=FEATURE_NAMES, 
        plot_type="violin", 
        max_display=len(FEATURE_NAMES), 
        show=False
    )
    plt.title("SHAP Feature Importance (Impact on Reconstruction Loss)")
    plt.savefig(f"{BASE_PATH}/{filename}", bbox_inches='tight', dpi=300)
    plt.close()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Load scores and first plots
    s_bg = np.load(f'{BASE_PATH}/scores_bg.npy')
    s_sig = np.load(f'{BASE_PATH}/scores_sig.npy')
    
    plot_loss_histogram(s_bg, s_sig)
    plot_roc_curve(s_bg, s_sig)
    plot_learning_curve(f"{BASE_PATH}/CLSF_train_losses.npy", f"{BASE_PATH}/CLSF_val_losses.npy")

    # SHAP Plot
    with h5py.File(DATA_BG, 'r') as f:
        X_bg_raw = np.concatenate([f['jet_kinematics'][:2000], f['jettiness'][:2000]], axis=1)
    
    with h5py.File(DATA_SIG, 'r') as f:
        X_sig_raw = np.concatenate([f['jet_kinematics'][:100], f['jettiness'][:100]], axis=1)

    scaler = StandardScaler().fit(X_bg_raw)

    ae = Autoencoder(n_inputs=N_INPUTS, layers=LAYERS, save_path="AE_models")
    ae._load_model(MODEL_PATH)
    ae.model.eval()
    run_shap_analysis(ae, scaler, X_sig_raw, X_bg_raw)