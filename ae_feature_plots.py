import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


with h5py.File('/beegfs/u/bbd1146/daten/events_b_sr.h5', 'r') as f:
    kin_b_sr = f['jet_kinematics'][:]
    jet_b_sr = f['jettiness'][:]


with h5py.File('/beegfs/u/bbd1146/daten/events_s_sr.h5', 'r') as f:
    kin_s_sr = f['jet_kinematics'][:]
    jet_s_sr = f['jettiness'][:]

# Jettiness und kinematics kombinieren
all_features_b = np.concatenate([kin_b_sr, jet_b_sr], axis=1)
all_features_s = np.concatenate([kin_s_sr, jet_s_sr], axis=1)


# Namen für die Achsen
feature_names = [
    "Mjj", "Delta_Eta_jj",                   # Event-Ebene (0, 1)
    "Jet1_pT", "Jet1_Eta", "Jet1_Phi", "Jet1_Mass",  # Jet 1 (2, 3, 4, 5)
    "Jet2_pT", "Jet2_Eta", "Jet2_Phi", "Jet2_Mass",  # Jet 2 (6, 7, 8, 9)
    "Jet3_pT", "Jet3_Eta", "Jet3_Phi", "Jet3_Mass",  # Jet 3 / Zusatz (10, 11, 12, 13)
    "Tau_Global_1", "Tau_Global_2", "Tau_Global_3","Tau_Global_4",  # Global Jettiness 
    "Tau1", "Tau2", "Tau3", "Tau4"  # Substruktur 
]

groups = [
    {"indices": [0, 1], "title": "Mjj_Delta_Eta"},
    {"indices": [2, 3, 4, 5], "title": "Jet 1 Kinematik"},
    {"indices": [6, 7, 8, 9], "title": "Jet 2 Kinematik"},
    {"indices": [10, 11, 12, 13], "title": "Jet 3 Kinematik"},
    {"indices": [14, 15, 16, 17], "title": "Global Jettiness"},
    {"indices": [18, 19, 20, 21], "title": "Subjettiness"}
]


for group in groups:
    indices = group["indices"]
    n = len(indices)
    
    # Erstelle ein Bild pro Gruppe
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(group["title"], fontsize=16, fontweight='bold')
    
    # Falls nur ein Index in der Gruppe wäre (z.B. bei ungerader Zahl)
    if n == 1: axes = [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        # Histogramme plotten (normiert auf Dichte für besseren Vergleich)
        ax.hist(all_features_b[:, idx], bins=40, alpha=0.5, label='Background', color='blue', density=False)
        ax.hist(all_features_s[:, idx], bins=40, alpha=0.5, label='Signal', color='red', density=False)
        
        ax.set_title(f"Feature {idx}: {feature_names[idx]}")
        ax.legend(fontsize='small')
        ax.grid(alpha=0.3)
        #ax.set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join("plots_input_signal_odichte", f"plot_{group['title']}.png")
    plt.savefig(save_path, dpi=150)
    plt.close() 


