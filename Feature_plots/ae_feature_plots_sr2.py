import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

# Daten laden
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    kin_b_sr = f['jet_kinematics'][:]
    jet_b_sr = f['jettiness'][:]

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    kin_s_sr = f['jet_kinematics'][:]
    jet_s_sr = f['jettiness'][:]

# Jettiness und Kinematics kombinieren
all_features_b = np.concatenate([kin_b_sr, jet_b_sr], axis=1)
all_features_s = np.concatenate([kin_s_sr, jet_s_sr], axis=1)

# Namen für die Achsen
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

# Zielverzeichnis erstellen
output_dir = "/beegfs/u/bbd1146/plots_gesamt"
os.makedirs(output_dir, exist_ok=True)

# Parameter für das große Raster
nrows = 7
ncols = 4

# Großes Bild erstellen
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 24))
#fig.suptitle("All Features: Signal vs. Background (SR)", fontsize=20, fontweight='bold', y=0.99)

axes_flat = axes.flatten()

# Zähler für das tatsächliche Feature aus der Datenmatrix
feature_idx = 0

for grid_idx in range(nrows * ncols):
    ax = axes_flat[grid_idx]
    
    if grid_idx == 3 or grid_idx == 2:
        ax.axis('off')
        continue
        
    # Wenn wir noch Features zum Plotten haben
    if feature_idx < len(feature_names):
        # Histogramme plotten 
        ax.hist(all_features_b[:, feature_idx], bins=40, alpha=0.5, label='Background', color='blue', density=True)
        ax.hist(all_features_s[:, feature_idx], bins=40, alpha=0.5, label='Signal', color='red', density=True)
        
        ax.set_title(f"Feature {feature_idx}: {feature_names[feature_idx]}", fontsize=15)
        ax.legend(fontsize='xx-small', loc='upper right')
        #ax.grid(alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
        # ax.set_yscale('log')
        
        # Nur hochzählen, wenn wir auch wirklich ein Feature geplottet haben
        feature_idx += 1
    else:
        # Falls am Ende noch Plätze im Grid übrig wären
        ax.axis('off')

# Layout optimieren
plt.tight_layout()

# Speichern des Gesamtbildes
save_path = os.path.join(output_dir, "all_features_grid_spaced.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()