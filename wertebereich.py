import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

useful_indices = list(range(26))

# Background
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    X_b_br = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)[:, useful_indices]

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    X_b_sr = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)[:, useful_indices]

# Signal
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    X_s_sr = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)[:, useful_indices]

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_br.h5', 'r') as f:
    X_s_br = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)[:, useful_indices]


# 2. StandardScaler auf die Gesamtmenge oder die BR-Baseline fitten
# Wir fitten auf der Baseline (b_br) und transformieren alle vier konsistent
scaler = StandardScaler()
scaler.fit(X_b_br)

X_b_br_trans = scaler.transform(X_b_br)
X_b_sr_trans = scaler.transform(X_b_sr)
X_s_sr_trans = scaler.transform(X_s_sr)
X_s_br_trans = scaler.transform(X_s_br)


# 3. Plot erstellen
feature_to_plot = 0  # Index des Features, das du beispielhaft zeigen willst
plt.figure(figsize=(11, 6))

# Histogramme für alle 4 Bereiche (Verwendung von 'step' für bessere Sichtbarkeit bei Overlaps)
plt.hist(X_b_br_trans[:, feature_to_plot], bins=50, histtype='step', linewidth=2, label='Background (BR)', density=True)
plt.hist(X_b_sr_trans[:, feature_to_plot], bins=50, histtype='step', linewidth=2, label='Background (SR)', density=True)
plt.hist(X_s_sr_trans[:, feature_to_plot], bins=50, histtype='step', linewidth=2, label='Signal (SR)', density=True)
plt.hist(X_s_br_trans[:, feature_to_plot], bins=50, histtype='step', linewidth=2, label='Signal (BR)', density=True)

# 4. Die Hardware-Grenzen einzeichnen
#plt.axvline(x=-15, color='crimson', linestyle='--', linewidth=2.5, label='Hardware Limit ($i=4$)')
#plt.axvline(x=15, color='crimson', linestyle='--', linewidth=2.5)

# Optisches Finetuning
#plt.title('Standardized Feature Distribution across all 4 Regions vs. Hardware Limits', fontsize=12, pad=15)
plt.xlabel('Standardized Value (Z-Score)', fontsize=15)
plt.ylabel('Probability Density', fontsize=15)

# Fokus auf das Zentrum, aber die Grenzen bei +-15 voll sichtbar halten
plt.xlim(-5, 5) 
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.savefig('/beegfs/u/bbd1146/daten_26F/all_4_regions_quantization_limits.png', dpi=300)