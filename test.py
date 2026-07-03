import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

useful_indices = list(range(26))
LIMIT = 16  # Definiere hier deine Grenze (z.B. -15 und +15)

# 1. Daten laden
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


# 2. StandardScaler fitten und transformieren
scaler = StandardScaler()
scaler.fit(X_b_br)

datasets = {
    'Background (BR)': scaler.transform(X_b_br),
    'Background (SR)': scaler.transform(X_b_sr),
    'Signal (SR)':     scaler.transform(X_s_sr),
    'Signal (BR)':     scaler.transform(X_s_br)
}

# 3. Auswertung für alle Features
print(f"Überprüfung auf Werte außerhalb von [-{LIMIT}, {LIMIT}]:\n")
print(f"{'Feature':<9} | {'Region':<17} | {'Ausreißer (Absolut)':<20} | {'Anteil (%)':<10}")
print("-" * 65)

for feat_idx in useful_indices:
    for name, data in datasets.items():
        # Extrahiere die Spalte für das aktuelle Feature
        feature_data = data[:, feat_idx]
        total_events = len(feature_data)
        
        # Zähle, wie viele Werte außerhalb der Grenzen liegen
        out_of_bounds = np.sum((feature_data < -LIMIT) | (feature_data > LIMIT))
        percentage = (out_of_bounds / total_events) * 100
        
        # Nur ausgeben, wenn es überhaupt Ausreißer gibt (macht es übersichtlicher)
        # Wenn du immer alle sehen willst, entferne einfach die nächste Zeile
        if out_of_bounds > 0:
            print(f"Feature {feat_idx:<2} | {name:<17} | {out_of_bounds:<20,} | {percentage:.4f}%")
            
    # Kleiner Trenner nach jedem Feature (nur wenn Ausreißer existierten)
    # print("-" * 65)