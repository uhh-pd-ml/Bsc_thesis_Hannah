import h5py
import numpy as np
import struct
from sklearn.preprocessing import StandardScaler
import os

# 1. StandardScaler mit den b-br Daten fitten
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    kinematics = f['jet_kinematics'][:]
    jettiness = f['jettiness'][:]

X_train_raw = np.concatenate([kinematics, jettiness], axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)


def to_fixed_point_int16(value, total_bits=16, integer_bits=5):
    fractional_bits = total_bits - integer_bits  # 11 Bits
    
    # Skalieren: Wert mit 2^11 multiplizieren und runden
    scaled_value = int(round(value * (2**fractional_bits)))
    
    # Clipping (Sättigung): Verhindert Überläufe außerhalb des Bereichs [-16, 15.99]
    max_val = (1 << (total_bits - 1)) - 1  # 32767
    min_val = -(1 << (total_bits - 1))     # -32768
    scaled_value = max(min(scaled_value, max_val), min_val)
    
    # Zweierkomplement für negative Zahlen (16-bit Maske)
    if scaled_value < 0:
        scaled_value = (1 << total_bits) + scaled_value
    
    return scaled_value & 0xFFFF


def export_to_vivado_hex(data_scaled, filename):
    with open(filename, 'w') as f:
        for row in data_scaled:
            hex_features = []
            
            for x in row:
                val = to_fixed_point_int16(x)
                hex_features.append(f"{val:04x}")
            
            # Feature-Reihenfolge umdrehen (F0 steht ganz rechts)
            reversed_features = hex_features[::-1]
            
            # Padding hinzufügen (24 Hex-Nullen für 12 Bytes) ganz links
            padding_hex = "0" * 24
            
            # Die komplette 128-Zeichen-Zeile temporär zusammenbauen
            final_hex_line = padding_hex + "".join(reversed_features)
            
            # NEU: In 16 Blöcke zu je 8 Zeichen aufteilen
            blocks = [final_hex_line[i:i+8] for i in range(0, len(final_hex_line), 8)]
            
            # NEU: Blöcke komplett umdrehen (hinterster Block nach ganz oben)
            reversed_blocks = blocks[::-1]
            
            # NEU: Jeden Block als eigene Zeile in das File schreiben
            for block in reversed_blocks:
                f.write(f"{block}\n")


# Deine Daten laden
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    all_features_b = np.concatenate([f['jet_kinematics'][:1000], f['jettiness'][:1000]], axis=1)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    all_features_s = np.concatenate([f['jet_kinematics'][:1000], f['jettiness'][:1000]], axis=1)

# Skalieren
X_b_sr_scaled = scaler.transform(all_features_b)
X_s_sr_scaled = scaler.transform(all_features_s)

# Exportieren (Jedes Event verbraucht nun genau 16 Zeilen im Textfile)
export_to_vivado_hex(X_b_sr_scaled, '/beegfs/u/bbd1146/daten_26F/test_daten_b_n.txt')
export_to_vivado_hex(X_s_sr_scaled, '/beegfs/u/bbd1146/daten_26F/test_daten_s_n.txt')