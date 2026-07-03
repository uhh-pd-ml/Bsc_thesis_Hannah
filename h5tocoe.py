import h5py
import numpy as np
import struct
from sklearn.preprocessing import StandardScaler
import os

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    kinematics = f['jet_kinematics'][:]
    jettiness = f['jettiness'][:]

X_train_raw = np.concatenate([kinematics, jettiness], axis=1)

# Standardization
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


def export_to_vivado_coe(data_scaled, filename):
    """Exportiert die skalierten Daten im Vivado COE-Format (Radix 16, 512-bit breit)."""
    with open(filename, 'w') as f:
        # Schritt 1: Den COE-Header schreiben
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        
        lines = []
        for row in data_scaled:
            hex_features = []
            for x in row:
                val = to_fixed_point_int16(x)
                hex_features.append(f"{val:04x}")
            
            # Da Feature 0 RECHTS stehen muss, drehen wir die Reihenfolge um
            reversed_features = hex_features[::-1]
            
            # 12 Null-Bytes Padding = 24 Hex-Nullen ganz links (höchste Bits)
            padding_hex = "0" * 24
            
            # 512-Bit Gesamt-Hex-String zusammenbauen
            final_hex_line = padding_hex + "".join(reversed_features)
            lines.append(final_hex_line)
        
        # Schritt 2: Die Zeilen mit Kommata trennen, die letzte Zeile mit Semikolon beenden
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                f.write(f"{line},\n")
            else:
                f.write(f"{line};\n") # Letzter Eintrag endet mit Semikolon


# Deine Daten laden
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    all_features_b = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    all_features_s = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)


X_b_sr_scaled = scaler.transform(all_features_b)
X_s_sr_scaled = scaler.transform(all_features_s)

# Exportieren
export_to_vivado_coe(X_b_sr_scaled[:3000], '/beegfs/u/bbd1146/daten_26F/test_daten_b.coe')
export_to_vivado_coe(X_s_sr_scaled[:3000], '/beegfs/u/bbd1146/daten_26F/test_daten_s.coe')