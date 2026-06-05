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


def to_fixed_point_int16(value, total_bits=16, integer_bits=6):
    """
    Wandelt einen Float in einen 16-Bit Integer für ap_fixed<16,6> um.
    Das entspricht 10 Nachkommastellen (Fractional Bits).
    """
    fractional_bits = total_bits - integer_bits  # 10 Bits
    
    # Skalieren: Wert mit 2^10 multiplizieren und runden
    scaled_value = int(round(value * (2**fractional_bits)))
    
    # Clipping (Sättigung): Verhindert Überläufe außerhalb des Bereichs [-32, 31.99]
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
                
                # --- NEU: LITTLE-ENDIAN BYTE-SWAP ---
                # Beispiel: val = 0x1234
                # (val & 0x00FF) << 8  -> 0x3400
                # (val & 0xFF00) >> 8  -> 0x0012
                # Ergebnis: 0x3412
                val_little_endian = ((val & 0x00FF) << 8) | ((val & 0xFF00) >> 8)
                
                # Jetzt als 4-stelliges Hex-Zeichen formatieren
                hex_features.append(f"{val_little_endian:04x}")
            
            # Da Feature 0 RECHTS stehen muss, drehen wir die Reihenfolge der Features um
            reversed_features = hex_features[::-1]
            
            # Padding für die höchsten Bits (ganz links im Gesamt-String)
            padding_hex = "0" * 24
            
            # Zusammenbauen: [Padding][Feature 25]...[Feature 0]
            final_hex_line = padding_hex + "".join(reversed_features)
            
            f.write(f"{final_hex_line}\n")


# Deine Daten laden
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    all_features_b = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    all_features_s = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)


X_b_sr_scaled = scaler.transform(all_features_b)
X_s_sr_scaled = scaler.transform(all_features_s)

# Exportieren
export_to_vivado_hex(X_b_sr_scaled, '/beegfs/u/bbd1146/daten_26F/test_daten_b_shift.txt')
export_to_vivado_hex(X_s_sr_scaled, '/beegfs/u/bbd1146/daten_26F/test_daten_s_shift.txt')


# --- ANGEPASSTER SANITY CHECK ---
first_event_scaled = X_s_sr_scaled[0]

with open('/beegfs/u/bbd1146/daten_26F/test_daten_s_shift.txt', 'r') as f:
    hex_line = f.readline().strip()

print("--- Sanity Check (Little-Endian) ---")
for i in range(3):
    start = -4 * (i + 1)
    end = -4 * i if i > 0 else None
    hex_val = hex_line[start:end] 
    
    # Hex-Wert aus der Datei lesen
    val_le = int(hex_val, 16)
    
    # --- NEU: Rücktransformation für den Sanity Check ---
    # Bytes wieder zurückdrehen, um den korrekten Integer-Wert zu berechnen
    val_int = ((val_le & 0x00FF) << 8) | ((val_le & 0xFF00) >> 8)
    
    # Zweierkomplement auflösen
    if val_int >= 0x8000:   
        val_int -= 0x10000

    reconstructed_float = val_int / 1024.0
    print(f"Feature {i}: Datei-Hex={hex_val} ({reconstructed_float:.4f}), Original={first_event_scaled[i]:.4f}")