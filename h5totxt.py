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


def export_to_vivado_hex(data_scaled, filename):
    with open(filename, 'w') as f:
        for row in data_scaled:
            hex_features = []
            
            for x in row:
                val = to_fixed_point_int16(x)
                # Direkt als 4-stelliges Hex-Zeichen formatieren (ohne Little-Endian-Drehung)
                hex_features.append(f"{val:04x}")
            
            # Da Feature 0 RECHTS stehen muss, drehen wir die Reihenfolge der Features um
            # Aus [F0, F1, ..., F25] wird [F25, ..., F1, F0]
            reversed_features = hex_features[::-1]
            
            # Jetzt fügen wir das geforderte Padding hinzu:
            # Du wolltest 12 Null-Bytes Padding (= 24 Hex-Nullen).
            # Da das Padding auf die HÖCHSTEN Bits (ganz links im Gesamt-String) soll,
            # setzen wir die Nullen einfach VORNE an den String dran.
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
export_to_vivado_hex(X_b_sr_scaled, '/beegfs/u/bbd1146/daten_26F/test_daten_b.txt')
export_to_vivado_hex(X_s_sr_scaled, '/beegfs/u/bbd1146/daten_26F/test_daten_s.txt')



# Nimm das allererste Event aus dem skalierten Array
first_event_scaled = X_s_sr_scaled[0]

# Lies die erste Zeile der exportierten Datei
with open('/beegfs/u/bbd1146/daten_26F/test_daten_s.txt', 'r') as f:
    hex_line = f.readline().strip()



for i in range(3):
    # Wir schneiden 4 Zeichen (16 Bit) für das Feature aus
    start = -4 * (i + 1)
    end = -4 * i if i > 0 else None
    hex_val = hex_line[start:end] 
    
    # Direkt konvertieren – KEIN ZUSÄTZLICHES DREHEN MEHR!
    val_int = int(hex_val, 16)
    
    # Zweierkomplement auflösen
    if val_int >= 0x8000:   
        val_int -= 0x10000

    reconstructed_float = val_int / 2048.0
    print(f"Feature {i}: Datei-Hex={hex_val} ({reconstructed_float:.4f}), Original={first_event_scaled[i]:.4f}")