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
    """
    Wandelt einen Float in einen 16-Bit Integer für ap_fixed<16,5> um.
    Das entspricht 11 Nachkommastellen (Fractional Bits).
    """
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


def export_to_vivado_hex(data_scaled, filename, total_features=26, padding_to_bits=512):
    """
    Fügt 26 Features à 16 Bit zu einem String zusammen und füllt auf 512 Bit auf.
    """
    with open(filename, 'w') as f:
        for row in data_scaled:
            # Jedes Feature konvertieren
            fixed_values = [to_fixed_point_int16(x) for x in row]
            
            # Bit-String bauen: Feature 0 steht an den niedrigsten Bits (ganz rechts)
            # Daher gehen wir die Liste rückwärts durch
            bit_string = ""
            for val in reversed(fixed_values):
                bit_string += format(val, '016b')
            
            # Aktuelle Länge: 26 * 16 = 416 Bits
            # Padding auf 512 Bit (96 Nullen vorne anfügen)
            padding_bits = "0" * (padding_to_bits - len(bit_string))
            full_bit_string = padding_bits + bit_string
            
            # Umwandlung in Hex (128 Zeichen für 512 Bit)
            full_hex = f"{int(full_bit_string, 2):0128x}"
            f.write(f"{full_hex}\n")


# Deine Daten laden
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    all_features_b = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    all_features_s = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)


X_b_sr_scaled = scaler.transform(all_features_b)
X_s_sr_scaled = scaler.transform(all_features_s)

# Exportieren
#export_to_vivado_hex(X_b_sr_scaled, '/beegfs/u/bbd1146/daten_26F/test_daten_b.txt')
#export_to_vivado_hex(X_s_sr_scaled, '/beegfs/u/bbd1146/daten_26F/test_daten_s.txt')



# Nimm das allererste Event aus dem skalierten Array
first_event_scaled = X_s_sr_scaled[0]

# Lies die erste Zeile der exportierten Datei
with open('/beegfs/u/bbd1146/daten_26F/test_daten_s.txt', 'r') as f:
    hex_line = f.readline().strip()



for i in range(3):
    start = -4 * (i + 1)
    end = -4 * i if i > 0 else None
    hex_val = hex_line[start:end]
    
    val_int = int(hex_val, 16)
    if val_int >= 0x8000:   
        val_int -= 0x10000

    # Umwandlung zurück in Float (vereinfacht für positive Werte)
    reconstructed_float = val_int / 2048.0
    print(f"Feature {i}: Datei={hex_val} ({reconstructed_float:.4f}), Original={first_event_scaled[i]:.4f}")