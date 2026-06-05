import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Daten laden und Scaler fitten ---
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    kinematics = f['jet_kinematics'][:]
    jettiness = f['jettiness'][:]

# WICHTIG: Merk dir die Spaltenanzahl für das spätere Zurück-Splitten
num_kinematics_features = kinematics.shape[1]  

X_train_raw = np.concatenate([kinematics, jettiness], axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)


# --- 2. Konvertierungs-Funktion ---
def to_fixed_point_int16_array(data_float, total_bits=16, integer_bits=5):
    fractional_bits = total_bits - integer_bits  # 11 Bits
    scaled_values = np.round(data_float * (2**fractional_bits)).astype(np.int32)
    
    max_val = (1 << (total_bits - 1)) - 1  
    min_val = -(1 << (total_bits - 1))     
    np.clip(scaled_values, min_val, max_val, out=scaled_values)
    
    return scaled_values.astype(np.int16)


# --- 3. Testdaten laden und transformieren ---
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    all_features_b = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    all_features_s = np.concatenate([f['jet_kinematics'][:], f['jettiness'][:]], axis=1)

X_b_sr_scaled = scaler.transform(all_features_b)
X_s_sr_scaled = scaler.transform(all_features_s)

X_b_sr_int16 = to_fixed_point_int16_array(X_b_sr_scaled)
X_s_sr_int16 = to_fixed_point_int16_array(X_s_sr_scaled)


# --- 4. Getrennt als HDF5-Dateien abspeichern ---
output_files = {
    '/beegfs/u/bbd1146/daten_26F/test_daten_b_int16.h5': X_b_sr_int16,
    '/beegfs/u/bbd1146/daten_26F/test_daten_s_int16.h5': X_s_sr_int16
}

for path, data_array in output_files.items():
    # Hier splitten wir das Array anhand der gemerkten Spaltenanzahl
    kin_data = data_array[:, :num_kinematics_features]
    jet_data = data_array[:, num_kinematics_features:]
    
    with h5py.File(path, 'w') as f:
        f.create_dataset('jet_kinematics', data=kin_data, compression="gzip")
        f.create_dataset('jettiness', data=jet_data, compression="gzip")
        print(f"Datei erfolgreich mit korrekten Keys gespeichert: {path}")