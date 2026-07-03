import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==============================================================================
# 1. Dateipfade definieren
# ==============================================================================
path_normal   = "/home/bbd1146/Bsc_thesis_Hannah/roc_kurve_hardware/autoencoder_results_b.csv"
path_anormal  = "/home/bbd1146/Bsc_thesis_Hannah/roc_kurve_hardware/autoencoder_results_s.csv"

SPALTEN_NAME = 'Hex_Data'

# ==============================================================================
# 2. Hilfsfunktion mit Slicing-Erweiterung für die Hex-Strings
# ==============================================================================
def load_loss_from_csv(file_path, column_name, slice_len=None):
    df = pd.read_csv(file_path)
    
    if column_name not in df.columns:
        raise ValueError(f"Spalte '{column_name}' wurde in {file_path} nicht gefunden!")
        
    # Wenn die Spalte Text/Hex-Strings enthält, schneiden wir sie zurecht
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        # Konvertiert in String und entfernt "0x", falls vorhanden
        clean_series = df[column_name].astype(str).apply(lambda x: x.strip().replace("0x", ""))
        
        # --- NEU: Hier schneiden wir die ersten X Zeichen ab ---
        if slice_len is not None:
            clean_series = clean_series.apply(lambda x: x[:slice_len])
            
        # Konvertiert den zugeschnittenen String in einen Integer (Hex-Basis 16)
        return clean_series.apply(lambda x: int(x, 16)).values
    else:
        # Falls es bereits Zahlen sind, unverändert zurückgeben
        return df[column_name].values

# ==============================================================================
# 3. Daten laden (mit den unterschiedlichen Zeichen-Längen) und Labels erstellen
# ==============================================================================
# Normale Daten (Background): Nimmt die ersten 16 Zeichen -> Label 0
losses_normal = load_loss_from_csv(path_normal, SPALTEN_NAME, slice_len=16)
labels_normal = np.zeros(len(losses_normal))

# Anormale Daten (Signal): Nimmt die ersten 32 Zeichen -> Label 1
losses_anormal = load_loss_from_csv(path_anormal, SPALTEN_NAME, slice_len=32)
labels_anormal = np.ones(len(losses_anormal))

print(f"Normale Datenpunkte geladen (16-Zeichen-Slice): {len(losses_normal)}")
print(f"Anormale Datenpunkte geladen (32-Zeichen-Slice): {len(losses_anormal)}")

# Beide Datensätze zusammenführen
all_losses = np.concatenate([losses_normal, losses_anormal])
all_labels = np.concatenate([labels_normal, labels_anormal])

all_losses = np.array(all_losses, dtype=float)
all_labels = np.array(all_labels, dtype=int)

# ==============================================================================
# 4. ROC-Kurve und AUC berechnen
# ==============================================================================
fpr, tpr, thresholds = roc_curve(all_labels, all_losses)
roc_auc = auc(fpr, tpr)

# ==============================================================================
# 5. ROC-Kurve plotten
# ==============================================================================
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AE Model (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=15)
plt.ylabel('True Positive Rate (TPR)', fontsize=15)
plt.legend(loc="lower right", fontsize='large')
plt.grid(True, linestyle=':', alpha=0.6)

# Speichern der Kurve
plt.savefig("/beegfs/u/bbd1146/roc_kurve.png", dpi=300)

# ==============================================================================
# 6. Optimalen Hardware-Schwellenwert bestimmen
# ==============================================================================
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("\n--- Auswertung ---")
print(f"Berechneter AUC-Wert: {roc_auc:.4f}")
print(f"Optimaler Loss-Schwellenwert (Dezimal): {optimal_threshold}")

try:
    print(f"Optimaler Schwellenwert in Hex (für deinen VHDL/Verilog Code): {hex(int(optimal_threshold))}")
except Exception:
    pass