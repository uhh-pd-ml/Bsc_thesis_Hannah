import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==============================================================================
# 1. Dateipfade definieren
# ==============================================================================
path_normal   = "/home/bbd1146/Bsc_thesis_Hannah/roc_kurve_hardware/ausgangs_daten_bhgq.csv"
path_anormal  = "/home/bbd1146/Bsc_thesis_Hannah/roc_kurve_hardware/ausgangs_daten_shgq.csv"
#path_normal   = "/home/bbd1146/Bsc_thesis_Hannah/roc_kurve_hardware/autoencoder_results_b_hgq.csv"
#path_anormal  = "/home/bbd1146/Bsc_thesis_Hannah/roc_kurve_hardware/autoencoder_results_s_hgq.csv"

# Welchen Namen hat die Spalte in deinen CSV-Dateien?
SPALTEN_NAME = 'Daten (Hex)' 
#SPALTEN_NAME = 'Hex_Data'

# ==============================================================================
# 2. Hilfsfunktion zum Einlesen und Konvertieren der Hex-Werte in Integer/Loss
# ==============================================================================
def load_loss_from_csv(file_path, column_name):
    df = pd.read_csv(file_path)
    
    if column_name not in df.columns:
        raise ValueError(f"Spalte '{column_name}' wurde in {file_path} nicht gefunden!")
        
    # Wir prüfen, ob die Spalte NICHT numerisch ist (also Strings/Hex enthält)
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        # Konvertiert jeden Wert sicher in einen Hex-Integer, ignoriert Leerzeichen
        return df[column_name].astype(str).apply(lambda x: int(x.strip(), 16)).values
    else:
        return df[column_name].values

# ==============================================================================
# 3. Daten laden und Labels (Ground Truth) erstellen
# ==============================================================================
# Normale Daten laden -> Label ist 0
losses_normal = load_loss_from_csv(path_normal, SPALTEN_NAME)[:953]
labels_normal = np.zeros(len(losses_normal))

# Anormale Daten laden -> Label ist 1
losses_anormal = load_loss_from_csv(path_anormal, SPALTEN_NAME)[:953]
labels_anormal = np.ones(len(losses_anormal))

print(f"Normale Datenpunkte geladen: {len(losses_normal)}")
print(f"Anormale Datenpunkte geladen: {len(losses_anormal)}")

# Beide Datensätze zu großen Arrays zusammenführen
all_losses = np.concatenate([losses_normal, losses_anormal])
all_labels = np.concatenate([labels_normal, labels_anormal])

# Sicherstellen, dass die Typen für Scikit-Learn absolut sauber sind (Floats/Ints)
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
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AE Model (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=15)
plt.ylabel('True Positiv Rate (TPR)', fontsize=15)
#plt.title('ROC-Kurve')
plt.legend(loc="lower right", fontsize='large')
plt.grid(True, linestyle=':', alpha=0.6)

# Speichern im garantiert beschreibbaren Beegfs-Temp-Verzeichnis
plt.savefig("/beegfs/u/bbd1146/roc_kurve_hs_hgq_953.png", dpi=300)


# ==============================================================================
# 6. Optimalen Hardware-Schwellenwert bestimmen
# ==============================================================================
# Findet den Punkt, der Youden's J-Statistik maximiert (am weitesten oben-links)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("\n--- Auswertung ---")
print(f"Berechneter AUC-Wert: {roc_auc:.3f}")
print(f"Optimaler Loss-Schwellenwert für deinen FPGA (Dezimal): {optimal_threshold}")

# Da wir wissen, dass wir Hex-Daten verarbeitet haben, wandeln wir den Threshold direkt wieder in Hex um:
try:
    print(f"Optimaler Schwellenwert in Hex (für deinen VHDL/Verilog Code): {hex(int(optimal_threshold))}")
except Exception:
    pass