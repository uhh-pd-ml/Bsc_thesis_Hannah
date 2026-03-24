import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# --- 1. DATEN LADEN ---
score_background = np.load('trash_ae/first_ae_22Features/scores_bg.npy')
score_signal = np.load('trash_ae/first_ae_22Features/scores_sig.npy')


all_scores = np.concatenate([score_background, score_signal])
labels = np.concatenate([np.zeros(len(score_background)), np.ones(len(score_signal))])

# --- 2. ROC-WERTE BERECHNEN ---
fpr, tpr, thresholds = roc_curve(labels, all_scores)

# --- 3. SIC BERECHNEN ---
epsilon = 1e-10
sic = tpr / np.sqrt(fpr + epsilon)

# --- 4. OPTIMALEN PUNKT FINDEN ---
idx_max = np.argmax(sic)
max_sic_val = sic[idx_max]
optimal_tpr = tpr[idx_max]
optimal_threshold = thresholds[idx_max]

# --- 5. PLOTTEN ---
plt.figure(figsize=(10, 6))

# Plot der SIC-Kurve
plt.plot(tpr, sic, label='SIC: $TPR / \sqrt{FPR}$', color='darkorange', lw=2)

# Zufalls-Linie zum Vergleich
plt.plot(tpr, np.sqrt(tpr + epsilon), linestyle='--', color='gray', label='Random Baseline')

# Markierung des besten Schwellenwerts
plt.axvline(optimal_tpr, color='red', linestyle=':', alpha=0.5)
plt.scatter(optimal_tpr, max_sic_val, color='red', label=f'Max SIC: {max_sic_val:.2f}')

plt.xlabel('Signal Efficiency (TPR)')
plt.ylabel('Significance Improvement (SIC)')
#plt.title('Performance Analyse: Significance Improvement Characteristic')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('sic.png')

print(f"Bester Schwellenwert: {optimal_threshold:.6f}")
print(f"Dort erreicht man eine Signal-Effizienz von: {optimal_tpr*100:.2f}%")