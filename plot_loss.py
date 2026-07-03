import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

save_path = '/beegfs/u/bbd1146/ae_data_output/run_20260626_133856'
datei_pfad = os.path.join(save_path, 'hls_torch_scores.npz')

# Datei laden
data = np.load(datei_pfad)

# Arrays über die vergebenen Namen herausholen
score_bkg_hls = data['score_bkg_hls']
score_sig_hls = data['score_sig_hls']
score_bkg_torch = data['score_bkg_torch']
score_sig_torch = data['score_sig_torch']

 
 
fig, ax = plt.subplots(1, 2, figsize=(12, 5)) # Höhe auf 5 reduziert, da 1x2 Grid
        
# Plot für die Hardware-Ergebnisse (HLS)
ax[0].hist(score_bkg_hls, bins=50, alpha=0.5, label='Bkg', density=True)
ax[0].hist(score_sig_hls, bins=50, alpha=0.5, label='Sig', density=True)
ax[0].set_xlim(0, 3)
ax[0].set_xlabel("Score", fontsize=12)
ax[0].set_ylabel("Probability Density", fontsize=12)
ax[0].legend()
ax[0].set_title("HLS Hardware Performance (C-Sim)")

# Plot für die Software-Ergebnisse (Torch)
ax[1].hist(score_bkg_torch, bins=50, alpha=0.5, label='Bkg', density=True)
ax[1].hist(score_sig_torch, bins=50, alpha=0.5, label='Sig', density=True)
ax[1].set_xlim(0, 3)
ax[1].set_xlabel("Score", fontsize=12)
ax[1].set_ylabel("Probability Density", fontsize=12)
ax[1].legend()
ax[1].set_title("PyTorch Software Performance")

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'hls_comparison_neu.png'))
plt.close()

fig = plt.figure(figsize=(10, 8))
# Grid: Oben der Hauptplot (Ratio 3), unten der Ratio-Plot (Ratio 1)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

ax_main = plt.subplot(gs[0])
ax_ratio = plt.subplot(gs[1], sharex=ax_main)

bins = np.linspace(0, 2.5, 60)

# --- 1. Hauptplot (Verteilungen) ---
# PyTorch (als gefüllte Flächen)
ax_main.hist(score_bkg_torch, bins=bins, alpha=0.5, label='PyTorch Background', density=True)
ax_main.hist(score_sig_torch, bins=bins, alpha=0.5,  label='PyTorch Signal', density=True)

# HLS (als Stufen/Linien, um den Overlay zu verdeutlichen)
hist_bkg_hls, _, _ = ax_main.hist(score_bkg_hls, bins=bins, histtype='step', linewidth=1.5, color='darkblue',
                                     label='HLS Background (C-Sim)', density=True)
hist_sig_hls, _, _ = ax_main.hist(score_sig_hls, bins=bins, histtype='step', linewidth=1.5, color='sienna',
                                     label='HLS Signal (C-Sim)', density=True)

ax_main.set_ylabel("Probability Density", fontsize=15)
ax_main.set_xlim(0, 2.5)
ax_main.legend(loc='upper right')
ax_main.grid(True, linestyle='--', alpha=0.5)
plt.setp(ax_main.get_xticklabels(), visible=False) # X-Achsen-Labels oben verstecken

# --- 2. Ratio-Plot (HLS / PyTorch) ---
# Wir berechnen die exakten Histogramm-Werte für die Division
counts_bkg_torch, _ = np.histogram(score_bkg_torch, bins=bins, density=True)
counts_sig_torch, _ = np.histogram(score_sig_torch, bins=bins, density=True)
counts_bkg_hls, _ = np.histogram(score_bkg_hls, bins=bins, density=True)
counts_sig_hls, _ = np.histogram(score_sig_hls, bins=bins, density=True)

# Berechne die Bin-Mitten für das Plotten der Punkte
bin_centers = (bins[:-1] + bins[1:]) / 2

# Division mit Guard gegen "Division by Zero"
with np.errstate(divide='ignore', invalid='ignore'):
    ratio_bkg = counts_bkg_hls / counts_bkg_torch
    ratio_sig = counts_sig_hls / counts_sig_torch

# Maske für valide Datenpunkte (wo Torch > 0 ist), um unschöne Ausreißer im Nichts zu vermeiden
mask_bkg = counts_bkg_torch > 0.01
mask_sig = counts_sig_torch > 0.01

# Punkte plotten
ax_ratio.scatter(bin_centers[mask_bkg], ratio_bkg[mask_bkg], color='darkblue', marker='o', s=20, label='Bkg Ratio')
ax_ratio.scatter(bin_centers[mask_sig], ratio_sig[mask_sig], color='darkgreen', marker='s', s=20, label='Sig Ratio')

# Toleranzbänder und Ideallinie
ax_ratio.axhline(1.0, color='black', linestyle='-', linewidth=1)
ax_ratio.fill_between(bins, 0.99, 1.01, color='gray', alpha=0.3, label='±1% Tolerance')
#ax_ratio.fill_between(bins, 0.95, 1.05, color='gray', alpha=0.1, label='±5% Tolerance')

ax_ratio.set_xlim(0, 2.5)
ax_ratio.set_ylim(0.9, 1.1)
ax_ratio.set_xlabel("Anomaly Score (MSE)", fontsize=15)
ax_ratio.set_ylabel("Ratio (HLS / Torch)", fontsize=15)
ax_ratio.grid(True, linestyle='--', alpha=0.5)

# Optional: Legende für den Ratio-Plot unten links platzieren
ax_ratio.legend(loc='lower left', fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'hls_comparison_ratio_neu.png'), dpi=300)
plt.close()

