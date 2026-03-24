import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# load data
score_bg = np.load('first_ae/scores_bg.npy')
score_sig = np.load('first_ae/scores_sig.npy')



#Loss Histogram
plt.figure(figsize=(8, 6))

plt.hist(score_bg, bins=100, density=True, alpha=0.5, 
         label='Background (QCD)', color='royalblue', range=(0, np.percentile(score_sig, 98)))

plt.hist(score_sig, bins=100, density=True, alpha=0.6, 
         label='Signal (BSM)', color='crimson', range=(0, np.percentile(score_sig, 98)))

#plt.yscale('log')
plt.xlabel('Autoencoder Reconstruction Loss (MSE)')
plt.ylabel('Probability Density (normalized)')
#plt.title('Anomaly Detection: Loss Distribution')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig('first_ae/loss_histogram.png')



#ROC-Kurve
y_true = np.concatenate([np.zeros(len(score_bg)), np.ones(len(score_sig))])
y_scores = np.concatenate([score_bg, score_sig])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('first_ae/ROC_Kurve.png')



#Learing courve
train_losses = np.load("first_ae/CLSF_train_losses.npy")
val_losses = np.load("first_ae/CLSF_val_losses.npy")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)

plt.title('Autoencoder Learning Curve', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig("first_ae/learning_curve.png", bbox_inches='tight')
