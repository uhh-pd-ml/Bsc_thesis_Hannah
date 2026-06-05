import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("/beegfs/u/bbd1146/regression/", f"run_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

##### 1. Load Data #####
def load_data(path):
    with h5py.File(path, 'r') as f:
        j1 = f['jet1_PFCands'][:, :60]
        j2 = f['jet2_PFCands'][:, :60]
        X = np.concatenate([j1, j2], axis=1)
        taus = f['jettiness'][:]
        kin  = f['jet_kinematics'][:]
        Y = np.concatenate([taus, kin], axis=1)
    return X, Y


X_br, Y_br = load_data('/beegfs/u/bbd1146/daten_26F/events_b_br.h5')
X_sr, Y_sr = load_data('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5')

X_all_bg = np.concatenate([X_br, X_sr], axis=0)
Y_all_bg = np.concatenate([Y_br, Y_sr], axis=0)


##### 2. Train-Test-Split (Nur auf dem Background!) #####
X_train_bg, X_test_bg, Y_train_bg, Y_test_bg = train_test_split(
    X_all_bg, Y_all_bg, test_size=0.2, random_state=42
)

##### 3. Preprocessing #####
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Fitte den Scaler nur auf den Trainings-Background!
X_train_scaled = scaler_x.fit_transform(X_train_bg)
Y_train_scaled = scaler_y.fit_transform(Y_train_bg)

# Test-Background und Signal nur transformieren
X_test_bg_scaled = scaler_x.transform(X_test_bg)
Y_test_bg_scaled = scaler_y.transform(Y_test_bg)

# Speichere Scaler für hls4ml / FPGA Einsatz
np.save(os.path.join(output_dir, "X_mean.npy"), scaler_x.mean_)
np.save(os.path.join(output_dir, "X_std.npy"), scaler_x.scale_)

# In Tensoren umwandeln
X_train_tensor = torch.from_numpy(X_train_scaled).float()
Y_train_tensor = torch.from_numpy(Y_train_scaled).float()

loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=256, shuffle=True)

##### 4. Model Definition #####
class FeatureRegressor(nn.Module):
    def __init__(self, input_dim=120, output_dim=26):
        super(FeatureRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeatureRegressor().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

##### 5. Training #####
epochs = 50
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_Y in loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), os.path.join(output_dir, "l1_feature.pth"))

##### 6. Evaluation (Regression Check) #####
model.eval()
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_s_sr.h5', 'r') as f:
    j1_s = f['jet1_PFCands'][:, :60]
    j2_s = f['jet2_PFCands'][:, :60]
    X_sig_raw = np.concatenate([j1_s, j2_s], axis=1)

with torch.no_grad():
    # 1. Training-Set für AE (Background BR) -> 26 Features
    X_br_tensor = torch.from_numpy(scaler_x.transform(X_br)).float().to(device)
    ae_train_bg = model(X_br_tensor).cpu().numpy()

    # 2. Test-Set für AE (Background SR) -> 26 Features
    X_sr_tensor = torch.from_numpy(scaler_x.transform(X_sr)).float().to(device)
    ae_test_bg = model(X_sr_tensor).cpu().numpy()

    # 3. Test-Set für AE (Signal SR) -> 26 Features
    X_sig_tensor = torch.from_numpy(scaler_x.transform(X_sig_raw)).float().to(device)
    ae_test_sig = model(X_sig_tensor).cpu().numpy()

    # Für die Plots (Validation auf dem Test-Split)
    bg_pred = model(torch.from_numpy(X_test_bg_scaled).float().to(device)).cpu().numpy()
    bg_true = Y_test_bg_scaled


# SPEICHERN FÜR AUTOENCODER
np.save(os.path.join(output_dir, "ae_train_bg.npy"), ae_train_bg)
np.save(os.path.join(output_dir, "ae_test_bg.npy"), ae_test_bg)
np.save(os.path.join(output_dir, "ae_test_sig.npy"), ae_test_sig)

def plot_res(true, pred, idx, name):
    plt.figure(figsize=(6,6))
    plt.scatter(true[:, idx], pred[:, idx], s=1, alpha=0.3)
    plt.plot([-3, 3], [-3, 3], color='red', linestyle='--')
    plt.title(f"Check Feature {name}")
    plt.savefig(os.path.join(output_dir, f"check_{name}.png"))
    plt.close()

plot_res(bg_true, bg_pred, 12, "mjj")
plot_res(bg_true, bg_pred, 14, "pTjet1")

diff = (bg_pred - bg_true)
plt.hist(diff[:, 14], bins=100, range=(-2, 2), alpha=0.7, label='pT Resolution')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Prediction - Truth (Standardized)")
plt.ylabel("Events")
plt.legend()
plt.savefig(os.path.join(output_dir, "histogram.png"))
