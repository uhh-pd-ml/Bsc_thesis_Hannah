import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = '/beegfs/u/bbd1146/results/'
os.makedirs(output_dir, exist_ok=True)


##### Load Data & Preprocess #####
with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_br.h5', 'r') as f:
    # Input: Top 10 Teilchen von Jet 1 + Top 10 von Jet 2 (jeweils pt, eta, phi)
    j1 = f['jet1_PFCands'][:, :30]
    j2 = f['jet2_PFCands'][:, :30]
    X_b = np.concatenate([j1, j2], axis=1)

    # Target: Die Features von FastJet
    taus = f['jettiness'][:]
    kin  = f['jet_kinematics'][:]
    Y_b = np.concatenate([taus, kin], axis=1)

with h5py.File('/beegfs/u/bbd1146/daten_26F/events_b_sr.h5', 'r') as f:
    # Input: Top 10 Teilchen von Jet 1 + Top 10 von Jet 2 (jeweils pt, eta, phi)
    j1 = f['jet1_PFCands'][:, :30]
    j2 = f['jet2_PFCands'][:, :30]
    X_s = np.concatenate([j1, j2], axis=1)

    # Target: Die Features von FastJet
    taus = f['jettiness'][:]
    kin  = f['jet_kinematics'][:]
    Y_s = np.concatenate([taus, kin], axis=1)

X = np.concatenate([X_b, X_s], axis=0)
Y = np.concatenate([Y_b, Y_s], axis=0)  

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

np.save(os.path.join(output_dir, "X_mean.npy"), scaler_x.mean_)
np.save(os.path.join(output_dir, "X_std.npy"), scaler_x.scale_)
np.save(os.path.join(output_dir, "Y_mean.npy"), scaler_y.mean_)
np.save(os.path.join(output_dir, "Y_std.npy"), scaler_y.scale_)

X_tensor = torch.from_numpy(X_scaled).float()
Y_tensor = torch.from_numpy(Y_scaled).float()

#### DataLoader #####
dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True)


##### Model Definition #####
class FeatureRegressor(nn.Module):
    def __init__(self, input_dim=60, output_dim=26):
        super(FeatureRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


#### Model, Optimizer, Loss #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeatureRegressor().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

#### Training Loop #####
epochs = 50
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_Y in loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")


torch.save(model.state_dict(), os.path.join(output_dir, "l1_feature.pth"))


##### Evaluation #####
model.eval()
with torch.no_grad():
    indices = np.random.choice(len(X_tensor), 2000, replace=False)
    test_pred = model(X_tensor[indices].to(device)).cpu().numpy()
    test_true = Y_tensor[indices].numpy()


plt.figure(figsize=(6,6))

plt.scatter(test_true[:, 12], test_pred[:, 12], s=1, alpha=0.5)
plt.plot([-3, 3], [-3, 3], color='red', linestyle='--')
plt.xlabel("Wahre Kinematik (normiert)")
plt.ylabel("Vorhergesagte Kinematik (normiert)")
plt.title("Regression Performance (mjj Approximation)")
plt.savefig(os.path.join(output_dir, "regression_check_mjj.png"))
plt.close()

plt.figure(figsize=(6,6))

plt.scatter(test_true[:, 14], test_pred[:, 14], s=1, alpha=0.5)
plt.plot([-3, 3], [-3, 3], color='red', linestyle='--')
plt.xlim(-2, 4)
plt.ylim(-2, 4)
plt.xlabel("Wahre Kinematik (normiert)")
plt.ylabel("Vorhergesagte Kinematik (normiert)")
plt.title("Regression Performance (pTjet1 Approximation)")
plt.savefig(os.path.join(output_dir, "regression_check_pTjet1.png"))
plt.close()