import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from models import VAE, Predictor

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pin_memory = device.type == "cuda"
batch_size = 128

# Load dataset
data = np.load("materials_data.npz")
X = torch.from_numpy(data["X"]).float()
y = torch.from_numpy(data["y"]).float()
dataset = TensorDataset(X, y)

# Split for VAE training
train_ds, val_ds = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

# Initialize VAE
vae = VAE(input_dim=X.shape[1], latent_dim=20).to(device)
vae_opt = Adam(vae.parameters(), lr=1e-4)
vae_sched = ReduceLROnPlateau(vae_opt, mode='min', factor=0.5, patience=5)

# VAE loss
def vae_loss(recon, x, mu, logvar):
    mse = F.mse_loss(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl

# Train VAE
best_vae = float('inf')
for epoch in range(1, 101):
    vae.train()
    train_loss = 0.0
    for xb, _ in train_loader:
        xb = xb.to(device)
        vae_opt.zero_grad()
        recon, mu, logvar = vae(xb)
        loss = vae_loss(recon, xb, mu, logvar)
        loss.backward()
        vae_opt.step()
        train_loss += loss.item()
    train_loss /= len(train_ds)

    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            val_loss += vae_loss(recon, xb, mu, logvar).item()
    val_loss /= len(val_ds)

    vae_sched.step(val_loss)
    print(f"[VAE] Epoch {epoch}, Train {train_loss:.3f}, Val {val_loss:.3f}")
    if val_loss < best_vae:
        best_vae = val_loss
        torch.save(vae.state_dict(), "vae_best.pth")

# Prepare latents for Predictor
vae.eval()
mus, labs = [], []
all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
with torch.no_grad():
    for xb, yb in all_loader:
        xb = xb.to(device)
        _, mu, _ = vae(xb)
        mus.append(mu.cpu())
        labs.append(yb)
mus = torch.cat(mus)
labs = torch.cat(labs)

# Split for Predictor training
ds_mu = TensorDataset(mus, labs)
train_mu, val_mu = random_split(ds_mu, [int(0.9 * len(ds_mu)), len(ds_mu) - int(0.9 * len(ds_mu))])
train_loader_mu = DataLoader(train_mu, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
val_loader_mu = DataLoader(val_mu, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

# Initialize Predictor
pred = Predictor(latent_dim=mus.shape[1], output_dim=labs.shape[1]).to(device)
pred_opt = Adam(pred.parameters(), lr=1e-3)
pred_sched = ReduceLROnPlateau(pred_opt, mode='min', factor=0.5, patience=5)

# Train Predictor
best_pred = float('inf')
for epoch in range(1, 51):
    pred.train()
    train_loss = 0.0
    for zb, yb in train_loader_mu:
        zb, yb = zb.to(device), yb.to(device)
        pred_opt.zero_grad()
        out = pred(zb)
        loss = F.mse_loss(out, yb)
        loss.backward()
        pred_opt.step()
        train_loss += loss.item()
    train_loss /= len(train_mu)

    pred.eval()
    val_loss = 0.0
    with torch.no_grad():
        for zb, yb in val_loader_mu:
            zb, yb = zb.to(device), yb.to(device)
            val_loss += F.mse_loss(pred(zb), yb).item()
    val_loss /= len(val_mu)

    pred_sched.step(val_loss)
    print(f"[PRED] Epoch {epoch}, Train {train_loss:.3f}, Val {val_loss:.3f}")
    if val_loss < best_pred:
        best_pred = val_loss
        torch.save(pred.state_dict(), "predictor.pth")

print("Training complete.")
