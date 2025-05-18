# models.py
import torch, torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim+cond_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim+cond_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def reparameterize(self, h):
        mu, logvar = h.chunk(2,1)
        std = (0.5*logvar).exp()
        return mu + std*torch.randn_like(std), mu, logvar
    def forward(self, x, c):
        h = self.encoder(torch.cat([x,c],dim=1))
        z, mu, logvar = self.reparameterize(h)
        recon = self.decoder(torch.cat([z,c],dim=1))
        return recon, mu, logvar

class Predictor(nn.Module):
    def __init__(self, latent_dim=20, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z):
        return self.net(z)
