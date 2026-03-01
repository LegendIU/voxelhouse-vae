from __future__ import annotations
import torch
import torch.nn as nn

class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(ch, ch, 3, 1, 1),
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(ch, ch, 3, 1, 1),
        )
    def forward(self, x):
        return x + self.net(x)

class VAE3D(nn.Module):
    def __init__(self, resolution: int = 64, latent_dim: int = 128, base_ch: int = 48):
        super().__init__()
        if resolution not in (32, 64):
            raise ValueError(f"resolution must be 32 or 64, got {resolution}")
        if base_ch % 8 != 0:
            raise ValueError(f"base_ch must be divisible by 8 for GroupNorm, got {base_ch}")
        self.resolution = resolution
        self.latent_dim = latent_dim

        # Encoder: downsample 3 times (R -> R/8)
        self.enc_in = nn.Conv3d(1, base_ch, 3, 1, 1)
        self.enc = nn.Sequential(
            ResBlock3D(base_ch),
            nn.Conv3d(base_ch, base_ch*2, 4, 2, 1),  # /2
            ResBlock3D(base_ch*2),
            nn.Conv3d(base_ch*2, base_ch*4, 4, 2, 1),  # /4
            ResBlock3D(base_ch*4),
            nn.Conv3d(base_ch*4, base_ch*6, 4, 2, 1),  # /8
            ResBlock3D(base_ch*6),
            nn.GroupNorm(8, base_ch*6),
            nn.SiLU(inplace=True),
        )
        feat_res = resolution // 8
        feat_dim = (base_ch*6) * feat_res * feat_res * feat_res
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, feat_dim)
        self.dec = nn.Sequential(
            ResBlock3D(base_ch*6),
            nn.ConvTranspose3d(base_ch*6, base_ch*4, 4, 2, 1),  # x2
            ResBlock3D(base_ch*4),
            nn.ConvTranspose3d(base_ch*4, base_ch*2, 4, 2, 1),
            ResBlock3D(base_ch*2),
            nn.ConvTranspose3d(base_ch*2, base_ch, 4, 2, 1),
            ResBlock3D(base_ch),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_ch, 1, 3, 1, 1),
        )
        self._feat_res = feat_res
        self._base_ch = base_ch

    def encode(self, x):
        h = self.enc(self.enc_in(x)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.shape[0], self._base_ch*6, self._feat_res, self._feat_res, self._feat_res)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

def kl_divergence(mu, logvar):
    return 0.5 * torch.mean(torch.sum(mu**2 + torch.exp(logvar) - 1.0 - logvar, dim=1))
