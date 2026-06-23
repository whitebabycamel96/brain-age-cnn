"""
model.py

Convolutional Autoencoder with Age Regression Head
for VBM gray matter slices (128x128, single channel).

Architecture
------------
Encoder:
  Conv2d(1,  16,  k=3, s=1, p=1) + BN + ReLU   -> 16 x 128 x 128
  Conv2d(16, 32,  k=3, s=2, p=1) + BN + ReLU   -> 32 x  64 x  64   [downsample 1]
  Conv2d(32, 32,  k=3, s=1, p=1) + BN + ReLU   -> 32 x  64 x  64
  Conv2d(32, 64,  k=3, s=2, p=1) + BN + ReLU   -> 64 x  32 x  32   [downsample 2]
  Conv2d(64, 64,  k=3, s=1, p=1) + BN + ReLU   -> 64 x  32 x  32
  Conv2d(64, 128, k=3, s=2, p=1) + BN + ReLU   -> 128 x 16 x  16   [downsample 3]
  Flatten -> Linear(128*16*16, latent_dim)       -> z in R^latent_dim

Decoder (mirror):
  Linear(latent_dim, 128*16*16) -> reshape            -> 128 x 16 x 16
  ConvTranspose2d(128,64, k=4, s=2, p=1) + BN + ReLU -> 64 x  32 x  32   [upsample 1]
  Conv2d(64, 64,  k=3, s=1, p=1)        + BN + ReLU -> 64 x  32 x  32
  ConvTranspose2d(64, 32, k=4, s=2, p=1) + BN + ReLU -> 32 x  64 x  64   [upsample 2]
  Conv2d(32, 32,  k=3, s=1, p=1)        + BN + ReLU -> 32 x  64 x  64
  ConvTranspose2d(32, 16, k=4, s=2, p=1) + BN + ReLU -> 16 x 128 x 128   [upsample 3]
  Conv2d(16, 1,   k=1)                               -> 1  x 128 x 128

Age regression head (on z):
  Linear(latent_dim, 1)                          -> age_hat (scalar)

Loss:
  L_total = L_recon + lambda_age * L_age
  L_recon = MSE(x_hat, x)
  L_age   = MSE(age_hat, age)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Encoder                                                            #
# ------------------------------------------------------------------ #

class Encoder(nn.Module):
    """
    Maps a single-channel 128x128 VBM slice to a latent vector z.

    Each conv block: Conv2d -> BatchNorm2d -> ReLU.
    Downsampling via stride-2 convolutions (not pooling),
    so the downsampling kernel is also learned.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # --- spatial feature extraction ---
        self.enc = nn.Sequential(
            # block 1: 1 -> 16 channels, same spatial size
            nn.Conv2d(1,  16,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # downsample 1: 128x128 -> 64x64, 16 -> 32 channels
            nn.Conv2d(16, 32,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # block 2: refine at 64x64
            nn.Conv2d(32, 32,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # downsample 2: 64x64 -> 32x32, 32 -> 64 channels
            nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # block 3: refine at 32x32
            nn.Conv2d(64, 64,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # downsample 3: 32x32 -> 16x16, 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # --- bottleneck projection ---
        # after enc: 128 channels x 16 x 16 = 32768 features
        self._flat_dim = 128 * 16 * 16
        self.proj = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, 128, 128)
        z : (B, latent_dim)
        """
        h = self.enc(x)                    # (B, 128, 16, 16)
        h = h.view(h.size(0), -1)          # (B, 32768)  -- flatten
        z = self.proj(h)                   # (B, latent_dim)
        return z


# ------------------------------------------------------------------ #
#  Decoder                                                            #
# ------------------------------------------------------------------ #

class Decoder(nn.Module):
    """
    Maps latent vector z back to a 128x128 single-channel image.

    Upsampling strategy: ConvTranspose2d (k=4, s=2, p=1) — a learned
    operation that spreads each input value into a 4x4 output patch via
    learned weights, doubling spatial size in one step.

    Output size formula for ConvTranspose2d:
        out = (in - 1) * stride - 2*padding + kernel_size
        e.g. (16-1)*2 - 2*1 + 4 = 32  ✓

    Each upsample is followed by a stride-1 Conv2d refinement block,
    same as the bilinear version -- the transposed conv handles spatial
    growth, the regular conv handles feature refinement.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self._flat_dim = 128 * 16 * 16

        # --- expand projection: z -> flat feature vector ---
        self.expand = nn.Linear(latent_dim, self._flat_dim)

        # --- spatial reconstruction ---
        self.dec = nn.Sequential(
            # upsample 1: 16x16 -> 32x32, 128 -> 64 channels
            # k=4, s=2, p=1: (16-1)*2 - 2 + 4 = 32 ✓
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # refine at 32x32
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # upsample 2: 32x32 -> 64x64, 64 -> 32 channels
            # k=4, s=2, p=1: (32-1)*2 - 2 + 4 = 64 ✓
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # refine at 64x64
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # upsample 3: 64x64 -> 128x128, 32 -> 16 channels
            # k=4, s=2, p=1: (64-1)*2 - 2 + 4 = 128 ✓
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # output conv: 16 -> 1 channel, 1x1 kernel (no spatial mixing)
            # no activation: data is z-normalised so output is unbounded
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z     : (B, latent_dim)
        x_hat : (B, 1, 128, 128)
        """
        h = self.expand(z)                         # (B, 32768)
        h = h.view(h.size(0), 128, 16, 16)         # (B, 128, 16, 16)
        x_hat = self.dec(h)                        # (B, 1, 128, 128)
        return x_hat


# ------------------------------------------------------------------ #
#  Age Regression Head                                                #
# ------------------------------------------------------------------ #

class AgeRegressionHead(nn.Module):
    """
    Single linear layer: z -> predicted age (scalar).

    W_r in R^(1 x latent_dim), output shape (B,).
    Keeping it linear preserves interpretability: each weight
    directly measures how much that latent dimension shifts
    the predicted age, holding all others fixed -- the exact
    multivariate regression setup for the per-dimension analysis.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z       : (B, latent_dim)
        age_hat : (B,)
        """
        return self.fc(z).squeeze(1)    # (B,)


# ------------------------------------------------------------------ #
#  Full Autoencoder                                                   #
# ------------------------------------------------------------------ #

class VBMAutoencoder(nn.Module):
    """
    Convolutional autoencoder with optional age regression head.

    Forward returns a dict:
        x_hat   : reconstructed slice  (B, 1, 128, 128)
        age_hat : predicted age        (B,)   -- only if age_head=True
        z       : latent vector        (B, latent_dim)

    Loss (computed externally in train loop for flexibility):
        L_total = MSE(x_hat, x) + lambda_age * MSE(age_hat, age)
    """

    def __init__(self, latent_dim: int = 128, age_head: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.age_head   = age_head

        self.encoder   = Encoder(latent_dim)
        self.decoder   = Decoder(latent_dim)
        self.regressor = AgeRegressionHead(latent_dim) if age_head else None

    def forward(self, x: torch.Tensor) -> dict:
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        out   = {"x_hat": x_hat, "z": z}
        if self.age_head:
            out["age_hat"] = self.regressor(z)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: extract latent vector only."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience method: decode from latent vector."""
        return self.decoder(z)


# ------------------------------------------------------------------ #
#  Loss                                                               #
# ------------------------------------------------------------------ #

def compute_loss(
    out:        dict,
    x:          torch.Tensor,
    age:        torch.Tensor,
    lambda_age: float = 1.0,
) -> dict:
    """
    Compute reconstruction + age regression loss.

    Returns a dict of scalar tensors so individual terms
    are available for logging without extra forward passes.

    out        : model output dict (x_hat, age_hat, z)
    x          : original slice   (B, 1, 128, 128)
    age        : true age         (B,)
    lambda_age : weight on age loss
    """
    l_recon = F.mse_loss(out["x_hat"], x)
    losses  = {"l_recon": l_recon, "l_age": torch.tensor(0.0, device=x.device)}

    if "age_hat" in out:
        l_age          = F.mse_loss(out["age_hat"], age)
        losses["l_age"] = l_age

    losses["l_total"] = losses["l_recon"] + lambda_age * losses["l_age"]
    return losses


# ------------------------------------------------------------------ #
#  Quick shape check                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    model = VBMAutoencoder(latent_dim=128, age_head=True)
    x     = torch.randn(4, 1, 128, 128)
    age   = torch.tensor([25.0, 34.0, 52.0, 61.0])

    out    = model(x)
    losses = compute_loss(out, x, age, lambda_age=1.0)

    print("=== shape check ===")
    print(f"  input       : {x.shape}")
    print(f"  z           : {out['z'].shape}")
    print(f"  x_hat       : {out['x_hat'].shape}")
    print(f"  age_hat     : {out['age_hat'].shape}")
    print(f"  l_recon     : {losses['l_recon'].item():.4f}")
    print(f"  l_age       : {losses['l_age'].item():.4f}")
    print(f"  l_total     : {losses['l_total'].item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {total_params:,}")