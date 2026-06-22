"""
2D convolutional autoencoder on VBM slices — reconstruction-focused,
leakage-safe, with Optuna hyperparameter tuning and a full diagnostic figure set.

Pipeline:
  1. load vbm_z60.npy (N,121,145) + participants_study_3.tsv (row-aligned)
  2. crop/pad each slice to 128x128
  3. one-time 70/15/15 subject-level split
  4. standardize with TRAIN stats only; build an in-brain mask from TRAIN
  5. [optional] Optuna search (--tune): TPE + median pruning, SQLite storage,
     objective = masked validation MSE
  6. train final AE (AdamW, cosine LR); checkpoint by masked val MSE
  7. write the diagnostic figures
"""

import os
import json
import argparse
import optuna
import optuna.visualization.matplotlib as ov
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use("Agg")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def crop_pad_128(sl):
    """(N,121,145) -> (N,128,128): centre-crop width 145->128, zero-pad height 121->128."""
    # cropped = sl[:, :, 8:136]                                        # 145 -> 128 (centre)
    return np.pad(sl, ((0, 0), (3, 4), (3, 4)),
                  mode="constant", constant_values=0)                # 121 -> 128

# def build_brain_mask(X_raw_train, frac=0.05):
#     """In-brain mask from the TRAIN mean image: voxels above frac * max.
# 
#     Built from training data only so it carries no information from val/test.
#     """
#     m = X_raw_train.mean(axis=0)
#     return m > (frac * m.max())  # (128,128) bool

# Input normalization (each mode pairs with a matching decoder output head)
OUT_ACT = {"global_z": None, "per_image_z": None,
           "global_minmax": "sigmoid", "scale_only": "softplus"}

class Normalizer:
    """Fit on TRAIN subjects (global modes) or per-image; transform any subjects;
    inverse-transform reconstructions back to GM-density units for comparison.

    Whether between-subject magnitude survives depends on global vs per-image:
      global_*   -> one transform for all images: between-subject level PRESERVED
      per_image_*-> each image its own stats:     between-subject level REMOVED
    The matching output activation is OUT_ACT[mode]."""
    def __init__(self, mode, X, tr):
        self.mode = mode
        if mode == "global_z":
            self.sd = float(X[tr].std()) + 1e-8
            self.mu = float(X[tr].mean())
        elif mode == "per_image_z":
            self.mu_i = X.mean(axis=(1, 2)).astype(np.float32)          # (N,)
            self.sd_i = X.std(axis=(1, 2)).astype(np.float32) + 1e-8
        elif mode == "global_minmax":
            self.lo = float(np.percentile(X[tr], 1))                    # robust min
            self.hi = float(np.percentile(X[tr], 99))                   # robust max
            self.rng = (self.hi - self.lo) + 1e-8
        elif mode == "scale_only":
            self.scale = float(np.percentile(X[tr], 99)) + 1e-8         # global, no shift
        else:
            raise ValueError(f"unknown --norm '{mode}'")

    def transform(self, X):
        if self.mode == "global_z":
            return ((X - self.mu) / self.sd).astype(np.float32)
        if self.mode == "per_image_z":
            return ((X - self.mu_i[:, None, None]) / self.sd_i[:, None, None]).astype(np.float32)
        if self.mode == "global_minmax":
            return np.clip((X - self.lo) / self.rng, 0.0, 1.0).astype(np.float32)
        if self.mode == "scale_only":
            return (X / self.scale).astype(np.float32)

    def inverse(self, A, idx=None):
        """Normalized -> GM-density units. idx = subject indices of A's rows (per-image only)."""
        if self.mode == "global_z":
            return A * self.sd + self.mu
        if self.mode == "per_image_z":
            return A * self.sd_i[idx][:, None, None] + self.mu_i[idx][:, None, None]
        if self.mode == "global_minmax":
            return A * self.rng + self.lo
        if self.mode == "scale_only":
            return A * self.scale

    def repr_params(self):
        """Representative (scale, shift) scalars for synthetic images (e.g. interpolation)."""
        if self.mode == "global_z":      return self.sd, self.mu
        if self.mode == "per_image_z":   return float(self.sd_i.mean()), float(self.mu_i.mean())
        if self.mode == "global_minmax": return self.rng, self.lo
        if self.mode == "scale_only":    return self.scale, 0.0


class ConvAE2D(nn.Module):
    """
    Encoder  : 4 strided Conv2d blocks  (Conv 3x3 s2 p1 + BN + ReLU)
               (1,128,128)->(16,64,64)->(32,32,32)->(64,16,16)->(128,8,8)
    Bottleneck: flatten 128*8*8=8192 -> Linear -> latent -> Linear -> 8192
    Decoder  : 4 ConvTranspose2d blocks (ConvT 4x4 s2 p1 + BN + ReLU)
               (128,8,8)->(64,16,16)->(32,32,32)->(16,64,64)->(1,128,128)
    No activation on the final layer (z-scored images -> unbounded output).
    """
    def __init__(self, latent: int = 512, out_act=None):
        super().__init__()
        self.out_act = out_act                                          # None | "sigmoid" | "softplus"

        def enc_block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            )

        def dec_block(ci, co, last=False):
            layers = [nn.ConvTranspose2d(ci, co, kernel_size=4, stride=2, padding=1)]
            if not last:
                layers += [nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            enc_block(1,   16), enc_block(16,  32),
            enc_block(32,  64), enc_block(64, 128),
        )
        self._flat = 128 * 8 * 8                                     # 8192
        self.to_latent   = nn.Linear(self._flat, latent)
        self.from_latent = nn.Linear(latent, self._flat)
        self.decoder = nn.Sequential(
            dec_block(128, 64), dec_block(64, 32),
            dec_block(32,  16), dec_block(16, 1, last=True),
        )

    def encode(self, x):
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z):
        x = self.decoder(self.from_latent(z).view(-1, 128, 8, 8))
        if self.out_act == "sigmoid":
            x = torch.sigmoid(x)                                        # bounded [0,1] target
        elif self.out_act == "softplus":
            x = F.softplus(x)                                           # non-negative target
        return x                                                       # else linear (signed target)

    def forward(self, x):
        return self.decode(self.encode(x))

# Masked-metric helpers
def masked_val_mse(model, loader, mask_t, device):
    """Mean squared error over in-brain voxels only (selection / pruning metric)."""
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            d2 = ((model(xb) - xb) ** 2)[:, 0][:, mask_t]
            sse += d2.sum().item()
            n += d2.numel()
    return sse / max(n, 1)

def masked_mse_per_subject(orig, recon, mask):
    """Per-subject masked MSE for arrays shaped (N,128,128); mask is (128,128) bool."""
    d2 = (orig - recon) ** 2
    return d2[:, mask].mean(axis=1)

def reconstruct_all(model, Xn, device, batch=64):
    model.eval()
    parts = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch):
            xb = torch.from_numpy(Xn[s:s + batch]).unsqueeze(1).to(device)
            parts.append(model(xb).cpu().numpy()[:, 0])
    return np.concatenate(parts, axis=0)

def encode_all(model, Xn, device, batch=64):
    model.eval()
    parts = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch):
            xb = torch.from_numpy(Xn[s:s + batch]).unsqueeze(1).to(device)
            parts.append(model.encode(xb).cpu().numpy())
    return np.concatenate(parts, axis=0)

# Training  (shared by tuning and final fit)
def train_autoencoder(Xn, tr, va, mask_t, device, *, latent, lr, weight_decay,
                      batch_size, noise_std, epochs, eta_min=1e-5,
                      trial=None, record=False, verbose=False,
                      snap_X=None, snap_epochs=(), out_act=None):
    """Train one AE. Gradient = full-image MSE; selection metric = masked val MSE.

    Returns (best_masked_val, best_state, history) where history is
    (train_full_mse, val_masked_mse, lr) lists (empty unless record=True).
    """
    Xtr_t = torch.from_numpy(Xn[tr]).unsqueeze(1)
    Xva_t = torch.from_numpy(Xn[va]).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(Xtr_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva_t), batch_size=batch_size, shuffle=False)

    model = ConvAE2D(latent, out_act=out_act).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    mse   = nn.MSELoss()

    best, best_state = float("inf"), None
    tr_hist, va_hist, lr_hist = [], [], []
    snaps = []

    for ep in range(epochs):
        model.train()
        tr_tot, ntr = 0.0, 0
        for (xb,) in train_loader:
            xb  = xb.to(device)
            inp = xb + noise_std * torch.randn_like(xb) if noise_std > 0 else xb
            opt.zero_grad()
            loss = mse(model(inp), xb)                               # full-image MSE
            loss.backward()
            opt.step()
            tr_tot += loss.item() * xb.size(0)
            ntr += xb.size(0)

        v = masked_val_mse(model, val_loader, mask_t, device)        # masked selection metric
        lr_hist.append(sched.get_last_lr()[0])
        sched.step()
        if record:
            tr_hist.append(tr_tot / ntr)
            va_hist.append(v)
        if v < best:
            best = v
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  train(full) {tr_tot/ntr:.4f}  val(masked) {v:.4f}")

        if snap_X is not None and ep in snap_epochs:                 # latent snapshot
            model.eval()
            with torch.no_grad():
                zs = [model.encode(torch.from_numpy(snap_X[s:s + 64]).unsqueeze(1).to(device)
                                   ).cpu().numpy() for s in range(0, len(snap_X), 64)]
            snaps.append((ep, np.concatenate(zs, axis=0)))
            model.train()

        if trial is not None:                                        # Optuna pruning
            trial.report(v, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best, best_state, (tr_hist, va_hist, lr_hist), snaps

# Hyperparameter tuning
def tune_hyperparams(Xn, tr, va, mask_t, device, out_dir, n_trials, tune_epochs, dataset_name,out_act=None):
    """TPE search with median pruning, persisted to SQLite. Returns best params/value."""
    def objective(trial):
        torch.manual_seed(trial.number)                             # fair per-trial init
        np.random.seed(trial.number)
        params = dict(
            latent       = trial.suggest_categorical("latent",       [64, 128, 256, 512]),
            lr           = trial.suggest_float(      "lr",           1e-4, 1e-2, log=True),
            weight_decay = trial.suggest_float(      "weight_decay", 1e-6, 1e-3, log=True),
            batch_size   = trial.suggest_categorical("batch_size",   [16, 32, 64]),
            noise_std    = trial.suggest_categorical("noise_std",    [0.0, 0.05, 0.1, 0.2]),
        )
        best, _, _, _ = train_autoencoder(Xn, tr, va, mask_t, device,
                                          epochs=tune_epochs, trial=trial,
                                          out_act=out_act, **params)
        return best

    storage = "sqlite:///" + os.path.abspath(os.path.join(out_dir, "optuna_study.db"))
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=storage, 
        study_name=dataset_name + "_ae_tuning",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  trials: {len(study.trials)} total, {n_pruned} pruned early")

    try:
        imp = optuna.importance.get_param_importances(study)
        print("  parameter importances (fANOVA):")
        for k, v in imp.items():
            print(f"    {k:13s} {v:.3f}")
    except Exception as e:                                           # needs >=2 completed trials
        print(f"  (importances unavailable: {e})")

    _save_optuna_plots(study, out_dir)
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({"best_value_masked_val_mse": study.best_value,
                   "best_params": study.best_params}, f, indent=2)
    return study.best_params, study.best_value

def _save_optuna_plots(study, out_dir):
    figs = {
        "optuna_history.png":           ov.plot_optimization_history,
        "optuna_param_importances.png": ov.plot_param_importances,
        "optuna_slice.png":             ov.plot_slice,
        "optuna_parallel.png":          ov.plot_parallel_coordinate,
    }
    for fname, fn in figs.items():
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"  ({fname} skipped: {e})")
    print("  saved: optuna_history / param_importances / slice / parallel")

# Diagnostic figures for the final model
def _representative_idx(mse, n=6):
    """Indices spanning best -> worst reconstruction, so figures show the full range."""
    order = np.argsort(mse)
    return order[np.linspace(0, len(order) - 1, n).astype(int)]

def plot_loss_and_lr(history, best_ep, out_dir):
    tr_hist, va_hist, lr_hist = history
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(tr_hist, label="train MSE (full image)")
    a1.plot(va_hist, label="val MSE (in-brain)")
    a1.axvline(best_ep, color="k", ls="--", lw=1, label=f"best epoch ({best_ep})")
    a1.set_xlabel("epoch"); a1.set_ylabel("MSE"); a1.set_yscale("log")
    a1.set_title("Reconstruction loss"); a1.legend()
    a2.plot(lr_hist, color="tab:green")
    a2.set_xlabel("epoch"); a2.set_ylabel("learning rate")
    a2.set_title("Cosine LR schedule")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120)
    plt.close(fig)

def plot_recon_panels(orig_raw, recon_raw, mask, per_mse, out_dir, n=6):
    idx = _representative_idx(per_mse, n)
    # shared display ranges so panels are comparable across rows
    gmax = float(np.percentile(orig_raw[idx][:, mask], 99))          # grayscale top (in-brain)
    errs = [np.abs(orig_raw[i] - recon_raw[i]) * mask for i in idx]
    emax = float(np.percentile(np.stack(errs)[:, mask], 99))         # shared error top
    emax = emax if emax > 0 else 1.0

    fig, axes = plt.subplots(n, 3, figsize=(9.5, 3 * n), constrained_layout=True)
    im_err = None
    for row, i in enumerate(idx):
        axes[row, 0].imshow(orig_raw[i].T,  cmap="gray", origin="lower", vmin=0, vmax=gmax)
        axes[row, 1].imshow(recon_raw[i].T, cmap="gray", origin="lower", vmin=0, vmax=gmax)
        im_err = axes[row, 2].imshow(errs[row].T, cmap="hot", origin="lower", vmin=0, vmax=emax)
        axes[row, 0].set_ylabel(f"masked MSE\n{per_mse[i]:.3f}", fontsize=9)
        for c in range(3):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    axes[0, 0].set_title("original"); axes[0, 1].set_title("reconstruction")
    axes[0, 2].set_title("|error| (in-brain)")
    # one shared colorbar on the right, spanning the error column, off the images
    cbar = fig.colorbar(im_err, ax=axes[:, 2], fraction=0.05, pad=0.02)
    cbar.set_label("|error|  (shared scale)")
    fig.suptitle("Test reconstructions: best -> worst", fontsize=12)
    fig.savefig(os.path.join(out_dir, "recon_panels.png"), dpi=120)
    plt.close(fig)

def plot_pixel_scatter(orig_raw, recon_raw, mask, out_dir):
    o = orig_raw[:, mask].ravel()
    r = recon_raw[:, mask].ravel()
    ss_res = np.sum((o - r) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    pear = np.corrcoef(o, r)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(o, r, gridsize=80, cmap="viridis", mincnt=1)
    plt.colorbar(hb, ax=ax, label="pixel count")
    lo, hi = min(o.min(), r.min()), max(o.max(), r.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x")
    ax.set_xlabel("original (GM density, in-brain)")
    ax.set_ylabel("reconstructed (GM density, in-brain)")
    ax.set_title(f"In-brain pixel scatter — test\nPearson r = {pear:.4f}   R² = {r2:.4f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pixel_scatter_masked.png"), dpi=120)
    plt.close(fig)
    return pear, r2

def plot_mean_error_map(orig_raw, recon_raw, mask, out_dir):
    mean_err = (np.abs(orig_raw - recon_raw).mean(axis=0)) * mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(orig_raw.mean(axis=0).T, cmap="gray", origin="lower")
    axes[0].set_title("mean original (test)"); axes[0].axis("off")
    im = axes[1].imshow(mean_err.T, cmap="hot", origin="lower")
    axes[1].set_title("mean |error| (in-brain)"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "mean_error_map.png"), dpi=120)
    plt.close(fig)

def plot_mse_hist(per_mse, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_mse, bins=30, color="tab:blue", alpha=0.8)
    ax.axvline(np.median(per_mse), color="k", ls="--", lw=1,
               label=f"median {np.median(per_mse):.3f}")
    ax.set_xlabel("per-subject masked MSE"); ax.set_ylabel("count")
    ax.set_title("Reconstruction quality across test subjects"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "per_subject_mse_hist.png"), dpi=120)
    plt.close(fig)

def plot_latent_diagnostics(Z, out_dir, color=None, color_label=None):
    # (a) PCA of the latent codes — structure of the learned representation
    pcs = PCA(n_components=2).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=color, cmap="viridis", s=18, alpha=0.85)
    if color is not None:
        plt.colorbar(sc, ax=ax, label=color_label or "")
    ax.set_xlabel("latent PC 1"); ax.set_ylabel("latent PC 2")
    ax.set_title(f"Latent space (PCA of {Z.shape[1]}-d codes, test)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "latent_pca.png"), dpi=120)
    plt.close(fig)

    # (b) per-unit latent std — how many of the 512 dimensions are actually used
    stds = np.sort(Z.std(axis=0))[::-1]
    active = int((stds > 0.01 * stds.max()).sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(stds, color="tab:purple")
    ax.set_xlabel("latent unit (sorted)"); ax.set_ylabel("std across test subjects")
    ax.set_title(f"Latent usage — {active}/{Z.shape[1]} units active")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "latent_usage.png"), dpi=120)
    plt.close(fig)

def _encoder_activations(model, x_sample, device):
    """List of (name, activation[C,H,W]) after each encoder block for one slice."""
    model.eval()
    acts = []
    with torch.no_grad():
        x = torch.from_numpy(x_sample[None, None]).to(device)        # (1,1,128,128)
        for i, blk in enumerate(model.encoder, 1):
            x = blk(x)
            acts.append((f"Enc{i}", x[0].cpu().numpy()))
    return acts

def _tile(maps, cols, pad=1):
    """Tile a (C,H,W) stack into one 2D grid; each map normalized to [0,1]."""
    norm = np.stack([(m - m.min()) / (m.max() - m.min() + 1e-8) for m in maps])
    norm = np.transpose(norm, (0, 2, 1))                             # brain orientation
    C, H, W = norm.shape
    rows = int(np.ceil(C / cols))
    grid = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad), np.nan)
    for k in range(C):
        r, c = divmod(k, cols)
        grid[r*(H+pad):r*(H+pad)+H, c*(W+pad):c*(W+pad)+W] = norm[k]
    return grid

_COLS = {16: 4, 32: 8, 64: 8, 128: 16}

def plot_feature_maps(model, x_sample, device, out_dir):
    """Per-channel activations at every encoder layer for one slice."""
    acts = _encoder_activations(model, x_sample, device)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (name, a) in zip(axes.ravel(), acts):
        ax.imshow(_tile(a, _COLS.get(a.shape[0], 8)), cmap="magma", origin="lower")
        ax.set_title(f"{name} · {a.shape[0]} channels @ {a.shape[1]}×{a.shape[2]}")
        ax.axis("off")
    fig.suptitle("Encoder feature maps (one test slice) — each tile is one channel", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "feature_maps.png"), dpi=120)
    plt.close(fig)

def plot_feature_maps_before_after(model_rnd, model_trained, x_sample, device, out_dir, layer=0):
    """Enc1 channels: random init vs trained — shows the channels learning."""
    name, a_rnd = _encoder_activations(model_rnd, x_sample, device)[layer]
    _,    a_tr  = _encoder_activations(model_trained, x_sample, device)[layer]
    cols = _COLS.get(a_rnd.shape[0], 8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (ttl, a) in zip(axes, [("random init (before)", a_rnd), ("trained (after)", a_tr)]):
        ax.imshow(_tile(a, cols), cmap="magma", origin="lower")
        ax.set_title(f"{name} — {ttl}"); ax.axis("off")
    fig.suptitle("What training does to the channels", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "feature_maps_before_after.png"), dpi=120)
    plt.close(fig)

def plot_latent_interpolation(model, Z, Xnp, sd, mu, device, out_dir, n=7):
    """Decode the straight-line path between the two PC1-extreme subjects."""
    pc1 = PCA(n_components=1).fit_transform(Z)[:, 0]
    iA, iB = int(np.argmin(pc1)), int(np.argmax(pc1))
    alphas = np.linspace(0, 1, n)
    codes = np.stack([(1 - a) * Z[iA] + a * Z[iB] for a in alphas]).astype(np.float32)
    model.eval()
    with torch.no_grad():
        dec = model.decode(torch.from_numpy(codes).to(device)).cpu().numpy()[:, 0]
    dec_raw = dec * sd + mu
    panels = ([("subject A", Xnp[iA] * sd + mu)]
              + [(f"α={a:.2f}", dec_raw[k]) for k, a in enumerate(alphas)]
              + [("subject B", Xnp[iB] * sd + mu)])
    fig, axes = plt.subplots(1, len(panels), figsize=(2 * len(panels), 2.6))
    for ax, (ttl, img) in zip(axes, panels):
        ax.imshow(img.T, cmap="gray", origin="lower"); ax.set_title(ttl, fontsize=9); ax.axis("off")
    fig.suptitle("Latent interpolation: decoding the path from subject A to subject B", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "latent_interpolation.png"), dpi=120)
    plt.close(fig)

def plot_latent_over_epochs(snaps, color, out_dir):
    """Validation latent space at several epochs, projected with fixed final-epoch PCA axes."""
    if not snaps:
        return
    pca = PCA(n_components=2).fit(snaps[-1][1])                       # shared axes
    fig, axes = plt.subplots(1, len(snaps), figsize=(4 * len(snaps), 4), squeeze=False)
    sc = None
    for ax, (ep, Z) in zip(axes[0], snaps):
        p = pca.transform(Z)
        sc = ax.scatter(p[:, 0], p[:, 1], c=color, cmap="viridis", s=14, alpha=0.85)
        ax.set_title(f"epoch {ep}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if color is not None:
        fig.colorbar(sc, ax=axes[0], fraction=0.025, label="age")
    fig.suptitle("Validation latent space organizing over training (fixed final-epoch axes)", fontsize=12)
    fig.savefig(os.path.join(out_dir, "latent_over_epochs.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

def encode_layer_features(model, Xnp, device, batch=64):
    """Per-subject mean activation of every channel at each encoder layer.
    Returns a list of 4 arrays: [(N,16), (N,32), (N,64), (N,128)] for Enc1..Enc4."""
    model.eval()
    accum = [[] for _ in range(len(model.encoder))]
    with torch.no_grad():
        for s in range(0, len(Xnp), batch):
            x = torch.from_numpy(Xnp[s:s + batch]).unsqueeze(1).to(device)
            for li, blk in enumerate(model.encoder):                 # step through the 4 blocks
                x = blk(x)
                accum[li].append(x.mean(dim=(2, 3)).cpu().numpy())   # (b,C) spatial mean
    return [np.concatenate(a, axis=0) for a in accum]

def _topvar_cols(F, k):
    """Highest-variance feature indices (age-agnostic selection)."""
    return list(np.argsort(F.var(axis=0))[::-1][:k])

def _scatter_vs_age(ax, age, feat, label, ylab):
    m = np.isfinite(age) & np.isfinite(feat)
    a, f = age[m], feat[m]
    ax.scatter(a, f, s=15, alpha=0.7, color="#1C7293", edgecolors="none")
    r = float("nan")
    if a.size >= 3 and np.ptp(a) > 0:
        b1, b0 = np.polyfit(a, f, 1)
        xs = np.array([a.min(), a.max()])
        ax.plot(xs, b0 + b1 * xs, color="#D98324", lw=1.8)
        r = float(np.corrcoef(a, f)[0, 1])
    ax.set_title(f"{label}    r = {r:+.2f}", fontsize=10)
    ax.set_xlabel("age"); ax.set_ylabel(ylab)
    return r

def latent_variance_order(Z):
    """Latent dimensions ordered by variance across subjects (highest first)."""
    return np.argsort(Z.var(axis=0))[::-1]

def plot_all_latent_features_vs_age(Z, age, out_dir, ncols=4, nrows=4):
    """
    Plot every latent dimension against age.

    Features are ordered by variance across subjects and saved
    as a single multi-page PDF.
    """
    if age is None:
        print("latent feature-age figures skipped: no age column in metadata")
        return

    age = np.asarray(age, dtype=float)
    order = np.argsort(Z.var(axis=0))[::-1]
    per_page = ncols * nrows
    n_pages = int(np.ceil(len(order) / per_page))

    pdf_path = os.path.join(out_dir, "latent_features_vs_age.pdf")
    print(
        f"creating latent feature-age PDF "
        f"({len(order)} dimensions across {n_pages} pages)"
    )
    with PdfPages(pdf_path) as pdf:
        for page_idx, start in enumerate(range(0, len(order), per_page), start=1,):
            dims = order[start:start + per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows), squeeze=False)
            flat_axes = axes.ravel()
            for ax_idx, (ax, latent_idx) in enumerate(zip(flat_axes, dims)):
                variance_rank = start + ax_idx + 1
                _scatter_vs_age(ax, age, Z[:, latent_idx],
                    f"Latent #{latent_idx} (var rank {variance_rank})",
                    "latent value",
                )

            for ax in flat_axes[len(dims):]:
                ax.axis("off")
            fig.suptitle(
                f"Latent dimensions vs age "
                f"(variance ranks {start+1}-{start+len(dims)})",
                fontsize=13,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=130)
            plt.close(fig)
    print(f"saved {pdf_path}")

def plot_latent_age_correlations(Z, age, out_dir):
    """
    Summary figure showing correlation with age for every latent
    dimension ordered by variance rank.
    """
    if age is None:
        print("latent age-correlation figure skipped: no age column in metadata")
        return

    age = np.asarray(age, dtype=float)
    order = latent_variance_order(Z)
    rs = np.full(len(order), np.nan, dtype=float)
    valid_age = np.isfinite(age)
    for i, latent_idx in enumerate(order):
        feat = Z[:, latent_idx]
        m = valid_age & np.isfinite(feat)
        if m.sum() >= 3 and np.ptp(feat[m]) > 0:
            rs[i] = np.corrcoef(age[m], feat[m])[0, 1]

    strongest = np.nanmax(np.abs(rs))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = np.where(rs >= 0, "#1C7293", "#D98324")
    ax.bar(np.arange(len(rs)), rs, color=colors, width=1.0)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("latent feature (ordered by variance rank)")
    ax.set_ylabel("Pearson r with age")
    ax.set_title(
        f"Age correlation across latent dimensions "
        f"(max |r| = {strongest:.2f})"
    )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latent_age_correlations.png"), dpi=130,)
    plt.close(fig)

def plot_top_latent_features_vs_age(Z, age, out_dir, topk=16):
    """
    Single-page overview of the highest-variance latent dimensions.
    Useful when you want a compact summary instead of all pages.
    """
    if age is None:
        print("top latent feature-age figure skipped: no age column in metadata")
        return
    age = np.asarray(age, dtype=float)

    order = latent_variance_order(Z)
    dims = order[:topk]

    ncols = 4
    nrows = int(np.ceil(topk / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows),squeeze=False,)
    flat_axes = axes.ravel()
    for rank, (ax, latent_idx) in enumerate(
        zip(flat_axes, dims),
        start=1,
    ):
        _scatter_vs_age(ax, age, Z[:, latent_idx], f"Latent #{latent_idx} (var rank {rank})", "latent value",)

    for ax in flat_axes[len(dims):]:
        ax.axis("off")

    fig.suptitle(f"Top {topk} latent dimensions vs age", fontsize=13,)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "latent_features_vs_age_top.png"),dpi=130,)
    plt.close(fig)

def plot_layer_features_vs_age(model, Xte_np, age, device, out_dir, topk=3):
    """Top-variance channels of each encoder layer (mean activation per subject) vs known age.
    Rows = encoder layers 1-4, columns = the layer's highest-variance channels. Variance ranking
    is age-agnostic, so the reported correlations are not selected on the outcome."""
    if age is None:
        print("layer feature-age figure skipped: no age column in metadata")
        return
    age = np.asarray(age, dtype=float)
    feats = encode_layer_features(model, Xte_np, device)             # [(N,16),(N,32),(N,64),(N,128)]
    fig, axes = plt.subplots(len(feats), topk, figsize=(4.3 * topk, 3.4 * len(feats)), squeeze=False)
    for li, F in enumerate(feats):
        for col, ch in enumerate(_topvar_cols(F, topk)):
            _scatter_vs_age(axes[li, col], age, F[:, ch],
                            f"Enc{li + 1}  ch #{ch} (var #{col + 1})", "mean activation")
    fig.suptitle("Top-variance encoder channels vs age (test)  ·  "
                 "rows = encoder layers 1-4, columns = highest-variance channels", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "layer_features_vs_age.png"), dpi=130)
    plt.close(fig)

def plot_all_layer_features_vs_age(model, Xte_np, age, device, out_dir, ncols=4, nrows=4):
    """
    Create one PDF per encoder layer.

    Each PDF contains scatterplots of ALL channels versus age,
    ordered by channel variance across test subjects.
    """

    if age is None:
        print("layer feature-age PDFs skipped: no age column")
        return

    age = np.asarray(age, dtype=float)
    feats = encode_layer_features(model, Xte_np, device)

    for layer_idx, F in enumerate(feats, start=1):
        n_channels = F.shape[1]
        order = np.argsort(F.var(axis=0))[::-1]
        per_page = ncols * nrows
        pdf_path = os.path.join(out_dir, f"layer{layer_idx}_features_vs_age.pdf")

        with PdfPages(pdf_path) as pdf:
            for start in range(0, n_channels, per_page):
                dims = order[start:start + per_page]
                fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows), squeeze=False,)
                flat_axes = axes.ravel()
                for ax_idx, (ax, ch) in enumerate(zip(flat_axes, dims)):
                    variance_rank = start + ax_idx + 1
                    _scatter_vs_age(ax, age, F[:, ch], (
                            f"Enc{layer_idx} "
                            f"ch #{ch} "
                            f"(var rank {variance_rank})"
                        ),
                        "mean activation",
                    )

                for ax in flat_axes[len(dims):]:
                    ax.axis("off")
                fig.suptitle(
                    f"Encoder layer {layer_idx} channels vs age\n"
                    f"variance ranks "
                    f"{start+1}-{start+len(dims)}",
                    fontsize=13,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig, dpi=130)
                plt.close(fig)
        print(f"saved {pdf_path}")

def plot_latent_variance_hist(Z, out_dir, bins=30):
    """Histogram comparing the variance of all latent features across test subjects."""
    v = Z.var(axis=0)                                                # (latent,)
    near_dead = int((v < 0.01 * v.max()).sum())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(v, bins=bins, color="#1C7293", alpha=0.85, edgecolor="white")
    ax.axvline(np.median(v), color="#D98324", ls="--", lw=1.5, label=f"median {np.median(v):.3g}")
    ax.set_xlabel("variance across test subjects")
    ax.set_ylabel("number of latent dimensions")
    ax.set_title(f"Latent feature variance  ·  {Z.shape[1]} dims, "
                 f"{near_dead} near-dead (< 1% of max)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "latent_variance_hist.png"), dpi=130)
    plt.close(fig)

def plot_latent_space(Z, age, out_dir):
    """Standalone latent-space overview: PCA & t-SNE embeddings (colored by age if available),
    explained-variance scree, and the correlation structure among latent dimensions."""
    from sklearn.manifold import TSNE
    N, D = Z.shape
    color = None if age is None else np.asarray(age, dtype=float)
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) linear PCA embedding
    pca = PCA(n_components=2).fit(Z)
    P = pca.transform(Z)
    evr = pca.explained_variance_ratio_
    sc = ax[0, 0].scatter(P[:, 0], P[:, 1], c=color, cmap="viridis", s=22, alpha=0.85)
    ax[0, 0].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax[0, 0].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax[0, 0].set_title("Latent PCA (linear)")
    if color is not None:
        fig.colorbar(sc, ax=ax[0, 0], fraction=0.046, label="age")

    # (0,1) nonlinear t-SNE embedding
    perp = int(min(30, max(5, N // 3)))
    try:
        T = TSNE(n_components=2, perplexity=perp, init="pca",
                 learning_rate="auto", random_state=0).fit_transform(Z)
        sc2 = ax[0, 1].scatter(T[:, 0], T[:, 1], c=color, cmap="viridis", s=22, alpha=0.85)
        ax[0, 1].set_xlabel("t-SNE 1"); ax[0, 1].set_ylabel("t-SNE 2")
        ax[0, 1].set_title(f"Latent t-SNE (perplexity {perp})")
        if color is not None:
            fig.colorbar(sc2, ax=ax[0, 1], fraction=0.046, label="age")
    except Exception as e:
        ax[0, 1].text(0.5, 0.5, f"t-SNE skipped\n{e}", ha="center", va="center", fontsize=9)
        ax[0, 1].set_axis_off()

    # (1,0) explained-variance scree (how many directions carry the variance)
    r = PCA().fit(Z).explained_variance_ratio_
    k = int(min(20, len(r)))
    cum = np.cumsum(r)
    n90 = int(np.argmax(cum >= 0.90) + 1)
    ax[1, 0].bar(np.arange(1, k + 1), r[:k], color="#1C7293", alpha=0.85, label="per-PC")
    ax[1, 0].plot(np.arange(1, k + 1), cum[:k], color="#D98324", marker="o", ms=3, label="cumulative")
    ax[1, 0].axhline(0.90, color="gray", ls=":", lw=1)
    ax[1, 0].set_xlabel("principal component"); ax[1, 0].set_ylabel("variance ratio")
    ax[1, 0].set_title(f"Explained variance  ·  {n90} PCs reach 90%")
    ax[1, 0].legend(fontsize=9)

    # (1,1) correlation among latent dimensions (redundancy / structure)
    C = np.nan_to_num(np.corrcoef(Z, rowvar=False))          # dead dims -> 0
    im = ax[1, 1].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax[1, 1].set_xlabel("latent dim"); ax[1, 1].set_ylabel("latent dim")
    ax[1, 1].set_title(f"Latent correlation ({D} dims)")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, label="correlation")

    fig.suptitle("Latent space overview (test set)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "latent_space_overview.png"), dpi=130)
    plt.close(fig)

def evaluate_age_decoding(Z, age, seed=0):
    """Cross-validated Ridge from latent code to age (the real 'does the code keep
    age?' test). Returns (R2, MAE_years) via 5-fold CV on the held-out latents, or None."""
    if age is None:
        return None
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    a = np.asarray(age, dtype=float)
    m = np.isfinite(a)
    Zf, af = Z[m], a[m]
    if len(af) < 10:
        return None
    k = int(min(5, len(af)))
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pred = cross_val_predict(pipe, Zf, af, cv=KFold(k, shuffle=True, random_state=seed))
    ss_res = float(np.sum((af - pred) ** 2)); ss_tot = float(np.sum((af - af.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(r2), float(np.mean(np.abs(af - pred)))

def mean_gm_age_corr(orig_density, mask, age):
    """Correlation of each subject's mean in-brain GM density with age — the probe that
    explains whether global level (which per-image norm discards) is age-informative."""
    if age is None:
        return None
    a = np.asarray(age, dtype=float)
    m = np.isfinite(a)
    meang = orig_density[:, mask].mean(axis=1)
    if m.sum() < 3:
        return None
    return float(np.corrcoef(meang[m], a[m])[0, 1])

def _append_norm_summary(norm, mse_density, mse_norm, age_dec, gm_corr, out_dir,
                         path="norm_comparison.csv"):
    """Append one row per run so the four --norm modes accumulate into one table."""
    import csv
    header = ["norm", "recon_mse_density", "recon_mse_normspace",
              "age_R2", "age_MAE_yr", "meanGM_age_r", "out_dir"]
    row = [norm, f"{mse_density:.6f}", f"{mse_norm:.6f}",
           "" if age_dec is None else f"{age_dec[0]:.4f}",
           "" if age_dec is None else f"{age_dec[1]:.4f}",
           "" if gm_corr is None else f"{gm_corr:.4f}", out_dir]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
    print(f"  appended comparison row to {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",         default="vbm_data_sliced/vbm_z60.npy")
    ap.add_argument("--tsv",         default="vbm_processing/participants_study_3.tsv")
    ap.add_argument("--out",         default="ae_run/ae_z60")
    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--batch-size",  type=int,   default=16)
    ap.add_argument("--lr",          type=float, default=8.3e-4)
    ap.add_argument("--weight-decay",type=float, default=3.9e-5)
    ap.add_argument("--latent",      type=int,   default=128)
    ap.add_argument("--noise-std",   type=float, default=0.1)
    ap.add_argument("--test-size",   type=float, default=0.15,
                    help="fraction for TEST (same for VAL) -> 70/15/15")
    ap.add_argument("--mask-frac",   type=float, default=0.05,
                    help="in-brain mask threshold as fraction of train-mean max")
    ap.add_argument("--norm",        default="global_z",
                    choices=["global_z", "per_image_z", "global_minmax", "scale_only"],
                    help="input normalization; sets the matching decoder output head")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n-vis",       type=int,   default=6)
    ap.add_argument("--tune",        action="store_true")
    ap.add_argument("--n-trials",    type=int,   default=30)
    ap.add_argument("--tune-epochs", type=int,   default=40)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. load + preprocess
    sl   = np.load(args.data)  # (N,128,128) float32
    meta = pd.read_csv(args.tsv, sep="\t")
    assert len(meta) == sl.shape[0], (len(meta), sl.shape[0])
    X   = crop_pad_128(sl).astype(np.float32)
    idx = np.arange(len(X))

    # 2. one-time subject-level split (70/15/15)
    tr_val, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    val_frac   = args.test_size / (1.0 - args.test_size)
    tr, va     = train_test_split(tr_val, test_size=val_frac, random_state=args.seed)

    # 3. normalization (TRAIN-fit for global modes) + TRAIN-only in-brain mask
    norm    = Normalizer(args.norm, X, tr)
    Xn      = norm.transform(X)
    out_act = OUT_ACT[args.norm]
    sd_repr, mu_repr = norm.repr_params()                           # for synthetic-image figures
    mask    = build_brain_mask(X[tr], frac=args.mask_frac)          # (128,128) bool, density space

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    mask_t = torch.from_numpy(mask).to(device)
    print(f"device: {device} | train {len(tr)} val {len(va)} test {len(te)} | "
          f"in-brain voxels {int(mask.sum())}/{mask.size}")

    # 4. [optional] Optuna search (objective = masked val MSE; test untouched)
    if args.tune:
        print(f"\nOptuna: {args.n_trials} trials x {args.tune_epochs} epochs "
              f"(TPE + median pruning, masked val MSE)")
        best_params, best_val = tune_hyperparams(
            Xn, tr, va, mask_t, device, args.out, args.n_trials, args.tune_epochs, os.path.basename(args.data).replace(".npy", ""),
            out_act=out_act)
        args.latent       = best_params["latent"]
        args.lr           = best_params["lr"]
        args.weight_decay = best_params["weight_decay"]
        args.batch_size   = best_params["batch_size"]
        args.noise_std    = best_params["noise_std"]
        print(f"  best masked val MSE {best_val:.4f} | params {best_params}")

    # 5. final training (gradient = full MSE; checkpoint = masked val MSE)
    n_params = sum(p.numel() for p in ConvAE2D(args.latent, out_act=out_act).parameters())
    print(f"\nFinal model: norm {args.norm} (head={out_act or 'linear'}) | "
          f"latent {args.latent} | params {n_params:,} | "
          f"lr {args.lr:.2e} | wd {args.weight_decay:.1e} | "
          f"batch {args.batch_size} | noise {args.noise_std}")
    torch.manual_seed(args.seed)
    snap_eps = sorted({0, args.epochs // 3, 2 * args.epochs // 3, args.epochs - 1})
    best_val, best_state, history, snaps = train_autoencoder(
        Xn, tr, va, mask_t, device,
        latent=args.latent, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, noise_std=args.noise_std,
        epochs=args.epochs, record=True, verbose=True,
        snap_X=Xn[va], snap_epochs=snap_eps, out_act=out_act)
    model = ConvAE2D(args.latent, out_act=out_act).to(device)
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(args.out, "ae_best.pth"))
    best_ep = int(np.argmin(history[1]))
    print(f"best masked val MSE {best_val:.4f} at epoch {best_ep}")

    # 6. SINGLE final evaluation on the sealed TEST set
    Xte_np = Xn[te]
    recon_np  = reconstruct_all(model, Xte_np, device)
    orig_raw  = norm.inverse(Xte_np, te)                             # back to GM density
    recon_raw = norm.inverse(recon_np, te)
    per_mse_norm = masked_mse_per_subject(Xte_np, recon_np, mask)    # normalized units (per-mode)
    per_mse      = masked_mse_per_subject(orig_raw, recon_raw, mask) # GM density (cross-mode comparable)
    print(f"[{args.norm}] test masked MSE  density mean {per_mse.mean():.5f} "
          f"median {np.median(per_mse):.5f} | norm-space mean {per_mse_norm.mean():.5f}")

    # 7. diagnostic figures
    plot_loss_and_lr(history, best_ep, args.out)
    plot_recon_panels(orig_raw, recon_raw, mask, per_mse, args.out, n=args.n_vis)
    pear, r2 = plot_pixel_scatter(orig_raw, recon_raw, mask, args.out)
    plot_mean_error_map(orig_raw, recon_raw, mask, args.out)
    plot_mse_hist(per_mse, args.out)

    Z_te = encode_all(model, Xte_np, device)
    np.save(os.path.join(args.out, "latent_features_test.npy"), Z_te)
    age = meta["age"].values[te] if "age" in meta.columns else None
    plot_latent_diagnostics(Z_te, args.out, color=age,
                            color_label="age" if age is not None else None)
    print(f"in-brain pixel scatter: r {pear:.4f}  R² {r2:.4f}")

    # 6b. normalization comparison: density-space recon (above) + latent->age decoding
    age_dec = evaluate_age_decoding(Z_te, age)
    gm_corr = mean_gm_age_corr(orig_raw, mask, age)
    if age_dec is not None:
        print(f"[{args.norm}] latent->age  CV R² {age_dec[0]:+.3f}  MAE {age_dec[1]:.2f} yr")
    if gm_corr is not None:
        print(f"[{args.norm}] mean in-brain GM density vs age:  r {gm_corr:+.3f}")
    _append_norm_summary(args.norm, per_mse.mean(), per_mse_norm.mean(),
                         age_dec, gm_corr, args.out)

    # 8. "how it works" figures for the presentation
    samp = int(_representative_idx(per_mse, 3)[1])                   # median-quality subject
    plot_feature_maps(model, Xte_np[samp], device, args.out)
    torch.manual_seed(args.seed + 1)
    rnd_model = ConvAE2D(args.latent, out_act=out_act).to(device)   # untrained reference
    plot_feature_maps_before_after(rnd_model, model, Xte_np[samp], device, args.out)
    plot_latent_interpolation(model, Z_te, Xte_np, sd_repr, mu_repr, device, args.out)
    va_age = meta["age"].values[va] if "age" in meta.columns else None
    plot_latent_over_epochs(snaps, va_age, args.out)
    plot_layer_features_vs_age(model, Xte_np, age, device, args.out)
    plot_all_layer_features_vs_age(model, Xte_np, age, device, args.out, ncols=4, nrows=4)
    plot_latent_variance_hist(Z_te, args.out)
    plot_latent_space(Z_te, age, args.out)

    # compact overview of highest-variance latent dimensions
    plot_top_latent_features_vs_age(Z_te, age, args.out, topk=16)
    # every latent dimension (all 128)
    plot_all_latent_features_vs_age(Z_te, age, args.out, ncols=4, nrows=4)
    # summary of age correlations across latent dimensions
    plot_latent_age_correlations(Z_te, age, args.out)

if __name__ == "__main__":
    main()