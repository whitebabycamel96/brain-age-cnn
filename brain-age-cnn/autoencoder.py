"""
2D convolutional autoencoder on z=60 VBM slices — reconstruction-focused,
leakage-safe, with Optuna hyperparameter tuning and a full diagnostic figure set.

Pipeline:
  1. load vbm_z60.npy (N,121,145) + participants_study_3.tsv (row-aligned)
  2. crop/pad each slice to 128x128
  3. one-time 70/15/15 subject-level split
  4. standardize with TRAIN stats only; build an in-brain mask from TRAIN
  5. [optional] Optuna search (--tune): TPE + median pruning, SQLite storage,
     objective = masked validation MSE
  6. train final AE (AdamW, cosine LR); checkpoint by masked val MSE
  7. write the diagnostic figures (see OUTPUTS at bottom)
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def crop_pad_128(sl):
    """(N,121,145) -> (N,128,128): centre-crop width 145->128, zero-pad height 121->128."""
    cropped = sl[:, :, 8:136]                                        # 145 -> 128 (centre)
    return np.pad(cropped, ((0, 0), (3, 4), (0, 0)),
                  mode="constant", constant_values=0)                # 121 -> 128

def build_brain_mask(X_raw_train, frac=0.05):
    """In-brain mask from the TRAIN mean image: voxels above frac * max.

    Built from training data only so it carries no information from val/test.
    """
    m = X_raw_train.mean(axis=0)
    return m > (frac * m.max())  # (128,128) bool

class ConvAE2D(nn.Module):
    """
    Encoder  : 4 strided Conv2d blocks  (Conv 3x3 s2 p1 + BN + ReLU)
               (1,128,128)->(16,64,64)->(32,32,32)->(64,16,16)->(128,8,8)
    Bottleneck: flatten 128*8*8=8192 -> Linear -> latent -> Linear -> 8192
    Decoder  : 4 ConvTranspose2d blocks (ConvT 4x4 s2 p1 + BN + ReLU)
               (128,8,8)->(64,16,16)->(32,32,32)->(16,64,64)->(1,128,128)
    No activation on the final layer (z-scored images -> unbounded output).
    """
    def __init__(self, latent: int = 512):
        super().__init__()

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
        return self.decoder(self.from_latent(z).view(-1, 128, 8, 8))

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
                      trial=None, record=False, verbose=False):
    """Train one AE. Gradient = full-image MSE; selection metric = masked val MSE.

    Returns (best_masked_val, best_state, history) where history is
    (train_full_mse, val_masked_mse, lr) lists (empty unless record=True).
    """
    Xtr_t = torch.from_numpy(Xn[tr]).unsqueeze(1)
    Xva_t = torch.from_numpy(Xn[va]).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(Xtr_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva_t), batch_size=batch_size, shuffle=False)

    model = ConvAE2D(latent).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    mse   = nn.MSELoss()

    best, best_state = float("inf"), None
    tr_hist, va_hist, lr_hist = [], [], []

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

        if trial is not None:                                        # Optuna pruning
            trial.report(v, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best, best_state, (tr_hist, va_hist, lr_hist)

# Hyperparameter tuning
def tune_hyperparams(Xn, tr, va, mask_t, device, out_dir, n_trials, tune_epochs):
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
        best, _, _ = train_autoencoder(Xn, tr, va, mask_t, device,
                                       epochs=tune_epochs, trial=trial, **params)
        return best

    storage = "sqlite:///" + os.path.abspath(os.path.join(out_dir, "optuna_study.db"))
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=storage, study_name="ae_recon", load_if_exists=True,
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
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, i in enumerate(idx):
        err = np.abs(orig_raw[i] - recon_raw[i]) * mask
        axes[row, 0].imshow(orig_raw[i].T, cmap="gray", origin="lower")
        axes[row, 1].imshow(recon_raw[i].T, cmap="gray", origin="lower")
        im = axes[row, 2].imshow(err.T, cmap="hot", origin="lower")
        axes[row, 0].set_ylabel(f"masked MSE\n{per_mse[i]:.3f}", fontsize=9)
        for c in range(3):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
    axes[0, 0].set_title("original"); axes[0, 1].set_title("reconstruction")
    axes[0, 2].set_title("|error| (in-brain)")
    fig.suptitle("Test reconstructions: best -> worst", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "recon_panels.png"), dpi=120)
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
    ax.set_title("Latent space (PCA of 512-d codes, test)")
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z60",         default="vbm_data_z60/vbm_z60.npy")
    ap.add_argument("--tsv",         default="vbm_processing/participants_study_3.tsv")
    ap.add_argument("--out",         default="ae_run")
    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--batch-size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight-decay",type=float, default=1e-5)
    ap.add_argument("--latent",      type=int,   default=512)
    ap.add_argument("--noise-std",   type=float, default=0.0)
    ap.add_argument("--test-size",   type=float, default=0.15,
                    help="fraction for TEST (same for VAL) -> 70/15/15")
    ap.add_argument("--mask-frac",   type=float, default=0.05,
                    help="in-brain mask threshold as fraction of train-mean max")
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
    sl   = np.load(args.z60)
    meta = pd.read_csv(args.tsv, sep="\t")
    assert len(meta) == sl.shape[0], (len(meta), sl.shape[0])
    X   = crop_pad_128(sl).astype(np.float32)
    idx = np.arange(len(X))

    # 2. one-time subject-level split (70/15/15)
    tr_val, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    val_frac   = args.test_size / (1.0 - args.test_size)
    tr, va     = train_test_split(tr_val, test_size=val_frac, random_state=args.seed)

    # 3. TRAIN-only standardization + TRAIN-only in-brain mask  (no leakage)
    mu, sd = X[tr].mean(), X[tr].std() + 1e-8
    Xn     = (X - mu) / sd
    mask   = build_brain_mask(X[tr], frac=args.mask_frac)            # (128,128) bool

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
            Xn, tr, va, mask_t, device, args.out, args.n_trials, args.tune_epochs)
        args.latent       = best_params["latent"]
        args.lr           = best_params["lr"]
        args.weight_decay = best_params["weight_decay"]
        args.batch_size   = best_params["batch_size"]
        args.noise_std    = best_params["noise_std"]
        print(f"  best masked val MSE {best_val:.4f} | params {best_params}")

    # 5. final training (gradient = full MSE; checkpoint = masked val MSE)
    n_params = sum(p.numel() for p in ConvAE2D(args.latent).parameters())
    print(f"\nFinal model: latent {args.latent} | params {n_params:,} | "
          f"lr {args.lr:.2e} | wd {args.weight_decay:.1e} | "
          f"batch {args.batch_size} | noise {args.noise_std}")
    torch.manual_seed(args.seed)
    best_val, best_state, history = train_autoencoder(
        Xn, tr, va, mask_t, device,
        latent=args.latent, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, noise_std=args.noise_std,
        epochs=args.epochs, record=True, verbose=True)
    model = ConvAE2D(args.latent).to(device)
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(args.out, "ae_best.pth"))
    best_ep = int(np.argmin(history[1]))
    print(f"best masked val MSE {best_val:.4f} at epoch {best_ep}")

    # 6. SINGLE final evaluation on the sealed TEST set
    Xte_np    = Xn[te]
    recon_np  = reconstruct_all(model, Xte_np, device)
    orig_raw  = Xte_np  * sd + mu                                    # back to GM density
    recon_raw = recon_np * sd + mu
    per_mse   = masked_mse_per_subject(Xte_np, recon_np, mask)       # std units, in-brain
    print(f"test masked MSE  mean {per_mse.mean():.4f}  "
          f"median {np.median(per_mse):.4f}  max {per_mse.max():.4f}")

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

if __name__ == "__main__":
    main()