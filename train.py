"""
train.py

Training loop for VBMAutoencoder.
Integrates with your existing VBMAgeDataset / GlobalZNormalizer pipeline.

Usage:
    python train.py                     # uses config.json
    python train.py --config my.json    # custom config path

Expected config.json fields:
    npy_path        : str   -- path to .npy file of raw VBM images
    tsv_path        : str   -- path to .tsv file with subject ages
    batch_size      : int   -- e.g. 16
    latent_dim      : int   -- bottleneck size, e.g. 128
    lambda_age      : float -- weight on age loss, e.g. 0.01
    lr              : float -- learning rate, e.g. 1e-3
    epochs          : int   -- e.g. 100
    checkpoint_dir  : str   -- directory to save checkpoints
    age_head        : bool  -- whether to use the regression head
    num_workers     : int   -- dataloader workers, 0 for debugging
"""

import argparse
import csv
import json
import logging
import math
import os
import time
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from VBMAgeDataset import VBMAgeDataset
from preprocessing import CropPad, GlobalZNormalizer
from model import VBMAutoencoder, compute_loss


# ------------------------------------------------------------------ #
#  Config                                                             #
# ------------------------------------------------------------------ #

def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("latent_dim",     128)
    cfg.setdefault("lambda_age",     0.01)
    cfg.setdefault("lr",             1e-4)
    cfg.setdefault("epochs",         100)
    cfg.setdefault("checkpoint_dir", "checkpoints")
    cfg.setdefault("age_head",       True)
    cfg.setdefault("num_workers",    4)
    cfg.setdefault("seed",           42)
    cfg.setdefault("test_size",      0.2)
    cfg.setdefault("dropout",        0.1)
    cfg.setdefault("weight_decay",   1e-4)
    # CropPad defaults match your preprocessing class defaults
    cfg.setdefault("pad_top",        3)
    cfg.setdefault("pad_bottom",     4)
    cfg.setdefault("pad_left",       0)
    cfg.setdefault("pad_right",      0)
    cfg.setdefault("crop_top",       0)
    cfg.setdefault("crop_bottom",    0)
    cfg.setdefault("crop_left",      8)
    cfg.setdefault("crop_right",     9)
    return cfg


# ------------------------------------------------------------------ #
#  Logging setup                                                      #
# ------------------------------------------------------------------ #

def setup_logging(log_dir: str) -> logging.Logger:
    """
    Sets up a logger that writes to both console and a timestamped log file.

    Console: INFO level, human-readable
    File:    DEBUG level, includes timestamps and module name

    Log file lives in log_dir/train_YYYYMMDD_HHMMSS.log
    so each run gets its own file and you never overwrite a previous run.
    """
    os.makedirs(log_dir, exist_ok=True)
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"train_{run_id}.log")

    logger = logging.getLogger("vbm_ae")
    logger.setLevel(logging.DEBUG)

    # console handler -- INFO and above, no timestamp (clean stdout)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    # file handler -- DEBUG and above, full timestamp + level
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"log file: {logfile}")
    return logger, run_id


# ------------------------------------------------------------------ #
#  CSV metrics writer                                                 #
# ------------------------------------------------------------------ #

class MetricsCSV:
    """
    Appends one row per epoch to a CSV file.
    Columns: epoch, split, l_total, l_recon, l_age, mae_age,
             grad_norm, lr, elapsed_s, timestamp

    Load for plotting:
        import pandas as pd
        df = pd.read_csv("checkpoints/metrics.csv")
        df[df.split == "train"].plot(x="epoch", y="l_recon")
    """

    FIELDS = [
        "epoch", "split", "l_total", "l_recon", "l_age",
        "mae_age", "grad_norm", "lr", "elapsed_s", "timestamp",
    ]

    def __init__(self, path: str):
        self.path = path
        # write header only if file doesn't exist yet
        # (so resuming a run appends rather than overwriting)
        write_header = not os.path.exists(path)
        self._f   = open(path, "a", newline="")
        self._csv = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        if write_header:
            self._csv.writeheader()
            self._f.flush()

    def write(self, row: dict):
        # fill any missing fields with empty string
        full_row = {k: row.get(k, "") for k in self.FIELDS}
        self._csv.writerow(full_row)
        self._f.flush()   # flush every row so the file is readable mid-run

    def close(self):
        self._f.close()


# ------------------------------------------------------------------ #
#  Metrics helpers                                                    #
# ------------------------------------------------------------------ #

class RunningMean:
    """Accumulates a batch-size-weighted running mean."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._n   = 0

    def update(self, val: float, n: int = 1):
        self._sum += val * n
        self._n   += n

    @property
    def mean(self) -> float:
        return self._sum / self._n if self._n > 0 else float("nan")


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    L2 norm of all parameter gradients, concatenated.
    Logged after each backward pass to catch exploding/vanishing gradients.
    Returns nan if no gradients exist yet.
    """
    total = 0.0
    n     = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
            n += 1
    return math.sqrt(total) if n > 0 else float("nan")


# ------------------------------------------------------------------ #
#  One epoch                                                          #
# ------------------------------------------------------------------ #

def run_epoch(
    model:      VBMAutoencoder,
    loader:     DataLoader,
    optimizer:  Optional[optim.Optimizer],
    device:     torch.device,
    lambda_age: float,
    training:   bool,
    logger:     logging.Logger,
) -> dict:
    """
    Run one full pass through loader.

    Returns dict of epoch-mean metrics:
        l_total, l_recon, l_age  -- MSE losses
        mae_age                  -- mean absolute error in years (interpretable)
        grad_norm                -- mean grad norm across batches (train only)
    """
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()

    meters = {k: RunningMean() for k in
              ["l_total", "l_recon", "l_age", "mae_age", "grad_norm"]}

    with ctx:
        for batch in loader:
            x   = batch["image"].to(device)        # (B, 1, 128, 128)
            age = batch["age"].float().to(device)  # (B,)

            if training:
                optimizer.zero_grad()

            out    = model(x)
            losses = compute_loss(out, x, age, lambda_age)

            if training:
                losses["l_total"].backward()
                # log grad norm BEFORE clipping so you see the raw signal
                gn = compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                gn = float("nan")

            b = x.size(0)
            meters["l_total"].update(losses["l_total"].item(), b)
            meters["l_recon"].update(losses["l_recon"].item(), b)
            meters["l_age"].update(losses["l_age"].item(), b)
            meters["grad_norm"].update(gn if not math.isnan(gn) else 0.0, b)

            # MAE on age -- only meaningful if regression head is active
            if "age_hat" in out:
                mae = (out["age_hat"].detach() - age).abs().mean().item()
                meters["mae_age"].update(mae, b)

    return {k: m.mean for k, m in meters.items()}


# ------------------------------------------------------------------ #
#  Main training loop                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    # ── logging ─────────────────────────────────────────────────────
    log_dir = cfg["checkpoint_dir"]
    logger, run_id = setup_logging(log_dir)

    # log full config at the top of every run for reproducibility
    logger.info("=" * 70)
    logger.info(f"run id  : {run_id}")
    logger.info(f"config  : {args.config}")
    logger.debug(f"full config:\n{json.dumps(cfg, indent=2)}")
    logger.info("=" * 70)

    # ── metrics CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(cfg["checkpoint_dir"], "metrics.csv")
    metrics_csv = MetricsCSV(csv_path)
    logger.info(f"metrics csv: {csv_path}")

    try:
        # ── device ──────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"device: {device}")
        if device.type == "cuda":
            logger.info(f"gpu: {torch.cuda.get_device_name(0)}")
            logger.debug(f"cuda memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # ── data ────────────────────────────────────────────────────
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        logger.info(f"seed: {cfg['seed']}")
        logger.info("loading data ...")

        # build CropPad entirely from config — no hardcoded values
        crop = CropPad(
            pad_top=cfg["pad_top"],
            pad_bottom=cfg["pad_bottom"],
            pad_left=cfg["pad_left"],
            pad_right=cfg["pad_right"],
            crop_top=cfg["crop_top"],
            crop_bottom=cfg["crop_bottom"],
            crop_left=cfg["crop_left"],
            crop_right=cfg["crop_right"],
        )
        logger.debug(
            f"CropPad: pad=({cfg['pad_top']},{cfg['pad_bottom']},"
            f"{cfg['pad_left']},{cfg['pad_right']}) "
            f"crop=({cfg['crop_top']},{cfg['crop_bottom']},"
            f"{cfg['crop_left']},{cfg['crop_right']})"
        )

        raw_images = np.load(cfg["npy_path"]).astype(np.float32)
        logger.info(f"raw images loaded: {raw_images.shape}")

        raw_images = np.stack([crop(img) for img in raw_images])
        logger.info(f"after crop/pad: {raw_images.shape}")

        idx = np.arange(len(raw_images))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=cfg["test_size"],
            random_state=cfg["seed"],
        )
        logger.info(
            f"train / test split: {len(train_idx)} / {len(test_idx)} "
            f"(test_size={cfg['test_size']}, seed={cfg['seed']})"
        )

        normalizer = GlobalZNormalizer()
        normalizer.fit(raw_images[train_idx])
        logger.debug(f"normalizer mu={normalizer.mu:.4f}, sd={normalizer.sd:.4f}")

        # apply normalization before passing to dataset
        norm_images = np.stack([normalizer.transform(img) for img in raw_images])
        logger.info(f"images normalized: mu={normalizer.mu:.4f}, sd={normalizer.sd:.4f}")

        train_ds = VBMAgeDataset(
            norm_images, cfg["tsv_path"],
            indices=train_idx,
        )
        test_ds = VBMAgeDataset(
            norm_images, cfg["tsv_path"],
            indices=test_idx,
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
        )

        # sanity check
        sample = train_ds[0]
        logger.info(f"sample image shape : {sample['image'].shape}")
        logger.info(f"sample age         : {sample['age']:.1f}")
        logger.info(f"sample subject_id  : {sample['subject_id']}")

        # ── model ───────────────────────────────────────────────────
        model = VBMAutoencoder(
            latent_dim=cfg["latent_dim"],
            age_head=cfg["age_head"],
            dropout=cfg["dropout"],
        ).to(device)

        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.debug(
            f"latent_dim={cfg['latent_dim']}, age_head={cfg['age_head']}, "
            f"lambda_age={cfg['lambda_age']}, dropout={cfg['dropout']}, "
            f"weight_decay={cfg['weight_decay']}"
        )

        # ── optimiser + scheduler ───────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        # ── checkpoint dir ──────────────────────────────────────────
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

        # ── table header ────────────────────────────────────────────
        header = (
            f"\n{'epoch':>6}  {'split':>6}  "
            f"{'l_total':>10}  {'l_recon':>10}  {'l_age':>10}  "
            f"{'mae_age':>8}  {'grad_norm':>10}  {'lr':>8}  {'time':>7}"
        )
        logger.info(header)
        logger.info("-" * 95)

        # ── training loop ───────────────────────────────────────────
        best_val_loss = float("inf")
        history       = []

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()

            # --- train ---
            train_m = run_epoch(
                model, train_loader, optimizer, device,
                cfg["lambda_age"], training=True, logger=logger,
            )

            # --- validate ---
            val_m = run_epoch(
                model, test_loader, None, device,
                cfg["lambda_age"], training=False, logger=logger,
            )

            scheduler.step(val_m["l_total"])
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # --- log both splits ---
            for split, m in [("train", train_m), ("val", val_m)]:
                gn_str = f"{m['grad_norm']:>10.4f}" if split == "train" else f"{'—':>10}"
                row = (
                    f"{epoch:>6}  {split:>6}  "
                    f"{m['l_total']:>10.4f}  {m['l_recon']:>10.4f}  {m['l_age']:>10.4f}  "
                    f"{m['mae_age']:>8.2f}  {gn_str}  {current_lr:>8.2e}  {elapsed:>6.1f}s"
                )
                logger.info(row)

                # write to CSV
                metrics_csv.write({
                    "epoch":     epoch,
                    "split":     split,
                    "l_total":   round(m["l_total"],   6),
                    "l_recon":   round(m["l_recon"],   6),
                    "l_age":     round(m["l_age"],     6),
                    "mae_age":   round(m["mae_age"],   4),
                    "grad_norm": round(m["grad_norm"], 6) if split == "train" else "",
                    "lr":        current_lr,
                    "elapsed_s": round(elapsed, 2)    if split == "train" else "",
                    "timestamp": ts,
                })

            # --- checkpoint if best ---
            if val_m["l_total"] < best_val_loss:
                best_val_loss = val_m["l_total"]
                ckpt_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
                torch.save({
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_loss":   best_val_loss,
                    "cfg":        cfg,
                    "run_id":     run_id,
                }, ckpt_path)
                logger.info(f"         -> best checkpoint saved  (val_loss={best_val_loss:.4f})")

            # --- early stopping signal (log only, not implemented as hard stop) ---
            if math.isnan(train_m["l_total"]):
                logger.error(f"NaN loss detected at epoch {epoch} — stopping early")
                logger.error("check lambda_age scaling; age MSE likely dominates and explodes")
                break

            history.append({
                "epoch":    epoch,
                "run_id":   run_id,
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}":   v for k, v in val_m.items()},
                "lr":       current_lr,
            })

        # ── save final checkpoint + history ─────────────────────────
        torch.save({
            "epoch":     cfg["epochs"],
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg":       cfg,
            "run_id":    run_id,
        }, os.path.join(cfg["checkpoint_dir"], "final.pt"))

        history_path = os.path.join(cfg["checkpoint_dir"], "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info("=" * 70)
        logger.info(f"training complete")
        logger.info(f"best val loss : {best_val_loss:.4f}")
        logger.info(f"checkpoints   : {cfg['checkpoint_dir']}")
        logger.info(f"metrics csv   : {csv_path}")
        logger.info(f"history json  : {history_path}")
        logger.info("=" * 70)

    except Exception:
        # log the full traceback to the file so cluster jobs don't lose errors
        logger.error("uncaught exception during training:")
        logger.error(traceback.format_exc())
        raise

    finally:
        metrics_csv.close()


if __name__ == "__main__":
    main()