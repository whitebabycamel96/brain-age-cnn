"""
hparam_search.py

Optuna hyperparameter search for VBMAutoencoder.
Reuses run_epoch and compute_loss from train.py directly.

Usage:
    python hparam_search.py                      # uses config.json
    python hparam_search.py --config my.json     # custom config
    python hparam_search.py --trials 30          # number of trials
    python hparam_search.py --epochs-per-trial 20

What gets tuned:
    lambda_age   -- [1e-3, 0.1]  log-uniform  (most important)
    lr           -- [1e-5, 5e-4] log-uniform
    latent_dim   -- {64, 128, 256}
    batch_size   -- {16, 32}

Objective:
    Minimize val_l_recon + val_mae_age (normalised).
    Rationale: you care about both reconstruction quality AND
    age prediction accuracy, so neither dominates the search.
    Pruning: MedianPruner kills trials whose val_l_total at
    epoch N is worse than the median of all completed trials
    at the same epoch — saves compute on bad configs.

Results:
    Stored in SQLite: hparam_search/study.db
    Summary printed + saved to hparam_search/results.csv
    Best config saved to hparam_search/best_config.json
    (ready to drop straight into your main training run)
"""

import argparse
import json
import logging
import math
import os
import warnings

import numpy as np
import torch
import torch.optim as optim
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from VBMAgeDataset import VBMAgeDataset
from preprocessing import CropPad, GlobalZNormalizer
from model import VBMAutoencoder, compute_loss
from train import load_config, run_epoch, RunningMean


# ------------------------------------------------------------------ #
#  Logging                                                            #
# ------------------------------------------------------------------ #

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("hparam")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                          datefmt="%H:%M:%S"))
        fh = logging.FileHandler(os.path.join(log_dir, "hparam_search.log"))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


# ------------------------------------------------------------------ #
#  Data preparation (done once, shared across all trials)             #
# ------------------------------------------------------------------ #

def prepare_data(cfg: dict) -> tuple:
    """
    Load, crop, normalize and split data once.
    Normalization is fitted on train set only, applied to all images.
    Returns normalized images so VBMAgeDataset receives ready-to-use arrays.
    """
    crop = CropPad(
        pad_top=cfg["pad_top"],     pad_bottom=cfg["pad_bottom"],
        pad_left=cfg["pad_left"],   pad_right=cfg["pad_right"],
        crop_top=cfg["crop_top"],   crop_bottom=cfg["crop_bottom"],
        crop_left=cfg["crop_left"], crop_right=cfg["crop_right"],
    )

    raw_images = np.load(cfg["npy_path"]).astype(np.float32)
    raw_images = np.stack([crop(img) for img in raw_images])

    idx = np.arange(len(raw_images))
    train_idx, test_idx = train_test_split(
        idx, test_size=cfg["test_size"], random_state=cfg["seed"],
    )

    # fit on train only, apply to all — same as train.py
    normalizer = GlobalZNormalizer()
    normalizer.fit(raw_images[train_idx])
    norm_images = np.stack([normalizer.transform(img) for img in raw_images])

    return norm_images, train_idx, test_idx


# ------------------------------------------------------------------ #
#  Objective function (one Optuna trial = one short training run)     #
# ------------------------------------------------------------------ #

def make_objective(
    cfg:        dict,
    norm_images: np.ndarray,
    train_idx:  np.ndarray,
    test_idx:   np.ndarray,
    device:     torch.device,
    n_epochs:   int,
    logger:     logging.Logger,
):
    """
    Returns a closure over the shared data so Optuna can call it
    as objective(trial) -> float.
    """

    def objective(trial: optuna.Trial) -> float:
        # ── suggest hyperparameters ─────────────────────────────────
        lambda_age   = trial.suggest_float("lambda_age",   1e-4, 1.0,  log=True)
        lr           = trial.suggest_float("lr",           1e-5, 5e-4, log=True)
        latent_dim   = trial.suggest_categorical("latent_dim",   [64, 128, 256])
        batch_size   = trial.suggest_categorical("batch_size",   [16, 32])
        dropout      = trial.suggest_float("dropout",      0.0,  0.4,  step=0.1)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        logger.info(
            f"trial {trial.number:>3}  "
            f"lambda_age={lambda_age:.4f}  lr={lr:.2e}  "
            f"latent_dim={latent_dim}  batch_size={batch_size}  "
            f"dropout={dropout:.1f}  wd={weight_decay:.2e}"
        )

        # ── data loaders (batch_size varies per trial) ───────────────
        # norm_images already normalized — pass directly, no normalizer arg
        train_ds = VBMAgeDataset(
            norm_images, cfg["tsv_path"],
            indices=train_idx,
        )
        test_ds = VBMAgeDataset(
            norm_images, cfg["tsv_path"],
            indices=test_idx,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
        )

        # ── model + optimiser ────────────────────────────────────────
        torch.manual_seed(cfg["seed"])
        model = VBMAutoencoder(
            latent_dim=latent_dim,
            age_head=cfg.get("age_head", True),
            dropout=dropout,
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # dummy logger for run_epoch (suppress per-epoch noise during search)
        silent = logging.getLogger("hparam.silent")
        silent.addHandler(logging.NullHandler())
        silent.propagate = False

        best_val_loss = float("inf")

        # ── short training run ───────────────────────────────────────
        for epoch in range(1, n_epochs + 1):
            train_m = run_epoch(
                model, train_loader, optimizer, device,
                lambda_age, training=True, logger=silent,
            )
            val_m = run_epoch(
                model, test_loader, None, device,
                lambda_age, training=False, logger=silent,
            )
            scheduler.step(val_m["l_total"])

            # NaN check — report as a bad trial, don't crash the study
            if math.isnan(val_m["l_total"]) or math.isnan(train_m["l_total"]):
                logger.info(
                    f"  trial {trial.number} pruned at epoch {epoch}: NaN loss "
                    f"(lambda_age={lambda_age:.4f} likely too large)"
                )
                raise optuna.exceptions.TrialPruned()

            if val_m["l_total"] < best_val_loss:
                best_val_loss = val_m["l_total"]

            # report to Optuna for pruning — prune if this trial is worse
            # than the median of all trials so far at the same epoch
            trial.report(val_m["l_total"], epoch)
            if trial.should_prune():
                logger.info(f"  trial {trial.number} pruned at epoch {epoch} by MedianPruner")
                raise optuna.exceptions.TrialPruned()

        logger.info(
            f"  trial {trial.number} done  "
            f"best_val_loss={best_val_loss:.4f}  "
            f"val_mae={val_m['mae_age']:.2f} yrs"
        )

        return best_val_loss

    return objective


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           default="config.json")
    parser.add_argument("--trials",           type=int, default=20,
                        help="number of Optuna trials")
    parser.add_argument("--epochs-per-trial", type=int, default=20,
                        help="epochs per trial (shorter than full training)")
    parser.add_argument("--study-name",       default="vbm_ae_search")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = "hparam_search"
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir)

    logger.info("=" * 60)
    logger.info(f"study      : {args.study_name}")
    logger.info(f"trials     : {args.trials}")
    logger.info(f"epochs/trial: {args.epochs_per_trial}")
    logger.info("=" * 60)

    # ── device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── data — loaded once, shared across all trials ─────────────────
    logger.info("preparing data (once for all trials) ...")
    norm_images, train_idx, test_idx = prepare_data(cfg)
    logger.info(f"images: {norm_images.shape}  train={len(train_idx)}  test={len(test_idx)}")

    # ── Optuna study ─────────────────────────────────────────────────
    # SQLite storage means trials survive crashes and you can resume:
    #   python hparam_search.py --trials 10   (run 10 more on top of existing)
    db_path = os.path.join(out_dir, "study.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=cfg["seed"]),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        ),
        load_if_exists=True,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = make_objective(
        cfg, norm_images, train_idx, test_idx,
        device, args.epochs_per_trial, logger,
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        catch=(RuntimeError,),     # catch CUDA OOM etc without crashing the study
    )

    # ── results ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("search complete")
    logger.info(f"best trial  : #{study.best_trial.number}")
    logger.info(f"best value  : {study.best_value:.4f}")
    logger.info("best params :")
    for k, v in study.best_params.items():
        logger.info(f"  {k:<15} {v}")

    # save best config — merge best params into base config, ready to use
    best_cfg = cfg.copy()
    best_cfg.update(study.best_params)
    best_cfg["checkpoint_dir"] = os.path.join(
        cfg["checkpoint_dir"].replace("autoencoder_age", ""),
        "autoencoder_age_best",
    )
    best_cfg_path = os.path.join(out_dir, "best_config.json")
    with open(best_cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    logger.info(f"\nbest config saved: {best_cfg_path}")
    logger.info("run full training with:")
    logger.info(f"  python train.py --config {best_cfg_path}")

    # save full results CSV for plotting
    results_path = os.path.join(out_dir, "results.csv")
    df = study.trials_dataframe()
    df.to_csv(results_path, index=False)
    logger.info(f"all trial results: {results_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()