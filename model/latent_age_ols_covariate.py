import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use("Agg")
from scipy import stats

def build_covariate_matrix(meta_df, subject_indices):
    """
    Build the nuisance covariate matrix C for the subjects in subject_indices.

    Covariates included (all chosen from participants_study_3.tsv):
      - sex       : binary dummy, female=1, male=0
                    Brain morphology differs systematically by sex.
      - tiv       : total intracranial volume, z-scored across test subjects
                    Standard VBM covariate; controls for head-size variation
                    that inflates grey-matter activation maps.
      - site      : one-hot (K-1 dummies, reference = most common site)
                    17 acquisition sites with very different age distributions
                    (site 4 mean=14.7 yr, site 46 mean=69.1 yr) — strong
                    site-age confound that must be removed.

    Excluded (zero-variance in study 3):
      - magnetic_field_strength  : all 3.0 T
      - acquisition_setting      : all 1.0
      - diagnosis                : all "control"

    Excluded (post-treatment mediators, not confounds):
      - csfv, gmv, wmv           : VBM-derived volumes; over-controls the
                                   signal of interest

    Returns
    -------
    C : (n, q)  float64 covariate matrix, NO intercept column
        (the intercept is added jointly with the feature in ols_covariate_batch)
    covariate_names : list of q strings
    """
    sub = meta_df.iloc[subject_indices].reset_index(drop=True)

    # sex dummy (female = 1)
    sex_dummy = (sub["sex"] == "female").astype(np.float64).values[:, None]

    # TIV z-scored across the supplied subjects (test set only)
    tiv = sub["tiv"].values.astype(np.float64)
    tiv_z = ((tiv - tiv.mean()) / (tiv.std() + 1e-8))[:, None]

    # site dummies (K-1, drop most-common site as reference)
    site_vals   = sub["site"].values
    ref_site    = meta_df["site"].value_counts().idxmax()   # site 4 (n=84)
    other_sites = sorted(s for s in np.unique(site_vals) if s != ref_site)
    if other_sites:
        site_dummies = np.column_stack([
            (site_vals == s).astype(np.float64) for s in other_sites
        ])                                         # (n, K-1)
        C = np.hstack([sex_dummy, tiv_z, site_dummies])
        covariate_names = (
            ["sex_female", "tiv_z"]
            + [f"site_{s}" for s in other_sites]
        )
    else:
        C = np.hstack([sex_dummy, tiv_z])          # all subjects same site
        covariate_names = ["sex_female", "tiv_z"]
    return C, covariate_names


def ols_covariate_batch(X_feat, y, C):
    """
    Vectorised partial OLS for V features simultaneously, controlling for
    the nuisance covariate matrix C.

    Model for feature v:
        y = beta0 + beta1_v * x_v + C @ gamma_v + e_v,   e_v ~ N(0, s2_v)

    Design matrix for feature v:
        D_v = [1  x_v  C]   shape (n, 1 + 1 + q) = (n, q+2)

    The coefficient of interest is beta1_v (position 1 in D_v).

    Implementation — Frisch-Waugh-Lovell (FWL) theorem
    ---------------------------------------------------
    Projecting out the nuisance columns first is numerically identical to
    fitting the full design matrix but avoids forming V different (q+2)×(q+2)
    systems.  Instead we do ONE regression of each x_v on [1, C] and ONE
    regression of y on [1, C], then bivariate OLS on the residuals.

    Let M = I - W(W^T W)^{-1} W^T   be the annihilator of W = [1, C].

        x_v_tilde = M x_v    (residual of x_v after regressing on [1,C])
        y_tilde   = M y      (residual of y   after regressing on [1,C])

    Then:
        beta1_v = (x_v_tilde . y_tilde) / (x_v_tilde . x_v_tilde)
        RSS_v   = ||y_tilde - beta1_v * x_v_tilde||^2
        df      = n - q - 2      (n minus intercept, feature, q covariates)
        s2_v    = RSS_v / df
        SE_v    = sqrt(s2_v / ||x_v_tilde||^2)
        t_v     = beta1_v / SE_v    ~ t(df) under H0: beta1_v = 0

    Parameters
    ----------
    X_feat : (n, V)  feature matrix (channels or latent dims)
    y      : (n,)    outcome (age)
    C      : (n, q)  nuisance covariates (NO intercept column)

    Returns
    -------
    beta1, se, t, p_val : each (V,),  df_resid : int
    """
    n, V = X_feat.shape
    q    = C.shape[1]
    df   = n - q - 2          # intercept + feature + q covariates

    # W = [intercept | covariates]
    W  = np.hstack([np.ones((n, 1)), C])          # (n, q+1)
    # projection onto W: hat matrix columns
    WtW_inv = np.linalg.pinv(W.T @ W)             # (q+1, q+1)
    H_W     = W @ WtW_inv @ W.T                   # (n, n)  — only needed via product
    # annihilator: M = I - H_W  →  M @ v = v - H_W @ v
    def annihilate(V_mat):
        return V_mat - H_W @ V_mat

    y_tilde   = annihilate(y)                     # (n,)
    X_tilde   = annihilate(X_feat)                # (n, V)

    # FWL bivariate OLS on residuals
    SXX = (X_tilde ** 2).sum(axis=0)              # (V,)
    SXY = X_tilde.T @ y_tilde                     # (V,)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta1 = np.where(SXX > 0, SXY / SXX, 0.0)

    resid = y_tilde[:, None] - beta1[None, :] * X_tilde   # (n, V)
    RSS   = (resid ** 2).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        s2  = RSS / df
        se  = np.where(SXX > 0, np.sqrt(s2 / SXX), np.inf)
        t   = np.where(se > 0, beta1 / se, 0.0)

    p_val = 2.0 * stats.t.sf(np.abs(t), df=df)
    return beta1, se, t, p_val, df

def bh_fdr(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR (1995).

    q_i = min_{j >= rank(i)} [ p_(j) · V / j ]   (step-up).
    Returns q-values and boolean rejection array.
    """
    V     = len(p_values)
    order = np.argsort(p_values)
    p_s   = p_values[order]
    q_s   = np.minimum.accumulate(
        (p_s * V / np.arange(1, V + 1))[::-1]
    )[::-1]
    q_s   = np.clip(q_s, 0.0, 1.0)
    q     = np.empty_like(q_s)
    q[order] = q_s
    return q, q < alpha

def run_stage(X_feat, y, C, alpha, label):
    """
    Partial OLS (covariate-adjusted) for every column of X_feat vs y,
    then BH-FDR.

    Parameters
    ----------
    X_feat : (n, V)  feature matrix
    y      : (n,)    outcome (age)
    C      : (n, q)  nuisance covariate matrix (sex, tiv_z, site dummies)
    alpha  : float   BH-FDR level
    label  : str     stage name

    Returns a tidy DataFrame with columns:
        stage, feature, channel, beta1, se, t, p_uncorr, q_bh, significant, df_resid
    """
    beta1, se, t, p, df_resid = ols_covariate_batch(X_feat, y, C)
    q_bh, rej                 = bh_fdr(p, alpha=alpha)

    V  = X_feat.shape[1]
    df = pd.DataFrame({
        "stage":       label,
        "feature":     [f"{label}_ch{c}" for c in range(V)],
        "channel":     np.arange(V),
        "beta1":       beta1,
        "se":          se,
        "t":           t,
        "p_uncorr":    p,
        "q_bh":        q_bh,
        "significant": rej,
        "df_resid":    df_resid,
    })
    n_sig = int(rej.sum())
    print(f"  {label:12s}  V={V:3d}  n={len(y)}  df={df_resid}"
          f"  sig={n_sig}/{V}"
          f"  |t|_max={np.abs(t).max():.2f}"
          f"  |t|_median={np.median(np.abs(t)):.2f}")
    return df

def _draw_coef_map(ax, df_stage, alpha, title_fontsize=10, show_legend_text=False):
    stage = df_stage["stage"].iloc[0]
    beta  = df_stage["beta1"].values
    sig   = df_stage["significant"].values
    V     = len(beta)

    colors = []
    for b, s in zip(beta, sig):
        if not s:
            colors.append("#CCCCCC")
        elif b > 0:
            colors.append("#D63B3B")
        else:
            colors.append("#1C7293")

    ax.bar(np.arange(V), beta, color=colors, alpha=0.9, width=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("β̂₁", fontsize=8)
    n_sig = int(sig.sum())
    # subtitle: stage info only — legend key goes in a separate textbox
    ax.set_title(
        f"{stage}  —  V={V},  sig={n_sig}/{V},  BH-FDR < {alpha}",
        fontsize=title_fontsize, pad=4
    )
    ax.tick_params(labelsize=7)

def plot_coef_map_single(df_stage, out_dir, alpha):
    """Save one stand-alone coef_map_<stage>.png."""
    stage = df_stage["stage"].iloc[0]
    V     = len(df_stage)
    fig, ax = plt.subplots(figsize=(max(6, V * 0.12), 3.5))
    _draw_coef_map(ax, df_stage, alpha)
    ax.set_xlabel("Channel index", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, f"coef_map_{stage}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")

def plot_coef_map_5x1_grid(stage_dfs, out_dir, alpha):
    Vs    = [len(df) for df in stage_dfs]
    V_max = max(Vs)

    # ── Figure-fraction layout constants ────────────────────────────────
    left        = 0.09    # left edge of every bar panel (aligned)
    right       = 0.97    # rightmost edge for the widest panel
    fig_top     = 0.93    # below the suptitle
    fig_bottom  = 0.04
    row_gap     = 0.030   # vertical gap between rows (gives breathing room)
    title_frac  = 0.22    # fraction of each slot reserved for the title text
                          # (remaining 1-title_frac is the bar area)

    n_rows      = len(stage_dfs)
    total_h     = fig_top - fig_bottom
    slot_h      = (total_h - row_gap * (n_rows - 1)) / n_rows
    bar_h       = slot_h * (1.0 - title_frac)
    title_h     = slot_h * title_frac
    panel_width = right - left

    fig = plt.figure(figsize=(14, 3.2 * n_rows))

    axes = []
    for row_idx, df_stage in enumerate(stage_dfs):
        V  = len(df_stage)
        w  = panel_width * (V / V_max)          # proportional width

        # slot bottom, then bar sits at bottom of slot, title gap above it
        slot_bottom = fig_top - (row_idx + 1) * slot_h - row_idx * row_gap
        bar_bottom  = slot_bottom                # bars flush at slot floor
        # title text lives in [bar_top … slot_top]; matplotlib set_title
        # puts it just above the axes, so we just need enough clearance
        # — achieved by making the axes only as tall as bar_h.

        ax = fig.add_axes([left, bar_bottom, w, bar_h])
        _draw_coef_map(ax, df_stage, alpha, title_fontsize=9)
        if row_idx == n_rows - 1:
            ax.set_xlabel("Channel / latent-dim index", fontsize=8)
        axes.append((ax, V, w, bar_bottom, bar_h))

    # ── Colour-key textbox in blank space right of row 0 (layer1) ───────
    ax0, V0, w0, b0, h0 = axes[0]

    # figure-fraction coordinates of the blank area to the right of row 0
    legend_left   = left + w0 + 0.015          # small gap after the bar panel
    legend_bottom = b0                          # align with bar bottom
    legend_top    = b0 + h0 + title_h          # up to top of title region
    legend_cx     = (legend_left + right) / 2  # horizontal centre
    legend_cy     = (legend_bottom + legend_top) / 2

    legend_lines = [
        ("■", "#D63B3B", " pos. slope, significant"),
        ("■", "#1C7293", " neg. slope, significant"),
        ("■", "#CCCCCC", " not significant (n.s.)"),
    ]

    # draw each swatch + label as separate text calls so we can colour the square
    line_spacing = 0.05   # in figure-fraction units
    n_lines      = len(legend_lines)
    y_starts     = [legend_cy + (i - (n_lines - 1) / 2) * line_spacing
                    for i in range(n_lines - 1, -1, -1)]

    # overlay each coloured square + black label
    for (sq, color, label), y in zip(legend_lines, y_starts):
        fig.text(legend_cx - 0.005, y, sq,
                 ha="right", va="center", fontsize=9,
                 color=color, transform=fig.transFigure,
                 fontweight="bold")
        fig.text(legend_cx - 0.005, y, label,
                 ha="left",  va="center", fontsize=8,
                 color="#333333", transform=fig.transFigure)

    # ── Suptitle ─────────────────────────────────────────────────────────
    fig.suptitle(
        "β̂₁ coefficient maps — univariate OLS (age ~ feature), test subjects  "
        f"|  BH-FDR α={alpha}  |  left-aligned, width ∝ V",
        fontsize=11, y=0.975
    )
    path = os.path.join(out_dir, "coef_map_5x1_grid.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# t-statistic distribution plot (empirical vs theoretical t(df))
# ─────────────────────────────────────────────────────────────────────────────
def plot_t_distribution(df_stage, n_subjects, out_dir, alpha):
    stage  = df_stage["stage"].iloc[0]
    t_vals = df_stage["t"].values.astype(float)
    sig    = df_stage["significant"].values
    V      = len(t_vals)
    df_t   = n_subjects - 2

    # x-range: cover the data plus the theoretical tails
    x_abs  = max(np.abs(t_vals).max() * 1.15, 4.0)
    xs     = np.linspace(-x_abs, x_abs, 600)
    null_y = stats.t.pdf(xs, df=df_t)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # ── theoretical null density ─────────────────────────────────────────
    ax.fill_between(xs, null_y, alpha=0.15, color="#888888")
    ax.plot(xs, null_y, color="#555555", lw=1.5, ls="--",
            label=f"t({df_t}) null")

    # ── empirical histogram of t-values ──────────────────────────────────
    # bins: Scott's rule, but at least 5 bins and at most V bins
    bw    = 3.49 * t_vals.std() * V ** (-1 / 3)
    n_bin = max(5, min(V, int(np.ceil(2 * x_abs / bw))))
    ax.hist(t_vals, bins=n_bin, range=(-x_abs, x_abs),
            density=True, color="#1C7293", alpha=0.55,
            edgecolor="white", linewidth=0.4,
            label=f"observed t  (V={V})")

    # ── critical thresholds ──────────────────────────────────────────────
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=df_t))   # uncorrected α
    ax.axvline( t_crit, color="#D63B3B", lw=1.2, ls=":",
                label=f"±t_{{α/2}}={t_crit:.2f}  (α={alpha}, uncorr.)")
    ax.axvline(-t_crit, color="#D63B3B", lw=1.2, ls=":")

    # ── rug: significant channels ────────────────────────────────────────
    t_sig = t_vals[sig]
    if len(t_sig) > 0:
        ax.plot(t_sig, np.full_like(t_sig, -0.003), "|",
                color="#D63B3B", ms=10, mew=1.5,
                label=f"BH-sig channels (n={len(t_sig)})")

    ax.set_xlabel("t-statistic", fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.set_xlim(-x_abs, x_abs)
    ax.set_ylim(bottom=-0.012)
    ax.set_title(
        f"{stage}  —  t-statistic distribution  (V={V} features, df={df_t})\n"
        f"Grey curve: t({df_t}) null  |  histogram: observed t-values  |  "
        f"BH-FDR sig={sig.sum()}/{V}",
        fontsize=10
    )
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, f"t_dist_{stage}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def plot_t_distribution_5x1_grid(stage_dfs, n_subjects, out_dir, alpha):
    """Save all t-distribution panels in a single 5-row figure."""
    n_rows = len(stage_dfs)
    df_t   = n_subjects - 2

    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, df_stage in zip(axes, stage_dfs):
        stage  = df_stage["stage"].iloc[0]
        t_vals = df_stage["t"].values.astype(float)
        sig    = df_stage["significant"].values
        V      = len(t_vals)

        x_abs  = max(np.abs(t_vals).max() * 1.15, 4.0)
        xs     = np.linspace(-x_abs, x_abs, 600)
        null_y = stats.t.pdf(xs, df=df_t)

        ax.fill_between(xs, null_y, alpha=0.15, color="#888888")
        ax.plot(xs, null_y, color="#555555", lw=1.5, ls="--",
                label=f"t({df_t}) null")

        bw    = 3.49 * t_vals.std() * V ** (-1 / 3)
        n_bin = max(5, min(V, int(np.ceil(2 * x_abs / bw))))
        ax.hist(t_vals, bins=n_bin, range=(-x_abs, x_abs),
                density=True, color="#1C7293", alpha=0.55,
                edgecolor="white", linewidth=0.4,
                label=f"observed t  (V={V})")

        t_crit = float(stats.t.ppf(1 - alpha / 2, df=df_t))
        ax.axvline( t_crit, color="#D63B3B", lw=1.2, ls=":",
                    label=f"±t_{{α/2}}={t_crit:.2f}  (α={alpha}, uncorr.)")
        ax.axvline(-t_crit, color="#D63B3B", lw=1.2, ls=":")

        t_sig = t_vals[sig]
        if len(t_sig) > 0:
            ax.plot(t_sig, np.full_like(t_sig, -0.003), "|",
                    color="#D63B3B", ms=8, mew=1.2,
                    label=f"BH-sig channels (n={len(t_sig)})")

        ax.set_xlim(-x_abs, x_abs)
        ax.set_ylim(bottom=-0.012)
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(
            f"{stage}  —  V={V},  BH-FDR sig={sig.sum()}/{V},  df={df_t}",
            fontsize=9, pad=3
        )
        ax.legend(fontsize=7, framealpha=0.85, loc="upper right")
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("t-statistic", fontsize=9)
    fig.suptitle(
        f"t-statistic distributions — univariate OLS (age ~ feature), test subjects  "
        f"|  BH-FDR α={alpha}",
        fontsize=11, y=1.002
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "t_dist_5x1_grid.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# –log₁₀(p) plot  (volcano-style, one panel per stage)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_pp_panel(ax, df_stage, n_subjects, alpha):
    """
    Draw a -log10 PP plot into *ax*.

    x: -log10(k/V)            expected -log10(p) under uniform null
    y: -log10(p_(k))          observed -log10(p), sorted ascending

    Reference lines
    ---------------
    y = x                     null diagonal (observed == expected)
    y = x - log10(alpha)      BH-FDR boundary: BH rejects p_(k) <= alpha*k/V
                              => -log10(p_(k)) >= -log10(k/V) - log10(alpha)
                              i.e. shifted -log10(alpha) above the diagonal
    """
    stage   = df_stage["stage"].iloc[0]
    p_obs   = df_stage["p_uncorr"].values.astype(float)
    sig     = df_stage["significant"].values
    V       = len(p_obs)

    order      = np.argsort(p_obs)              # ascending p
    p_sorted   = np.clip(p_obs[order], 1e-300, 1.0)
    sig_sorted = sig[order]

    expected = np.arange(1, V + 1) / V          # k/V for k=1..V
    x_vals   = -np.log10(expected)              # expected -log10(p)
    y_vals   = -np.log10(p_sorted)              # observed -log10(p)

    xy_max = max(x_vals.max(), y_vals.max()) * 1.08
    ax.set_xlim(0, xy_max)
    ax.set_ylim(0, xy_max)

    # y = x null diagonal
    diag = np.array([0, xy_max])
    ax.plot(diag, diag, color="#888888", lw=1.0, ls="-",
            label="y = x  (null)", zorder=1)

    # BH-FDR boundary: y = x + (-log10(alpha))
    shift = -np.log10(alpha)
    ax.plot(diag, diag + shift, color="#D98324", lw=1.3, ls="--",
            label=f"BH-FDR boundary  (alpha={alpha})", zorder=2)

    # observed vs expected — steel blue line
    ax.plot(x_vals, y_vals, color="#4682B4", lw=1.6, zorder=3,
            label="observed vs expected")

    # significant points in red
    if sig_sorted.any():
        ax.scatter(x_vals[sig_sorted], y_vals[sig_sorted],
                   color="#D63B3B", s=28, zorder=4, edgecolors="none",
                   label=f"BH-sig  (n={int(sig_sorted.sum())})")

    n_sig = int(sig.sum())
    ax.set_title(
        f"{stage}  -  V={V},  sig={n_sig}/{V},  df={n_subjects - 2}",
        fontsize=9, pad=3
    )
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pp_single(df_stage, n_subjects, out_dir, alpha):
    """
    -log10 PP plot for one stage, saved as  pp_<stage>.png.
    """
    stage = df_stage["stage"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _draw_pp_panel(ax, df_stage, n_subjects, alpha)
    ax.set_xlabel("Expected  -log10(p)", fontsize=10)
    ax.set_ylabel("Observed  -log10(p)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85, loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, f"pp_{stage}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def plot_pp_5x1_grid(stage_dfs, n_subjects, out_dir, alpha):
    """
    All five PP panels stacked in a 5-row x 1-col figure.
    Saved as  pp_5x1_grid.png.
    Panels share the same square aspect ratio; only bottom panel gets
    x-label; legend appears once on the top panel.
    """
    n_rows = len(stage_dfs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 5.0 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for i, (ax, df_stage) in enumerate(zip(axes, stage_dfs)):
        _draw_pp_panel(ax, df_stage, n_subjects, alpha)
        ax.set_ylabel("Observed  -log10(p)", fontsize=9)
        if i == n_rows - 1:
            ax.set_xlabel("Expected  -log10(p)", fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, framealpha=0.85, loc="upper left")

    fig.suptitle(
        "-log10 PP plots -- univariate OLS (age ~ feature), test subjects  "
        f"|  BH-FDR alpha={alpha}",
        fontsize=11, y=1.002
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "pp_5x1_grid.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")




# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Covariate-adjusted univariate OLS on TEST subjects: "
                    "age ~ feature + sex + tiv + site"
    )
    ap.add_argument("--ae-out",  required=True,
                    help="Directory written by autoencoder.py (contains "
                         "latent_all.npy, layer_features_all.npz, "
                         "ages_all.npy, split_indices.npz)")
    ap.add_argument("--tsv",     required=True,
                    help="Path to participants_study_3.tsv (row-aligned with "
                         "the VBM slice array)")
    ap.add_argument("--out",     default=None,
                    help="Output directory (default: <ae-out>/ols_covariate)")
    ap.add_argument("--alpha",   type=float, default=0.05,
                    help="BH-FDR level (default 0.05)")
    args = ap.parse_args()

    ae_out  = args.ae_out
    out_dir = args.out or os.path.join(ae_out, "ols_covariate")
    os.makedirs(out_dir, exist_ok=True)

    # ── load artefacts ────────────────────────────────────────────────────
    print("loading artefacts from", ae_out)
    Z_all      = np.load(os.path.join(ae_out, "latent_all.npy"))
    ages_all   = np.load(os.path.join(ae_out, "ages_all.npy"))
    splits     = np.load(os.path.join(ae_out, "split_indices.npz"))
    layer_data = np.load(os.path.join(ae_out, "layer_features_all.npz"))
    te         = splits["te"]

    # ── load metadata (row-aligned with VBM array) ────────────────────────
    print("loading metadata from", args.tsv)
    meta = pd.read_csv(args.tsv, sep="\t")
    assert len(meta) == len(ages_all), (
        f"TSV rows ({len(meta)}) != ages_all length ({len(ages_all)})"
    )

    N, n_latent = Z_all.shape
    print(f"  total subjects N={N}  |  test set n_te={len(te)}")

    # ── TEST subjects with valid age ──────────────────────────────────────
    age_te_all = ages_all[te]
    valid      = np.isfinite(age_te_all)
    if not valid.all():
        print(f"  dropping {(~valid).sum()} test subjects with missing age")
    te_valid   = te[valid]                         # absolute row indices
    # positions of te_valid within te (for meta indexing)
    te_pos     = np.where(valid)[0]                # positions inside te
    y          = age_te_all[valid].astype(np.float64)
    Z_te       = Z_all[te_valid].astype(np.float64)

    # ── build covariate matrix for test subjects ──────────────────────────
    C, cov_names = build_covariate_matrix(meta, te_valid)
    q   = C.shape[1]
    df_resid = len(y) - q - 2    # n - (intercept + feature + q covariates)

    print(f"\n── Covariates ───────────────────────────────────────────────")
    print(f"  {q} covariates: {cov_names}")
    print(f"  n={len(y)} test subjects | df_resid = n - q - 2 = {df_resid}")
    print(f"  age [{y.min():.1f}, {y.max():.1f}] yr")

    # ══════════════════════════════════════════════════════════════════════
    # Covariate-adjusted univariate OLS at every encoder stage
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Covariate-adjusted OLS per stage ─────────────────────────")
    stage_dfs = []

    for key in sorted(layer_data.files):            # layer1 … layer4
        layer_idx  = int(key.replace("layer", ""))
        F_te       = layer_data[key][te_valid].astype(np.float64)
        stage_name = f"layer{layer_idx}"
        df_s       = run_stage(F_te, y, C, args.alpha, stage_name)
        stage_dfs.append(df_s)

        csv_path = os.path.join(out_dir, f"{stage_name}_univariate_summary.csv")
        df_s.to_csv(csv_path, index=False, float_format="%.6g")
        print(f"    -> {csv_path}")

    # latent space
    df_latent = run_stage(Z_te, y, C, args.alpha, "latent")
    stage_dfs.append(df_latent)
    csv_latent = os.path.join(out_dir, "latent_univariate_summary.csv")
    df_latent.to_csv(csv_latent, index=False, float_format="%.6g")
    print(f"    -> {csv_latent}")

    # ── cross-stage summary ───────────────────────────────────────────────
    print("\n── Cross-stage signal summary ───────────────────────────────")
    rows = []
    for df_s in stage_dfs:
        abs_t = np.abs(df_s["t"].values)
        rows.append({
            "stage":         df_s["stage"].iloc[0],
            "n_features":    len(df_s),
            "n_significant": int(df_s["significant"].sum()),
            "prop_sig":      round(df_s["significant"].mean(), 4),
            "median_abs_t":  round(float(np.median(abs_t)), 4),
            "max_abs_t":     round(float(abs_t.max()), 4),
            "df_resid":      df_resid,
        })
    summary_tbl = pd.DataFrame(rows)
    print(summary_tbl.to_string(index=False))
    tbl_path = os.path.join(out_dir, "cross_stage_summary.csv")
    summary_tbl.to_csv(tbl_path, index=False)
    print(f"\n  saved {tbl_path}")

    # ── coefficient-map figures ───────────────────────────────────────────
    print("\n── Coefficient-map figures ──────────────────────────────────")
    for df_s in stage_dfs:
        plot_coef_map_single(df_s, out_dir, args.alpha)
    plot_coef_map_5x1_grid(stage_dfs, out_dir, args.alpha)

    # ── t-statistic distribution (empirical vs t(df) null) ───────────────
    print("\n── t-distribution plots ─────────────────────────────────────")
    for df_s in stage_dfs:
        plot_t_distribution(df_s, len(y), out_dir, args.alpha)
    plot_t_distribution_5x1_grid(stage_dfs, len(y), out_dir, args.alpha)

    # ── PP plots (-log10 observed vs expected) ────────────────────────────
    print("\n── -log10 PP plots ──────────────────────────────────────────")
    for df_s in stage_dfs:
        plot_pp_single(df_s, len(y), out_dir, args.alpha)
    plot_pp_5x1_grid(stage_dfs, len(y), out_dir, args.alpha)

    print("\ndone. outputs in:", out_dir)


if __name__ == "__main__":
    main()