"""
Visualize each VBM slice type through crop_pad_128 (no brain mask):

    z = 60 (axial, 121x145) ->  128x128
    z = 40 (axial, 121x145) ->  128x128
    y = 80 (coronal, 121x121) -> 128x128

For one subject, each row shows the original slice and the crop-padded result.
The annotations show the geometry generically: any axis longer than 128 is
centre-cropped (red, "drop"), any axis shorter than 128 is zero-padded (blue,
"pad"). So the axial rows crop the 145 axis and pad the 121 axis, while the
coronal row pads both 121 axes and crops nothing.

Run:
    python vbm_croppad_vis.py
    python vbm_croppad_vis.py --subject 42
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, GREEN = "#D1495B", "#2E6E9E", "#2E8B57"


def crop_pad_128(sl):
    """(N,H,W) -> (N,128,128): centre-crop axes longer than 128, zero-pad axes shorter."""
    def fit(a, axis):
        n = a.shape[axis]
        if n > 128:                                   # centre-crop
            start = (n - 128) // 2
            idx = [slice(None)] * a.ndim
            idx[axis] = slice(start, start + 128)
            return a[tuple(idx)]
        if n < 128:                                   # zero-pad (centred)
            before = (128 - n) // 2
            pad = [(0, 0)] * a.ndim
            pad[axis] = (before, 128 - n - before)
            return np.pad(a, pad, mode="constant", constant_values=0)
        return a
    return fit(fit(sl, 1), 2)


def _plan(n):
    """How axis of length n maps to 128: ('crop'|'pad'|'none', before, after)."""
    if n > 128:
        b = (n - 128) // 2
        return ("crop", b, n - 128 - b)
    if n < 128:
        b = (128 - n) // 2
        return ("pad", b, 128 - n - b)
    return ("none", 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z60", default="vbm_data_sliced/vbm_z60.npy")
    ap.add_argument("--z40", default="vbm_data_sliced/vbm_z40.npy")
    ap.add_argument("--y80", default="vbm_data_sliced/vbm_y80.npy")
    ap.add_argument("--out", default="vbm_processing/vbm_croppad_demo.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subject", type=int, default=None,
                    help="subject row index (shared across the three files); random if unset")
    args = ap.parse_args()

    rows = [("z = 60 (axial)", args.z60),
            ("z = 40 (axial)", args.z40),
            ("y = 80 (coronal)", args.y80)]
    data = [(label, np.load(path).astype(np.float32)) for label, path in rows]
    N = data[0][1].shape[0]

    rng = np.random.default_rng(args.seed)
    s = args.subject if args.subject is not None else int(rng.integers(N))

    # shared grayscale across panels (same subject -> comparable GM density)
    pooled = np.concatenate([arr[s][arr[s] > 0].ravel() for _, arr in data]
                            or [np.array([1.0])])
    vmax = float(np.percentile(pooled, 99)) if pooled.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(len(rows), 2, figsize=(8.5, 12))
    for r, (label, arr) in enumerate(data):
        raw = arr[s]                                  # original (H,W)
        cp = crop_pad_128(raw[None])[0]               # after crop_pad (128,128)
        H, W = raw.shape
        py, px = _plan(H), _plan(W)                   # axis0 rows / axis1 cols

        # left: original, annotate any CROP (what gets dropped)
        a0 = ax[r, 0]
        a0.imshow(raw, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        if px[0] == "crop":
            a0.axvspan(-0.5, px[1] - 0.5, color=RED, alpha=0.20)
            a0.axvspan(W - px[2] - 0.5, W - 0.5, color=RED, alpha=0.20)
            a0.axvline(px[1] - 0.5, color=GREEN, ls="--", lw=1.1)
            a0.axvline(W - px[2] - 0.5, color=GREEN, ls="--", lw=1.1)
            a0.text(px[1] / 2, H / 2, f"drop\n{px[1]}", color=RED, ha="center", va="center", fontsize=8, weight="bold")
            a0.text(W - px[2] / 2, H / 2, f"drop\n{px[2]}", color=RED, ha="center", va="center", fontsize=8, weight="bold")
        if py[0] == "crop":
            a0.axhspan(-0.5, py[1] - 0.5, color=RED, alpha=0.20)
            a0.axhspan(H - py[2] - 0.5, H - 0.5, color=RED, alpha=0.20)
        a0.set_title(f"{label}\noriginal  ({H} × {W})", fontsize=10)

        # right: after crop_pad, annotate any PAD (zeros added)
        a1 = ax[r, 1]
        a1.imshow(cp, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        if py[0] == "pad":
            a1.axhspan(-0.5, py[1] - 0.5, color=BLUE, alpha=0.28)
            a1.axhspan(128 - py[2] - 0.5, 127.5, color=BLUE, alpha=0.28)
            a1.text(64, py[1] / 2, f"pad {py[1]}", color=BLUE, ha="center", va="center", fontsize=8, weight="bold")
            a1.text(64, 128 - py[2] / 2, f"pad {py[2]}", color=BLUE, ha="center", va="center", fontsize=8, weight="bold")
        if px[0] == "pad":
            a1.axvspan(-0.5, px[1] - 0.5, color=BLUE, alpha=0.28)
            a1.axvspan(128 - px[2] - 0.5, 127.5, color=BLUE, alpha=0.28)
            a1.text(px[1] / 2, 64, f"pad\n{px[1]}", color=BLUE, ha="center", va="center", fontsize=8, weight="bold")
            a1.text(128 - px[2] / 2, 64, f"pad\n{px[2]}", color=BLUE, ha="center", va="center", fontsize=8, weight="bold")
        a1.set_title("after crop_pad_128  (128 × 128)", fontsize=10)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"VBM slices · crop & zero-pad to 128×128 · subject #{s}", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"subject {s} | original shapes:", [arr[s].shape for _, arr in data])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()