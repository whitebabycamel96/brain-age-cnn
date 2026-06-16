"""
Brain slice viewer with interactive sliders.
Usage:
    python3 brain_viewer.py path/to/file.npy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── load ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python3 brain_viewer.py <path_to_npy_file>")
    sys.exit(1)

path = sys.argv[1]
img  = np.load(path).squeeze()          # drop any leading (1,1,...) dims
assert img.ndim == 3, f"Expected 3D volume after squeeze, got shape {img.shape}"
X, Y, Z = img.shape
print(f"Loaded: {path}  |  shape: {img.shape}  |  range: [{img.min():.3f}, {img.max():.3f}]")

# ── initial slices ────────────────────────────────────────────────────────────
cx, cy, cz = X // 2, Y // 2, Z // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
plt.subplots_adjust(bottom=0.25)
fig.suptitle("Gray Matter Density — use sliders to scroll through slices", fontsize=13)

im0 = axes[0].imshow(img[cx, :, :], cmap="gray", origin="lower", vmin=img.min(), vmax=img.max())
im1 = axes[1].imshow(img[:, cy, :], cmap="gray", origin="lower", vmin=img.min(), vmax=img.max())
im2 = axes[2].imshow(img[:, :, cz], cmap="gray", origin="lower", vmin=img.min(), vmax=img.max())

titles = ["Sagittal (side)  x", "Coronal (front)  y", "Axial (top-down)  z"]
for ax, t in zip(axes, titles):
    ax.set_title(t)
    ax.axis("off")

# ── sliders ───────────────────────────────────────────────────────────────────
ax_x = plt.axes([0.10, 0.13, 0.25, 0.04])
ax_y = plt.axes([0.40, 0.13, 0.25, 0.04])
ax_z = plt.axes([0.70, 0.13, 0.25, 0.04])

sl_x = Slider(ax_x, f"Sagittal (0–{X-1})", 0, X-1, valinit=cx, valstep=1)
sl_y = Slider(ax_y, f"Coronal  (0–{Y-1})", 0, Y-1, valinit=cy, valstep=1)
sl_z = Slider(ax_z, f"Axial    (0–{Z-1})", 0, Z-1, valinit=cz, valstep=1)

# ── update callbacks ──────────────────────────────────────────────────────────
def update(_):
    ix = int(sl_x.val)
    iy = int(sl_y.val)
    iz = int(sl_z.val)
    im0.set_data(img[ix, :, :])
    im1.set_data(img[:, iy, :])
    im2.set_data(img[:, :, iz])
    axes[0].set_title(f"Sagittal  x={ix}")
    axes[1].set_title(f"Coronal   y={iy}")
    axes[2].set_title(f"Axial     z={iz}")
    fig.canvas.draw_idle()

sl_x.on_changed(update)
sl_y.on_changed(update)
sl_z.on_changed(update)

plt.show()