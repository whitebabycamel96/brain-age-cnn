import numpy as np, nibabel as nib
import matplotlib.pyplot as plt
from nilearn.masking import unmask
from nilearn import plotting

vbm = np.load("vbm_processing/vbm_study_3.npy", mmap_mode="r")          # (508, 519945)

# the SAME mask the dataset used, binarized the same way (threshold 0.05)
mask_prob = nib.load("cat12vbm_space-MNI152_desc-gm_TPM.nii.gz")
mask_bin = nib.Nifti1Image(
    (mask_prob.get_fdata() > 0.05).astype(np.int8), mask_prob.affine)
print("in-mask voxels:", int((mask_prob.get_fdata() > 0.05).sum()))  # expect 519945

i = 0
flat = np.asarray(vbm[i])                  # (519945,) — one participant
vol_img = unmask(flat, mask_bin)           # -> Nifti1Image, 3D (121, 145, 121)
print("reconstructed shape:", vol_img.shape)

plotting.plot_anat(vol_img, title=f"VBM subject {i}")   # or plot_stat_map / plot_img
plotting.show()

# ---- three slices side by side: z=60, z=40 (axial) and y=80 (coronal) ----
subj = 0
panels = [
    ("vbm_data_sliced/vbm_z60.npy", "z = 60 (axial)"),
    ("vbm_data_sliced/vbm_z40.npy", "z = 40 (axial)"),
    ("vbm_data_sliced/vbm_y80.npy", "y = 80 (coronal)"),
]

# load the chosen subject's slice from each file (axial -> 121x145, coronal -> 121x121)
slices = [np.load(path)[subj] for path, _ in panels]

# shared grayscale so the three panels are directly comparable (GM density units)
vmax = max((np.percentile(s[s > 0], 99) for s in slices if np.any(s > 0)), default=1.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
im = None
for ax, s, (path, title) in zip(axes, slices, panels):
    im = ax.imshow(s.T, cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=vmax)
    ax.set_title(f"VBM subject {subj}  —  {title}")
    ax.set_xlabel("voxels")
    ax.set_ylabel("voxels")

cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="GM density")
fig.savefig("vbm_processing/vbm_subject0_slices.png", dpi=150)
plt.show()