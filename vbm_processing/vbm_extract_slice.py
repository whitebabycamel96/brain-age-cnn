"""
Extract two slices from each subject's VBM volume and stack them into arrays for
a 2D autoencoder:
 
  vbm_z40.npy   shape (N, 121, 145) float32   <- axial slice   z = 40  (vol[:, :, z])
  vbm_y80.npy   shape (N, 121, 121) float32   <- coronal slice y = 80  (vol[:, y, :])
 
Reads the flat VBM directly (vbm_study_3.npy), unmasks each subject to the
121x145x121 grid, takes one axial and one coronal slice, and stacks. Row i of
each output matches row i of participants_study_3.tsv.
 
Note the two outputs have *different* shapes: axial is 121x145 (X x Y), coronal is
121x121 (X x Z). The autoencoder's crop_pad_128 was written for the 121x145 axial
slice, so the coronal 121x121 slice needs its own crop/pad (pad both axes 121->128).
 
Run:
  python vbm_extract_slice.py --vbm vbm_study_3.npy --tsv participants_study_3.tsv \
      --mask cat12vbm_space-MNI152_desc-gm_TPM.nii.gz --z 40 --y 80 \
      --out-z vbm_data_sliced/vbm_z40.npy --out-y vbm_data_sliced/vbm_y80.npy
"""
 
import argparse
import os
import numpy as np
import pandas as pd
import nibabel as nib
 
VBM_SIZE = 519945
GRID = (121, 145, 121)  # (X, Y, Z)
 
 
def main():
    ap = argparse.ArgumentParser(description="Extract an axial z-slice and a coronal y-slice for a 2D model.")
    ap.add_argument("--vbm", default="vbm_processing/vbm_study_3.npy")
    ap.add_argument("--tsv", default="vbm_processing/participants_study_3.tsv")
    ap.add_argument("--mask", default="cat12vbm_space-MNI152_desc-gm_TPM.nii.gz")
    ap.add_argument("--z", type=int, default=40, help="axial slice index (axis 2, 0..120)")
    ap.add_argument("--y", type=int, default=80, help="coronal slice index (axis 1, 0..144)")
    ap.add_argument("--out-z", default="vbm_data_sliced/vbm_z40.npy")
    ap.add_argument("--out-y", default="vbm_data_sliced/vbm_y80.npy")
    ap.add_argument("--thr", type=float, default=0.05)
    args = ap.parse_args()
 
    vbm = np.load(args.vbm, mmap_mode="r")
    meta = pd.read_csv(args.tsv, sep="\t")
    if len(meta) != vbm.shape[0]:
        raise ValueError("Alignment: tsv {0} vs vbm {1}.".format(len(meta), vbm.shape[0]))
    if vbm.shape[1] != VBM_SIZE:
        raise ValueError("vbm width {0} != {1}.".format(vbm.shape[1], VBM_SIZE))
    if not 0 <= args.z < GRID[2]:
        raise ValueError("z={0} out of range [0,{1}).".format(args.z, GRID[2]))
    if not 0 <= args.y < GRID[1]:
        raise ValueError("y={0} out of range [0,{1}).".format(args.y, GRID[1]))
 
    mask_bool = nib.load(args.mask).get_fdata() > args.thr
    if int(mask_bool.sum()) != VBM_SIZE:
        raise ValueError("Mask has {0} voxels at thr={1}, expected {2}.".format(
            int(mask_bool.sum()), args.thr, VBM_SIZE))
 
    N = vbm.shape[0]
    slices_z = np.zeros((N, GRID[0], GRID[1]), dtype=np.float32)   # axial   (X, Y) = 121x145
    slices_y = np.zeros((N, GRID[0], GRID[2]), dtype=np.float32)   # coronal (X, Z) = 121x121
    vol = np.zeros(GRID, dtype=np.float32)
    for i in range(N):
        vol[:] = 0
        vol[mask_bool] = np.asarray(vbm[i], dtype=np.float32)
        slices_z[i] = vol[:, :, args.z]          # axial slice at z
        slices_y[i] = vol[:, args.y, :]          # coronal slice at y
        if i % 100 == 0:
            print("  {0}/{1}".format(i, N))
 
    for path, arr, tag, plane in [(args.out_z, slices_z, "z={0}".format(args.z), "axial"),
                                  (args.out_y, slices_y, "y={0}".format(args.y), "coronal")]:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        np.save(path, arr)
        nz = int((arr[0] != 0).sum())
        print("saved {0}  shape={1}  {2:.1f} MB  ({3} {4}, slice0 {5} nonzero of {6})".format(
            path, arr.shape, arr.nbytes / 1e6, plane, tag, nz, arr.shape[1] * arr.shape[2]))
 
 
if __name__ == "__main__":
    main()