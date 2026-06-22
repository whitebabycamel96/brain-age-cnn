"""
Reconstruct the 3D VBM gray-matter volume for every participant in
vbm_study_3.npy and save one file per subject:

  vbm_data/sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy

Each output is a (121, 145, 121) float32 array: the 519945 in-mask GM values
scattered back into the MNI grid (out-of-mask voxels = 0). This is the exact
inverse of the masking that produced the flat vectors, and matches
nilearn.masking.unmask for a boolean mask.

Run:
  python reconstruct_vbm.py --vbm vbm_study_3.npy --tsv participants_study_3.tsv \
      --mask cat12vbm_space-MNI152_desc-gm_TPM.nii.gz --out vbm_data
"""

import os
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

VBM_SIZE = 519945                 # in-mask GM voxels = flat vector length
GRID = (121, 145, 121)            # MNI grid the volume is rebuilt into

def main():
    ap = argparse.ArgumentParser(description="Reconstruct per-subject VBM volumes.")
    ap.add_argument("--vbm", default="vbm_processing/vbm_study_3.npy")
    ap.add_argument("--tsv", default="vbm_processing/participants_study_3.tsv")
    ap.add_argument("--mask", default="cat12vbm_space-MNI152_desc-gm_TPM.nii.gz")
    ap.add_argument("--out", default="vbm_data")
    ap.add_argument("--thr", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    vbm = np.load(args.vbm, mmap_mode="r")                    # (N, 519945)
    meta = pd.read_csv(args.tsv, sep="\t", dtype={"participant_id": str})
    if "participant_id" not in meta.columns:
        raise KeyError("No 'participant_id' in {0}. Columns: {1}".format(
            args.tsv, meta.columns.tolist()))
    if len(meta) != vbm.shape[0]:
        raise ValueError("Alignment broken: tsv {0} vs vbm {1}.".format(
            len(meta), vbm.shape[0]))
    if vbm.shape[1] != VBM_SIZE:
        raise ValueError("vbm width {0} != {1}; wrong file?".format(
            vbm.shape[1], VBM_SIZE))

    # Binarize the GM mask exactly as the dataset did; the count IS the check.
    mask_prob = nib.load(args.mask)
    mask_bool = mask_prob.get_fdata() > args.thr
    n_in = int(mask_bool.sum())
    if n_in != VBM_SIZE:
        raise ValueError("Mask has {0} voxels at thr={1}, expected {2}. "
                         "Wrong mask/threshold.".format(n_in, args.thr, VBM_SIZE))
    if mask_bool.shape != GRID:
        raise ValueError("Mask grid {0} != {1}.".format(mask_bool.shape, GRID))

    pids = meta["participant_id"].str.strip().tolist()
    made, skipped = 0, 0
    for i, pid in enumerate(pids):
        pid = pid[4:] if pid.startswith("sub-") else pid     # avoid sub-sub-
        fpath = os.path.join(
            args.out, "sub-{0}_preproc-cat12vbm_desc-gm_T1w.npy".format(pid))
        if os.path.exists(fpath):                            # resumable
            skipped += 1
            continue
        vol = np.zeros(GRID, dtype=np.float32)
        vol[mask_bool] = np.asarray(vbm[i], dtype=np.float32)  # scatter -> 3D
        np.save(fpath, vol)
        made += 1
        if i % 50 == 0:
            print("  {0}/{1}  sub-{2}  shape={3}".format(
                i + 1, len(pids), pid, vol.shape))

    per_mb = np.prod(GRID) * 4 / 1e6
    print("done: {0} written, {1} already existed | {2}/ "
          "(~{3:.1f} MB each, ~{4:.1f} GB total)".format(
              made, skipped, args.out, per_mb, per_mb * len(pids) / 1000))


if __name__ == "__main__":
    main()