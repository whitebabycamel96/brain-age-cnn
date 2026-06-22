"""
The downloaded `train.npy` is a (N, 3_659_572) float32 matrix in which 
every row concatenates six derived blocks. 

This script keeps only the vbm blocks we actually use, 
only for the subjects with `study == 3`
"""

import ast
import os
import json
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd

# Read participant_id in train.tsv and participants.tsv
train = pd.read_csv("org_data/train.tsv", sep="\t", dtype={"participant_id": str})
participants = pd.read_csv("participants.tsv", sep="\t", dtype={"participant_id": str})
train["participant_id"] = train["participant_id"].str.strip()
participants["participant_id"] = participants["participant_id"].str.strip()

# The lookup must have unique ids; warn + keep first if not.
dups = participants["participant_id"].duplicated().sum()
if dups:
    print(f"WARNING: {dups} duplicate participant_id in participants.tsv; keeping first")
    participants = participants.drop_duplicates("participant_id", keep="first")
study_map = participants.set_index("participant_id")["study"]
train["study"] = train["participant_id"].map(study_map)

# Verify if every train subject exists in participants.
missing = train["study"].isna().sum()
print(f"rows: {len(train)} | study filled: {train['study'].notna().sum()} | missing: {missing}")
if missing:
    print("unmatched participant_ids:",
          train.loc[train["study"].isna(), "participant_id"].head(10).tolist())

# Write to a NEW file
train.to_csv("org_data/train_with_study.tsv", sep="\t", index=False)

# VBM is the first block of the concatenated feature vector
VBM_SIZE = 519945 # 519945 in-mask gray-matter voxels

def load_intact(path):
    """Memory-map only the rows physically present, ignoring a truncated tail.
 
    Returns (memmap, n_full). n_full may be < the header's declared row count
    when the file is short of what its header claims.
    """
    with open(path, "rb") as f:
        f.read(6)                                  # magic \x93NUMPY
        major = f.read(1)[0]
        f.read(1)                                  # minor
        hlen = int.from_bytes(f.read(2 if major == 1 else 4), "little")
        meta = ast.literal_eval(f.read(hlen).decode())
        start = f.tell()
    shape, dt = meta["shape"], np.dtype(meta["descr"])
    row_bytes = int(np.prod(shape[1:])) * dt.itemsize
    n_full = (os.path.getsize(path) - start) // row_bytes
    n_decl = shape[0]
    if n_full < n_decl:
        print("WARNING: {0} declares {1} rows but only {2} are present; "
              "using the {2} intact rows.".format(path, n_decl, n_full))
    arr = np.memmap(path, dtype=dt, mode="r", offset=start,
                    shape=(int(n_full), shape[1]))
    return arr, int(n_full)

def main():
    ap = argparse.ArgumentParser(description="Extract VBM for study==3 subjects.")
    ap.add_argument("--npy", default="org_data/train.npy")
    ap.add_argument("--tsv", default="org_data/train_with_study.tsv")
    ap.add_argument("--study-col", default="study")
    ap.add_argument("--study-val", type=int, default=3)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = pd.read_csv(args.tsv, sep="\t")
    X, n_full = load_intact(args.npy)
    if X.shape[1] < VBM_SIZE:
        raise ValueError("npy width {0} < VBM block {1}; wrong file?".format(
            X.shape[1], VBM_SIZE))
    
    # Trim metadata to the rows that actually exist, then assert alignment.
    if len(meta) > n_full:
        print("Trimming tsv from {0} to {1} rows to match intact npy.".format(
            len(meta), n_full))
        meta = meta.iloc[:n_full].reset_index(drop=True)
    if len(meta) != X.shape[0]:
        raise ValueError("Alignment broken: tsv {0} vs npy {1}.".format(
            len(meta), X.shape[0]))
    
    # Row indices for study == study_val
    idx = meta.index[meta[args.study_col] == args.study_val].to_numpy()
    if idx.size == 0:
        raise ValueError("No subjects with {0}=={1}. Counts: {2}".format(
            args.study_col, args.study_val,
            meta[args.study_col].value_counts().to_dict()))
 
    sub = meta.loc[idx].reset_index(drop=True)
 
    # Age range
    amin, amax = float(sub["age"].min()), float(sub["age"].max())
    print("study=={0}: {1} subjects | AGE RANGE {2:.1f} - {3:.1f} "
          "(mean {4:.1f}, median {5:.1f})".format(
              args.study_val, len(idx), amin, amax,
              float(sub["age"].mean()), float(sub["age"].median())))
    
    # Extract VBM block, row by row (peak RAM = output, not the source)
    out = np.empty((len(idx), VBM_SIZE), dtype=np.float32)
    for i, row in enumerate(idx):
        if i % 100 == 0:
            print("  extracting {0}/{1}".format(i, len(idx)))
        out[i] = np.asarray(X[row, :VBM_SIZE], dtype=np.float32)
 
    npy_path = os.path.join(args.out, "vbm_processing", "vbm_study_3.npy")
    tsv_path = os.path.join(args.out, "vbm_processing", "participants_study_3.tsv")
    np.save(npy_path, out)
    sub.to_csv(tsv_path, sep="\t", index=False)
 
    print("saved {0}  shape={1}  {2:.1f} MB".format(
        npy_path, out.shape, out.nbytes / 1e6))
    print("saved {0}  rows={1}".format(tsv_path, len(sub)))
    print("NOTE: vbm_study_3.npy[i] <-> participants_study_3.tsv row i "
          "(aligned by construction).")
    
if __name__ == "__main__":
    main()