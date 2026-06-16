#!/bin/bash

# Source directory
SRC="./openBHB/train/derivatives"

# Destination directory
DEST="./brain_scans"

# Create destination folder if it doesn't exist
mkdir -p "$DEST"

# Find and copy all GM T1w files
find "$SRC" -type f -name "*_preproc-cat12vbm_desc-gm_T1w.npy" -exec cp {} "$DEST" \;

echo "Done! Files copied to $DEST"