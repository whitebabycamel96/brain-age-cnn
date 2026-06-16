import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class BrainAgeDataset(Dataset):
    def __init__(self, metadata_path, image_dir):

        self.df = pd.read_csv(metadata_path, sep="\t")
        self.image_dir = image_dir

        available_files = set(os.listdir(image_dir))

        valid_rows = []

        for _, row in self.df.iterrows():

            participant_id = str(row["participant_id"])

            filename = (
                f"sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy"
            )

            if filename in available_files:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(f"{len(self.df)} subjects with available scans found.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        participant_id = str(row["participant_id"])
        age = row["age"]

        filename = (
            f"sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy"
        )

        path = os.path.join(self.image_dir, filename)

        brain = np.load(path)
        brain = np.squeeze(brain)

        z_mid = brain.shape[2] // 2
        x = brain[:, :, z_mid]

        x = x.astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-8) # why is there this nomalization step?

        x = torch.tensor(x).unsqueeze(0)

        y = torch.tensor(age, dtype=torch.float32)

        return x, y