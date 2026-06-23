import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VBMAgeDataset(Dataset):
    def __init__(self, images, tsv_path, indices=None):
        self.images = images.astype(np.float32)
        self.meta = pd.read_csv(tsv_path, sep="\t")

        assert len(self.images) == len(self.meta)

        self.indices = np.arange(len(self.images)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        image = self.images[real_idx]
        age = self.meta.iloc[real_idx]["age"]
        subject_id = self.meta.iloc[real_idx]["participant_id"]

        return {
            "image": torch.tensor(image, dtype=torch.float32).unsqueeze(0),
            "age": torch.tensor(age, dtype=torch.float32),
            "subject_id": subject_id,
            "index": real_idx,
        }