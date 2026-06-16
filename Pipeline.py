from BrainAgeDataset import BrainAgeDataset
from torch.utils.data import Dataset, DataLoader


dataset = BrainAgeDataset(
    metadata_path="participants.tsv",
    image_dir="brain_scans"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

for X, y in loader:
    print(X.shape)  # (batch_size, 1, H, W)
    print(y.shape)  # (batch_size,)
    break