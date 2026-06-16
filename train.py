import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from BrainAgeDataset import BrainAgeDataset
from Age2DCNN import Age2DCNN

def train():
    dataset = BrainAgeDataset(
        metadata_path="participants.tsv",
        image_dir="brain_scans"
    )

    loader = DataLoader(
        dataset,
        batch_size=30,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Age2DCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_mae = 0.0

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            preds = model(X)

            loss = criterion(preds, y)
            mae = torch.mean(torch.abs(preds - y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_mae += mae.item() * X.size(0)

        avg_loss = total_loss / len(dataset)
        avg_mae = total_mae / len(dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"MSE: {avg_loss:.4f} | "
            f"MAE: {avg_mae:.4f}"
        )

    torch.save(model.state_dict(), "age_cnn_model.pth")
    print("Model saved to age_cnn_model.pth")


if __name__ == "__main__":
    train()