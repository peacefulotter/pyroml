import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyModel(nn.Module):
    def __init__(self, in_dim=16):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.ReLU(),
            nn.Linear(in_dim * 4, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.module(x)


class DummyDataset(Dataset):
    def __init__(self, size=1024, in_dim=16):
        self.size = size
        self.in_dim = in_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(self.in_dim)
        y = 2 * x  # x**2 + 0.3 * x + 0.1
        return x, y


if __name__ == "__main__":
    ds = DummyDataset()
    model = DummyModel()
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    for x, y in loader:
        print(x, y)
        print(x.shape, y.shape)
        output = model(x)
        print(output.shape)
        assert output.shape == y.shape
        break
