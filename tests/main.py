import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .dummy import DummyDataset, DummyModel
from ..src.config import Config

if __name__ == "__main__":
    config = Config(
        name="test_config", max_iterations=1024, wandb_project="test_project"
    )
    ds = DummyDataset()
    model = DummyModel()
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    for x, y in loader:
        print(x.shape, y.shape)
        output = model(x)
        print(output.shape)
        assert output.shape == y.shape
        break
