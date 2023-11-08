import sys

sys.path.append("..")

import torch
import torch.nn as nn
from dummy import DummyModel, DummyDataset
from src.trainer import Trainer
from src.config import Config

if __name__ == "__main__":
    in_dim = 16
    model = DummyModel(in_dim=in_dim)
    ds = DummyDataset(size=8, in_dim=in_dim)
    config = Config(
        name="pyro_main_test_v2",
        max_iterations=1,
        wandb=False,
    )
    trainer = Trainer.from_pretrained(model, config, 3, 5120)
    x, y = ds[:]
    output = model(x)
    mse = nn.MSELoss()(output, y)
    print(torch.allclose(mse, torch.tensor([0.0]), atol=1e-2))
