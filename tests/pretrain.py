import sys

sys.path.append("..")

import torch
import torch.nn as nn
from dummy import DummyModel, DummyDataset
from pyroml.trainer import Trainer
from pyroml.config import Config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("need to pass checkpoints path")
    path = sys.argv[1]

    in_dim = 16
    model = DummyModel(in_dim=in_dim)
    ds = DummyDataset(size=1024, in_dim=in_dim)
    config = Config(
        name="pyro_main_test_v2",
        max_iterations=256,
        batch_size=16,
        lr=0.05,
        wandb=False,
        evaluate=False,
        compile=False,
        verbose=True,
    )
    trainer = Trainer.from_pretrained(model, config, path, resume=False)
    trainer.fit(ds)
    x, y = ds[:]
    output = model(x)
    mse = nn.MSELoss()(output, y)
    print(mse)
    print(torch.allclose(mse, torch.tensor([0.0]), atol=1e-2))
