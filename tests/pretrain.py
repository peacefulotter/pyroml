import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader

from setup import setup_test

from dummy.regression import DummyRegressionModel, DummyRegressionDataset
from pyroml.trainer import Trainer

if __name__ == "__main__":
    setup_test()

    in_dim = 16
    model = DummyRegressionModel(in_dim=in_dim)
    ds = DummyRegressionDataset(size=256, in_dim=in_dim)

    # First train a model
    trainer = Trainer(
        max_epochs=2,
        batch_size=16,
        lr=0.05,
        wandb=False,
        evaluate=False,
        compile=False,
        log_level=logging.INFO,
    )
    trainer.fit(model, ds)

    # Reset the model and load the checkpoint
    model = DummyRegressionModel(in_dim=in_dim)
    trainer = Trainer.load(trainer.checkpoint_folder)

    x, y = next(iter(DataLoader(ds, batch_size=16)))
    output = model(x)
    mse = nn.MSELoss()(output, y)
    print(mse)
    print(torch.allclose(mse, torch.tensor([0.0]), atol=1e-2))
