import sys
import torch

sys.path.append("..")

from pyroml import Config, Trainer, Stage
from dummy import DummyRegressionDataset, DummyRegressionModel

if __name__ == "__main__":
    te_ds = DummyRegressionDataset(size=64)
    model = DummyRegressionModel()

    config = Config(
        dtype=torch.bfloat16,
        batch_size=4,
        wandb=False,
        num_workers=0,
    )
    trainer = Trainer(model, config)
    tracker = trainer.test(te_ds)
    print(tracker.records[Stage.TEST])
