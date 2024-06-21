import sys

sys.path.append("..")

from dummy import DummyRegressionDataset, DummyRegressionModel
from pyroml.config import Config
from pyroml.trainer import Trainer

if __name__ == "__main__":
    te_ds = DummyRegressionDataset(size=64)
    model = DummyRegressionModel()

    config = Config(
        name="trainer_test",
        batch_size=4,
        wandb=False,
        num_workers=0,
    )
    trainer = Trainer(model, config)
    tracker = trainer.test(te_ds)
    print(tracker.stats)
