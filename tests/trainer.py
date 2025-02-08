import sys

sys.path.append("..")

import pyroml as p
from tests.dummy.regression import DummyRegressionDataset, DummyRegressionModel

if __name__ == "__main__":
    ds = DummyRegressionDataset(size=1024)
    model = DummyRegressionModel()

    trainer = p.Trainer(
        lr=0.01,
        evaluate_on=False,
        batch_size=32,
        max_epochs=32,
        wandb=False,
        num_workers=0,
    )
    trainer.fit(model, ds)
