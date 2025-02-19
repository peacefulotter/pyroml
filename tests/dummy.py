import sys

import torch

from tests.dummy.regression import DummyRegressionDataset, DummyRegressionModel

sys.path.append("..")

from pyroml import Trainer

if __name__ == "__main__":
    tr_ds = DummyRegressionDataset(size=256)
    ev_ds = DummyRegressionDataset(size=64)
    te_ds = DummyRegressionDataset(size=64)
    model = DummyRegressionModel()

    # Test dataset works with model
    x, y = tr_ds[0]
    output = model(x)
    assert output.shape == y.shape

    lr = 1e-2
    max_iterations = 256
    trainer = Trainer(
        lr=lr,
        dtype=torch.bfloat16,
        max_steps=max_iterations,
        batch_size=16,
        grad_norm_clip=None,
        wandb_project="pyro_test",
        evaluate_on="step",
        evaluate_every=4,
        wandb=False,
        compile=False,
        num_workers=0,
    )
    metrics = trainer.fit(model=model, tr_dataset=tr_ds, ev_dataset=ev_ds)
    print(metrics)

    metrics = trainer.evaluate(model=model, dataset=te_ds)
    print(metrics)
