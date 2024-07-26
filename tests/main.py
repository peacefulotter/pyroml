import sys
import torch

sys.path.append("..")

from dummy.regression import DummyRegressionDataset, DummyRegressionModel
from pyroml.trainer import Trainer

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
    scheduler = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = {
        "max_lr": lr,
        "pct_start": 0.02,
        "anneal_strategy": "cos",
        "cycle_momentum": False,
        "div_factor": 1e2,
        "final_div_factor": 0.05,
        "total_steps": max_iterations,
    }

    trainer = Trainer(
        lr=lr,
        dtype=torch.bfloat16,
        max_steps=max_iterations,
        batch_size=16,
        grad_norm_clip=None,
        wandb_project="pyro_test",
        evaluate=True,
        evaluate_every=4,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        wandb=False,
        compile=False,
        num_workers=0,
    )
    metrics = trainer.fit(model=model, tr_dataset=tr_ds, ev_dataset=ev_ds)
    print(metrics)

    metrics = trainer.test(model=model, dataset=te_ds)
    print(metrics)
