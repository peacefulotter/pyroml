import sys
import torch

sys.path.append("..")

from dummy import DummyRegressionDataset, DummyRegressionModel
from pyroml.config import Config
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

    config = Config(
        name="main_test",
        lr=lr,
        max_iterations=max_iterations,
        batch_size=16,
        grad_norm_clip=None,
        wandb_project="pyro_test",
        evaluate=True,
        evaluate_every=2,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        verbose=False,
        wandb=False,
        compile=False,
        num_workers=0,
    )
    trainer = Trainer(model, config)
    trainer.fit(tr_ds, ev_ds)

    print(trainer.tracker.stats)

    te_tracker = trainer.test(te_ds)
    print(te_tracker.stats)
