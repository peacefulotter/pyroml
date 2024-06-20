import sys
import torch

sys.path.append("..")

from dummy import DummyDataset, DummyModel
from pyroml.config import Config
from pyroml.trainer import Trainer

if __name__ == "__main__":
    tr_ds = DummyDataset()
    ev_ds = DummyDataset(size=128)
    model = DummyModel()

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
        name="pyro_main_test_v2",
        max_iterations=max_iterations,
        lr=lr,
        batch_size=64,
        metrics=[],
        grad_norm_clip=None,
        wandb_project="pyro_main_test",
        evaluate_every=10,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        verbose=False,
        wandb=False,
        compile=False,
    )
    trainer = Trainer(model, config)
    trainer.fit(tr_ds, ev_ds)

    # TODO: not necessarily static, don't need to recreate a trainer
    trainer = Trainer.from_pretrained(model, config, trainer.cp_path, resume=True)
    trainer.fit(tr_ds, ev_ds)
