import sys

sys.path.append("..")

from dummy import DummyDataset, DummyModel
from pyroml.config import Config
from pyroml.trainer import Trainer
from pyroml.metrics import Accuracy, RMSE

if __name__ == "__main__":
    tr_ds = DummyDataset()
    ev_ds = DummyDataset(size=128)
    model = DummyModel()

    # Test dataset works with model
    x, y = tr_ds[0]
    output = model(x)
    assert output.shape == y.shape

    def on_epoch_end(trainer, **kwargs):
        pass
        # trainer.save_model()

    max_iterations = 256
    config = Config(
        name="pyro_main_test_v2",
        max_iterations=max_iterations,
        lr=1e-2,
        batch_size=64,
        metrics=[Accuracy(), RMSE()],
        grad_norm_clip=None,
        wandb_project="pyro_main_test",
        evaluate_every=10,
        verbose=False,
        wandb=True,
    )
    trainer = Trainer(model, config)
    trainer.add_callback("on_epoch_end", on_epoch_end)
    trainer.run(tr_ds, ev_ds)
