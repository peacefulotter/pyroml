import sys

sys.path.append("..")

from dummy import DummyDataset, DummyModel
from src.config import Config
from src.trainer import Trainer

if __name__ == "__main__":
    tr_ds = DummyDataset()
    ev_ds = DummyDataset(size=16)
    model = DummyModel()

    # Test dataset works with model
    x, y = tr_ds[0]
    output = model(x)
    assert output.shape == y.shape

    def on_epoch_end(trainer, **kwargs):
        trainer.save_model()

    max_iterations = 100_000
    config = Config(
        name="pyro_main_test",
        max_iterations=max_iterations,
        wandb_project="pyro_main_test",
        evaluate_every=10,
        verbose=False,
    )
    trainer = Trainer(model, config)
    trainer.add_callback("on_epoch_end", on_epoch_end)
    trainer.run(tr_ds, ev_ds)
