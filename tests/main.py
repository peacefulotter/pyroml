from dummy import DummyDataset, DummyModel
from src.config import Config
from src.trainer import Trainer

if __name__ == "__main__":
    config = Config(
        name="test_config", max_iterations=1024, wandb_project="test_project"
    )
    tr_ds = DummyDataset()
    ev_ds = DummyDataset(size=16)
    model = DummyModel()

    # Test dataset works with model
    x, y = tr_ds[0]
    output = model(x)
    assert output.shape == y.shape

    def on_batch_end(trainer, model, **kwargs):
        print("on_batch_end")

    def on_epoch_end(trainer, model, **kwargs):
        trainer.save_model()

    max_iterations = len()
    config = Config(
        name="main_test",
        max_iterations=max_iterations,
        wandb_project="main_test_project",
        verbose=True,
    )
    trainer = Trainer(model, config)
    trainer.add_callback("on_batch_end", on_batch_end)
    trainer.add_callback("on_epoch_end", on_epoch_end)
    trainer.run(tr_ds, ev_ds)
