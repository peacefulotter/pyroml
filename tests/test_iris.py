import rich
import torch
import torch.utils
import torch.utils.data

import pyroml as p
from pyroml.callbacks.progress.tqdm_progress import TQDMProgress
from pyroml.loop import Loop
from pyroml.template.iris import IrisDataset, IrisModel


class ScheduledIrisModel(IrisModel):
    def configure_optimizers(self, loop: "Loop"):
        tr = self.trainer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=tr.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.99
        )

    def step(self, data, stage):
        self.log(lr=self.scheduler.get_last_lr()[0])
        return super().step(data, stage)


def test_iris_training():
    dataset = IrisDataset(split="train")
    tr_ds, te_ds = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.7), int(len(dataset) * 0.3)]
    )

    model = ScheduledIrisModel()

    trainer = p.Trainer(
        lr=0.03,
        max_epochs=32,
        batch_size=16,
        device="cpu",
        dtype=torch.bfloat16,
        compile=False,
        evaluate_on=False,
        wandb=False,
        pin_memory=False,
        num_workers=0,
        callbacks=[TQDMProgress()],
    )

    tr_tracker = trainer.fit(model, tr_ds)
    rich.print(tr_tracker.records)

    # Uncomment to see plots: tr_tracker.plot(epoch=True)

    te_tracker = trainer.evaluate(model, te_ds)
    rich.print(te_tracker.records)

    _, te_preds = trainer.predict(model, te_ds)
    rich.print("Test Predictions", te_preds)
