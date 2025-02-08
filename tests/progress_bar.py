import sys

sys.path.append("..")

import random
import time

import pyroml as p
from pyroml.callbacks.progress import RichProgressBar
from pyroml.loop.predict import PredictLoop
from tests.dummy.classification import DummyClassificationModel

if __name__ == "__main__":
    epochs = 3
    iters = 10

    model = DummyClassificationModel()
    trainer = p.Trainer()
    loop = PredictLoop(trainer, model)
    progress = RichProgressBar()

    metrics = {
        "loss": random.random(),
        "acc": random.random(),
    }

    with progress.progress:
        progress.on_train_start(trainer, loop, **dict())

        for e in range(epochs):
            tr_loader = range(30)
            loop.loader = tr_loader

            progress.on_train_epoch_start(trainer, loop, **dict(epoch=e))

            # Training across dataset
            for i in tr_loader:
                progress.on_train_iter_start(trainer, loop, **dict(epoch=e))
                # --- Training step
                metrics["loss"] = random.random()
                metrics["acc"] = random.random()
                time.sleep(random.random())
                # --- End of training step
                progress.on_train_iter_end(trainer, loop, metrics=metrics, epoch=e)

                # --- Validation at some point during training
                if i % iters == 0 and i > 0:
                    ev_loader = range(10)
                    progress.on_validation_start(trainer, loop, epoch=e)

                    for j in ev_loader:
                        progress.on_validation_iter_start(
                            trainer, loop, **dict(epoch=e)
                        )
                        # --- Validation step
                        metrics["ev_loss"] = random.random()
                        metrics["ev_acc"] = random.random()
                        time.sleep(random.random())
                        # --- End of validation step
                        progress.on_validation_iter_end(
                            trainer, loop, metrics=metrics, epoch=e
                        )

                    progress.on_validation_end(trainer, loop, epoch=e)
                # --- End of validation

            progress.on_train_epoch_end(trainer, loop, **dict(epoch=e))
