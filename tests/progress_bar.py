import sys

sys.path.append("..")

import time
import random
from pyroml import Stage, ProgressBar

if __name__ == "__main__":
    epochs = 3
    iters = 10

    trainer = None
    p = ProgressBar()

    metrics = {
        "loss": random.random(),
        "acc": random.random(),
    }

    with p.bar:
        p.on_train_start(trainer, **dict())

        for e in range(epochs):
            tr_loader = range(30)

            p.on_train_epoch_start(trainer, **dict())

            # Training across dataset
            for i in tr_loader:
                p.on_train_iter_start(trainer, **dict())
                # --- Training step
                metrics["loss"] = random.random()
                metrics["acc"] = random.random()
                time.sleep(random.random())
                # --- End of training step
                p.on_train_iter_end(trainer, metrics=metrics)

                # --- Validation at some point during training
                if i % iters == 0 and i > 0:
                    ev_loader = range(10)
                    p.on_validation_start(trainer)

                    for j in ev_loader:
                        p.on_validation_iter_start(trainer, **dict())
                        # --- Validation step
                        metrics["ev_loss"] = random.random()
                        metrics["ev_acc"] = random.random()
                        time.sleep(random.random())
                        # --- End of validation step
                        p.on_validation_iter_end(trainer, metrics=metrics)

                    p.on_validation_end(trainer)
                # --- End of validation

            p.on_train_epoch_end(trainer, **dict())
