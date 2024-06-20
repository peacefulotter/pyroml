import sys

sys.path.append("..")

import time
import random
from pyroml import Stage, Progress

if __name__ == "__main__":
    epochs = 3
    iters = 10

    p = Progress()

    metrics = {
        "loss": random.random(),
        "acc": random.random(),
    }

    with p.bar:
        for e in range(epochs):
            p.new_epoch(e)

            real_tr_loader = range(30)
            p.set_stage(Stage.TRAIN, real_tr_loader)

            # Training across dataset
            for i in real_tr_loader:
                metrics["loss"] = random.random()
                metrics["acc"] = random.random()
                time.sleep(0.1)

                # --- Validation at some point during training
                if i % iters == 0 and i > 0:
                    ev_loader = range(10)
                    p.set_stage(Stage.VAL, ev_loader)

                    for j in ev_loader:
                        metrics["ev_loss"] = random.random()
                        metrics["ev_acc"] = random.random()
                        p.advance(metrics)
                        time.sleep(0.1)

                    p.set_stage(Stage.TRAIN)
                # --- End of validation

                p.advance(metrics)
