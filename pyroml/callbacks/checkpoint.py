import os
import re
import warnings
from pathlib import Path
from typing import Literal

from pyroml.callbacks.callback import Callback, CallbackArgs


class CheckpointCallback(Callback):
    def __init__(
        self,
        on=Literal["epoch", "step"],
        every: int = 1,
        checkpoint_folder: Path | str = "./checkpoints",
    ):
        super().__init__()
        self.on = on
        self.every = every
        self.version = -1
        self.checkpoint_folder = checkpoint_folder

    def _fetch_version(self):
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        for f in os.scandir(self.checkpoint_folder):
            if not f.is_dir() or not re.match(r"v_(\d+)", f.name):
                continue
            version = int(f.name[2:])
            self.version = max(self.version, version)

        self.version += 1
        self.checkpoint_folder = self.checkpoint_folder / f"v_{self.version}"

    def save_checkpoints(self, args: "CallbackArgs"):
        if self.version < 0:
            self._fetch_version()
        warnings.warn(
            "CheckpointCallback will override previous checkpoint on all save"
        )
        args.model.save(folder=self.checkpoint_folder)
        args.trainer.save(folder=self.checkpoint_folder)

    def on_train_iter_end(self, args):
        if self.on == "step" and self.every % args.loop.step == 0:
            self.save_checkpoints(args)

    def on_train_epoch_end(self, args):
        if self.on == "epoch" and self.every % args.loop.step == 0:
            self.save_checkpoints(args)
