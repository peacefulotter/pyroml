import torch

from typing import Any
from typing import cast
from pathlib import Path


from pyroml.trainer import Trainer


class UndefinedFolder(Exception):
    pass


class UnexpectedFileContent(Exception):
    pass


class UnexpectedAttribute(Exception):
    pass


class AttributeNotFound(Exception):
    pass


class WithHyperParameters:
    def __init__(self, hparams_file: Path | str, **kwargs):
        super().__init__(**kwargs)
        assert (
            hparams_file is not None
        ), "hparams_file in WithHyperParameters class is not set properly"
        self.hparams_file = hparams_file
        self.hparams: set[str] = set()

    def _resolve_hparams_folder(self, folder: Path | str | None = None):
        if folder is not None:
            return folder
        elif isinstance(self, Trainer):
            return self.checkpoint_folder
        elif hasattr(self, "trainer"):
            return cast(self.trainer, Trainer)._resolve_checkpoint_folder()
        raise UndefinedFolder(
            "Attempting to save/load hyperparameters without a trainer assigned, a checkpoint folder must be specified"
        )

    def _resolve_hparams_file(self, file: Path | str = None):
        return file or self.hparams_file

    def _resolve_path(self, folder=None, file=None):
        folder = self._resolve_hparams_folder(folder=folder)
        file = self._resolve_hparams_file(file=file)
        return Path(folder) / Path(file)

    def register_hparams(self, *hparams_attributes):
        for attr in hparams_attributes:
            if not isinstance(attr, str):
                msg = f"The method `register_hparams` takes one or multiple attributes as string, gave {type(attr)}"
                raise UnexpectedAttribute(msg)
            if not hasattr(self, attr):
                msg = f"The attribute {attr} is not in {self}"
                raise AttributeNotFound(msg)

            self.hparams.add(attr)

    def save_hparams(self, folder=None, file=None):
        hparams = {}
        for attr in self.hparams:
            hparams[attr] = self.__getattribute__(attr)

        f = self._resolve_path(folder=folder, file=file)
        torch.save(hparams, f)

    def load_hparams(self, folder=None, file=None):
        f = self._resolve_path(folder=folder, file=file)
        hparams = torch.load(f, map_location="cpu")

        if not isinstance(hparams, dict):
            raise UnexpectedFileContent(
                f"While attempting to load hyperparameters from a file, the file at {f} does not contain a dictionnary of hyperparameters, but is instead of type {type(hparams)}"
            )

        for attr, val in hparams.items():
            self.__setattr__(attr, val)
