import torch

from typing import cast
from pathlib import Path

import pyroml as p
from pyroml.utils import get_classname


class UndefinedFolder(Exception):
    pass


class UnexpectedFileContent(Exception):
    pass


class UnexpectedAttribute(Exception):
    pass


class AttributeNotFound(Exception):
    pass


log = p.get_logger(__name__)


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
        elif isinstance(self, p.Trainer):
            return self.checkpoint_folder
        elif hasattr(self, "trainer"):
            return cast(self.trainer, p.Trainer)._resolve_checkpoint_folder()
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
        log.info(f"Saving {get_classname(self)} hparams to {f}")
        torch.save(hparams, f)

    @staticmethod
    def _load_hparams(f: Path):
        log.info(f"Loading hyperparameters from {f}")
        hparams = torch.load(f, map_location="cpu")

        if not isinstance(hparams, dict):
            raise UnexpectedFileContent(
                f"While attempting to load hyperparameters from a file, the file at {f} does not contain a dictionnary of hyperparameters, but is instead of type {type(hparams)}"
            )

        return hparams

    def load_hparams(self, folder=None, file=None):
        f = self._resolve_path(folder=folder, file=file)
        hparams = WithHyperParameters._load_hparams(f)
        for attr, val in hparams.items():
            self.__setattr__(attr, val)
