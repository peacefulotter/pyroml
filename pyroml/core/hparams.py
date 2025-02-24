from pathlib import Path
from typing import TYPE_CHECKING, Optional

import safetensors.torch as st
import torch

from pyroml.utils import get_classname
from pyroml.utils.date import get_date
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    pass


class UndefinedFolder(Exception):
    pass


class UnexpectedFileContent(Exception):
    pass


class UnexpectedAttribute(Exception):
    pass


class AttributeNotFound(Exception):
    pass


log = get_logger(__name__)


class WithHyperParameters:
    def __init__(self, hparams_file: Path | str) -> None:
        self.hparams_file = hparams_file
        self._hparams: set[str] = set()

    def _resolve_hparams_file(self, file: Optional[Path | str] = None) -> Path | str:
        return file or self.hparams_file

    def _resolve_path(self, folder: Path | str, file: Optional[Path | str] = None):
        return Path(folder) / self._resolve_hparams_file(file=file)

    def register_hparams(self, *hparams_attributes):
        for attr in hparams_attributes:
            if not isinstance(attr, str):
                msg = f"The method `register_hparams` takes one or multiple attributes as string, gave {type(attr)}"
                raise UnexpectedAttribute(msg)
            if not hasattr(self, attr):
                msg = f"The attribute {attr} is not in {self}"
                raise AttributeNotFound(msg)

            self._hparams.add(attr)

    def save_hparams(self, folder: Path | str, file: Optional[Path | str] = None):
        hparams = {"date": get_date()}
        for attr in self._hparams:
            hparams[attr] = self.__getattribute__(attr)

        f = self._resolve_path(folder=folder, file=file)
        log.info(f"Saving {get_classname(self)} hparams to {f}")
        torch.save(hparams, f)

    @staticmethod
    def _load_hparams(f: Path):
        log.info(f"Loading hyperparameters from {f}")
        hparams = st.load_file(f, device="cpu")

        if not isinstance(hparams, dict):
            raise UnexpectedFileContent(
                f"While attempting to load hyperparameters from a file, the file at {f} does not contain a dictionnary of hyperparameters, but is instead of type {type(hparams)}"
            )

        return hparams

    def load_hparams(self, folder: Path | str, file: Optional[Path | str] = None):
        f = self._resolve_path(folder=folder, file=file)
        hparams = WithHyperParameters._load_hparams(f)
        for attr, val in hparams.items():
            self.__setattr__(attr, val)
