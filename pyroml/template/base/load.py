import os
from os import PathLike
from pathlib import Path

import datasets


def load_dataset(
    dataset: str, folder: PathLike | str, split: str = "train", save: bool = True
) -> datasets.Dataset:
    folder = Path(folder) / split
    if os.path.isdir(folder):
        ds = datasets.load_from_disk(folder)
    else:
        ds = datasets.load_dataset(dataset, split=split)
        if save:
            ds.save_to_disk(folder)

    ds = ds.with_format("torch")
    return ds
