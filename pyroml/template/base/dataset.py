import os
from os import PathLike
from pathlib import Path
from typing import Optional

import datasets
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform


class TemplateDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        folder: PathLike | str,
        transform: Optional[Transform] = None,
        split: str = "train",
        save: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.folder = folder
        self.transform = transform
        self.split = split
        self.save = save

        self.ds = self.load_dataset(**kwargs)

    def _load_dataset(self, **kwargs) -> datasets.Dataset:
        folder = Path(self.folder) / self.split
        if os.path.isdir(folder):
            ds = datasets.load_from_disk(folder, **kwargs)
        else:
            ds = datasets.load_dataset(self.dataset_path, split=self.split, **kwargs)
            if self.save:
                ds.save_to_disk(folder)

        ds = ds.with_format("torch")
        return ds

    def load_dataset(self, **kwargs):
        """
        In case you wish to implement more complex loading while wrapping the huggingface dataset loading process (_load_dataset)
        using e.g. ds.map(_)
        """
        ds = self._load_dataset(**kwargs)
        return ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        if self.transform:
            item["img"] = self.transform(item["img"])
        return item
