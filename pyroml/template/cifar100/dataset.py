from os import PathLike
from typing import Optional

from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform

from pyroml.template.base import load_dataset


class Cifar100Dataset(Dataset):
    def __init__(
        self,
        folder: PathLike | str = "data/cifar100",
        transform: Optional[Transform] = None,
        split: str = "train",
        save: bool = True,
    ):
        super().__init__()
        self.folder = folder
        self.transform = transform
        self.split = split
        self.save = save

        self.ds = self._load_dataset()
        self.fine_labels = self.ds.info.features["fine_label"].names
        self.coarse_labels = self.ds.info.features["coarse_label"].names

    def _load_dataset(self):
        ds = load_dataset(
            dataset="uoft-cs/cifar100",
            folder=self.folder,
            split=self.split,
            save=self.save,
        )
        return ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        if self.transform:
            item["img"] = self.transform(item["img"] / 255.0)
        return item
