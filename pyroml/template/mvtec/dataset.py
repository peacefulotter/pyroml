from os import PathLike
from typing import Optional

from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform

from pyroml.template.base import load_dataset


class MVTecDataset(Dataset):
    def __init__(
        self,
        folder: PathLike | str = "data/mvtec",
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

    def _load_dataset(self):
        ds = load_dataset(
            dataset="Voxel51/mvtec-ad",
            folder=self.folder,
            split=self.split,
            save=self.save,
        )
        return ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        print(item.keys())
        if self.transform:
            item["img"] = self.transform(item["img"] / 255.0)
        return item
