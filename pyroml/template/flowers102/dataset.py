from os import PathLike
from typing import Optional

from torchvision.transforms.v2 import Transform

from pyroml.template.base import TemplateDataset


class Flowers102Dataset(TemplateDataset):
    def __init__(
        self,
        folder: PathLike | str = "data/flowers102",
        transform: Optional[Transform] = None,
        split: str = "train",
        save: bool = True,
    ):
        super().__init__(
            dataset_path="nelorth/oxford-flowers",
            folder=folder,
            transform=transform,
            split=split,
            save=save,
        )
