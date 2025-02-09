from pathlib import Path
from typing import Optional

from torchvision.transforms.v2 import Transform

from pyroml.template.base.dataset import TemplateDataset


class Imagenet1kDataset(TemplateDataset):
    def __init__(
        self,
        folder: Path | str = "data/imagenet1k",
        transform: Optional[Transform] = None,
        split: str = "train",
        save: bool = True,
    ):
        super().__init__(
            dataset_path="ILSVRC/imagenet-1k",
            folder=folder,
            transform=transform,
            split=split,
            save=save,
            trust_remote_code=True,
        )
