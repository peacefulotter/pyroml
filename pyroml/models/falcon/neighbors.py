from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from pyroml.models.backbone import Backbone


def _neighbors_path(dataset: Dataset, backbone: str, num_neighbors: int):
    if hasattr(dataset, "folder"):
        dataset_folder = Path(str(dataset.folder))
    else:
        dataset_folder = Path("data") / dataset.__class__.__name__

    return (
        dataset_folder
        / f"neighbors_backbone={backbone}_num-neighbors={num_neighbors}.pth"
    )


def gather_representations(loader, model):
    num_samples = len(loader.dataset)
    actual = torch.zeros(num_samples, dtype=torch.long)
    actual_coarse = torch.zeros(num_samples, dtype=torch.long)
    feats = None
    bs = loader.batch_size
    with tqdm(total=len(loader.dataset)) as progress_bar:
        with torch.no_grad():
            for idx, data in enumerate(loader):
                x = data["inputs"]
                y_fine = data["fine_label"]
                y_coarse = data["coarse_label"]
                x = x[1].cuda()
                y_fine = y_fine.cuda()
                actual[idx * bs : (idx + 1) * bs] = y_fine
                actual_coarse[idx * bs : (idx + 1) * bs] = y_coarse
                feats_ = model(x)
                if feats is None:
                    feats = torch.zeros(
                        num_samples, feats_.shape[-1], dtype=torch.float32
                    )
                feats[idx * bs : (idx + 1) * bs] = feats_.cpu()
                progress_bar.update(x.size(0))

    feats = F.normalize(feats, dim=-1)
    return feats, actual, actual_coarse


def neighours_with_coarse_(backbone: str, dataset: Dataset, num_neighbors: int):
    _FAISS_AVAILABLE = False
    try:
        import faiss

        _FAISS_AVAILABLE = True
    except Exception:
        pass

    model = Backbone.load(backbone, pre_trained=True)
    model.eval()
    feats, actual, actual_coarse = gather_representations(loader, model)

    neighbors = torch.zeros((feats.shape[0], num_neighbors), dtype=torch.long)
    indices = torch.arange(feats.shape[0])
    bar = tqdm(actual_coarse.unique())
    for coarse_y in bar:
        coarse_indices = indices[actual_coarse == coarse_y]

        if _FAISS_AVAILABLE:
            faiss_index = faiss.IndexFlatIP(feats.shape[-1])
            faiss_index.add(feats[actual_coarse == coarse_y].numpy())
            neighbors[actual_coarse == coarse_y] = coarse_indices[
                faiss_index.search(
                    feats[actual_coarse == coarse_y].numpy(), num_neighbors + 1
                )[1][:, 1:]
            ]
        else:
            _feat = feats[actual_coarse == coarse_y].cuda()
            _idx = (
                (_feat @ _feat.T)
                .topk(num_neighbors + 1, dim=-1, largest=True)[1][:, 1:]
                .cpu()
            )
            neighbors[actual_coarse == coarse_y] = coarse_indices[_idx]

        bar.set_description(f"Finished coarse class {coarse_y}")

    neighbors_classes = actual[neighbors]
    return neighbors, neighbors_classes, actual


def _compute_neighbors(dataset: Dataset, num_neighbors: int):
    neighbors_with_coarse, neighbors_classes_with_coarse, actual = (
        neighours_with_coarse_(model, train_loader, num_neighbors, cfg.USE_FAISS)
    )
    correct_neighbors_with_coarse_percentage = (
        (actual.unsqueeze(1).repeat(1, num_neighbors) == neighbors_classes_with_coarse)
        .float()
        .mean()
    )
    print(
        f"Accuracy of neighbors w/ coarse: {correct_neighbors_with_coarse_percentage * 100}%"
    )

    return neighbors_with_coarse


def _load_or_compute_neighbors(
    dataset: Dataset, backbone, num_neighbors: int
) -> torch.Tensor:
    path = _neighbors_path(dataset, backbone, num_neighbors)
    try:
        return torch.load(path)
    except FileNotFoundError:
        neighbors = _compute_neighbors(dataset, backbone, num_neighbors)
        torch.save(neighbors, path)
        return neighbors


class NeighborsWrapper(Dataset):
    def __init__(self, dataset: Dataset, backbone: str, num_neighbors: int):
        self.dataset = dataset
        self.neighbors = _load_or_compute_neighbors(dataset, backbone, num_neighbors)
        self.num_neighbors = num_neighbors
        assert num_neighbors <= self.neighbors.shape[-1]

    def _find_neighbors(self, idx):
        neighbors_indices = np.random.choice(
            self.neighbors[idx], self.num_neighbors, replace=False
        )
        neighbors = torch.cat(
            [self.dataset[_idx]["img"].unsqueeze(0) for _idx in neighbors_indices],
            dim=0,
        )
        return neighbors

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["neighbors"] = self._find_neighbors(idx)
        return item

    def __len__(self):
        return len(self.dataset)
