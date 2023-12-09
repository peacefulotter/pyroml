# Inspired by https://github.com/yangzhangalmo/pytorch-iris/blob/master/main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import sys

sys.path.append("..")

from pyroml.config import Config
from pyroml.trainer import Trainer
from pyroml.metrics import Accuracy


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Linear(100, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.module(x)


class IrisDataset(Dataset):
    def __init__(self, ds):
        super().__init__()
        self.x = torch.empty(len(ds), 4)
        self.y = torch.zeros(len(ds), 1, dtype=torch.int64)
        for i, iris in enumerate(ds):
            self.x[i] = torch.stack(
                (
                    iris["SepalLengthCm"],
                    iris["SepalWidthCm"],
                    iris["PetalLengthCm"],
                    iris["PetalWidthCm"],
                )
            )
            self.y[i] = iris["Species"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx].float()
        y = (
            F.one_hot(
                self.y[idx],
                num_classes=3,
            )
            .squeeze()
            .float()
        )
        return x, y


if __name__ == "__main__":
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    def map_species(x):
        x["Species"] = species.index(x["Species"])
        return x

    ds = load_dataset("scikit-learn/iris", split="train")
    ds = ds.map(map_species)
    ds = ds.with_format("torch")
    dataset = IrisDataset(ds)

    # train_X, test_X, train_y, test_y = train_test_split(
    #     dataset[dataset.columns[0:4]].values, dataset.species.values, test_size=0.8
    # )

    model = IrisNet()

    a, b = dataset[:3]
    c = model(a)
    assert c.shape == b.shape

    config = Config(
        name="IrisModelV1",
        max_iterations=2048,
        batch_size=64,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        lr=0.01,
        evaluate=False,
        wandb=True,
        wandb_project="pyro_main_test",
        verbose=False,
    )

    trainer = Trainer(model, config)
    trainer.run(dataset)

    # predict_out = model(test_X)
    # _, predict_y = torch.max(predict_out, 1)

    # print 'prediction accuracy', accuracy_score(test_y.data, predict_y.data)

    # print 'macro precision', precision_score(test_y.data, predict_y.data, average='macro')
    # print 'micro precision', precision_score(test_y.data, predict_y.data, average='micro')
    # print 'macro recall', recall_score(test_y.data, predict_y.data, average='macro')
    # print 'micro recall', recall_score(test_y.data, predict_y.data, average='micro')
