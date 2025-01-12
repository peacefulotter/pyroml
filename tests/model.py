# Inspired by https://github.com/yangzhangalmo/pytorch-iris/blob/master/main.py

import os
import torch
import datasets
import numpy as np
import torch.nn as nn

import sys

sys.path.append("..")

from pyroml.trainer import Trainer

from dummy.regression import DummyRegressionDataset, DummyRegressionModel
from dummy.classification import DummyClassificationDataset, DummyClassificationModel


def forward(model, dataset):
    for loss in [nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()]:
        x, y = classification_dataset[0]
        preds = classification_model(x)
        loss(preds, y)


if __name__ == "__main__":

    classification_model = DummyClassificationModel()
    classification_dataset = DummyClassificationDataset()
    forward(classification_model, classification_dataset)

    regression_model = DummyRegressionModel()
    regression_dateset = DummyClassificationDataset()
    forward(regression_model, regression_dateset)
