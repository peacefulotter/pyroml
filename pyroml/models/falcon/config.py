import torch

from dataclasses import dataclass


@dataclass
class FalconConfig:
    """
    Default values are mostly taken from https://github.com/mlbio-epfl/falcon/blob/main/configs/cifar100/coarse2fine/base.yaml
    """

    # Number of fine and coarse classes
    fine_classes: int
    coarse_classes: int

    # Model
    embed_dim: int  # Model embedding size (last layer output shape)
    head_type: str = "Linear"

    # Training parameters
    beta_reg: float = 0.1
    time_limit: float = 30  # in seconds
    soft_labels_epochs: str = 30
    solve_every: int = 20

    ## Loss
    loss_temp: float = 0.9
    loss_lambda1: float = 0.5
    loss_lambda2: float = 0.5
    loss_lambda3: float = 0.5
