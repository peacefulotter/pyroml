import torch
import torch.nn as nn
from torch.optim import AdamW

from .utils import Record


def Config(
    name,
    max_iterations,
    device="auto",
    lr=1e-4,
    epochs=None,
    batch_size=64,
    dropout=0.1,
    compile=True,
    stats_every=1,
    evaluate=True,
    evaluate_every=10,
    metrics=[],
    grad_norm_clip=1.0,
    num_workers=4,
    eval_num_workers=0,
    optimizer=AdamW,
    criterion=nn.MSELoss,
    scheduler=None,
    optimizer_params=None,
    scheduler_params=None,
    wandb=True,
    wandb_project=None,
    checkpoint_folder="./checkpoints",
    verbose=False,
):
    """
    Returns a configuration object with the specified hyperparameters.

    Args:
        name (str): Name of the configuration.
        max_iterations (int): Maximum number of iterations.
        device (str, optional): Device to train on. Defaults to "auto" which will use GPU if available.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        epochs (int, optional): Number of epochs (if max_iterations is not defined). Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 64.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        compile (bool, optional): Whether to compile the model, this can significantly improve training time but is not supported on all GPUs. Defaults to True.
        stats_every (int, optional): Compute statistics every `stats_every` iterations. Defaults to 1.
        evaluate (bool or str, optional): Whether to periodically evaluate the model on the evaluation dataset, or 'epoch' to evaluate every epoch. Defaults to True.
        evaluate_every (int, optional): Evaluate every `evaluate_every` iterations / or epoch if evaluate is set to 'epoch'. Defaults to 10.
        metrics (list, optional): List of metric classes to compute. Defaults to []. Possible values: metrics.Accuracy, metrics.RMSE, or your own class inheriting Metrics.
        grad_norm_clip (float, optional): Gradient norm clipping. Defaults to 1.0.
        num_workers (int, optional): Number of workers for the dataloader. Defaults to 4.
        eval_num_workers (int, optional): Number of workers for the evaluation dataloader. Note that a value > 0 can cause an AssertionError: 'can only test a child process during evaluation'. Defaults to 0.
        optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to AdamW.
        criterion (torch.nn.modules.loss._Loss, optional): Loss function. Defaults to nn.MSELoss.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler. Defaults to None.
        optimizer_params (dict, optional): Optimizer parameters. Defaults to None.
        scheduler_params (dict, optional): Scheduler parameters. Defaults to None.
        wandb (bool, optional): Whether to use wandb. Defaults to True.
        wandb_project (str, optional): Wandb project name, if wandb is set to True. Defaults to None.
        checkpoint_folder (str, optional): Folder to save checkpoints. Defaults to "./checkpoints".
        verbose (bool, optional): Whether to print details of whats going on in the system. Defaults to False.

    Returns:
        Record: Configuration object with the specified hyperparameters.
    """
    if optimizer_params == None:
        optimizer_params = {}

    if scheduler == None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR
        scheduler_params = dict(
            {
                "max_lr": lr,
                "pct_start": 0.02,
                "anneal_strategy": "cos",
                "cycle_momentum": False,
                "div_factor": 1e2,
                "final_div_factor": 0.05,
                "total_steps": max_iterations,
            }
            if scheduler_params == None
            else scheduler_params
        )

    return Record(
        name=name,
        max_iterations=max_iterations,
        device=device,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        compile=compile,
        stats_every=stats_every,
        evaluate=evaluate,
        evaluate_every=evaluate_every,
        metrics=metrics,
        grad_norm_clip=grad_norm_clip,
        num_workers=num_workers,
        eval_num_workers=eval_num_workers,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        wandb=wandb,
        wandb_project=wandb_project,
        checkpoint_folder=checkpoint_folder,
        verbose=verbose,
    )
