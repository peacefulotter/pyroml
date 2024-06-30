from typing import Callable

from torch.utils.data import Dataset, DataLoader, RandomSampler

import pyroml as p
from pyroml.utils import Stage
from pyroml.progress_bar import ProgressBar


class BatchIterator:
    @staticmethod
    def _get_dataloader(
        trainer: "p.Trainer", dataset: Dataset, config: "p.Config", stage: Stage
    ):
        is_training = stage == Stage.TRAIN
        batch_size = config.batch_size if is_training else config.eval_batch_size
        num_workers = config.num_workers if is_training else config.eval_num_workers
        sampler = RandomSampler(dataset, replacement=True) if is_training else None

        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=str(trainer.type_ctx.device) != "cpu",
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @staticmethod
    def iterate(
        trainer: "p.Trainer",
        dataset: Dataset,
        task_name: str,
        progress: ProgressBar,
        cb: Callable,
    ):
        status = trainer.status
        config = trainer.config
        stage = status.stage

        loader = BatchIterator._get_dataloader(trainer, dataset, config, stage)
        data_iter = iter(loader)

        training = stage == Stage.TRAIN
        max_epochs = config.max_epochs if training else 1
        max_steps = config.max_steps if training else config.eval_max_steps

        if max_epochs is None and config.max_steps is None:
            msg = "Either max_epochs or max_steps must be defined for training"
            raise ValueError(msg)

        trainer._trigger_callback("epoch_start")
        progress.add_stage(length=len(loader), task_name=task_name)

        while True:
            if max_steps and status.step >= max_steps:
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                trainer._trigger_callback("epoch_end")

                if max_epochs is not None and status.epoch + 1 >= max_epochs:
                    break

                data_iter = iter(loader)
                batch = next(data_iter)
                status.advance_epoch()

                trainer._trigger_callback("epoch_start")

            trainer._trigger_callback("iter_start")

            metrics = cb(batch)
            status.advance_step()

            trainer._trigger_callback("iter_end", metrics=metrics)
