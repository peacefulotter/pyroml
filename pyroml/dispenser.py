from torch.utils.data import Dataset, DataLoader, RandomSampler

from pyroml.status import Status
from pyroml.config import Config
from pyroml.utils import Stage, Callback


class Dispenser:

    def __init__(self, config: Config, dataset: Dataset, stage: Stage):
        self.config = config
        self.dataset = dataset
        self.stage = stage

        self.loader = self._get_dataloader()
        self.status = Status(stage)

    def _get_dataloader(self):
        is_training = self.stage == Stage.TRAIN
        batch_size = (
            self.config.batch_size if is_training else self.config.eval_batch_size
        )
        num_workers = (
            self.config.num_workers if is_training else self.config.eval_num_workers
        )
        sampler = RandomSampler(self.dataset, replacement=True) if is_training else None

        return DataLoader(
            self.dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=str(self.type_ctx.device) != "cpu",
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def iterate(self):

        training = self.stage == Stage.TRAIN
        max_epochs = self.config.max_epochs if training else 1
        max_steps = self.config.max_steps if training else self.config.eval_max_steps

        if max_epochs is None and self.config.max_steps is None:
            msg = "Either max_epochs or max_steps must be defined for training"
            raise ValueError(msg)

        data_iter = iter(self.loader)

        self.trigger_callback(Callback.ON_EPOCH_START(self.stage))

        while True:
            if max_steps and self.status.step >= max_steps:
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                self.trigger_callback(Callback.ON_EPOCH_END(self.stage))

                if max_epochs is not None and self.status.epoch + 1 >= max_epochs:
                    break

                data_iter = iter(self.loader)
                batch = next(data_iter)
                self.status.epoch += 1

                self.trigger_callback(Callback.ON_EPOCH_START(self.stage))

            self.status.step += 1
            yield batch
