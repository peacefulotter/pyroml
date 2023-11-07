import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .wandb import Wandb
from .stats import Statistics
from .utils import to_device, get_lr, Callbacks


class Trainer(Callbacks):
    def __init__(self, model, config):
        self.config = config
        if config.device == "auto":
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Trainer] set on device {self.config.device}")

        self.model = model.to(device=config.device)
        # self.model = torch.compile(self.model)

        self.epoch = 0
        self.iteration = 0
        self.optimizer = self.config.optimizer(
            self.model.parameters(), lr=self.config.lr, **self.config.optimizer_params
        )
        self.criterion = self.config.criterion()
        self.scheduler = None
        if self.config.scheduler:
            self.scheduler = self.config.scheduler(
                self.optimizer, **self.config.scheduler_params
            )

        if config.wandb:
            self.wandb = Wandb(config)

        Callbacks.__init__(self)

    @staticmethod
    def get_checkpoint_path(checkpoint_folder, name, epoch, iteration):
        folder = os.path.join(checkpoint_folder, name)
        file = f"epoch={epoch:03d}_iter={iteration:06d}.pt"
        return folder, os.path.join(folder, file)

    def save_model(self):
        state = {
            "epoch": self.epoch,
            "iter": self.iteration,
            "config": self.config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler != None
            else None,
        }

        folder, cp_path = Trainer.get_checkpoint_path(
            self.config.checkpoint_folder, self.config.name, self.epoch, self.iteration
        )
        if not os.path.exists(folder):
            os.makedirs(folder)

        if self.config.verbose:
            print(
                f"\t> Saving model {self.config.name} at epoch {self.epoch}, iter {self.iteration} to {cp_path}"
            )
        torch.save(state, cp_path)

    def _load_state_dict(self, checkpoint):
        self.epoch = checkpoint["epoch"]
        self.iteration = checkpoint["iteration"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.config.scheduler != None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    @staticmethod
    def from_pretrained(model, checkpoint_folder, model_name, epoch, iteration):
        _, cp_path = Trainer.get_checkpoint_path(
            checkpoint_folder, model_name, epoch, iteration
        )
        checkpoint = torch.load(cp_path)
        config = checkpoint["config"]
        trainer = Trainer(model, config)
        trainer._load_state_dict(checkpoint)
        return trainer

    def on_batch_end(self, statistics, output, target, loss):
        self.trigger_callbacks("on_batch_end")

        if self.config.stats_every == None or (
            self.iteration != 0 and self.iteration % self.config.stats_every != 0
        ):
            return

        stats = statistics.register(output, target, loss, self.epoch, self.iteration)

        if self.config.wandb:
            self.wandb.log(stats)

        self.trigger_callbacks("on_stats", **stats)

        if self.config.verbose:
            if "eval" in stats:
                print(
                    f"[{self.epoch:03d} | {self.iteration:05d}:{self.config.max_iterations:05d}] | [Loss] tr: {stats['train']['loss']:.4f}, ev: {stats['eval']['loss']:.4f} | [Acc] tr: {stats['train']['acc']:.4f}, ev: {stats['eval']['acc']:.4f} | [RMSE] tr: {stats['train']['rmse']:.4f}, ev: {stats['eval']['rmse']:.4f} | [Lr] {stats['lr']:.4f}"
                )
            else:
                print(
                    f"[{self.epoch:03d} | {self.iteration:05d}:{self.config.max_iterations:05d}] | [Loss] tr: {stats['train']['loss']:.4f} | [Acc] tr: {stats['train']['acc']:.4f} | [RMSE] tr: {stats['train']['rmse']:.4f} | [Lr] {stats['lr']:.4f}"
                )

    def run(self, train_dataset, eval_dataset=None):
        device = self.config.device
        self.model.train()

        statistics = Statistics(
            self.model,
            self.optimizer,
            self.criterion,
            self.scheduler,
            self.config,
            eval_dataset,
        )

        train_loader = DataLoader(
            train_dataset,
            sampler=torch.utils.data.RandomSampler(
                train_dataset, replacement=True, num_samples=int(1e5)
            ),
            shuffle=False,
            pin_memory=False,  # FIXME: set this to true
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        data_iter = iter(train_loader)

        if self.config.wandb:
            self.wandb.init(self.model, self.optimizer, self.criterion, self.scheduler)

        while self.iteration < self.config.max_iterations and (
            self.config.epochs == None or self.epoch < self.config.epochs
        ):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                self.epoch += 1
                self.trigger_callbacks("on_epoch_end")
                batch = next(data_iter)

            self.trigger_callbacks("on_batch_start")
            data, target = batch
            data, target = to_device(data, device), to_device(target, device)

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            if self.config.grad_norm_clip != None and self.config.grad_norm_clip != 0.0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.on_batch_end(statistics, output, target, loss)

            self.iteration += 1

        return statistics
