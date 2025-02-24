import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pyroml.callbacks import Callback
from pyroml.core.stage import Stage

from .utils import compute_assignment_from_cost, cosine_annealing, entropy

if TYPE_CHECKING:
    from pyroml.callbacks.callback import CallbackArgs

    from .model import Falcon


class FalconLoss(nn.Module, Callback):
    def __init__(
        self,
        model: "Falcon",
        fine_classes: int,
        coarse_classes: int,
        beta_reg: float,
        time_limit: float,
        soft_labels_epochs: int,
        solve_every: int,
        loss_temp: float,
        loss_lambda1: float,
        loss_lambda2: float,
        loss_lambda3: float,
    ):
        super().__init__()
        self.model = model
        self.fine_classes = fine_classes
        self.coarse_classes = coarse_classes
        self.beta_reg = beta_reg
        self.time_limit = time_limit
        self.soft_labels_epochs = soft_labels_epochs
        self.solve_every = solve_every
        self.temp = loss_temp
        self.lambda1 = loss_lambda1
        self.lambda2 = loss_lambda2
        self.lambda3 = loss_lambda3

        self.tau = 0.0
        self.fine_preds: list[torch.Tensor] = []
        self.coarse_labels: list[torch.Tensor] = []

        self.M: torch.Tensor

    def __repr__(self):
        """
        Overriding default nn.Module to prevent recursion error since model -> loss -> model -> ...
        """
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            if key == "model":
                continue
            else:
                mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    @property
    def device(self):
        return self.model.device

    def compute_assignment_from_cost(self, cost: torch.Tensor):
        return compute_assignment_from_cost(
            cost.cpu().numpy(),
            reg_coef=self.beta_reg,
            time_limit=self.time_limit,
        )

    def on_train_start(self, args: "CallbackArgs"):
        # TODO: retrieve fine/coarse classes from either kwargs or => trainer
        # Add a way to retrieve current loop, current status and current dataset
        cost = torch.randn(self.fine_classes, self.coarse_classes).softmax(-1)
        self.M = self.compute_assignment_from_cost(cost).to(self.device)

    def on_train_epoch_start(self, args: "CallbackArgs"):
        epoch: int = args.loop.epoch
        self.tau = (
            cosine_annealing(1.0, 0.0, epoch - 1, self.soft_labels_epochs)
            if epoch <= self.soft_labels_epochs
            else 0.0
        )
        self.beta_reg = self.beta_reg if epoch <= self.soft_labels_epochs else 0.0
        print(f"Tau: {self.tau}")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        stage: Stage,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor
        x, y_coarse, neighbors = batch["img"], batch["coarse_label"], batch["neighbors"]
        if stage == Stage.TRAIN:
            self.coarse_labels.append(y_coarse)

        x, x_ema = x[0].to(self.device), x[1].to(self.device)
        y_coarse = y_coarse.to(self.device)

        N, num_neig = neighbors.shape[:2]
        pattern = (
            "n k c h w -> (n k) c h w"
            if len(neighbors.shape) == 5
            else "n k c -> (n k) c"
        )
        neighbors = rearrange(neighbors, pattern)

        with torch.amp.autocast(
            enabled=self.model.trainer.dtype == torch.float16
            if self.model.trainer is not None
            else False
        ):
            with torch.no_grad():
                with self.model.ema.average_parameters():
                    logits_ema: torch.Tensor = self(x_ema)
                    logits_ema = logits_ema.detach()
                    ema_probs = logits_ema.softmax(-1)

                    if stage == Stage.TRAIN:
                        self.fine_preds.append(ema_probs)

                    neighbors = neighbors.to(self.device)
                    neighbors = self(neighbors)
                    neighbors = rearrange(
                        neighbors.softmax(-1).detach(),
                        "(n k) c -> n k c",
                        n=N,
                        k=num_neig,
                    )

            logits: torch.Tensor = self.model(x)
            probs: torch.Tensor = logits.softmax(-1)

            log_coarse_prob = torch.logsumexp(
                logits.unsqueeze(-1)
                .repeat(1, 1, self.M.shape[1])
                .masked_fill_(
                    ~self.M.unsqueeze(0).repeat(logits.shape[0], 1, 1).bool(),
                    float("-inf"),
                ),
                dim=1,
            ) - torch.logsumexp(logits, dim=-1, keepdim=True)

            loss_consist = (
                -torch.einsum("n c, n k c -> n k", probs, neighbors).log().mean()
            )
            assert loss_consist.requires_grad

            mask = ~self.M.T[y_coarse].bool()
            q_fine_soft = (
                (logits_ema / self.temp).masked_fill(mask, float("-inf")).softmax(-1)
            )

            q_fine_hard = F.one_hot(
                q_fine_soft.argmax(-1), q_fine_soft.shape[-1]
            ).float()
            pseudo_fine_lbls = self.tau * q_fine_soft + (1 - self.tau) * q_fine_hard

            loss_cls_fine = (
                -(F.log_softmax(logits, dim=-1) * pseudo_fine_lbls).sum(-1).mean()
            )
            loss_cls_coarse = F.cross_entropy(log_coarse_prob, y_coarse)

            avg_prob = probs.mean(0)
            loss_reg_fine: torch.Tensor = -entropy(avg_prob) + math.log(probs.shape[1])

            loss_total = (
                self.lambda1 * loss_cls_coarse
                + self.lambda2 * (loss_cls_fine + loss_consist)
                + self.lambda3 * loss_reg_fine
            )

            self.model.log(
                loss_reg=loss_reg_fine.item(),
                loss_fine=loss_cls_fine.item(),
                loss_coarse=loss_cls_coarse.item(),
                loss=loss_total.item(),
            )

        return loss_total

    def on_train_iter_end(self, args: "CallbackArgs"):
        # FIXME: doubt its on train iter end
        step: int = args.loop.step
        if step > 0 and step % self.solve_every == 0:
            preds = torch.cat(self.fine_preds, dim=0).to(self.device)
            labels = torch.cat(self.coarse_labels, dim=0).to(self.device)
            coarse_gt_oh = F.one_hot(labels, self.coarse_classes).float()
            cost = (preds.T @ coarse_gt_oh) / coarse_gt_oh.shape[0]
            self.M = self.compute_assignment_from_cost(cost).to(self.device)
            self.coarse_labels, self.fine_preds = [], []
