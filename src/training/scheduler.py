"""Learning rate schedulers for STT training."""

import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupScheduler(LRScheduler):
    """Cosine annealing with linear warmup.

    Linearly increases LR from 0 to base_lr over warmup_steps,
    then decreases following a cosine curve to min_lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class NoamScheduler(LRScheduler):
    """Noam learning rate schedule (from 'Attention Is All You Need').

    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 10000,
        scale: float = 1.0,
        last_epoch: int = -1,
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        lr = self.scale * (self.d_model ** -0.5) * min(
            step ** -0.5, step * self.warmup_steps ** -1.5
        )
        return [lr for _ in self.base_lrs]
