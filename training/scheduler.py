"""
Learning rate scheduling and curriculum learning for weather model training.

Provides:
  1. Warmup + cosine decay (standard)
  2. Curriculum learning schedule for progressive lead-time training
"""

import math
import torch


class CosineWarmupScheduler:
    """
    Linear warmup followed by cosine decay to eta_min.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of linear warmup epochs
        total_epochs: Total training epochs
        eta_min: Minimum learning rate (as fraction of base_lr)
        base_lr: Base learning rate
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        eta_min: float = 0.01,
        base_lr: float = 1e-3,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        """Update learning rate for the current epoch."""
        self.current_epoch += 1
        lr = self._get_lr(self.current_epoch - 1)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return self.base_lr * (self.eta_min + (1 - self.eta_min) * 0.5 * (
                1 + math.cos(math.pi * progress)
            ))

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class CurriculumScheduler:
    """
    Manages progressive lead-time training.

    Phase 1: Single lead time (Δt = 1 month / 4 weeks)
    Phase 2: Multi lead time (Δt ∈ {1, 2, 3} months)

    The model is trained with increasing difficulty:
      - Start with short lead times (easier)
      - Gradually introduce longer lead times
    """
    def __init__(
        self,
        phase_epochs: list = None,
        phase_lead_times: list = None,
    ):
        if phase_epochs is None:
            phase_epochs = [50, 30, 20]  # Epochs per phase
        if phase_lead_times is None:
            phase_lead_times = [[1], [1, 2], [1, 2, 3]]

        self.phase_epochs = phase_epochs
        self.phase_lead_times = phase_lead_times
        self.cumulative_epochs = []
        cumsum = 0
        for pe in phase_epochs:
            cumsum += pe
            self.cumulative_epochs.append(cumsum)

    def get_lead_times(self, epoch: int) -> list:
        """Return the list of lead times active at the given epoch."""
        for i, cum_ep in enumerate(self.cumulative_epochs):
            if epoch < cum_ep:
                return self.phase_lead_times[i]
        return self.phase_lead_times[-1]

    def get_phase(self, epoch: int) -> int:
        """Return the current phase index."""
        for i, cum_ep in enumerate(self.cumulative_epochs):
            if epoch < cum_ep:
                return i
        return len(self.cumulative_epochs) - 1
