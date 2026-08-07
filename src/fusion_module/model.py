"""Model architecture for the fusion module: the shared MLP head, an optional
grade-stratified loss variant, and the early-stopping rule used by both.
"""

import torch
import torch.nn as nn


def make_fusion_mlp(input_dim: int, hidden_layers=(256, 64), dropout: float = 0.2, output_dim: int = 1) -> nn.Module:
    """Build the fusion MLP head. Only `input_dim` varies across variants.
    Every fusion variant (identity-only through full 4-way fusion) shares
    this exact architecture by design, so gains are attributable to the
    inputs, not to per-variant architecture tuning.
    """
    h1, h2 = hidden_layers
    return nn.Sequential(
        nn.Linear(input_dim, h1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h2, output_dim),
    )


class EarlyStopper:
    """Strict-improvement early stopping on a monitored metric.

    Stops when the metric has not improved by more than `min_delta` for
    `patience` consecutive epochs. Stores the best epoch/value so the
    caller can restore weights afterwards.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    def step(self, current: float, epoch: int) -> bool:
        """Return True if training should stop after this epoch."""
        if current < self.best - self.min_delta:
            self.best = current
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class GradeStratifiedHuberLoss(nn.Module):
    """Huber loss with a per-row delta selected by PSA grade.

    Used in a sensitivity probe on the best fusion variant: PSA 10 rows get
    a smaller delta (more sensitive to tail errors) than PSA 8/9 rows.
    """

    def __init__(self, delta_psa10: float, delta_default: float):
        super().__init__()
        self.delta_psa10 = delta_psa10
        self.delta_default = delta_default

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, grade: torch.Tensor) -> torch.Tensor:
        residual = (y_pred - y_true).abs()
        delta = torch.where(
            grade == 10,
            torch.full_like(residual, self.delta_psa10),
            torch.full_like(residual, self.delta_default),
        )
        quadratic = 0.5 * residual ** 2
        linear = delta * (residual - 0.5 * delta)
        loss = torch.where(residual < delta, quadratic, linear)
        return loss.mean()
