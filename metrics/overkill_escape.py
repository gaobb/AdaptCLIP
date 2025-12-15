"""OverkillEscape metric implementation.

Computes overkill rate (false positive rate) at fixed escape rate threshold.
- Overkill rate: proportion of normal samples incorrectly flagged as anomalous
- Escape rate: proportion of actual anomalies that are missed (false negative rate)
"""

import logging

import torch
from torchmetrics import Metric

logger = logging.getLogger(__name__)


class OverkillEscape(Metric):
    """OverkillEscape metric for computing overkill rate at fixed escape rate threshold.

    Args:
        escape_rate (float): Escape rate threshold percentage (proportion of anomalies missed). Default: 2.0
        **kwargs: Additional arguments to parent Metric class.

    Examples:
        >>> overkill_escape = OverkillEscape(escape_rate=2.0)
        >>> overkill_rate = overkill_escape(preds, target)
    """

    full_state_update: bool = False

    def __init__(
        self,
        escape_rate: float = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Fixed escape rate threshold for evaluation
        self.escape_rate = escape_rate
        # Add states for accumulating predictions and targets
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Anomaly scores, shape (N,)
            target: Labels (0=normal, 1=abnormal), shape (N,)
        """
        self.preds.append(preds.flatten())
        self.target.append(target.flatten())

    def compute(self) -> float:
        """Compute overkill rate at specified escape rate threshold.

        Returns:
            float: Overkill rate (proportion of normal samples incorrectly flagged)
        """
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        normal_mask = (target == 0)
        abnormal_mask = (target == 1)

        normal_scores = preds[normal_mask]
        abnormal_scores = preds[abnormal_mask]

        if len(normal_scores) == 0:
            raise ValueError("No normal samples found (label=0)")
        if len(abnormal_scores) == 0:
            raise ValueError("No abnormal samples found (label=1)")

        # Calculate threshold at escape rate percentile
        threshold = torch.quantile(abnormal_scores, self.escape_rate / 100.0)

        # Compute overkill rate (proportion of normal samples incorrectly flagged)
        overkill_rate = torch.mean((normal_scores >= threshold).float())

        return overkill_rate

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> float:
        """Forward pass: update -> compute -> reset.

        Args:
            preds: Anomaly scores
            target: Labels (0=normal, 1=abnormal)

        Returns:
            float: Overkill rate at specified escape rate threshold
        """
        self.update(preds, target)
        results = self.compute()
        self.reset()
        return results

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
