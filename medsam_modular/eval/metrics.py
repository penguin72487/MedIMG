"""Segmentation and detection metric entry points."""

from medsam_modular.eval.evaluate import (
    compute_bce_batch_tensor,
    compute_metrics_batch_tensor,
    compute_metrics_tensor,
)

__all__ = [
    "compute_bce_batch_tensor",
    "compute_metrics_batch_tensor",
    "compute_metrics_tensor",
]
