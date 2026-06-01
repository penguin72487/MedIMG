"""Evaluation package public API.

The implementation is being split out of the historical monolithic
``eval.py`` while keeping ``from medsam_modular.eval import ...`` stable.
"""

from medsam_modular.eval.evaluate import (
    OODDetector,
    TTAPredictor,
    compute_bce_batch_tensor,
    compute_metrics_batch_tensor,
    compute_metrics_tensor,
    evaluate_dataset,
    evaluate_dataset_ood_only,
    evaluate_dataset_ood_tta,
)

__all__ = [
    "OODDetector",
    "TTAPredictor",
    "compute_bce_batch_tensor",
    "compute_metrics_batch_tensor",
    "compute_metrics_tensor",
    "evaluate_dataset",
    "evaluate_dataset_ood_only",
    "evaluate_dataset_ood_tta",
]
