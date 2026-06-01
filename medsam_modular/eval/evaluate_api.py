"""Dataset evaluation entry points."""

from medsam_modular.eval.evaluate import (
    evaluate_dataset,
    evaluate_dataset_ood_only,
    evaluate_dataset_ood_tta,
)

__all__ = [
    "evaluate_dataset",
    "evaluate_dataset_ood_only",
    "evaluate_dataset_ood_tta",
]
