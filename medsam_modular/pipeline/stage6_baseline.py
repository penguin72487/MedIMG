"""Stage 6 baseline evaluation entry points."""

from typing import Any, Dict, Tuple

from medsam_modular.eval import evaluate_dataset


def evaluate_baseline_dataset(**kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return evaluate_dataset(**kwargs)


__all__ = ["evaluate_baseline_dataset"]

