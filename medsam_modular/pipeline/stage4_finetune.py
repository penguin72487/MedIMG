"""Stage 4/5 fine-tuning entry point."""

from typing import Any

from medsam_modular.train import maybe_finetune


def run_finetune(**kwargs: Any) -> Any:
    return maybe_finetune(**kwargs)


__all__ = ["run_finetune"]

