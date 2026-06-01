"""Autobatch tuning entry points.

This module exposes the current private tuner for internal use while the
evaluation package is being separated.
"""

from medsam_modular.eval.evaluate import _auto_tune_eval_batch_size

__all__ = ["_auto_tune_eval_batch_size"]
