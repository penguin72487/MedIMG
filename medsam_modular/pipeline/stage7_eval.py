"""Stage 7: test OOD and TTA evaluation."""

import time
from pathlib import Path
from typing import Any, Dict

from medsam_modular.cache import PredictionCache
from medsam_modular.eval import OODDetector, TTAPredictor, evaluate_dataset_ood_tta


def evaluate_test_ood_tta(
    *,
    model: Any,
    processor: Any,
    device: str,
    test_sets: Dict[str, Any],
    ood_detector: OODDetector,
    tta_predictor: TTAPredictor,
    pred_cache: PredictionCache,
    profiler: Any,
    output_dir: Path,
    baseline_all_stats: Dict[str, Dict[str, Any]],
    file_tag: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    from medsam_modular.runner import _fmt_metric, _save_json, _timed_log

    all_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset_name, dataset in test_sets.items():
        if len(dataset) == 0:
            print(f"\n  ⚠️ skip empty dataset: {dataset_name}")
            continue

        print(f"\n=== Evaluating {dataset_name} ({len(dataset)} samples) [{file_tag}] ===")
        t_ds = time.time()
        with _timed_log(f"Stage 7: evaluate {dataset_name} [{file_tag}]"):
            with profiler.section_and_flush(f"eval.{dataset_name}.{file_tag}.ood_tta.total"):
                ood_results, ood_stats, tta_results, tta_stats = evaluate_dataset_ood_tta(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    model=model,
                    processor=processor,
                    device=device,
                    ood_detector=ood_detector,
                    tta_predictor=tta_predictor,
                    pred_cache=pred_cache,
                    profiler=profiler,
                    profile_prefix=f"eval.{dataset_name}.{file_tag}",
                )

        ood_stats["ood_threshold"] = float(getattr(ood_detector, "threshold", 0.5))
        ood_stats["ood_method"] = str(getattr(ood_detector, "method", "entropy"))
        tta_stats["ood_threshold"] = float(getattr(ood_detector, "threshold", 0.5))
        tta_stats["ood_method"] = str(getattr(ood_detector, "method", "entropy"))

        baseline_stats = baseline_all_stats.get(dataset_name, {})
        all_stats[dataset_name] = {
            "baseline": baseline_stats,
            "ood": ood_stats,
            "tta": tta_stats,
        }
        baseline_dice = baseline_stats.get("mean_dice", baseline_stats.get("dice_mean"))
        tta_dice = tta_stats.get("mean_dice", tta_stats.get("dice_mean"))
        print(
            f"  [{dataset_name}] 完成  ({time.time()-t_ds:.1f}s)  "
            f"baseline_dice={_fmt_metric(baseline_dice)}  "
            f"tta_dice={_fmt_metric(tta_dice)}"
        )

        with _timed_log(f"Stage 7: save {dataset_name} [{file_tag}] outputs"):
            _save_json(output_dir / f"{dataset_name.lower()}_{file_tag}_ood_results.json", ood_results)
            _save_json(output_dir / f"{dataset_name.lower()}_{file_tag}_ood_stats.json", ood_stats)
            _save_json(output_dir / f"{dataset_name.lower()}_{file_tag}_tta_results.json", tta_results)
            _save_json(output_dir / f"{dataset_name.lower()}_{file_tag}_tta_stats.json", tta_stats)
    return all_stats


__all__ = ["evaluate_test_ood_tta"]
