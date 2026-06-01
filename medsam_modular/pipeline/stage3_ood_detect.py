"""Stage 3: train-split OOD detection."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from medsam_modular.cache import PredictionCache
from medsam_modular.data import prepare_datasets_by_split
from medsam_modular.eval import OODDetector, evaluate_dataset_ood_only


def detect_ood_train_subset(
    *,
    model: Any,
    processor: Any,
    data_paths: Dict[str, str],
    split_root: Path,
    image_size: int,
    device: str,
    ood_detector: OODDetector,
    pred_cache: Optional[PredictionCache],
    profiler: Any,
    output_dir: Path,
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    from medsam_modular.runner import _save_json, _timed_log

    with _timed_log("Stage 3: prepare train datasets for OOD detection"):
        train_sets = prepare_datasets_by_split(
            data_paths=data_paths,
            split_root=split_root,
            split_name="train",
            image_size=image_size,
        )

    subset_by_name: Dict[str, Set[str]] = {}
    summary: Dict[str, Any] = {}

    for dataset_name, dataset in train_sets.items():
        if len(dataset) == 0:
            summary[dataset_name] = {"num_samples": 0, "num_ood": 0, "ood_ratio": 0.0}
            subset_by_name[dataset_name] = set()
            continue

        print(f"\n=== Baseline OOD detect on train: {dataset_name} ({len(dataset)} samples) ===")
        with _timed_log(f"Stage 3: OOD detect train dataset {dataset_name}"):
            results, stats = evaluate_dataset_ood_only(
                dataset=dataset,
                dataset_name=dataset_name,
                model=model,
                processor=processor,
                device=device,
                ood_detector=ood_detector,
                pred_cache=pred_cache,
                profiler=profiler,
                profile_prefix=f"train_ood_detect.{dataset_name}",
            )

        ood_names = {str(r.get("name", "")) for r in results if bool(r.get("is_ood", False))}
        ood_names.discard("")
        subset_by_name[dataset_name] = ood_names

        num_samples = int(len(results))
        num_ood = int(len(ood_names))
        ratio = float(num_ood / max(1, num_samples))
        summary[dataset_name] = {
            "num_samples": num_samples,
            "num_ood": num_ood,
            "ood_ratio": ratio,
            "ood_threshold": float(getattr(ood_detector, "threshold", 0.5)),
            "ood_method": str(getattr(ood_detector, "method", "entropy")),
            "eval_stats": stats,
        }

        with _timed_log(f"Stage 3: save train OOD outputs for {dataset_name}"):
            _save_json(output_dir / f"{dataset_name.lower()}_train_ood_detect_results.json", results)
            _save_json(output_dir / f"{dataset_name.lower()}_train_ood_detect_stats.json", summary[dataset_name])
        print(f"  [{dataset_name}] train OOD: {num_ood}/{num_samples} ({ratio:.2%})")

    with _timed_log("Stage 3: save train OOD subset summary"):
        _save_json(output_dir / "train_ood_subset_summary.json", summary)
    return subset_by_name, summary


def load_cached_ood_train_subset(
    *,
    output_dir: Path,
    dataset_names: List[str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    subset_by_name: Dict[str, Set[str]] = {name: set() for name in dataset_names}
    summary: Dict[str, Any] = {}

    summary_path = output_dir / "train_ood_subset_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                summary = payload
        except Exception:
            summary = {}

    for dataset_name in dataset_names:
        res_path = output_dir / f"{dataset_name.lower()}_train_ood_detect_results.json"
        if not res_path.exists():
            continue
        try:
            results = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(results, list):
            continue

        names = {
            str(r.get("name", ""))
            for r in results
            if bool(r.get("is_ood", False)) and str(r.get("name", ""))
        }
        subset_by_name[dataset_name] = names

        row = summary.get(dataset_name, {}) if isinstance(summary, dict) else {}
        if not isinstance(row, dict):
            row = {}
        if "num_samples" not in row:
            row["num_samples"] = int(len(results))
        if "num_ood" not in row:
            row["num_ood"] = int(len(names))
        row["ood_ratio"] = float(int(row.get("num_ood", 0)) / max(1, int(row.get("num_samples", 0))))
        summary[dataset_name] = row

    for dataset_name in dataset_names:
        if dataset_name not in summary:
            summary[dataset_name] = {
                "num_samples": 0,
                "num_ood": int(len(subset_by_name.get(dataset_name, set()))),
                "ood_ratio": 0.0,
            }

    return subset_by_name, summary


__all__ = ["detect_ood_train_subset", "load_cached_ood_train_subset"]
