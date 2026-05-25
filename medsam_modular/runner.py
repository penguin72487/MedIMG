import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from medsam_modular.cache import PredictionCache
from medsam_modular.config import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_ID, DEFAULT_OUTPUT_DIR_REL, ENV_DEFAULTS
from medsam_modular.data import prepare_datasets_by_split
from medsam_modular.eval import OODDetector, TTAPredictor, evaluate_dataset, evaluate_dataset_ood_only, evaluate_dataset_ood_tta
from medsam_modular.io_async import get_global_async_writer, shutdown_global_async_writer
from medsam_modular.model import load_medsam, load_state_dict_compat
from medsam_modular.train import maybe_finetune
from medsam_modular.visualize import build_comparison_table, save_comparison_chart


_TRUE_SET = {"1", "true", "yes", "y", "on"}


class _NullProfiler:
    enabled = False

    @contextmanager
    def section_and_flush(self, _section: str):
        yield

    def flush(self) -> Dict[str, Any]:
        return {}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, ENV_DEFAULTS.get(name, "1" if default else "0")).strip().lower()
    return raw in _TRUE_SET


def _env(name: str) -> str:
    return os.getenv(name, ENV_DEFAULTS.get(name, ""))


def _resolve_split_root(project_root: Path) -> Path:
    split_root_raw = _env("MEDSAM_SPLIT_ROOT").strip()
    if split_root_raw:
        return Path(split_root_raw)
    return project_root / "splits"


def _cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _auto_cpu_threads(device: str) -> int:
    _ = device
    return _cpu_count()


def _setup_cuda_accel() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "enabled": False,
        "flash_sdp": None,
        "mem_efficient_sdp": None,
        "math_sdp": None,
        "matmul_precision": None,
    }
    if not torch.cuda.is_available():
        return status

    status["enabled"] = True
    try:
        torch.set_float32_matmul_precision("high")
        status["matmul_precision"] = "high"
    except Exception:
        status["matmul_precision"] = "<unsupported>"

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    try:
        status["flash_sdp"] = bool(torch.backends.cuda.flash_sdp_enabled())
        status["mem_efficient_sdp"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        status["math_sdp"] = bool(torch.backends.cuda.math_sdp_enabled())
    except Exception:
        pass

    return status


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_path_is_valid(dataset_name: str, candidate: Path) -> bool:
    if not candidate.exists():
        return False

    if dataset_name in {"TN3K", "TG3K"}:
        return (
            (candidate / "test-image").exists()
            or (candidate / "test" / "images").exists()
            or (candidate / "trainval-image").exists()
        )

    if dataset_name == "DDTI":
        return (
            (candidate / "test" / "annotations").exists()
            or any(candidate.glob("*.xml"))
        )

    if dataset_name == "TN5000":
        return (
            (candidate / "test" / "annotations").exists()
            or (candidate / "Annotations").exists()
        )

    return candidate.exists()


def _resolve_data_paths(project_root: Path) -> Dict[str, str]:
    defaults = {
        "TN3K": str(project_root / "TN3K"),
        "TG3K": str(project_root / "TG3K"),
        "DDTI": str(project_root / "DDTI"),
        "TN5000": str(project_root / "TN5000"),
    }
    resolved = dict(defaults)
    data_root = _env("MEDSAM_DATA_ROOT").strip()

    for name, default_path in defaults.items():
        specific = _env(f"MEDSAM_{name}_PATH").strip()
        if specific and _dataset_path_is_valid(name, Path(specific)):
            resolved[name] = specific
            continue

        if data_root:
            base = Path(data_root)
            candidates = [
                base / name,
                base / name / f"{name}_forReview",
            ]
            picked = next((p for p in candidates if _dataset_path_is_valid(name, p)), None)
            if picked is not None:
                resolved[name] = str(picked)
                continue

        local = Path(default_path)
        if _dataset_path_is_valid(name, local):
            resolved[name] = str(local)

    return resolved


def _resolve_baseline_weight_path(project_root: Path) -> str:
    candidates = [
        project_root / "results" / "medsam_vit_b.pth",
        project_root / "results" / "medsam_finetuned_best.pth",
        project_root / "results" / "medsam_finetuned.pth",
    ]
    picked = next((p for p in candidates if p.exists()), None)
    return str(picked) if picked is not None else ""


def _resolve_resume_weight_path(project_root: Path, output_dir: Path) -> str:
    candidates = [
        output_dir / "medsam_OOD_finetuned_best.pth",
        output_dir / "medsam_OOD_finetuned_last.pth",
        output_dir / "medsam_finetuned_best.pth",
        output_dir / "medsam_finetuned_last.pth",
        output_dir / "medsam_finetuned.pth",
        project_root / "results" / "medsam_OOD_finetuned_best.pth",
        project_root / "results" / "medsam_OOD_finetuned_last.pth",
        project_root / "results" / "medsam_finetuned_best.pth",
        project_root / "results" / "medsam_finetuned_last.pth",
        project_root / "results" / "medsam_finetuned.pth",
        project_root / "results" / "medsam_vit_b.pth",
    ]
    picked = next((p for p in candidates if p.exists()), None)
    return str(picked) if picked is not None else ""


def _save_json(path: Path, payload: Any) -> None:
    writer = get_global_async_writer()
    writer.submit_text(path, json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_existing_baseline_stats(test_sets: Dict[str, Any], output_dir: Path) -> Dict[str, Dict[str, Any]]:
    baseline_all_stats: Dict[str, Dict[str, Any]] = {}
    for dataset_name in test_sets:
        stats_path = output_dir / f"{dataset_name.lower()}_baseline_stats.json"
        if not stats_path.exists():
            continue
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                baseline_all_stats[dataset_name] = payload
        except Exception:
            continue
    return baseline_all_stats


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _build_train_config(
    project_root: Path,
    data_paths: Dict[str, str],
    image_size: int,
    device: str,
    output_dir: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    split_root = _resolve_split_root(project_root)
    resume_weight_path = _resolve_resume_weight_path(project_root=project_root, output_dir=output_dir)
    cfg: Dict[str, Any] = {
        "split_root": split_root,
        "image_size": image_size,
        "data_paths": data_paths,
        "device": device,
        "output_dir": output_dir,
        "resume_weight_path": resume_weight_path,
        "skip_finetune": _env("MEDSAM_SKIP_FINETUNE"),
        "finetune_train_backbone": _env("MEDSAM_FINETUNE_TRAIN_BACKBONE"),
        "finetune_epochs": _env("MEDSAM_FINETUNE_EPOCHS"),
        "finetune_batch": _env("MEDSAM_FINETUNE_BATCH"),
        "finetune_lr": _env("MEDSAM_FINETUNE_LR"),
        "finetune_weight_decay": _env("MEDSAM_FINETUNE_WEIGHT_DECAY"),
        "finetune_adamw_beta1": _env("MEDSAM_FINETUNE_ADAMW_BETA1"),
        "finetune_adamw_beta2": _env("MEDSAM_FINETUNE_ADAMW_BETA2"),
        "finetune_adamw_eps": _env("MEDSAM_FINETUNE_ADAMW_EPS"),
        "finetune_val_ratio": _env("MEDSAM_FINETUNE_VAL_RATIO"),
        "finetune_patience": _env("MEDSAM_FINETUNE_PATIENCE"),
        "finetune_min_epochs": _env("MEDSAM_FINETUNE_MIN_EPOCHS"),
        "finetune_min_delta": _env("MEDSAM_FINETUNE_MIN_DELTA"),
        "finetune_use_plateau_scheduler": _env("MEDSAM_FINETUNE_USE_PLATEAU_SCHEDULER"),
        "finetune_plateau_factor": _env("MEDSAM_FINETUNE_PLATEAU_FACTOR"),
        "finetune_plateau_patience": _env("MEDSAM_FINETUNE_PLATEAU_PATIENCE"),
        "finetune_plateau_cooldown": _env("MEDSAM_FINETUNE_PLATEAU_COOLDOWN"),
        "finetune_plateau_min_lr": _env("MEDSAM_FINETUNE_PLATEAU_MIN_LR"),
        "finetune_early_stop_require_min_lr": _env("MEDSAM_FINETUNE_EARLY_STOP_REQUIRE_MIN_LR"),
        "finetune_grad_accum": _env("MEDSAM_FINETUNE_GRAD_ACCUM"),
        "finetune_grad_clip": _env("MEDSAM_FINETUNE_GRAD_CLIP"),
        "finetune_workers": _env("MEDSAM_FINETUNE_WORKERS"),
        "finetune_max_samples": _env("MEDSAM_FINETUNE_MAX_SAMPLES"),
        "finetune_use_fused_adamw": _env("MEDSAM_FINETUNE_USE_FUSED_ADAMW"),
    }
    if extra:
        cfg.update(extra)
    return cfg


def _detect_ood_train_subset(
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
            summary[dataset_name] = {
                "num_samples": 0,
                "num_ood": 0,
                "ood_ratio": 0.0,
            }
            subset_by_name[dataset_name] = set()
            continue

        print(f"\n=== Baseline OOD detect on train: {dataset_name} ({len(dataset)} samples) ===")
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
            "min_lesion_ratio": float(getattr(ood_detector, "min_lesion_ratio", 0.0003)),
            "eval_stats": stats,
        }

        _save_json(output_dir / f"{dataset_name.lower()}_train_ood_detect_results.json", results)
        _save_json(output_dir / f"{dataset_name.lower()}_train_ood_detect_stats.json", summary[dataset_name])
        print(f"  [{dataset_name}] train OOD: {num_ood}/{num_samples} ({ratio:.2%})")

    _save_json(output_dir / "train_ood_subset_summary.json", summary)
    return subset_by_name, summary


def _evaluate_test_ood_tta(
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
    all_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset_name, dataset in test_sets.items():
        if len(dataset) == 0:
            print(f"\n  ⚠️ skip empty dataset: {dataset_name}")
            continue

        print(f"\n=== Evaluating {dataset_name} ({len(dataset)} samples) [{file_tag}] ===")
        t_ds = time.time()
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
        ood_stats["min_lesion_ratio"] = float(getattr(ood_detector, "min_lesion_ratio", 0.0003))
        tta_stats["ood_threshold"] = float(getattr(ood_detector, "threshold", 0.5))
        tta_stats["ood_method"] = str(getattr(ood_detector, "method", "entropy"))
        tta_stats["min_lesion_ratio"] = float(getattr(ood_detector, "min_lesion_ratio", 0.0003))

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

        _save_json(output_dir / f"{dataset_name.lower()}_{file_tag}_ood_results.json", ood_results)
        _save_json(output_dir / f"{dataset_name.lower()}_{file_tag}_ood_stats.json", ood_stats)
        _save_json(output_dir / f"{dataset_name.lower()}_{file_tag}_tta_results.json", tta_results)
        _save_json(output_dir / f"{dataset_name.lower()}_{file_tag}_tta_stats.json", tta_stats)
    return all_stats


def main() -> None:
    project_root = _project_root()
    output_dir_raw = _env("MEDSAM_OUTPUT_DIR").strip()
    output_dir = Path(output_dir_raw) if output_dir_raw else (project_root / DEFAULT_OUTPUT_DIR_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_accel = _setup_cuda_accel()
    raw_cpu_threads = int(_env("MEDSAM_CPU_THREADS"))
    cpu_threads = _auto_cpu_threads(device) if raw_cpu_threads <= 0 else max(1, raw_cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(8, cpu_threads // 2)))
    except RuntimeError:
        # set_num_interop_threads may be called only once in some runtimes.
        pass

    image_size = int(_env("MEDSAM_IMAGE_SIZE") or str(DEFAULT_IMAGE_SIZE))
    model_id = _env("MEDSAM_MODEL_ID") or DEFAULT_MODEL_ID
    data_paths = _resolve_data_paths(project_root)
    baseline_weight_path = _resolve_baseline_weight_path(project_root)
    resume_weight_path = _resolve_resume_weight_path(project_root, output_dir)
    profiler = _NullProfiler()

    print("=" * 80)
    print("MedSAM Modular Runner")
    print("=" * 80)
    print(f"device       : {device}")
    if cuda_accel.get("enabled", False):
        print(
            "cuda accel   : "
            f"matmul={cuda_accel.get('matmul_precision')} "
            f"flash_sdp={cuda_accel.get('flash_sdp')} "
            f"mem_eff_sdp={cuda_accel.get('mem_efficient_sdp')} "
            f"math_sdp={cuda_accel.get('math_sdp')}"
        )
    print(f"cpu threads  : {cpu_threads}")
    print(f"model_id     : {model_id}")
    print(f"image_size   : {image_size}")
    print(f"baseline wt  : {baseline_weight_path or '<missing vit_b checkpoint>'}")
    print(f"resume wt    : {resume_weight_path or '<none>'}")
    for k, v in data_paths.items():
        print(f"  {k:8s}: {v}")

    print("\n[Stage 1/7] 載入模型 ...")
    t1 = time.time()
    with profiler.section_and_flush("stage.load_model"):
        model, processor, compile_report = load_medsam(
            model_id=model_id,
            device=device,
            image_size=image_size,
            local_weight_path=baseline_weight_path,
        )
    compile_backend = compile_report.get("backend", "<none>")
    compile_mode = compile_report.get("compile_mode", "<none>")
    compile_dynamic = compile_report.get("compile_dynamic", "<unknown>")
    warmup_batches = compile_report.get("warmup_batches", "<unknown>")
    print(f"  compile    : {compile_report.get('compiled', False)}  ({time.time()-t1:.1f}s)")
    print(f"  compile cfg: backend={compile_backend}, mode={compile_mode}, dynamic={compile_dynamic}, warmup_batches={warmup_batches}")
    if not compile_report.get('compiled', False):
        err = compile_report.get('error', '')
        if err:
            print(f"  compile err: {err[:120].strip()} ...")

    require_compile = _env_bool("MEDSAM_REQUIRE_COMPILE", False)
    if require_compile and not bool(compile_report.get("compiled", False)):
        raise RuntimeError(f"torch.compile(inductor) required but unavailable: {compile_report}")

    print("\n[Stage 2/7] 準備測試資料 ...")
    t3 = time.time()
    split_root = _resolve_split_root(project_root)
    with profiler.section_and_flush("stage.prepare_test_data"):
        test_sets = prepare_datasets_by_split(
            data_paths=data_paths,
            split_root=split_root,
            split_name="test",
            image_size=image_size,
        )
    total_test = sum(len(ds) for ds in test_sets.values())
    print(f"  資料準備耗時: {time.time()-t3:.1f}s")
    for name, ds in test_sets.items():
        print(f"  {name:8s}: {len(ds)} samples")
    print(f"  共計    : {total_test} samples")

    ood_detector = OODDetector(
        threshold=float(_env("MEDSAM_OOD_THRESHOLD")),
        method=_env("MEDSAM_OOD_METHOD"),
        min_lesion_ratio=float(_env("MEDSAM_OOD_MIN_LESION_RATIO") or 0.0003),
    )

    tta_fusion_mode = _env("MEDSAM_TTA_FUSION")
    tta_augmentations = None
    tta_augs_str = _env("MEDSAM_TTA_AUGMENTATIONS")
    if tta_augs_str:
        tta_augmentations = [aug.strip() for aug in tta_augs_str.split(",")]
    tta_predictor = TTAPredictor(
        augmentations=tta_augmentations,
        fusion_mode=tta_fusion_mode,
    )

    baseline_pred_cache = PredictionCache(output_dir / "pred_cache_baseline")
    train_ood_detect_cache = PredictionCache(output_dir / "pred_cache_train_ood_detect")
    ood_finetuned_pred_cache = PredictionCache(output_dir / "pred_cache_ood_finetuned")
    finetuned_pred_cache = PredictionCache(output_dir / "pred_cache_finetuned")

    print(f"\n=== OOD Configuration ===")
    print(f"  Threshold: {float(getattr(ood_detector, 'threshold', 0.5)):.4f}")
    print(f"  Method: {str(getattr(ood_detector, 'method', 'entropy'))}")
    print(f"\n=== TTA Configuration ===")
    print(f"  Fusion mode: {tta_fusion_mode}")
    print(f"  Augmentations: {tta_predictor.augmentations}")
    print(f"  Number of augmentations: {len(tta_predictor.augmentations)}")

    finetune_only = _env_bool("MEDSAM_FINETUNE_ONLY", False)
    skip_finetune = _env_bool("MEDSAM_SKIP_FINETUNE", True)
    run_only_stage7 = _env_bool("MEDSAM_RUN_ONLY_STAGE7", False)
    if run_only_stage7:
        skip_finetune = True
        finetune_only = False
        print("\n[Mode] stage7-only enabled: skip Stage 3~6 and run Stage 7 only.")

    split_root = _resolve_split_root(project_root)

    ood_subset_by_name: Dict[str, Set[str]] = {}
    ood_subset_summary: Dict[str, Any] = {}
    ood_finetuned_best_path = output_dir / "medsam_OOD_finetuned_best.pth"
    full_finetuned_best_path = output_dir / "medsam_finetuned_best.pth"

    if not skip_finetune:
        print("\n[Stage 3/7] baseline 偵測 train split OOD ...")
        ood_subset_by_name, ood_subset_summary = _detect_ood_train_subset(
            model=model,
            processor=processor,
            data_paths=data_paths,
            split_root=split_root,
            image_size=image_size,
            device=device,
            ood_detector=ood_detector,
            pred_cache=train_ood_detect_cache,
            profiler=profiler,
            output_dir=output_dir,
        )

        total_ood = int(sum(len(v) for v in ood_subset_by_name.values()))
        total_all = int(sum(int(v.get("num_samples", 0)) for v in ood_subset_summary.values()))
        print(f"  OOD train subset: {total_ood}/{total_all} samples")

        if total_ood > 0:
            print("\n[Stage 4/7] OOD 子集微調（使用 TTA 增強資料）...")
            if baseline_weight_path and Path(baseline_weight_path).exists():
                load_state_dict_compat(model, Path(baseline_weight_path), map_location=device)

            t2 = time.time()
            with profiler.section_and_flush("stage.ood_finetune"):
                model = maybe_finetune(
                    model=model,
                    processor=processor,
                    config=_build_train_config(
                        project_root=project_root,
                        data_paths=data_paths,
                        image_size=image_size,
                        device=device,
                        output_dir=output_dir,
                        extra={
                            "skip_finetune": "0",
                            "resume_weight_path": "",
                            "finetune_subset_by_name": {k: sorted(v) for k, v in ood_subset_by_name.items()},
                            "finetune_use_tta_augment": True,
                            "finetune_tta_augmentations": list(tta_predictor.augmentations),
                            "finetune_weight_prefix": "medsam_OOD_finetuned",
                            "finetune_stats_prefix": "ood_finetune",
                        },
                    ),
                    profiler=profiler,
                )
            print(f"  OOD 微調耗時: {time.time()-t2:.1f}s")
        else:
            print("\n[Stage 4/7] OOD 子集為空，略過 OOD 微調。")

        print("\n[Stage 5/7] 全資料微調（輸出 medsam_finetuned_best.pth）...")
        if ood_finetuned_best_path.exists():
            load_state_dict_compat(model, ood_finetuned_best_path, map_location=device)
            print(f"  📌 全資料微調起始權重: {ood_finetuned_best_path}")
        elif baseline_weight_path and Path(baseline_weight_path).exists():
            load_state_dict_compat(model, Path(baseline_weight_path), map_location=device)
            print(f"  📌 全資料微調起始權重: {baseline_weight_path}")

        t2 = time.time()
        with profiler.section_and_flush("stage.full_finetune"):
            model = maybe_finetune(
                model=model,
                processor=processor,
                config=_build_train_config(
                    project_root=project_root,
                    data_paths=data_paths,
                    image_size=image_size,
                    device=device,
                    output_dir=output_dir,
                    extra={
                        "skip_finetune": "0",
                        "resume_weight_path": "",
                        "finetune_subset_by_name": {},
                        "finetune_use_tta_augment": False,
                        "finetune_weight_prefix": "medsam_finetuned",
                        "finetune_stats_prefix": "finetune",
                    },
                ),
                profiler=profiler,
            )
        print(f"  全資料微調耗時: {time.time()-t2:.1f}s")
    else:
        print("\n[Stage 3/7] 已設定 skip_finetune，略過 OOD 與全資料微調。")

    if finetune_only:
        print("\n[Stage 6/7] 已啟用 finetune-only，略過後續 baseline / OOD / TTA 評估。")
        shutdown_global_async_writer()
        return

    baseline_all_results: Dict[str, Any] = {}
    baseline_all_stats: Dict[str, Dict[str, Any]] = {}
    if run_only_stage7:
        print("\n[Stage 6/7] stage7-only：略過 baseline 評估，嘗試載入既有 baseline stats ...")
        baseline_all_stats = _load_existing_baseline_stats(test_sets=test_sets, output_dir=output_dir)
        if baseline_all_stats:
            print(f"  已載入 baseline stats: {', '.join(sorted(baseline_all_stats.keys()))}")
        else:
            print("  ⚠️ 找不到 baseline stats，Stage 7 仍會執行，但 comparison 可能不完整。")
    else:
        print("\n[Stage 6/7] 基線評估 (vit_b) ...")
        t_eval_start = time.time()
        baseline_weight = Path(baseline_weight_path) if baseline_weight_path else None
        if baseline_weight is not None and baseline_weight.exists():
            load_state_dict_compat(model, baseline_weight, map_location=device)
            print(f"  📌 baseline 使用權重: {baseline_weight}")
        else:
            print("  ⚠️ baseline 權重不存在，將使用目前模型權重進行 baseline 評估。")

        for dataset_name, dataset in test_sets.items():
            if len(dataset) == 0:
                print(f"\n  ⚠️ skip empty dataset: {dataset_name}")
                continue

            print(f"\n=== Baseline {dataset_name} ({len(dataset)} samples) ===")
            t_ds = time.time()
            with profiler.section_and_flush(f"eval.{dataset_name}.baseline.total"):
                baseline_results, baseline_stats = evaluate_dataset(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    model=model,
                    processor=processor,
                    device=device,
                    use_ood=False,
                    use_tta=False,
                    ood_detector=None,
                    tta_predictor=None,
                    pred_cache=baseline_pred_cache,
                    profiler=profiler,
                    profile_prefix=f"eval.{dataset_name}.baseline",
                )
            baseline_all_results[dataset_name] = baseline_results
            baseline_all_stats[dataset_name] = baseline_stats
            baseline_dice = baseline_stats.get("mean_dice", baseline_stats.get("dice_mean"))
            print(f"  [{dataset_name}] 完成  ({time.time()-t_ds:.1f}s)  baseline_dice={_fmt_metric(baseline_dice)}")
            _save_json(output_dir / f"{dataset_name.lower()}_baseline_results.json", baseline_results)
            _save_json(output_dir / f"{dataset_name.lower()}_baseline_stats.json", baseline_stats)

    all_stats_ood_finetuned: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if ood_finetuned_best_path.exists():
        print("\n[Stage 7/7] 測試 OOD finetuned 模型：先 OOD 判斷，再 TTA inference ...")
        load_state_dict_compat(model, ood_finetuned_best_path, map_location=device)
        print(f"  📌 OOD finetuned 評估權重: {ood_finetuned_best_path}")
        all_stats_ood_finetuned = _evaluate_test_ood_tta(
            model=model,
            processor=processor,
            device=device,
            test_sets=test_sets,
            ood_detector=ood_detector,
            tta_predictor=tta_predictor,
            pred_cache=ood_finetuned_pred_cache,
            profiler=profiler,
            output_dir=output_dir,
            baseline_all_stats=baseline_all_stats,
            file_tag="ood_finetuned",
        )
        _save_json(output_dir / "summary_ood_finetuned.json", all_stats_ood_finetuned)
    else:
        print("\n[Stage 7/7] 找不到 medsam_OOD_finetuned_best.pth，略過 OOD finetuned 模型測試。")

    if skip_finetune and resume_weight_path and Path(resume_weight_path).exists():
        load_state_dict_compat(model, Path(resume_weight_path), map_location=device)
        print(f"  📌 評估使用權重: {resume_weight_path}")
    elif full_finetuned_best_path.exists():
        load_state_dict_compat(model, full_finetuned_best_path, map_location=device)
        print(f"  📌 評估使用權重: {full_finetuned_best_path}")
    else:
        print("  📌 評估使用權重: <finetuned model in-memory>")

    print("\n[Stage 7/7] 測試全資料 finetuned 模型：OOD 判斷 + TTA inference ...")
    all_stats = _evaluate_test_ood_tta(
        model=model,
        processor=processor,
        device=device,
        test_sets=test_sets,
        ood_detector=ood_detector,
        tta_predictor=tta_predictor,
        pred_cache=finetuned_pred_cache,
        profiler=profiler,
        output_dir=output_dir,
        baseline_all_stats=baseline_all_stats,
        file_tag="full_finetuned",
    )

    # 維持相容輸出檔名（預設指向 full_finetuned 評估結果）
    for dataset_name in all_stats:
        src_ood = output_dir / f"{dataset_name.lower()}_full_finetuned_ood_results.json"
        src_ood_stats = output_dir / f"{dataset_name.lower()}_full_finetuned_ood_stats.json"
        src_tta = output_dir / f"{dataset_name.lower()}_full_finetuned_tta_results.json"
        src_tta_stats = output_dir / f"{dataset_name.lower()}_full_finetuned_tta_stats.json"
        if src_ood.exists():
            shutil.copy2(src_ood, output_dir / f"{dataset_name.lower()}_ood_results.json")
        if src_ood_stats.exists():
            shutil.copy2(src_ood_stats, output_dir / f"{dataset_name.lower()}_ood_stats.json")
        if src_tta.exists():
            shutil.copy2(src_tta, output_dir / f"{dataset_name.lower()}_tta_results.json")
        if src_tta_stats.exists():
            shutil.copy2(src_tta_stats, output_dir / f"{dataset_name.lower()}_tta_stats.json")

    if not all_stats:
        raise RuntimeError("No test datasets were loaded. Check dataset paths and split files.")

    comparison_path = output_dir / "comparison_table.csv"
    chart_path = output_dir / "performance_comparison.png"
    if run_only_stage7 and not baseline_all_stats:
        print("\n[Stage 7/7] baseline stats 缺失，略過 comparison table/chart 產生。")
    else:
        with profiler.section_and_flush("stage.build_comparison"):
            comparison_table = build_comparison_table(all_stats)
        with profiler.section_and_flush("stage.save_comparison_csv"):
            comparison_table.to_csv(comparison_path, index=False)
        with profiler.section_and_flush("stage.save_comparison_chart"):
            chart_path = save_comparison_chart(all_stats, output_dir)
    top_results_dir = project_root / "results"
    top_results_dir.mkdir(parents=True, exist_ok=True)
    top_chart_path = top_results_dir / chart_path.name
    with profiler.section_and_flush("stage.copy_chart"):
        if chart_path.resolve() != top_chart_path.resolve():
            shutil.copy2(chart_path, top_chart_path)
    _save_json(output_dir / "summary.json", all_stats)

    print("\nOutputs:")
    if comparison_path.exists():
        print(f"- comparison_table: {comparison_path}")
    if chart_path.exists():
        print(f"- comparison_chart: {chart_path}")
        print(f"- comparison_chart_top: {top_chart_path}")
    print(f"- summary: {output_dir / 'summary.json'}")
    if all_stats_ood_finetuned:
        print(f"- summary_ood_finetuned: {output_dir / 'summary_ood_finetuned.json'}")
    shutdown_global_async_writer()


if __name__ == "__main__":
    main()
