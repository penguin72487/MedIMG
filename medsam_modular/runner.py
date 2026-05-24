import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from medsam_modular.cache import PredictionCache
from medsam_modular.data import prepare_datasets_by_split
from medsam_modular.eval import OODDetector, TTAPredictor, evaluate_dataset, evaluate_dataset_ood_tta
from medsam_modular.io_async import get_global_async_writer, shutdown_global_async_writer
from medsam_modular.model import load_medsam, load_state_dict_compat
from medsam_modular.profiler import PerformanceProfiler, set_active_profiler
from medsam_modular.train import maybe_finetune
from medsam_modular.visualize import build_comparison_table, save_comparison_chart


_TRUE_SET = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in _TRUE_SET


def _cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _auto_cpu_threads(device: str) -> int:
    cores = _cpu_count()
    if device == "cuda":
        # Reserve headroom for DataLoader/I/O workers while keeping tensor ops responsive.
        return max(2, min(12, cores // 2))
    return max(1, cores - 1)


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
    data_root = os.getenv("MEDSAM_DATA_ROOT", "").strip()

    for name, default_path in defaults.items():
        specific = os.getenv(f"MEDSAM_{name}_PATH", "").strip()
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
        output_dir / "medsam_finetuned_best.pth",
        output_dir / "medsam_finetuned_last.pth",
        output_dir / "medsam_finetuned.pth",
        project_root / "results" / "medsam_finetuned_best.pth",
        project_root / "results" / "medsam_finetuned_last.pth",
        project_root / "results" / "medsam_finetuned.pth",
        project_root / "results" / "medsam_vit_b.pth",
    ]
    picked = next((p for p in candidates if p.exists()), None)
    return str(picked) if picked is not None else ""


def _save_json(path: Path, payload: Any) -> None:
    profiler = None
    try:
        from medsam_modular.profiler import get_active_profiler

        profiler = get_active_profiler()
    except Exception:
        profiler = None
    t0 = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
    writer = get_global_async_writer()
    writer.submit_text(path, json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if profiler is not None and profiler.enabled:
        profiler.record_duration("io.save_json", time.perf_counter() - t0)


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _build_train_config(project_root: Path, data_paths: Dict[str, str], image_size: int, device: str, output_dir: Path) -> Dict[str, Any]:
    split_root = Path(os.getenv("MEDSAM_SPLIT_ROOT", str(project_root / "splits")))
    resume_weight_path = _resolve_resume_weight_path(project_root=project_root, output_dir=output_dir)
    return {
        "split_root": split_root,
        "image_size": image_size,
        "data_paths": data_paths,
        "device": device,
        "output_dir": output_dir,
        "resume_weight_path": resume_weight_path,
        "skip_finetune": os.getenv("MEDSAM_SKIP_FINETUNE", "1"),
        "finetune_train_backbone": os.getenv("MEDSAM_FINETUNE_TRAIN_BACKBONE", "0"),
        "finetune_epochs": os.getenv("MEDSAM_FINETUNE_EPOCHS", "1000"),
        "finetune_batch": os.getenv("MEDSAM_FINETUNE_BATCH", "8"),
        "finetune_lr": os.getenv("MEDSAM_FINETUNE_LR", "1e-4"),
        "finetune_weight_decay": os.getenv("MEDSAM_FINETUNE_WEIGHT_DECAY", "1e-3"),
        "finetune_adamw_beta1": os.getenv("MEDSAM_FINETUNE_ADAMW_BETA1", "0.9"),
        "finetune_adamw_beta2": os.getenv("MEDSAM_FINETUNE_ADAMW_BETA2", "0.999"),
        "finetune_adamw_eps": os.getenv("MEDSAM_FINETUNE_ADAMW_EPS", "1e-8"),
        "finetune_val_ratio": os.getenv("MEDSAM_FINETUNE_VAL_RATIO", "0.1"),
        "finetune_patience": os.getenv("MEDSAM_FINETUNE_PATIENCE", "20"),
        "finetune_min_delta": os.getenv("MEDSAM_FINETUNE_MIN_DELTA", "1e-4"),
        "finetune_grad_accum": os.getenv("MEDSAM_FINETUNE_GRAD_ACCUM", "2"),
        "finetune_grad_clip": os.getenv("MEDSAM_FINETUNE_GRAD_CLIP", "1.0"),
        "finetune_workers": os.getenv("MEDSAM_FINETUNE_WORKERS", "16"),
        "finetune_max_samples": os.getenv("MEDSAM_FINETUNE_MAX_SAMPLES", "0"),
        "finetune_use_fused_adamw": os.getenv("MEDSAM_FINETUNE_USE_FUSED_ADAMW", "1"),
    }


def main() -> None:
    project_root = _project_root()
    output_dir = Path(os.getenv("MEDSAM_OUTPUT_DIR", str(project_root / "results" / "modular")))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_cpu_threads = int(os.getenv("MEDSAM_CPU_THREADS", "0"))
    cpu_threads = _auto_cpu_threads(device) if raw_cpu_threads <= 0 else max(1, raw_cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(8, cpu_threads // 2)))
    except RuntimeError:
        # set_num_interop_threads may be called only once in some runtimes.
        pass

    image_size = int(os.getenv("MEDSAM_IMAGE_SIZE", "512"))
    model_id = os.getenv("MEDSAM_MODEL_ID", "facebook/sam-vit-base")
    data_paths = _resolve_data_paths(project_root)
    baseline_weight_path = _resolve_baseline_weight_path(project_root)
    resume_weight_path = _resolve_resume_weight_path(project_root, output_dir)
    profile_enabled = _env_bool("MEDSAM_PROFILE", True)
    profile_path_env = os.getenv("MEDSAM_PROFILE_PATH", "").strip()
    profile_path = Path(profile_path_env) if profile_path_env else (output_dir / "bottleneck_profile.json")
    profiler = PerformanceProfiler(enabled=profile_enabled, run_name="medsam_pipeline")
    profiler.configure_output(profile_path)
    set_active_profiler(profiler)
    profiler.set_metadata("device", device)
    profiler.set_metadata("cpu_threads", cpu_threads)
    profiler.set_metadata("model_id", model_id)
    profiler.set_metadata("image_size", image_size)
    profiler.set_metadata("data_paths", data_paths)
    profiler.snapshot_cuda("startup")

    print("=" * 80)
    print("MedSAM Modular Runner")
    print("=" * 80)
    print(f"device       : {device}")
    print(f"cpu threads  : {cpu_threads}")
    print(f"model_id     : {model_id}")
    print(f"image_size   : {image_size}")
    print(f"baseline wt  : {baseline_weight_path or '<missing vit_b checkpoint>'}")
    print(f"resume wt    : {resume_weight_path or '<none>'}")
    for k, v in data_paths.items():
        print(f"  {k:8s}: {v}")

    print("\n[Stage 1/3] 載入模型 ...")
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

    require_compile = _env_bool("MEDSAM_REQUIRE_COMPILE", True)
    if require_compile and not bool(compile_report.get("compiled", False)):
        raise RuntimeError(f"torch.compile(inductor) required but unavailable: {compile_report}")

    print("\n[Stage 2/4] 準備測試資料 / 基線評估 ...")
    t3 = time.time()
    split_root = Path(os.getenv("MEDSAM_SPLIT_ROOT", str(project_root / "splits")))
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
        threshold=float(os.getenv("MEDSAM_OOD_THRESHOLD", "0.5")),
        method=os.getenv("MEDSAM_OOD_METHOD", "entropy"),
    )

    tta_fusion_mode = os.getenv("MEDSAM_TTA_FUSION", "entropy_weighted")
    tta_fast_mode = os.getenv("MEDSAM_TTA_FAST", "0").lower() in ("1", "true", "yes")
    tta_augmentations = None
    tta_augs_str = os.getenv("MEDSAM_TTA_AUGMENTATIONS", "")
    if tta_augs_str:
        tta_augmentations = [aug.strip() for aug in tta_augs_str.split(",")]
    tta_predictor = TTAPredictor(
        augmentations=tta_augmentations,
        fusion_mode=tta_fusion_mode,
        use_fast_mode=tta_fast_mode,
    )

    baseline_pred_cache = PredictionCache(output_dir / "pred_cache_baseline")
    finetuned_pred_cache = PredictionCache(output_dir / "pred_cache_finetuned")

    print(f"\n=== TTA Configuration ===")
    print(f"  Fusion mode: {tta_fusion_mode}")
    print(f"  Fast mode: {tta_fast_mode}")
    print(f"  Augmentations: {tta_predictor.augmentations}")
    print(f"  Number of augmentations: {len(tta_predictor.augmentations)}")

    baseline_all_results: Dict[str, Any] = {}
    baseline_all_stats: Dict[str, Dict[str, Any]] = {}
    print("\n[Stage 2/4] 基線評估 (vit_b) ...")
    t_eval_start = time.time()
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
        latest_profile = profiler.flush() or {}
        limit_info = latest_profile.get("optimization_limit_analysis", {})
        print(
            f"  [{dataset_name}] optimization headroom: {limit_info.get('status', 'unknown')} | "
            f"{limit_info.get('message', '')}"
        )

    finetune_only = _env_bool("MEDSAM_FINETUNE_ONLY", False)
    skip_finetune = _env_bool("MEDSAM_SKIP_FINETUNE", True)

    print("\n[Stage 3/4] 訓練 / 微調 ...")
    t2 = time.time()
    with profiler.section_and_flush("stage.finetune"):
        model = maybe_finetune(
            model=model,
            processor=processor,
            config=_build_train_config(
                project_root=project_root,
                data_paths=data_paths,
                image_size=image_size,
                device=device,
                output_dir=output_dir,
            ),
            profiler=profiler,
        )
    print(f"  微調耗時: {time.time()-t2:.1f}s")

    if finetune_only:
        print("\n[Stage 4/4] 已啟用 finetune-only，略過後續 OOD/TTA 評估。")
        profiler.flush()
        shutdown_global_async_writer()
        return

    if skip_finetune and resume_weight_path and Path(resume_weight_path).exists():
        load_state_dict_compat(model, Path(resume_weight_path), map_location=device)
        print(f"  📌 評估使用權重: {resume_weight_path}")
    else:
        print(f"  📌 評估使用權重: <finetuned model>")

    print("\n[Stage 4/4] OOD / TTA 評估 ...")
    t_eval_start = time.time()
    all_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset_name, dataset in test_sets.items():
        if len(dataset) == 0:
            print(f"\n  ⚠️ skip empty dataset: {dataset_name}")
            continue

        print(f"\n=== Evaluating {dataset_name} ({len(dataset)} samples) ===")
        t_ds = time.time()
        with profiler.section_and_flush(f"eval.{dataset_name}.ood_tta.total"):
            ood_results, ood_stats, tta_results, tta_stats = evaluate_dataset_ood_tta(
                dataset=dataset,
                dataset_name=dataset_name,
                model=model,
                processor=processor,
                device=device,
                ood_detector=ood_detector,
                tta_predictor=tta_predictor,
                pred_cache=finetuned_pred_cache,
                profiler=profiler,
                profile_prefix=f"eval.{dataset_name}",
            )

        baseline_stats = baseline_all_stats.get(dataset_name, {})
        baseline_results = baseline_all_results.get(dataset_name, [])
        all_stats[dataset_name] = {
            "baseline": baseline_stats,
            "ood": ood_stats,
            "tta": tta_stats,
        }
        baseline_dice = baseline_stats.get("mean_dice", baseline_stats.get("dice_mean"))
        tta_dice = tta_stats.get("mean_dice", tta_stats.get("dice_mean"))
        print(f"  [{dataset_name}] 完成  ({time.time()-t_ds:.1f}s)  "
              f"baseline_dice={_fmt_metric(baseline_dice)}  "
              f"tta_dice={_fmt_metric(tta_dice)}")

        _save_json(output_dir / f"{dataset_name.lower()}_ood_results.json", ood_results)
        _save_json(output_dir / f"{dataset_name.lower()}_ood_stats.json", ood_stats)
        _save_json(output_dir / f"{dataset_name.lower()}_tta_results.json", tta_results)
        _save_json(output_dir / f"{dataset_name.lower()}_tta_stats.json", tta_stats)
        latest_profile = profiler.flush() or {}
        limit_info = latest_profile.get("optimization_limit_analysis", {})
        print(
            f"  [{dataset_name}] optimization headroom: {limit_info.get('status', 'unknown')} | "
            f"{limit_info.get('message', '')}"
        )

    if not all_stats:
        raise RuntimeError("No test datasets were loaded. Check dataset paths and split files.")

    with profiler.section_and_flush("stage.build_comparison"):
        comparison_table = build_comparison_table(all_stats)
    comparison_path = output_dir / "comparison_table.csv"
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
    profiler.add_counter("datasets_evaluated", float(len(all_stats)))
    profiler.add_counter("total_test_samples", float(total_test))
    profiler.add_counter("eval_total_sec", float(time.time() - t_eval_start))
    profiler.snapshot_cuda("end")
    profile_payload = profiler.save_json(profile_path)
    top_bottlenecks = profile_payload.get("top_bottlenecks", [])

    print("\nOutputs:")
    print(f"- comparison_table: {comparison_path}")
    print(f"- comparison_chart: {chart_path}")
    print(f"- comparison_chart_top: {top_chart_path}")
    print(f"- summary: {output_dir / 'summary.json'}")
    print(f"- bottleneck_profile: {profile_path}")
    if top_bottlenecks:
        print("\nTop bottlenecks:")
        for item in top_bottlenecks[:5]:
            print(
                f"  - {item.get('section')}: {float(item.get('total_sec', 0.0)):.3f}s "
                f"({100.0 * float(item.get('ratio', 0.0)):.1f}%)"
            )
    limit_info = profile_payload.get("optimization_limit_analysis", {})
    if limit_info:
        print("\nOptimization limit analysis:")
        print(f"  - status: {limit_info.get('status', 'unknown')}")
        print(f"  - confidence: {float(limit_info.get('confidence', 0.0)):.2f}")
        print(f"  - message: {limit_info.get('message', '')}")
    shutdown_global_async_writer()
    set_active_profiler(None)


if __name__ == "__main__":
    main()
