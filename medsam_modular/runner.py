import json
import os
import shutil
import time
import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from medsam_modular.cache import PredictionCache
from medsam_modular.config import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_ID, DEFAULT_OUTPUT_DIR_REL, ENV_DEFAULTS
from medsam_modular.data import prepare_datasets_by_split
from medsam_modular.eval import OODDetector, TTAPredictor, evaluate_dataset_ood_only, evaluate_dataset_ood_tta
from medsam_modular.io_async import get_global_async_writer, shutdown_global_async_writer
from medsam_modular.model import (
    build_inputs_batch,
    load_medsam,
    load_state_dict_compat,
    normalize_pred_masks_to_4d,
    predict_prob_masks_from_inputs,
    resolve_amp_dtype,
)
from medsam_modular.pipeline.stage3_ood_detect import (
    detect_ood_train_subset as pipeline_detect_ood_train_subset,
    load_cached_ood_train_subset as pipeline_load_cached_ood_train_subset,
)
from medsam_modular.pipeline.stage4_finetune import run_finetune as pipeline_run_finetune
from medsam_modular.pipeline.stage6_baseline import evaluate_baseline_dataset as pipeline_evaluate_baseline_dataset
from medsam_modular.pipeline.stage7_eval import evaluate_test_ood_tta as pipeline_evaluate_test_ood_tta
from medsam_modular.pipeline.stage8_report import run_stage8_plotting as pipeline_run_stage8_plotting
from medsam_modular.visualize import (
    build_comparison_table,
    merge_stage8_stats,
    save_cache_throughput_trend_chart,
    save_cost_breakdown_chart,
    save_delta_chart,
    save_four_way_variant_chart,
    save_method_overview_chart,
    save_ood_train_test_count_chart,
    save_ood_detection_chart,
    save_quality_throughput_frontier,
    save_top_bottom_case_comparison_chart,
    save_tta_cache_hit_chart,
)


_TRUE_SET = {"1", "true", "yes", "y", "on"}


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


@contextmanager
def _timed_log(label: str):
    start = time.perf_counter()
    print(f"[START] {label}", flush=True)
    try:
        yield
    finally:
        print(f"[DONE]  {label} | elapsed={_fmt_elapsed(time.perf_counter() - start)}", flush=True)


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


def _env_float(name: str, default: float) -> float:
    raw = _env(name).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


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
        "allow_tf32": None,
        "cudnn_allow_tf32": None,
        "cudnn_benchmark": None,
        "flash_sdp": None,
        "mem_efficient_sdp": None,
        "math_sdp": None,
        "matmul_precision": None,
    }
    if not torch.cuda.is_available():
        return status

    status["enabled"] = True
    matmul_precision = str(_env("MEDSAM_CUDA_MATMUL_PRECISION") or "high").strip().lower()
    try:
        torch.set_float32_matmul_precision(matmul_precision)
        status["matmul_precision"] = matmul_precision
    except Exception:
        status["matmul_precision"] = "<unsupported>"

    allow_tf32 = _env_bool("MEDSAM_CUDA_ALLOW_TF32", True)
    cudnn_benchmark = _env_bool("MEDSAM_CUDA_CUDNN_BENCHMARK", True)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        status["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        status["allow_tf32"] = "<unsupported>"
    try:
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        status["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        status["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        status["cudnn_allow_tf32"] = "<unsupported>"
        status["cudnn_benchmark"] = "<unsupported>"

    try:
        torch.backends.cuda.enable_flash_sdp(_env_bool("MEDSAM_CUDA_ENABLE_FLASH_SDP", True))
        torch.backends.cuda.enable_mem_efficient_sdp(_env_bool("MEDSAM_CUDA_ENABLE_MEM_EFFICIENT_SDP", True))
        torch.backends.cuda.enable_math_sdp(_env_bool("MEDSAM_CUDA_ENABLE_MATH_SDP", True))
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


def _resolve_clinical_weight_path(project_root: Path, output_dir: Path) -> str:
    configured = _env("MEDSAM_CLINICAL_WEIGHT_PATH").strip()
    if configured and Path(configured).exists():
        return configured
    candidates = [
        output_dir / "medsam_OOD_finetuned_best.pth",
        output_dir / "medsam_OOD_finetuned_last.pth",
        project_root / "results" / "modular" / "medsam_OOD_finetuned_best.pth",
        project_root / "results" / "modular" / "medsam_OOD_finetuned_last.pth",
        project_root / "results" / "medsam_OOD_finetuned_best.pth",
        project_root / "results" / "medsam_OOD_finetuned_last.pth",
    ]
    picked = next((p for p in candidates if p.exists()), None)
    return str(picked) if picked is not None else ""


def _get_train_base_model(model: Any) -> Any:
    if hasattr(model, "_orig_mod") and getattr(model, "_orig_mod") is not None:
        return model._orig_mod
    return model


def _configure_online_finetune_params(model: Any) -> int:
    base = _get_train_base_model(model)
    for p in base.parameters():
        p.requires_grad = False
    if hasattr(base, "mask_decoder"):
        for p in base.mask_decoder.parameters():
            p.requires_grad = True
    if hasattr(base, "prompt_encoder"):
        for p in base.prompt_encoder.parameters():
            p.requires_grad = True
    if _env_bool("MEDSAM_FINETUNE_TRAIN_BACKBONE", False) and hasattr(base, "vision_encoder"):
        for p in base.vision_encoder.parameters():
            p.requires_grad = True
    return int(sum(int(p.numel()) for p in base.parameters() if bool(getattr(p, "requires_grad", False))))


def _online_seg_loss(outputs: Any, gt_mask: torch.Tensor) -> torch.Tensor:
    logits = normalize_pred_masks_to_4d(outputs.pred_masks)
    target = gt_mask.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = torch.nn.functional.interpolate(target, size=logits.shape[-2:], mode="nearest")
    target = target.to(dtype=logits.dtype)

    probs = torch.sigmoid(logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

    flat_p = probs.reshape(probs.shape[0], -1)
    flat_t = target.reshape(target.shape[0], -1)
    numerator = 2.0 * (flat_p * flat_t).sum(dim=1) + 1.0
    denominator = flat_p.pow(2).sum(dim=1) + flat_t.pow(2).sum(dim=1) + 1.0
    dice = 1.0 - (numerator / denominator).mean()
    return bce + dice


def _online_finetune_single_sample(
    *,
    model: Any,
    processor: Any,
    image: Any,
    bbox: List[int],
    gt_mask: Any,
    device: str,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    steps: int,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    if gt_mask is None:
        return {"applied": False, "reason": "missing_gt_mask", "loss_last": None, "steps": 0}

    gt = gt_mask if isinstance(gt_mask, torch.Tensor) else torch.as_tensor(gt_mask)
    if int(gt.numel()) <= 0:
        return {"applied": False, "reason": "empty_gt_mask", "loss_last": None, "steps": 0}

    gt = gt.to(device=device, dtype=torch.float32, non_blocking=(device == "cuda")).unsqueeze(0)
    base = _get_train_base_model(model)
    base.train()

    loss_last: Optional[float] = None
    ran_steps = 0
    for _ in range(max(1, int(steps))):
        inputs = build_inputs_batch(processor=processor, images=[image], input_boxes=[[bbox]])
        inputs = {
            k: (v.to(device=device, non_blocking=(device == "cuda")) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = base(**inputs)
                loss = _online_seg_loss(outputs, gt)
        else:
            outputs = base(**inputs)
            loss = _online_seg_loss(outputs, gt)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_last = float(loss.detach().item())
        ran_steps += 1

    base.eval()
    return {"applied": True, "reason": "ok", "loss_last": loss_last, "steps": ran_steps}


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or (("cuda" in msg) and ("memory" in msg))


def _is_cuda_runtime_unready_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        ("cuda" in msg)
        and (
            "device not ready" in msg
            or "device-side assert" in msg
            or "illegal memory access" in msg
            or "unspecified launch failure" in msg
        )
    )


def _stabilize_tta_for_clinical(tta_predictor: TTAPredictor, device: str) -> None:
    # Clinical mode prioritizes run stability; tuner can spike memory on first sample.
    if hasattr(tta_predictor, "_autotune_enabled"):
        setattr(tta_predictor, "_autotune_enabled", False)
    if hasattr(tta_predictor, "_chunk_size_tuned"):
        setattr(tta_predictor, "_chunk_size_tuned", True)
    if hasattr(tta_predictor, "_build_inputs_on_cpu"):
        setattr(tta_predictor, "_build_inputs_on_cpu", bool(device == "cuda"))

    if hasattr(tta_predictor, "infer_chunk_size"):
        forced_chunk_raw = _env("MEDSAM_CLINICAL_TTA_CHUNK_SIZE").strip()
        forced_chunk = int(forced_chunk_raw) if forced_chunk_raw else 0
        if forced_chunk > 0:
            setattr(tta_predictor, "infer_chunk_size", max(1, forced_chunk))
            return

        # Clinical mode defaults to the most stable chunk to avoid CUDA driver resets.
        if device == "cuda":
            setattr(tta_predictor, "infer_chunk_size", 1)
        else:
            setattr(tta_predictor, "infer_chunk_size", 1)


def _estimate_clinical_tta_max_chunk(device: str) -> int:
    if device != "cuda" or not torch.cuda.is_available():
        return 1
    try:
        total_gb = float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory) / (1024.0 ** 3)
    except Exception:
        total_gb = 12.0

    # Deterministic chunk policy by VRAM class (no runtime trial).
    if total_gb <= 12.5:
        return 1
    if total_gb <= 16.5:
        return 2
    if total_gb <= 24.5:
        return 3
    return 4


def _set_tta_augmentations(tta_predictor: TTAPredictor, augmentations: List[str]) -> None:
    tta_predictor.augmentations = list(augmentations)
    tta_predictor._aug_to_id = {name: idx for idx, name in enumerate(tta_predictor.augmentations)}


def _build_clinical_tta_aug_levels(base_augmentations: List[str]) -> List[List[str]]:
    levels: List[List[str]] = []

    full = [str(v) for v in base_augmentations if str(v)]
    if full:
        levels.append(full)

    four_way = [aug for aug in ["none", "hflip", "vflip", "hvflip"] if aug in full]
    if len(four_way) >= 2 and four_way not in levels:
        levels.append(four_way)

    two_way = [aug for aug in ["none", "hflip"] if aug in full]
    if len(two_way) < 2:
        two_way = full[:2]
    if len(two_way) >= 2 and two_way not in levels:
        levels.append(two_way)

    return levels or [["none", "hflip"]]


def _run_clinical_mode(
    *,
    model: Any,
    processor: Any,
    device: str,
    test_sets: Dict[str, Any],
    ood_detector: OODDetector,
    tta_predictor: TTAPredictor,
    output_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    clinical_weight_path = _resolve_clinical_weight_path(project_root=project_root, output_dir=output_dir)
    if not clinical_weight_path:
        raise RuntimeError("Clinical mode requires OOD-finetuned weight, but none was found.")

    with _timed_log("Clinical mode: load OOD-finetuned weights"):
        load_state_dict_compat(model, Path(clinical_weight_path), map_location=device)

    trainable_count = _configure_online_finetune_params(model)
    base = _get_train_base_model(model)
    base.eval()

    clinical_steps = max(1, int(_env("MEDSAM_CLINICAL_FINETUNE_STEPS") or "2"))
    clinical_lr = float(_env("MEDSAM_CLINICAL_FINETUNE_LR") or "5e-6")
    clinical_wd = float(_env("MEDSAM_CLINICAL_FINETUNE_WEIGHT_DECAY") or "1e-4")
    clinical_use_fused = _env_bool("MEDSAM_CLINICAL_FINETUNE_USE_FUSED_ADAMW", True)
    amp_dtype = resolve_amp_dtype(device)
    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    params = [p for p in base.parameters() if bool(getattr(p, "requires_grad", False))]
    optimizer_kwargs: Dict[str, Any] = {"lr": clinical_lr, "weight_decay": clinical_wd}
    if device == "cuda":
        optimizer_kwargs["fused"] = bool(clinical_use_fused)
    optimizer = torch.optim.AdamW(params, **optimizer_kwargs)

    all_rows: List[Dict[str, Any]] = []
    no_ood_ood_times: List[float] = []
    no_ood_tta_times: List[float] = []
    no_ood_total_times: List[float] = []
    ood_branch_ood_times: List[float] = []
    ood_branch_ft_times: List[float] = []
    ood_branch_tta_times: List[float] = []
    ood_branch_total_times: List[float] = []
    finetune_applied_count = 0
    finetune_skipped_count = 0

    print("\n=== Clinical Mode ===")
    print(f"  weight: {clinical_weight_path}")
    print(f"  online finetune steps: {clinical_steps}")
    print(f"  online finetune lr/wd: {clinical_lr:.2e} / {clinical_wd:.2e}")
    print(f"  online finetune trainable params: {trainable_count:,}")

    tta_default_chunk = int(getattr(tta_predictor, "infer_chunk_size", 1) or 1)
    clinical_chunk_target_raw = _env("MEDSAM_CLINICAL_TTA_MAX_CHUNK_SIZE").strip()
    if clinical_chunk_target_raw:
        clinical_chunk_target = max(1, int(clinical_chunk_target_raw))
    else:
        clinical_chunk_target = _estimate_clinical_tta_max_chunk(device=device)

    _stabilize_tta_for_clinical(tta_predictor=tta_predictor, device=device)
    if hasattr(tta_predictor, "infer_chunk_size"):
        base_chunk = int(getattr(tta_predictor, "infer_chunk_size", 1) or 1)
        setattr(tta_predictor, "infer_chunk_size", max(1, min(base_chunk, clinical_chunk_target)))
    print(
        f"  clinical TTA: autotune=off, chunk_size={int(getattr(tta_predictor, 'infer_chunk_size', 1) or 1)}, "
        f"max_chunk_target={clinical_chunk_target}"
    )

    progress_enabled = _env_bool("MEDSAM_PROGRESS", True)
    progress_interval = max(0.2, _env_float("MEDSAM_PROGRESS_INTERVAL", 1.0))
    total_clinical_samples = int(sum(len(ds) for ds in test_sets.values() if len(ds) > 0))
    processed_samples = 0

    with tqdm(
        total=total_clinical_samples,
        desc="Clinical Mode",
        unit="sample",
        dynamic_ncols=False,
        mininterval=progress_interval,
        disable=not progress_enabled,
    ) as clinical_bar:
        for dataset_name, dataset in test_sets.items():
            if len(dataset) == 0:
                continue
            print(f"\n[Clinical] dataset={dataset_name}, samples={len(dataset)}")
            for idx in range(len(dataset)):
                branch = "unknown"
                sample_name = f"sample_{idx}"
                try:
                    sample = dataset[idx]
                    image = sample.get("image")
                    bbox = sample.get("bbox")
                    sample_name = str(sample.get("name", f"sample_{idx}"))
                    if image is None:
                        continue
                    if not isinstance(bbox, list) or len(bbox) < 4:
                        if hasattr(image, "size") and isinstance(getattr(image, "size"), tuple):
                            w, h = image.size
                        else:
                            w, h = 1024, 1024
                        bbox = [0, 0, max(0, int(w) - 1), max(0, int(h) - 1)]

                    t_sample_start = time.perf_counter()

                    t_ood_start = time.perf_counter()
                    prob_for_ood = _predict_prob_single(
                        model=model,
                        processor=processor,
                        tta_predictor=tta_predictor,
                        image=image,
                        bbox=bbox,
                        device=device,
                        use_tta=False,
                    )
                    ood_info = ood_detector.detect_tensor(prob_for_ood)
                    ood_elapsed = float(time.perf_counter() - t_ood_start)

                    is_ood = bool(ood_info.get("is_ood", False))
                    ft_elapsed = 0.0
                    ft_info: Dict[str, Any] = {"applied": False, "reason": "not_needed", "loss_last": None, "steps": 0}

                    if is_ood:
                        t_ft_start = time.perf_counter()
                        ft_info = _online_finetune_single_sample(
                            model=model,
                            processor=processor,
                            image=image,
                            bbox=bbox,
                            gt_mask=sample.get("mask"),
                            device=device,
                            optimizer=optimizer,
                            scaler=scaler,
                            steps=clinical_steps,
                            amp_dtype=amp_dtype,
                        )
                        ft_elapsed = float(time.perf_counter() - t_ft_start)
                        if device == "cuda":
                            try:
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        if bool(ft_info.get("applied", False)):
                            finetune_applied_count += 1
                        else:
                            finetune_skipped_count += 1

                    t_tta_start = time.perf_counter()
                    tta_retries = 0
                    tta_aug_levels = _build_clinical_tta_aug_levels(list(getattr(tta_predictor, "augmentations", [])))
                    tta_aug_level_idx = 0
                    while True:
                        try:
                            _ = tta_predictor.predict(
                                model=model,
                                processor=processor,
                                image=image,
                                bbox=bbox,
                                device=device,
                            )
                            break
                        except Exception as exc:
                            if device != "cuda":
                                raise

                            cur_chunk = int(getattr(tta_predictor, "infer_chunk_size", 1) or 1)
                            if _is_cuda_oom_error(exc):
                                next_chunk = max(1, cur_chunk // 2)
                                if next_chunk < cur_chunk:
                                    setattr(tta_predictor, "infer_chunk_size", next_chunk)
                                    print(
                                        f"  [Clinical TTA retry] {dataset_name}/{sample_name}: chunk {cur_chunk} -> {next_chunk}",
                                        flush=True,
                                    )

                                if tta_aug_level_idx + 1 < len(tta_aug_levels):
                                    tta_aug_level_idx += 1
                                    next_augs = tta_aug_levels[tta_aug_level_idx]
                                    _set_tta_augmentations(tta_predictor, next_augs)
                                    print(
                                        f"  [Clinical TTA degrade] {dataset_name}/{sample_name}: augmentations -> {next_augs}",
                                        flush=True,
                                    )

                                if tta_retries < 6:
                                    tta_retries += 1
                                    try:
                                        torch.cuda.empty_cache()
                                        torch.cuda.synchronize()
                                    except Exception:
                                        pass
                                    continue

                            if _is_cuda_runtime_unready_error(exc):
                                if tta_retries < 1:
                                    setattr(tta_predictor, "infer_chunk_size", 1)
                                    tta_retries += 1
                                    try:
                                        torch.cuda.empty_cache()
                                        torch.cuda.synchronize()
                                    except Exception:
                                        pass
                                    print(
                                        f"  [Clinical TTA recover] {dataset_name}/{sample_name}: CUDA unstable, retry with chunk=1",
                                        flush=True,
                                    )
                                    continue
                                raise RuntimeError(
                                    "CUDA runtime became unstable during clinical TTA inference. "
                                    "Try rerun with smaller chunk via MEDSAM_TTA_CHUNK_SIZE=1 "
                                    "or restart the Python process/GPU driver."
                                ) from exc
                            raise
                    tta_elapsed = float(time.perf_counter() - t_tta_start)

                    total_elapsed = float(time.perf_counter() - t_sample_start)

                    branch = "ood_branch" if is_ood else "no_ood_branch"
                    if is_ood:
                        ood_branch_ood_times.append(ood_elapsed)
                        ood_branch_ft_times.append(ft_elapsed)
                        ood_branch_tta_times.append(tta_elapsed)
                        ood_branch_total_times.append(total_elapsed)
                    else:
                        no_ood_ood_times.append(ood_elapsed)
                        no_ood_tta_times.append(tta_elapsed)
                        no_ood_total_times.append(total_elapsed)

                    all_rows.append(
                        {
                            "dataset": dataset_name,
                            "index": int(idx),
                            "name": sample_name,
                            "branch": branch,
                            "is_ood": is_ood,
                            "ood_score": float(ood_info.get("ood_score", 0.0)),
                            "ood_reason_codes": list(ood_info.get("ood_reason_codes", [])),
                            "ood_time_sec": ood_elapsed,
                            "finetune_time_sec": ft_elapsed,
                            "finetune_applied": bool(ft_info.get("applied", False)),
                            "finetune_reason": str(ft_info.get("reason", "")),
                            "finetune_steps": int(ft_info.get("steps", 0)),
                            "finetune_loss_last": ft_info.get("loss_last", None),
                            "tta_time_sec": tta_elapsed,
                            "total_time_sec": total_elapsed,
                        }
                    )
                finally:
                    processed_samples += 1
                    clinical_bar.update(1)
                    clinical_bar.set_postfix(
                        dataset=dataset_name,
                        sample=sample_name,
                        branch=branch,
                        done=f"{processed_samples}/{max(1, total_clinical_samples)}",
                        refresh=False,
                    )

    n_total = len(all_rows)
    n_no_ood = len(no_ood_total_times)
    n_ood = len(ood_branch_total_times)
    clinical_total_time = float(np.sum(np.asarray([r["total_time_sec"] for r in all_rows], dtype=np.float64))) if all_rows else 0.0

    summary: Dict[str, Any] = {
        "mode": "clinical",
        "weight_path": clinical_weight_path,
        "clinical_finetune_steps": int(clinical_steps),
        "clinical_finetune_lr": float(clinical_lr),
        "clinical_finetune_weight_decay": float(clinical_wd),
        "clinical_finetune_use_fused_adamw": bool(clinical_use_fused),
        "num_samples": int(n_total),
        "num_no_ood_branch": int(n_no_ood),
        "num_ood_branch": int(n_ood),
        "finetune_applied_count": int(finetune_applied_count),
        "finetune_skipped_count": int(finetune_skipped_count),
        "total_time_sec": clinical_total_time,
        "overall_avg_time_per_sample_sec": float(clinical_total_time / max(1, n_total)),
        "overall_throughput_samples_per_sec": float(n_total / clinical_total_time) if clinical_total_time > 0 else 0.0,
        "no_ood_branch": {
            "avg_ood_time_sec": _safe_mean(no_ood_ood_times),
            "avg_finetune_time_sec": 0.0,
            "avg_tta_time_sec": _safe_mean(no_ood_tta_times),
            "avg_total_time_sec": _safe_mean(no_ood_total_times),
            "throughput_samples_per_sec": float(n_no_ood / max(1e-8, float(np.sum(np.asarray(no_ood_total_times, dtype=np.float64))))) if n_no_ood > 0 else 0.0,
        },
        "ood_branch": {
            "avg_ood_time_sec": _safe_mean(ood_branch_ood_times),
            "avg_finetune_time_sec": _safe_mean(ood_branch_ft_times),
            "avg_tta_time_sec": _safe_mean(ood_branch_tta_times),
            "avg_total_time_sec": _safe_mean(ood_branch_total_times),
            "throughput_samples_per_sec": float(n_ood / max(1e-8, float(np.sum(np.asarray(ood_branch_total_times, dtype=np.float64))))) if n_ood > 0 else 0.0,
        },
    }

    _save_json(output_dir / "clinical_mode_results.json", all_rows)
    _save_json(output_dir / "clinical_mode_stats.json", summary)

    print("\n[Clinical] Summary")
    print(f"  total samples: {n_total}")
    print(f"  no-OOD branch: {n_no_ood} | avg_total={summary['no_ood_branch']['avg_total_time_sec']:.4f}s")
    print(f"  OOD branch   : {n_ood} | avg_total={summary['ood_branch']['avg_total_time_sec']:.4f}s")
    print(f"  overall throughput: {summary['overall_throughput_samples_per_sec']:.3f} samples/s")

    return summary


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


def _all_have_baseline_stats(all_stats: Dict[str, Dict[str, Dict[str, Any]]]) -> bool:
    if not all_stats:
        return False
    for _, modes in all_stats.items():
        baseline = modes.get("baseline")
        if not isinstance(baseline, dict) or not baseline:
            return False
    return True


def _run_stage8_plotting(
    *,
    all_stats: Dict[str, Dict[str, Dict[str, Any]]],
    all_stats_ood_finetuned: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    project_root: Path,
    profiler: Any,
) -> Tuple[Path, Path, Path, Dict[str, Path], Optional[Path]]:
    comparison_path = output_dir / "comparison_table.csv"
    chart_path = output_dir / "performance_comparison_4way.png"
    top_chart_path = (project_root / "results") / chart_path.name
    stage8_paths: Dict[str, Path] = {}
    stage8_history_path: Optional[Path] = None
    stage8_plot_stats = merge_stage8_stats(
        full_summary=all_stats,
        ood_finetuned_summary=all_stats_ood_finetuned,
    )

    if not _all_have_baseline_stats(all_stats):
        print("\n[Stage 8/8] baseline stats 缺失，略過 comparison table/chart 產生。")
    else:
        with _timed_log("Stage 8: build comparison table"):
            with profiler.section_and_flush("stage.build_comparison"):
                comparison_table = build_comparison_table(all_stats)
        with _timed_log("Stage 8: save comparison CSV"):
            with profiler.section_and_flush("stage.save_comparison_csv"):
                comparison_table.to_csv(comparison_path, index=False)
        with _timed_log("Stage 8: save 4-way comparison chart"):
            with profiler.section_and_flush("stage.save_comparison_chart_4way"):
                chart_path = save_four_way_variant_chart(
                    full_summary=all_stats,
                    ood_finetuned_summary=all_stats_ood_finetuned or {},
                    output_dir=output_dir,
                )

    with _timed_log("Stage 8: save method overview chart"):
        with profiler.section_and_flush("stage.save_stage8_method_overview"):
            stage8_paths["method_overview"] = save_method_overview_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save delta chart"):
        with profiler.section_and_flush("stage.save_stage8_delta"):
            stage8_paths["delta_vs_baseline"] = save_delta_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save cost breakdown chart"):
        with profiler.section_and_flush("stage.save_stage8_cost_breakdown"):
            stage8_paths["cost_breakdown"] = save_cost_breakdown_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save quality-throughput frontier"):
        with profiler.section_and_flush("stage.save_stage8_frontier"):
            stage8_paths["quality_throughput_frontier"] = save_quality_throughput_frontier(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save OOD detection chart"):
        with profiler.section_and_flush("stage.save_stage8_ood_detection"):
            stage8_paths["ood_detection_quality"] = save_ood_detection_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save TTA cache chart"):
        with profiler.section_and_flush("stage.save_stage8_tta_cache"):
            stage8_paths["tta_cache_hits"] = save_tta_cache_hit_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save cache throughput trend"):
        with profiler.section_and_flush("stage.save_stage8_cache_throughput_trend"):
            trend_path, history_path = save_cache_throughput_trend_chart(stage8_plot_stats, output_dir)
            stage8_paths["cache_throughput_trend"] = trend_path
            stage8_history_path = history_path

    top_results_dir = project_root / "results"
    top_results_dir.mkdir(parents=True, exist_ok=True)
    top_chart_path = top_results_dir / chart_path.name
    with _timed_log("Stage 8: copy top-level comparison chart"):
        with profiler.section_and_flush("stage.copy_chart"):
            if chart_path.exists() and chart_path.resolve() != top_chart_path.resolve():
                shutil.copy2(chart_path, top_chart_path)

    return comparison_path, chart_path, top_chart_path, stage8_paths, stage8_history_path


def _dataset_name_to_index_map(dataset: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list):
        for idx, s in enumerate(samples):
            if isinstance(s, dict):
                name = str(s.get("name", s.get("image_id", f"sample_{idx}")))
            else:
                name = f"sample_{idx}"
            out[name] = idx
        return out

    if hasattr(dataset, "__len__"):
        for idx in range(int(len(dataset))):
            sample = dataset[idx]
            out[str(sample.get("name", f"sample_{idx}"))] = idx
    return out


def _select_top_bottom_samples(results: List[Dict[str, Any]], top_k: int = 3, bottom_k: int = 3) -> List[Dict[str, Any]]:
    finite_results: List[Dict[str, Any]] = []
    for r in results:
        try:
            _ = float(r.get("dice", float("nan")))
            finite_results.append(r)
        except Exception:
            continue
    if not finite_results:
        return []

    sorted_results = sorted(finite_results, key=lambda r: float(r.get("dice", float("nan"))))
    bottom = sorted_results[: max(0, int(bottom_k))]

    used_names = {str(r.get("name", "")) for r in bottom}
    top_candidates = list(reversed(sorted_results))
    top: List[Dict[str, Any]] = []
    for r in top_candidates:
        name = str(r.get("name", ""))
        if name in used_names:
            continue
        top.append(r)
        if len(top) >= max(0, int(top_k)):
            break

    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(top):
        rec = dict(r)
        rec["rank_label"] = f"Best #{idx + 1}"
        out.append(rec)
    for idx, r in enumerate(bottom):
        rec = dict(r)
        rec["rank_label"] = f"Worst #{idx + 1}"
        out.append(rec)
    return out


def _mask_has_positive_label(mask_like: Any) -> bool:
    if mask_like is None:
        return False
    if isinstance(mask_like, torch.Tensor):
        if int(mask_like.numel()) == 0:
            return False
        return bool((mask_like > 0.5).any().item())

    try:
        arr = np.asarray(mask_like)
    except Exception:
        return False
    if int(arr.size) == 0:
        return False
    return bool(np.any(arr > 0.5))


def _predict_prob_single(
    *,
    model: Any,
    processor: Any,
    tta_predictor: TTAPredictor,
    image: Any,
    bbox: Any,
    device: str,
    use_tta: bool,
) -> torch.Tensor:
    if hasattr(image, "size") and isinstance(getattr(image, "size"), tuple):
        width, height = image.size
    else:
        arr = np.asarray(image)
        height = int(arr.shape[0]) if arr.ndim >= 2 else 1024
        width = int(arr.shape[1]) if arr.ndim >= 2 else 1024

    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0, 0, max(0, int(width) - 1), max(0, int(height) - 1)]

    if use_tta:
        prob_t, _ = tta_predictor.predict(
            model=model,
            processor=processor,
            image=image,
            bbox=bbox,
            device=device,
        )
        return prob_t

    inputs = build_inputs_batch(processor=processor, images=[image], input_boxes=[[bbox]])
    prob_batch = predict_prob_masks_from_inputs(
        model=model,
        inputs=inputs,
        device=device,
        output_size=(int(height), int(width)),
        use_amp=True,
        inputs_already_on_device=False,
    )[:, 0]
    return prob_batch[0]


def _generate_top_bottom_case_charts(
    *,
    output_dir: Path,
    test_sets: Dict[str, Any],
    model: Any,
    processor: Any,
    device: str,
    ood_detector: OODDetector,
    tta_predictor: TTAPredictor,
    baseline_weight_path: str,
    ood_finetuned_best_path: Path,
    full_finetuned_best_path: Path,
    resume_weight_path: str,
    file_tag: str,
) -> Dict[str, Path]:
    case_dir = output_dir / "case_comparisons"
    case_dir.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, Path] = {}

    variant_cfgs: List[Dict[str, Any]] = [
        {
            "variant_key": "baseline",
            "result_suffix": "baseline_results.json",
            "weight_path": (Path(baseline_weight_path) if baseline_weight_path else None),
            "use_tta": False,
        },
        {
            "variant_key": "ood_finetune",
            "result_suffix": "ood_finetuned_ood_results.json",
            "weight_path": ood_finetuned_best_path,
            "use_tta": False,
        },
        {
            "variant_key": "full_finetune",
            "result_suffix": "full_finetuned_ood_results.json",
            "weight_path": (
                full_finetuned_best_path
                if full_finetuned_best_path.exists()
                else (Path(resume_weight_path) if resume_weight_path else None)
            ),
            "use_tta": False,
        },
        {
            "variant_key": "ood_finetune_tta",
            "result_suffix": "ood_finetuned_tta_results.json",
            "weight_path": ood_finetuned_best_path,
            "use_tta": True,
        },
    ]

    for cfg in variant_cfgs:
        variant_key = str(cfg["variant_key"])
        weight_path = cfg.get("weight_path")
        if weight_path is None or not Path(weight_path).exists():
            continue

        load_state_dict_compat(model, Path(weight_path), map_location=device)

        for dataset_name, dataset in test_sets.items():
            result_path = output_dir / f"{dataset_name.lower()}_{cfg['result_suffix']}"
            if not result_path.exists():
                continue

            try:
                variant_results = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(variant_results, list) or not variant_results:
                continue

            name_to_idx = _dataset_name_to_index_map(dataset)
            valid_name_set: Set[str] = set()
            for name, idx in name_to_idx.items():
                try:
                    sample = dataset[idx]
                except Exception:
                    continue
                if _mask_has_positive_label(sample.get("mask")):
                    valid_name_set.add(name)

            filtered_results = [
                r
                for r in variant_results
                if str(r.get("name", "")) in valid_name_set
            ]
            if not filtered_results:
                continue

            picked = _select_top_bottom_samples(filtered_results, top_k=3, bottom_k=3)
            if not picked:
                continue

            case_entries: List[Dict[str, Any]] = []
            for row in picked:
                sample_name = str(row.get("name", ""))
                if sample_name not in name_to_idx:
                    continue

                sample = dataset[name_to_idx[sample_name]]
                image = sample.get("image")
                if image is None:
                    continue

                gt_mask = sample.get("mask")
                if not _mask_has_positive_label(gt_mask):
                    continue
                if isinstance(gt_mask, torch.Tensor):
                    gt_mask_np = gt_mask.detach().cpu().numpy()
                else:
                    gt_mask_np = gt_mask

                bbox = sample.get("bbox", None)
                prob_t = _predict_prob_single(
                    model=model,
                    processor=processor,
                    tta_predictor=tta_predictor,
                    image=image,
                    bbox=bbox,
                    device=device,
                    use_tta=bool(cfg.get("use_tta", False)),
                )
                pred_mask_np = (prob_t > 0.5).to(torch.uint8).detach().cpu().numpy()
                ood_pred = ood_detector.detect_tensor(prob_t)

                case_entries.append(
                    {
                        "rank_label": row.get("rank_label", ""),
                        "name": sample_name,
                        "dice": float(row.get("dice", float("nan"))),
                        "ood_score": float(ood_pred.get("ood_score", 0.0)),
                        "is_ood": bool(ood_pred.get("is_ood", False)),
                        "image": image,
                        "gt_mask": gt_mask_np,
                        "bbox": bbox,
                        "pred_mask": pred_mask_np,
                    }
                )

            out_path = save_top_bottom_case_comparison_chart(
                dataset_name=dataset_name,
                case_entries=case_entries,
                output_dir=case_dir,
                file_tag=f"{file_tag}_{variant_key}",
            )
            if out_path is not None:
                out_paths[f"{dataset_name}_{variant_key}"] = out_path

    return out_paths


def _save_train_test_ood_summary(
    *,
    output_dir: Path,
    train_ood_summary: Dict[str, Any],
    test_all_stats: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[Path, Path, Optional[Path]]:
    datasets = sorted(set(train_ood_summary.keys()) | set(test_all_stats.keys()))
    payload: Dict[str, Dict[str, Any]] = {}
    chart_in: Dict[str, Dict[str, float]] = {}

    for dataset_name in datasets:
        train_row = train_ood_summary.get(dataset_name, {}) if isinstance(train_ood_summary, dict) else {}
        test_row = (test_all_stats.get(dataset_name, {}) or {}).get("ood", {})

        train_n = int(train_row.get("num_samples", 0) or 0)
        train_ood = int(train_row.get("num_ood", train_row.get("num_ood_detected", 0)) or 0)
        test_n = int(test_row.get("num_samples", 0) or 0)
        test_ood = int(test_row.get("num_ood_detected", 0) or 0)

        payload[dataset_name] = {
            "train_samples": train_n,
            "train_ood": train_ood,
            "train_ood_ratio": float(train_ood / max(1, train_n)),
            "test_samples": test_n,
            "test_ood": test_ood,
            "test_ood_ratio": float(test_ood / max(1, test_n)),
        }
        chart_in[dataset_name] = {
            "train_ood": float(train_ood),
            "test_ood": float(test_ood),
            "train_ood_ratio": float(train_ood / max(1, train_n)),
            "test_ood_ratio": float(test_ood / max(1, test_n)),
        }

    ood_finetune_stats_path = output_dir / "ood_finetune_stats.json"
    ood_finetune_stats: Dict[str, Any] = {}
    if ood_finetune_stats_path.exists():
        try:
            loaded = json.loads(ood_finetune_stats_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                ood_finetune_stats = loaded
        except Exception:
            ood_finetune_stats = {}

    total_train_ood = int(sum(int((train_ood_summary.get(ds, {}) or {}).get("num_ood", 0) or 0) for ds in datasets))
    total_train_samples = int(sum(int((train_ood_summary.get(ds, {}) or {}).get("num_samples", 0) or 0) for ds in datasets))
    ood_ft_total_sec = float(ood_finetune_stats.get("total_finetune_sec", 0.0) or 0.0)
    ood_ft_epochs_ran = int(ood_finetune_stats.get("epochs_ran", 0) or 0)
    if ood_ft_epochs_ran <= 0:
        hist = ood_finetune_stats.get("history", {})
        if isinstance(hist, dict) and isinstance(hist.get("val_loss"), list):
            ood_ft_epochs_ran = int(len(hist.get("val_loss", [])))

    ood_ft_convergence_epoch = int(ood_finetune_stats.get("convergence_epoch", 0) or 0)
    if ood_ft_convergence_epoch <= 0:
        hist = ood_finetune_stats.get("history", {})
        val_loss_hist = hist.get("val_loss", []) if isinstance(hist, dict) else []
        if isinstance(val_loss_hist, list) and val_loss_hist:
            try:
                ood_ft_convergence_epoch = int(np.argmin(np.asarray(val_loss_hist, dtype=np.float64))) + 1
            except Exception:
                ood_ft_convergence_epoch = 0

    if ood_ft_total_sec <= 0:
        epoch_durations = ood_finetune_stats.get("epoch_durations_sec", [])
        if isinstance(epoch_durations, list) and epoch_durations:
            try:
                ood_ft_total_sec = float(np.sum(np.asarray(epoch_durations, dtype=np.float64)))
            except Exception:
                ood_ft_total_sec = 0.0
    ood_ft_avg_epoch_sec = float(
        ood_finetune_stats.get("avg_epoch_sec", (ood_ft_total_sec / max(1, ood_ft_epochs_ran)))
        or 0.0
    )
    avg_sec_per_ood_sample = float(ood_ft_total_sec / max(1, total_train_ood))

    payload["__overall__"] = {
        "ood_train_samples_total": total_train_ood,
        "train_samples_total": total_train_samples,
        "ood_finetune_train_samples": int(ood_finetune_stats.get("train_samples", 0) or 0),
        "ood_finetune_epochs_ran": ood_ft_epochs_ran,
        "ood_finetune_convergence_epoch": ood_ft_convergence_epoch,
        "ood_finetune_avg_epoch_sec": ood_ft_avg_epoch_sec,
        "ood_finetune_total_sec": ood_ft_total_sec,
        "ood_finetune_avg_sec_per_ood_sample": avg_sec_per_ood_sample,
    }

    json_path = output_dir / "ood_train_test_counts.json"
    _save_json(json_path, payload)

    csv_path = output_dir / "ood_train_test_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "train_samples",
            "train_ood",
            "train_ood_ratio",
            "test_samples",
            "test_ood",
            "test_ood_ratio",
            "ood_finetune_epochs_ran",
            "ood_finetune_convergence_epoch",
            "ood_finetune_avg_epoch_sec",
            "ood_finetune_avg_sec_per_ood_sample",
        ])
        for dataset_name in datasets:
            row = payload[dataset_name]
            writer.writerow([
                dataset_name,
                row["train_samples"],
                row["train_ood"],
                f"{float(row['train_ood_ratio']):.6f}",
                row["test_samples"],
                row["test_ood"],
                f"{float(row['test_ood_ratio']):.6f}",
                ood_ft_epochs_ran,
                ood_ft_convergence_epoch,
                f"{ood_ft_avg_epoch_sec:.6f}",
                f"{avg_sec_per_ood_sample:.6f}",
            ])

    chart_path = save_ood_train_test_count_chart(chart_in, output_dir)
    return json_path, csv_path, chart_path


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
        "skip_finetune": "0",
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
            summary[dataset_name] = {
                "num_samples": 0,
                "num_ood": 0,
                "ood_ratio": 0.0,
            }
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


def _load_cached_ood_train_subset(
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


def main() -> None:
    pipeline_start = time.perf_counter()
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
            f"tf32={cuda_accel.get('allow_tf32')} "
            f"cudnn_tf32={cuda_accel.get('cudnn_allow_tf32')} "
            f"cudnn_bench={cuda_accel.get('cudnn_benchmark')} "
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

    print("\n[Stage 1/8] 載入模型 ...")
    t1 = time.time()
    with _timed_log("Stage 1/8: load model"):
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

    print("\n[Stage 2/8] 準備測試資料 ...")
    t3 = time.time()
    split_root = _resolve_split_root(project_root)
    with _timed_log("Stage 2/8: prepare test datasets"):
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

    run_stage3_detect_train_ood = _env_bool("MEDSAM_RUN_STAGE3_DETECT_TRAIN_OOD", True)
    run_stage4_ood_finetune = _env_bool("MEDSAM_RUN_STAGE4_OOD_FINETUNE", True)
    run_stage5_full_finetune = _env_bool("MEDSAM_RUN_STAGE5_FULL_FINETUNE", True)
    run_stage6_baseline_eval = _env_bool("MEDSAM_RUN_STAGE6_BASELINE_EVAL", True)
    run_stage7_eval_ood_finetuned = _env_bool("MEDSAM_RUN_STAGE7_EVAL_OOD_FINETUNED", True)
    run_stage7_eval_full_finetuned = _env_bool("MEDSAM_RUN_STAGE7_EVAL_FULL_FINETUNED", True)
    run_stage8_plotting = _env_bool("MEDSAM_RUN_STAGE8_PLOTTING", True)
    run_clinical_mode = _env_bool("MEDSAM_RUN_CLINICAL_MODE", False)

    print("\n=== Pipeline Stage Switches ===")
    print(f"  Stage3 detect train OOD       : {run_stage3_detect_train_ood}")
    print(f"  Stage4 OOD finetune           : {run_stage4_ood_finetune}")
    print(f"  Stage5 full finetune          : {run_stage5_full_finetune}")
    print(f"  Stage6 baseline eval          : {run_stage6_baseline_eval}")
    print(f"  Stage7 eval OOD-finetuned     : {run_stage7_eval_ood_finetuned}")
    print(f"  Stage7 eval full-finetuned    : {run_stage7_eval_full_finetuned}")
    print(f"  Stage8 plotting               : {run_stage8_plotting}")
    print(f"  Clinical mode                 : {run_clinical_mode}")

    if run_clinical_mode:
        print("\n[Clinical Mode] 啟用：使用 OOD finetuned 權重進行線上決策流程。")
        with _timed_log("Clinical mode total"):
            _run_clinical_mode(
                model=model,
                processor=processor,
                device=device,
                test_sets=test_sets,
                ood_detector=ood_detector,
                tta_predictor=tta_predictor,
                output_dir=output_dir,
                project_root=project_root,
            )
        with _timed_log("Shutdown async writer"):
            shutdown_global_async_writer()
        print(f"\nPipeline total elapsed: {_fmt_elapsed(time.perf_counter() - pipeline_start)}")
        return

    split_root = _resolve_split_root(project_root)

    ood_subset_by_name: Dict[str, Set[str]] = {}
    ood_subset_summary: Dict[str, Any] = {}
    total_ood = 0
    total_all = 0
    ood_finetuned_best_path = output_dir / "medsam_OOD_finetuned_best.pth"
    full_finetuned_best_path = output_dir / "medsam_finetuned_best.pth"

    if run_stage3_detect_train_ood:
        print("\n[Stage 3/8] baseline 偵測 train split OOD ...")
        with _timed_log("Stage 3/8: detect train split OOD"):
            ood_subset_by_name, ood_subset_summary = pipeline_detect_ood_train_subset(
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
    else:
        print("\n[Stage 3/8] 依設定略過 train OOD 偵測，嘗試載入既有結果 ...")
        ood_subset_by_name, ood_subset_summary = pipeline_load_cached_ood_train_subset(
            output_dir=output_dir,
            dataset_names=list(test_sets.keys()),
        )
        total_ood = int(sum(len(v) for v in ood_subset_by_name.values()))
        total_all = int(sum(int((ood_subset_summary.get(k, {}) or {}).get("num_samples", 0)) for k in test_sets.keys()))
        if total_all > 0:
            print(f"  已載入 cached OOD subset: {total_ood}/{total_all} samples")
        else:
            print("  ⚠️ 找不到可用 cached OOD subset。")

    if run_stage4_ood_finetune and total_ood > 0:
        print("\n[Stage 4/8] OOD 子集微調（使用 TTA 增強資料）...")
        if baseline_weight_path and Path(baseline_weight_path).exists():
            with _timed_log("Stage 4: load baseline weights before OOD fine-tune"):
                load_state_dict_compat(model, Path(baseline_weight_path), map_location=device)

        t2 = time.time()
        with _timed_log("Stage 4/8: OOD subset fine-tune"):
            with profiler.section_and_flush("stage.ood_finetune"):
                model = pipeline_run_finetune(
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
    elif run_stage4_ood_finetune:
        print("\n[Stage 4/8] OOD 子集為空，略過 OOD 微調。")
    else:
        print("\n[Stage 4/8] 依設定略過 OOD 微調。")

    if run_stage5_full_finetune:
        print("\n[Stage 5/8] 全資料微調（輸出 medsam_finetuned_best.pth）...")
        vit_b_weight_path = project_root / "results" / "medsam_vit_b.pth"
        if vit_b_weight_path.exists():
            with _timed_log("Stage 5: load medsam_vit_b weights before full fine-tune"):
                load_state_dict_compat(model, vit_b_weight_path, map_location=device)
            print(f"  📌 全資料微調起始權重: {vit_b_weight_path}")
        elif baseline_weight_path and Path(baseline_weight_path).exists():
            with _timed_log("Stage 5: load fallback baseline weights before full fine-tune"):
                load_state_dict_compat(model, Path(baseline_weight_path), map_location=device)
            print(f"  ⚠️ 找不到 medsam_vit_b.pth，改用 fallback 起始權重: {baseline_weight_path}")
        else:
            print("  ⚠️ 找不到可用起始權重，將沿用目前模型權重進行全資料微調。")

        t2 = time.time()
        with _timed_log("Stage 5/8: full-data fine-tune"):
            with profiler.section_and_flush("stage.full_finetune"):
                model = pipeline_run_finetune(
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
        print("\n[Stage 5/8] 依設定略過全資料微調。")

    baseline_all_results: Dict[str, Any] = {}
    baseline_all_stats: Dict[str, Dict[str, Any]] = {}
    if run_stage6_baseline_eval:
        print("\n[Stage 6/8] 基線評估 (vit_b) ...")
        t_eval_start = time.time()
        baseline_weight = Path(baseline_weight_path) if baseline_weight_path else None
        if baseline_weight is not None and baseline_weight.exists():
            with _timed_log("Stage 6: load baseline weights"):
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
            with _timed_log(f"Stage 6: baseline evaluate {dataset_name}"):
                with profiler.section_and_flush(f"eval.{dataset_name}.baseline.total"):
                    baseline_results, baseline_stats = pipeline_evaluate_baseline_dataset(
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
            with _timed_log(f"Stage 6: save baseline outputs for {dataset_name}"):
                _save_json(output_dir / f"{dataset_name.lower()}_baseline_results.json", baseline_results)
                _save_json(output_dir / f"{dataset_name.lower()}_baseline_stats.json", baseline_stats)
    else:
        print("\n[Stage 6/8] 依設定略過 baseline 評估，嘗試載入既有 baseline stats ...")
        baseline_all_stats = _load_existing_baseline_stats(test_sets=test_sets, output_dir=output_dir)
        if baseline_all_stats:
            print(f"  已載入 baseline stats: {', '.join(sorted(baseline_all_stats.keys()))}")
        else:
            print("  ⚠️ 找不到 baseline stats；若啟用 Stage 7/8，comparison 可能不完整。")

    all_stats_ood_finetuned: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if run_stage7_eval_ood_finetuned and ood_finetuned_best_path.exists():
        print("\n[Stage 7/8] 測試 OOD finetuned 模型：先 OOD 判斷，再 TTA inference ...")
        with _timed_log("Stage 7: load OOD fine-tuned weights"):
            load_state_dict_compat(model, ood_finetuned_best_path, map_location=device)
        print(f"  📌 OOD finetuned 評估權重: {ood_finetuned_best_path}")
        with _timed_log("Stage 7/8: evaluate OOD fine-tuned model"):
            all_stats_ood_finetuned = pipeline_evaluate_test_ood_tta(
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
        with _timed_log("Stage 7: save OOD fine-tuned summary"):
            _save_json(output_dir / "summary_ood_finetuned.json", all_stats_ood_finetuned)
    elif run_stage7_eval_ood_finetuned:
        print("\n[Stage 7/8] 找不到 medsam_OOD_finetuned_best.pth，略過 OOD finetuned 模型測試。")
    else:
        print("\n[Stage 7/8] 依設定略過 OOD finetuned 模型測試。")

    all_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if run_stage7_eval_full_finetuned:
        if full_finetuned_best_path.exists():
            with _timed_log("Stage 7: load full fine-tuned weights"):
                load_state_dict_compat(model, full_finetuned_best_path, map_location=device)
            print(f"  📌 評估使用權重: {full_finetuned_best_path}")
        elif resume_weight_path and Path(resume_weight_path).exists():
            with _timed_log("Stage 7: load resume weights"):
                load_state_dict_compat(model, Path(resume_weight_path), map_location=device)
            print(f"  📌 評估使用權重: {resume_weight_path}")
        else:
            print("  📌 評估使用權重: <finetuned model in-memory>")

        print("\n[Stage 7/8] 測試全資料 finetuned 模型：OOD 判斷 + TTA inference ...")
        with _timed_log("Stage 7/8: evaluate full fine-tuned model"):
            all_stats = pipeline_evaluate_test_ood_tta(
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

        with _timed_log("Stage 7: save full fine-tuned summary"):
            _save_json(output_dir / "summary.json", all_stats)
    else:
        print("\n[Stage 7/8] 依設定略過 full finetuned 模型測試。")
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            try:
                loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded_summary, dict):
                    all_stats = loaded_summary
                    print(f"  已載入既有 summary: {summary_path}")
            except Exception:
                pass

    if not all_stats:
        raise RuntimeError("No evaluation summary available. Enable Stage 7 full-finetuned eval or provide existing summary.json.")

    case_chart_paths: Dict[str, Path] = {}
    if run_stage7_eval_full_finetuned:
        with _timed_log("Stage 7: generate top/bottom case charts"):
            case_chart_paths = _generate_top_bottom_case_charts(
                output_dir=output_dir,
                test_sets=test_sets,
                model=model,
                processor=processor,
                device=device,
                ood_detector=ood_detector,
                tta_predictor=tta_predictor,
                baseline_weight_path=baseline_weight_path,
                ood_finetuned_best_path=ood_finetuned_best_path,
                full_finetuned_best_path=full_finetuned_best_path,
                resume_weight_path=resume_weight_path,
                file_tag="4way",
            )

    with _timed_log("Stage 7: save train/test OOD summary"):
        ood_summary_json, ood_summary_csv, ood_summary_chart = _save_train_test_ood_summary(
            output_dir=output_dir,
            train_ood_summary=ood_subset_summary,
            test_all_stats=all_stats,
        )

    comparison_path = output_dir / "comparison_table.csv"
    chart_path = output_dir / "performance_comparison_4way.png"
    top_chart_path = (project_root / "results") / chart_path.name
    stage8_paths: Dict[str, Path] = {}
    stage8_history_path: Optional[Path] = None

    if not run_stage8_plotting:
        print("\n[Stage 8/8] 依設定略過繪圖階段。")
    else:
        print("\n[Stage 8/8] 產生 comparison table / chart ...")
        with _timed_log("Stage 8/8: plotting and report charts"):
            comparison_path, chart_path, top_chart_path, stage8_paths, stage8_history_path = pipeline_run_stage8_plotting(
                all_stats=all_stats,
                all_stats_ood_finetuned=all_stats_ood_finetuned,
                output_dir=output_dir,
                project_root=project_root,
                profiler=profiler,
            )

    print("\nOutputs:")
    if comparison_path.exists():
        print(f"- comparison_table: {comparison_path}")
    if chart_path.exists():
        print(f"- comparison_chart: {chart_path}")
        print(f"- comparison_chart_top: {top_chart_path}")
    for key in sorted(stage8_paths.keys()):
        path = stage8_paths[key]
        if path.exists():
            print(f"- stage8_{key}: {path}")
    if stage8_history_path is not None and stage8_history_path.exists():
        print(f"- stage8_run_history: {stage8_history_path}")
    print(f"- summary: {output_dir / 'summary.json'}")
    if all_stats_ood_finetuned:
        print(f"- summary_ood_finetuned: {output_dir / 'summary_ood_finetuned.json'}")
    if case_chart_paths:
        for ds_name in sorted(case_chart_paths.keys()):
            print(f"- top_bottom_cases_{ds_name}: {case_chart_paths[ds_name]}")
    if ood_summary_json.exists():
        print(f"- ood_train_test_counts_json: {ood_summary_json}")
    if ood_summary_csv.exists():
        print(f"- ood_train_test_counts_csv: {ood_summary_csv}")
    if ood_summary_chart is not None and ood_summary_chart.exists():
        print(f"- ood_train_test_counts_chart: {ood_summary_chart}")
    with _timed_log("Shutdown async writer"):
        shutdown_global_async_writer()
    print(f"\nPipeline total elapsed: {_fmt_elapsed(time.perf_counter() - pipeline_start)}")


if __name__ == "__main__":
    main()
