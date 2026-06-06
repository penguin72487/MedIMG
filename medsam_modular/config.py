import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_MODEL_ID = "facebook/sam-vit-base"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_OUTPUT_DIR_REL = "results/modular"
DEFAULT_CONFIG_FILENAME = "medsam_config.json"

_TRUE_SET = {"1", "true", "yes", "y", "on"}
_FALSE_SET = {"0", "false", "no", "n", "off"}


DEFAULT_SETTINGS: Dict[str, Any] = {
    "data_root": "",
    "tn3k_path": "",
    "tg3k_path": "",
    "ddti_path": "",
    "tn5000_path": "",
    "split_root": "",
    "model_id": DEFAULT_MODEL_ID,
    "weight_path": "",
    "image_size": DEFAULT_IMAGE_SIZE,
    "output_dir": "",
    "cuda_alloc_conf": "expandable_segments:True,max_split_size_mb:128",
    "amp_dtype": "bf16",
    "cuda_matmul_precision": "high",
    "cuda_allow_tf32": True,
    "cuda_cudnn_benchmark": True,
    "cuda_enable_flash_sdp": True,
    "cuda_enable_mem_efficient_sdp": True,
    "cuda_enable_math_sdp": True,
    "progress": False,
    "enable_compile": False,
    "enable_cuda_graph": False,
    "require_compile": False,
    "compile_dynamic": None,
    "compile_mode": "reduce-overhead",
    "compile_fullgraph": False,
    "compile_warmup_batches": "",
    "finetune_train_backbone": False,
    "finetune_epochs": 1000,
    "finetune_batch": 8,
    "finetune_lr": 1e-4,
    "finetune_weight_decay": 1e-3,
    "finetune_adamw_beta1": 0.9,
    "finetune_adamw_beta2": 0.999,
    "finetune_adamw_eps": 1e-8,
    "finetune_val_ratio": 0.1,
    "finetune_patience": 20,
    "finetune_min_epochs": 30,
    "finetune_min_delta": 1e-4,
    "finetune_use_plateau_scheduler": True,
    "finetune_plateau_factor": 0.5,
    "finetune_plateau_patience": 5,
    "finetune_plateau_cooldown": 2,
    "finetune_plateau_min_lr": 1e-6,
    "finetune_early_stop_require_min_lr": True,
    "finetune_grad_accum": 2,
    "finetune_grad_clip": 1.0,
    "finetune_workers": 0,
    "finetune_prefetch": 1,
    "finetune_safe_batch_emb": 2,
    "finetune_safe_batch_12gb": 1,
    "vram_limit_gb": 12.0,
    "low_vram_empty_cache_every": 1,
    "finetune_max_samples": 0,
    "finetune_use_fused_adamw": True,
    "finetune_optimizer": "adamw",
    "run_stage3_detect_train_ood": True,
    "run_stage4_ood_finetune": True,
    "run_stage5_full_finetune": True,
    "run_stage6_baseline_eval": True,
    "run_stage7_eval_ood_finetuned": True,
    "run_stage7_eval_full_finetuned": True,
    "run_stage8_plotting": True,
    "run_clinical_mode": False,
    "clinical_weight_path": "",
    "clinical_finetune_steps": 2,
    "clinical_finetune_lr": 5e-6,
    "clinical_finetune_weight_decay": 1e-4,
    "clinical_finetune_use_fused_adamw": True,
    "ood_threshold": 0.5,
    "ood_method": "entropy",
    "ood_enable_collapse_detection": True,
    "ood_collapse_max_prob_threshold": 0.5,
    "ood_enable_entropy_detection": True,
    "ood_entropy_threshold": 0.5,
    "ood_entropy_active_prob_threshold": 0.05,
    "ood_enable_fragmentation_detection": True,
    "ood_fragment_prob_threshold": 0.5,
    "ood_fragment_min_area": 80,
    "ood_fragment_max_large_components": 3,
    "eval_workers": 0,
    "eval_batch": 0,
    "eval_forward_chunk": 0,
    "eval_empty_cache_after_forward": True,
    "eval_input_cache": True,
    "cpu_threads": 0,
    "use_fast_preprocess": True,
    "tta_fusion": "entropy_weighted",
    "tta_augmentations": "",
    "tta_chunk_size": 8,
    "tta_fixed_batch": 0,
    "tta_build_on_cpu": False,
    "precompute_batch": 0,
    "precompute_workers": 0,
    "precompute_prefetch": 1,
    "precompute_gpu_chunk": 1,
    "precompute_empty_cache_every": 10,
    "precompute_embeddings": True,
    "emb_cache_keep_ram": False,
    "processor_cache": True,
    "cache_ram_entries": 256,
    "cache_async_write": True,
    "eval_warm_cache": True,
    "eval_warm_samples": 16,
    "eval_autobatch": True,
    "eval_autobatch_warmup_samples": 2,
    "eval_autobatch_max": 0,
    "eval_autobatch_bench_warmup": 1,
    "eval_autobatch_bench_rounds": 2,
    "eval_autobatch_candidate_growth": 2,
    "eval_autobatch_safety": 0.9,
    "ood_max_side": 64,
    "tta_autotune": True,
    "eval_prefetch": 4,
}


SETTING_ENV_MAP: Dict[str, str] = {
    "data_root": "MEDSAM_DATA_ROOT",
    "tn3k_path": "MEDSAM_TN3K_PATH",
    "tg3k_path": "MEDSAM_TG3K_PATH",
    "ddti_path": "MEDSAM_DDTI_PATH",
    "tn5000_path": "MEDSAM_TN5000_PATH",
    "split_root": "MEDSAM_SPLIT_ROOT",
    "model_id": "MEDSAM_MODEL_ID",
    "weight_path": "MEDSAM_WEIGHT_PATH",
    "image_size": "MEDSAM_IMAGE_SIZE",
    "output_dir": "MEDSAM_OUTPUT_DIR",
    "cuda_alloc_conf": "PYTORCH_CUDA_ALLOC_CONF",
    "amp_dtype": "MEDSAM_AMP_DTYPE",
    "cuda_matmul_precision": "MEDSAM_CUDA_MATMUL_PRECISION",
    "cuda_allow_tf32": "MEDSAM_CUDA_ALLOW_TF32",
    "cuda_cudnn_benchmark": "MEDSAM_CUDA_CUDNN_BENCHMARK",
    "cuda_enable_flash_sdp": "MEDSAM_CUDA_ENABLE_FLASH_SDP",
    "cuda_enable_mem_efficient_sdp": "MEDSAM_CUDA_ENABLE_MEM_EFFICIENT_SDP",
    "cuda_enable_math_sdp": "MEDSAM_CUDA_ENABLE_MATH_SDP",
    "progress": "MEDSAM_PROGRESS",
    "enable_compile": "MEDSAM_ENABLE_COMPILE",
    "enable_cuda_graph": "MEDSAM_ENABLE_CUDA_GRAPH",
    "require_compile": "MEDSAM_REQUIRE_COMPILE",
    "compile_dynamic": "MEDSAM_COMPILE_DYNAMIC",
    "compile_mode": "MEDSAM_COMPILE_MODE",
    "compile_fullgraph": "MEDSAM_COMPILE_FULLGRAPH",
    "compile_warmup_batches": "MEDSAM_COMPILE_WARMUP_BATCHES",
    "finetune_train_backbone": "MEDSAM_FINETUNE_TRAIN_BACKBONE",
    "finetune_epochs": "MEDSAM_FINETUNE_EPOCHS",
    "finetune_batch": "MEDSAM_FINETUNE_BATCH",
    "finetune_lr": "MEDSAM_FINETUNE_LR",
    "finetune_weight_decay": "MEDSAM_FINETUNE_WEIGHT_DECAY",
    "finetune_adamw_beta1": "MEDSAM_FINETUNE_ADAMW_BETA1",
    "finetune_adamw_beta2": "MEDSAM_FINETUNE_ADAMW_BETA2",
    "finetune_adamw_eps": "MEDSAM_FINETUNE_ADAMW_EPS",
    "finetune_val_ratio": "MEDSAM_FINETUNE_VAL_RATIO",
    "finetune_patience": "MEDSAM_FINETUNE_PATIENCE",
    "finetune_min_epochs": "MEDSAM_FINETUNE_MIN_EPOCHS",
    "finetune_min_delta": "MEDSAM_FINETUNE_MIN_DELTA",
    "finetune_use_plateau_scheduler": "MEDSAM_FINETUNE_USE_PLATEAU_SCHEDULER",
    "finetune_plateau_factor": "MEDSAM_FINETUNE_PLATEAU_FACTOR",
    "finetune_plateau_patience": "MEDSAM_FINETUNE_PLATEAU_PATIENCE",
    "finetune_plateau_cooldown": "MEDSAM_FINETUNE_PLATEAU_COOLDOWN",
    "finetune_plateau_min_lr": "MEDSAM_FINETUNE_PLATEAU_MIN_LR",
    "finetune_early_stop_require_min_lr": "MEDSAM_FINETUNE_EARLY_STOP_REQUIRE_MIN_LR",
    "finetune_grad_accum": "MEDSAM_FINETUNE_GRAD_ACCUM",
    "finetune_grad_clip": "MEDSAM_FINETUNE_GRAD_CLIP",
    "finetune_workers": "MEDSAM_FINETUNE_WORKERS",
    "finetune_prefetch": "MEDSAM_FINETUNE_PREFETCH",
    "finetune_safe_batch_emb": "MEDSAM_FINETUNE_SAFE_BATCH_EMB",
    "finetune_safe_batch_12gb": "MEDSAM_FINETUNE_SAFE_BATCH_12GB",
    "vram_limit_gb": "MEDSAM_VRAM_LIMIT_GB",
    "low_vram_empty_cache_every": "MEDSAM_LOW_VRAM_EMPTY_CACHE_EVERY",
    "finetune_max_samples": "MEDSAM_FINETUNE_MAX_SAMPLES",
    "finetune_use_fused_adamw": "MEDSAM_FINETUNE_USE_FUSED_ADAMW",
    "finetune_optimizer": "MEDSAM_FINETUNE_OPTIMIZER",
    "run_stage3_detect_train_ood": "MEDSAM_RUN_STAGE3_DETECT_TRAIN_OOD",
    "run_stage4_ood_finetune": "MEDSAM_RUN_STAGE4_OOD_FINETUNE",
    "run_stage5_full_finetune": "MEDSAM_RUN_STAGE5_FULL_FINETUNE",
    "run_stage6_baseline_eval": "MEDSAM_RUN_STAGE6_BASELINE_EVAL",
    "run_stage7_eval_ood_finetuned": "MEDSAM_RUN_STAGE7_EVAL_OOD_FINETUNED",
    "run_stage7_eval_full_finetuned": "MEDSAM_RUN_STAGE7_EVAL_FULL_FINETUNED",
    "run_stage8_plotting": "MEDSAM_RUN_STAGE8_PLOTTING",
    "run_clinical_mode": "MEDSAM_RUN_CLINICAL_MODE",
    "clinical_weight_path": "MEDSAM_CLINICAL_WEIGHT_PATH",
    "clinical_finetune_steps": "MEDSAM_CLINICAL_FINETUNE_STEPS",
    "clinical_finetune_lr": "MEDSAM_CLINICAL_FINETUNE_LR",
    "clinical_finetune_weight_decay": "MEDSAM_CLINICAL_FINETUNE_WEIGHT_DECAY",
    "clinical_finetune_use_fused_adamw": "MEDSAM_CLINICAL_FINETUNE_USE_FUSED_ADAMW",
    "ood_threshold": "MEDSAM_OOD_THRESHOLD",
    "ood_method": "MEDSAM_OOD_METHOD",
    "ood_enable_collapse_detection": "MEDSAM_OOD_ENABLE_COLLAPSE_DETECTION",
    "ood_collapse_max_prob_threshold": "MEDSAM_OOD_COLLAPSE_MAX_PROB_THRESHOLD",
    "ood_enable_entropy_detection": "MEDSAM_OOD_ENABLE_ENTROPY_DETECTION",
    "ood_entropy_threshold": "MEDSAM_OOD_ENTROPY_THRESHOLD",
    "ood_entropy_active_prob_threshold": "MEDSAM_OOD_ENTROPY_ACTIVE_PROB_THRESHOLD",
    "ood_enable_fragmentation_detection": "MEDSAM_OOD_ENABLE_FRAGMENTATION_DETECTION",
    "ood_fragment_prob_threshold": "MEDSAM_OOD_FRAGMENT_PROB_THRESHOLD",
    "ood_fragment_min_area": "MEDSAM_OOD_FRAGMENT_MIN_AREA",
    "ood_fragment_max_large_components": "MEDSAM_OOD_FRAGMENT_MAX_LARGE_COMPONENTS",
    "eval_workers": "MEDSAM_EVAL_WORKERS",
    "eval_batch": "MEDSAM_EVAL_BATCH",
    "eval_forward_chunk": "MEDSAM_EVAL_FORWARD_CHUNK",
    "eval_empty_cache_after_forward": "MEDSAM_EVAL_EMPTY_CACHE_AFTER_FORWARD",
    "eval_input_cache": "MEDSAM_EVAL_INPUT_CACHE",
    "cpu_threads": "MEDSAM_CPU_THREADS",
    "use_fast_preprocess": "MEDSAM_USE_FAST_PREPROCESS",
    "tta_fusion": "MEDSAM_TTA_FUSION",
    "tta_augmentations": "MEDSAM_TTA_AUGMENTATIONS",
    "tta_chunk_size": "MEDSAM_TTA_CHUNK_SIZE",
    "tta_fixed_batch": "MEDSAM_TTA_FIXED_BATCH",
    "tta_build_on_cpu": "MEDSAM_TTA_BUILD_ON_CPU",
    "precompute_batch": "MEDSAM_PRECOMPUTE_BATCH",
    "precompute_workers": "MEDSAM_PRECOMPUTE_WORKERS",
    "precompute_prefetch": "MEDSAM_PRECOMPUTE_PREFETCH",
    "precompute_gpu_chunk": "MEDSAM_PRECOMPUTE_GPU_CHUNK",
    "precompute_empty_cache_every": "MEDSAM_PRECOMPUTE_EMPTY_CACHE_EVERY",
    "precompute_embeddings": "MEDSAM_PRECOMPUTE_EMBEDDINGS",
    "emb_cache_keep_ram": "MEDSAM_EMB_CACHE_KEEP_RAM",
    "processor_cache": "MEDSAM_PROCESSOR_CACHE",
    "cache_ram_entries": "MEDSAM_CACHE_RAM_ENTRIES",
    "cache_async_write": "MEDSAM_CACHE_ASYNC_WRITE",
    "eval_warm_cache": "MEDSAM_EVAL_WARM_CACHE",
    "eval_warm_samples": "MEDSAM_EVAL_WARM_SAMPLES",
    "eval_autobatch": "MEDSAM_EVAL_AUTOBATCH",
    "eval_autobatch_warmup_samples": "MEDSAM_EVAL_AUTOBATCH_WARMUP_SAMPLES",
    "eval_autobatch_max": "MEDSAM_EVAL_AUTOBATCH_MAX",
    "eval_autobatch_bench_warmup": "MEDSAM_EVAL_AUTOBATCH_BENCH_WARMUP",
    "eval_autobatch_bench_rounds": "MEDSAM_EVAL_AUTOBATCH_BENCH_ROUNDS",
    "eval_autobatch_candidate_growth": "MEDSAM_EVAL_AUTOBATCH_CANDIDATE_GROWTH",
    "eval_autobatch_safety": "MEDSAM_EVAL_AUTOBATCH_SAFETY",
    "ood_max_side": "MEDSAM_OOD_MAX_SIDE",
    "tta_autotune": "MEDSAM_TTA_AUTOTUNE",
    "eval_prefetch": "MEDSAM_EVAL_PREFETCH",
}


BOOL_SETTING_KEYS = {
    "require_compile",
    "enable_compile",
    "enable_cuda_graph",
    "compile_fullgraph",
    "cuda_allow_tf32",
    "cuda_cudnn_benchmark",
    "cuda_enable_flash_sdp",
    "cuda_enable_mem_efficient_sdp",
    "cuda_enable_math_sdp",
    "finetune_train_backbone",
    "progress",
    "finetune_use_plateau_scheduler",
    "finetune_early_stop_require_min_lr",
    "finetune_use_fused_adamw",
    "run_stage3_detect_train_ood",
    "run_stage4_ood_finetune",
    "run_stage5_full_finetune",
    "run_stage6_baseline_eval",
    "run_stage7_eval_ood_finetuned",
    "run_stage7_eval_full_finetuned",
    "run_stage8_plotting",
    "run_clinical_mode",
    "clinical_finetune_use_fused_adamw",
    "ood_enable_collapse_detection",
    "ood_enable_entropy_detection",
    "ood_enable_fragmentation_detection",
    "cache_async_write",
    "precompute_embeddings",
    "emb_cache_keep_ram",
    "processor_cache",
    "eval_input_cache",
    "eval_empty_cache_after_forward",
    "use_fast_preprocess",
    "eval_warm_cache",
    "eval_autobatch",
    "tta_autotune",
    "tta_build_on_cpu",
}

INT_SETTING_KEYS = {
    "image_size",
    "finetune_epochs",
    "finetune_batch",
    "finetune_patience",
    "finetune_min_epochs",
    "finetune_plateau_patience",
    "finetune_plateau_cooldown",
    "finetune_grad_accum",
    "finetune_workers",
    "finetune_prefetch",
    "finetune_safe_batch_emb",
    "finetune_safe_batch_12gb",
    "low_vram_empty_cache_every",
    "finetune_max_samples",
    "eval_workers",
    "eval_batch",
    "eval_forward_chunk",
    "cpu_threads",
    "tta_chunk_size",
    "tta_fixed_batch",
    "precompute_batch",
    "precompute_workers",
    "precompute_prefetch",
    "precompute_gpu_chunk",
    "precompute_empty_cache_every",
    "cache_ram_entries",
    "eval_warm_samples",
    "eval_autobatch_warmup_samples",
    "eval_autobatch_max",
    "eval_autobatch_bench_warmup",
    "eval_autobatch_bench_rounds",
    "eval_autobatch_candidate_growth",
    "ood_max_side",
    "ood_fragment_min_area",
    "ood_fragment_max_large_components",
    "eval_prefetch",
    "clinical_finetune_steps",
}

FLOAT_SETTING_KEYS = {
    "finetune_lr",
    "finetune_weight_decay",
    "finetune_adamw_beta1",
    "finetune_adamw_beta2",
    "finetune_adamw_eps",
    "finetune_val_ratio",
    "finetune_min_delta",
    "finetune_plateau_factor",
    "finetune_plateau_min_lr",
    "finetune_grad_clip",
    "vram_limit_gb",
    "ood_threshold",
    "ood_collapse_max_prob_threshold",
    "ood_entropy_threshold",
    "ood_entropy_active_prob_threshold",
    "ood_fragment_prob_threshold",
    "eval_autobatch_safety",
    "clinical_finetune_lr",
    "clinical_finetune_weight_decay",
}


def default_config_path(project_root: Path) -> Path:
    return project_root / DEFAULT_CONFIG_FILENAME


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_SET:
            return True
        if lowered in _FALSE_SET:
            return False
    raise ValueError(f"invalid bool value: {value!r}")


def _coerce_setting_value(key: str, value: Any) -> Any:
    if key == "compile_dynamic":
        if value is None:
            return None
        return _coerce_bool(value)
    if key in BOOL_SETTING_KEYS:
        return _coerce_bool(value)
    if key in INT_SETTING_KEYS:
        return int(value)
    if key in FLOAT_SETTING_KEYS:
        return float(value)
    return str(value)


def load_settings(project_root: Path, config_path: Optional[Path] = None) -> Tuple[Dict[str, Any], Path]:
    resolved_path = (config_path or default_config_path(project_root)).expanduser()
    data: Dict[str, Any] = {}

    if resolved_path.exists():
        with resolved_path.open("r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if not isinstance(loaded, dict):
            raise ValueError(f"config must be a JSON object: {resolved_path}")
        data = loaded

    settings = dict(DEFAULT_SETTINGS)
    for key, value in data.items():
        if key not in settings:
            continue
        settings[key] = _coerce_setting_value(key, value)

    return settings, resolved_path


def settings_to_env(settings: Dict[str, Any]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for key, env_name in SETTING_ENV_MAP.items():
        if key not in settings:
            continue
        value = settings[key]
        if value is None:
            continue
        if key in BOOL_SETTING_KEYS:
            env[env_name] = "1" if bool(value) else "0"
            continue
        env[env_name] = str(value)
    return env


ENV_DEFAULTS = settings_to_env(DEFAULT_SETTINGS)
