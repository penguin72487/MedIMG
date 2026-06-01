import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torch.nn.functional as F
from tqdm import tqdm

from medsam_modular.config import ENV_DEFAULTS
from medsam_modular.data import compute_bbox_from_mask_np, prepare_datasets_by_split
from medsam_modular.model import load_state_dict_compat, normalize_pred_masks_to_4d


class _AsyncCheckpointSaver:
    """非同步檢查點保存器（GPU訓練繼續，I/O後台運行）"""
    def __init__(self):
        self._save_thread: Optional[threading.Thread] = None
        self._pending_save = False

    def save_async(self, state_dict: Dict[str, Any], path: Path) -> None:
        """後台保存權重，訓練繼續不等待"""
        if self._save_thread is not None:
            self._save_thread.join()
        
        def _save_worker():
            torch.save(state_dict, path)
        
        self._save_thread = threading.Thread(target=_save_worker, daemon=False)
        self._save_thread.start()
        self._pending_save = True

    def wait_for_save(self) -> None:
        """等待待定的儲存完成（在 epoch 結束時呼叫）"""
        if self._save_thread is not None and self._pending_save:
            self._save_thread.join()
            self._pending_save = False


def _env_bool(name: str, default: bool = False) -> bool:
    """讀取環境變數作為布林值"""
    raw_default = ENV_DEFAULTS.get(name, "1" if default else "0")
    raw = os.getenv(name, raw_default).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _cuda_total_memory_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        return float(props.total_memory) / (1024.0 ** 3)
    except Exception:
        return None


def _cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _auto_train_workers(device: str) -> int:
    _ = device
    return _cpu_count()


def _setup_cuda_backends() -> None:
    """Enable Tensor Core optimizations (TF32 + cuDNN benchmark)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _choose_amp_dtype() -> torch.dtype:
    """BF16 on Ampere+ (no GradScaler needed), FP16 elsewhere."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _precompute_image_embeddings(
    model: Any,
    dataset: Dataset,
    device: str,
    batch_size: int = 4,
    amp_dtype: torch.dtype = torch.float16,
    num_workers: int = 0,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Run the frozen ViT once for every sample and cache FP16 embeddings in CPU RAM.

    Returns dict {dataset_index -> cached tensors}.
    Calling this before the training loop removes ~95% of per-epoch GPU compute
    (the frozen vision encoder accounts for ~95% of SAM-ViT-B's FLOPs).
    """
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is None:
        return {}

    was_training = model.training
    model.eval()

    def _pv_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
            "input_boxes": torch.stack([item["input_boxes"] for item in batch], dim=0),
            "original_sizes": torch.stack([item["original_sizes"] for item in batch], dim=0),
            "reshaped_input_sizes": torch.stack([item["reshaped_input_sizes"] for item in batch], dim=0),
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=_pv_collate,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    embeddings: Dict[int, Dict[str, torch.Tensor]] = {}
    global_idx = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="  Pre-computing ViT embeddings", leave=False, unit="batch"):
            pv_batch = batch["pixel_values"].to(device, non_blocking=(device == "cuda"))
            try:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                    emb = vision_encoder(pv_batch)[0]  # [B, 256, 64, 64]
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device == "cuda":
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during embedding precompute. "
                        "Try smaller MEDSAM_PRECOMPUTE_BATCH (e.g. 2 or 1) "
                        "and/or MEDSAM_FINETUNE_BATCH (e.g. 2)."
                    ) from exc
                raise
            emb_cpu = emb.detach().to(dtype=torch.float16).cpu()
            boxes_cpu = batch["input_boxes"].to(dtype=torch.float32).cpu()
            orig_cpu = batch["original_sizes"].cpu()
            reshaped_cpu = batch["reshaped_input_sizes"].cpu()

            bsz = emb_cpu.shape[0]
            for i in range(bsz):
                embeddings[global_idx + i] = {
                    "image_embedding": emb_cpu[i],
                    "input_boxes": boxes_cpu[i],
                    "original_sizes": orig_cpu[i],
                    "reshaped_input_sizes": reshaped_cpu[i],
                }
            global_idx += bsz

    if was_training:
        model.train()
    if device == "cuda":
        torch.cuda.empty_cache()
    return embeddings


class FinetuneEmbeddingDataset(Dataset):
    """Wraps FinetuneProcessorDataset; substitutes pixel_values with pre-computed embedding."""

    def __init__(self, base: Dataset, embeddings: Dict[int, Dict[str, torch.Tensor]]) -> None:
        self.base = base
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cached = self.embeddings.get(idx)
        if cached is None:
            # Fallback path (should be rare): use original processor-based sample.
            return dict(self.base[idx])

        if hasattr(self.base, "base_dataset"):
            raw = self.base.base_dataset[idx]
            name = str(raw.get("name", f"sample_{idx}"))
            gt_mask = raw["mask"].float()
        else:
            item = dict(self.base[idx])
            item.pop("pixel_values", None)
            return item

        return {
            "image_embedding": cached["image_embedding"],
            "input_boxes": cached["input_boxes"],
            "original_sizes": cached["original_sizes"],
            "reshaped_input_sizes": cached["reshaped_input_sizes"],
            "gt_mask": gt_mask,
            "name": name,
        }


class FinetuneProcessorDataset(Dataset):
    def __init__(self, base_dataset: Dataset, processor: Any):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        image = sample["image"]
        bbox = sample["bbox"]
        mask = sample["mask"]

        inputs = self.processor(images=[image], input_boxes=[[bbox]], return_tensors="pt")
        packed = {k: v.squeeze(0) for k, v in inputs.items()}
        packed["gt_mask"] = mask.float()
        packed["name"] = str(sample.get("name", f"sample_{idx}"))
        return packed


class NameFilteredDataset(Dataset):
    """Filter a dataset by sample names while preserving original sample payload."""

    def __init__(self, base: Dataset, keep_names: Set[str]) -> None:
        self.base = base
        self.indices: List[int] = []
        for idx in range(len(base)):
            sample_name = self._extract_name(base, idx)
            if sample_name in keep_names:
                self.indices.append(idx)

    @staticmethod
    def _extract_name(base: Dataset, idx: int) -> str:
        samples = getattr(base, "samples", None)
        if isinstance(samples, list) and 0 <= idx < len(samples):
            entry = samples[idx]
            if isinstance(entry, dict):
                if "name" in entry:
                    return str(entry["name"])
                if "image_id" in entry:
                    return str(entry["image_id"])

        item = base[idx]
        return str(item.get("name", f"sample_{idx}"))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.base[self.indices[idx]]


class TTAAugmentedRawDataset(Dataset):
    """Deterministically expands raw train samples with TTA-style augmentations."""

    def __init__(self, base: Dataset, augmentations: List[str]) -> None:
        self.base = base
        self.augmentations = self._canonicalize(augmentations)
        self.base_len = len(base)

    @staticmethod
    def _canonicalize(augmentations: List[str]) -> List[str]:
        canonical_map = {"rotate_180": "hvflip"}
        valid = {"none", "hflip", "vflip", "hvflip", "rotate_90", "rotate_270"}
        out: List[str] = []
        seen: Set[str] = set()
        for aug in augmentations:
            name = canonical_map.get(str(aug).strip(), str(aug).strip())
            if name not in valid:
                continue
            if name not in seen:
                out.append(name)
                seen.add(name)
        return out or ["none"]

    def __len__(self) -> int:
        return self.base_len * len(self.augmentations)

    def _apply_aug(self, sample: Dict[str, Any], aug: str) -> Dict[str, Any]:
        image = sample["image"]
        mask_t = sample["mask"] if isinstance(sample["mask"], torch.Tensor) else torch.as_tensor(sample["mask"])
        mask_t = mask_t.to(dtype=torch.float32)

        if aug == "hflip":
            image = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            mask_t = torch.flip(mask_t, dims=[1])
        elif aug == "vflip":
            image = image.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
            mask_t = torch.flip(mask_t, dims=[0])
        elif aug == "hvflip":
            image = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
            mask_t = torch.flip(mask_t, dims=[0, 1])
        elif aug == "rotate_90":
            image = image.transpose(method=Image.Transpose.ROTATE_90)
            mask_t = torch.rot90(mask_t, k=1, dims=[0, 1])
        elif aug == "rotate_270":
            image = image.transpose(method=Image.Transpose.ROTATE_270)
            mask_t = torch.rot90(mask_t, k=3, dims=[0, 1])

        mask_np = (mask_t.detach().cpu().numpy() >= 0.5).astype(np.uint8)
        bbox = compute_bbox_from_mask_np(mask_np)
        name = str(sample.get("name", "sample"))
        return {
            "image": image,
            "mask": mask_t,
            "bbox": bbox,
            "name": f"{name}__aug_{aug}",
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = idx % self.base_len
        aug_idx = idx // self.base_len
        aug = self.augmentations[aug_idx]
        sample = self.base[base_idx]
        if aug == "none":
            keep = dict(sample)
            keep_name = str(keep.get("name", f"sample_{base_idx}"))
            keep["name"] = f"{keep_name}__aug_none"
            return keep
        return self._apply_aug(sample, aug)


def _finetune_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated: Dict[str, Any] = {}
    # Pre-computed embedding path: no pixel_values, use image_embedding instead
    if "image_embedding" in batch[0]:
        collated["image_embedding"] = torch.stack([item["image_embedding"] for item in batch], dim=0)
    else:
        collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch], dim=0)
    for k in ["input_boxes", "original_sizes", "reshaped_input_sizes"]:
        collated[k] = torch.stack([item[k] for item in batch], dim=0)
    collated["gt_mask"] = torch.stack([item["gt_mask"] for item in batch], dim=0)
    collated["name"] = [item["name"] for item in batch]
    return collated


def _build_finetune_datasets(config: Dict[str, Any], processor: Any) -> Tuple[Dataset, Dataset]:
    split_root = config["split_root"]
    image_size = int(config["image_size"])
    data_paths = config["data_paths"]

    train_sets = prepare_datasets_by_split(
        data_paths=data_paths,
        split_root=split_root,
        split_name="train",
        image_size=image_size,
    )
    val_sets = prepare_datasets_by_split(
        data_paths=data_paths,
        split_root=split_root,
        split_name="val",
        image_size=image_size,
    )

    subset_by_name_raw = config.get("finetune_subset_by_name", {})
    subset_by_name: Dict[str, Set[str]] = {}
    if isinstance(subset_by_name_raw, dict):
        for ds_name, values in subset_by_name_raw.items():
            if values is None:
                continue
            subset_by_name[str(ds_name)] = {str(v) for v in values}

    filtered_train_parts: List[Dataset] = []
    for ds_name, ds in train_sets.items():
        if len(ds) == 0:
            continue
        keep_names = subset_by_name.get(ds_name)
        if keep_names is None:
            filtered_train_parts.append(ds)
            continue
        filtered = NameFilteredDataset(ds, keep_names)
        if len(filtered) > 0:
            filtered_train_parts.append(filtered)

    train_concat = ConcatDataset(filtered_train_parts)
    if len(train_concat) == 0:
        raise RuntimeError("No train samples found for fine-tune. Check split files and dataset paths.")

    val_non_empty = [ds for ds in val_sets.values() if len(ds) > 0]
    val_concat = ConcatDataset(val_non_empty) if val_non_empty else None

    if val_concat is None or len(val_concat) == 0:
        val_ratio = float(config.get("finetune_val_ratio", 0.1))
        val_ratio = float(np.clip(val_ratio, 0.01, 0.5))
        val_size = max(1, int(round(len(train_concat) * val_ratio)))
        val_size = min(val_size, max(1, len(train_concat) - 1))
        train_size = len(train_concat) - val_size
        train_concat, val_concat = random_split(
            train_concat,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    train_raw: Dataset = train_concat
    use_tta_aug = _env_bool("MEDSAM_FINETUNE_USE_TTA_AUG", False)
    cfg_tta_aug = config.get("finetune_use_tta_augment", use_tta_aug)
    if isinstance(cfg_tta_aug, bool):
        use_tta_aug = cfg_tta_aug
    elif cfg_tta_aug is not None:
        use_tta_aug = str(cfg_tta_aug).strip().lower() in {"1", "true", "yes", "y", "on"}
    tta_augs_cfg = config.get("finetune_tta_augmentations", ["none", "hflip", "vflip", "hvflip"])
    if isinstance(tta_augs_cfg, str):
        tta_augs = [v.strip() for v in tta_augs_cfg.split(",") if v.strip()]
    elif isinstance(tta_augs_cfg, list):
        tta_augs = [str(v).strip() for v in tta_augs_cfg if str(v).strip()]
    else:
        tta_augs = ["none", "hflip", "vflip", "hvflip"]

    if use_tta_aug:
        train_raw = TTAAugmentedRawDataset(train_concat, tta_augs)

    return FinetuneProcessorDataset(train_raw, processor), FinetuneProcessorDataset(val_concat, processor)


def _configure_trainable_params(model: Any, train_backbone: bool) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "mask_decoder"):
        for p in model.mask_decoder.parameters():
            p.requires_grad = True

    if hasattr(model, "prompt_encoder"):
        for p in model.prompt_encoder.parameters():
            p.requires_grad = True

    if train_backbone and hasattr(model, "vision_encoder"):
        for p in model.vision_encoder.parameters():
            p.requires_grad = True


def _move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    non_blocking = device == "cuda"
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if k == "input_boxes" and v.dtype == torch.float64:
                v = v.to(torch.float32)
            moved[k] = v.to(device, non_blocking=non_blocking)
        else:
            moved[k] = v
    return moved


def _dice_loss(probs: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """L_Dice = 1 - (2 * Σ(g*s)) / (Σ(g²) + Σ(s²))  (paper Section 3.3)."""
    flat_p = probs.reshape(probs.shape[0], -1)
    flat_t = target.reshape(target.shape[0], -1).to(probs.dtype)
    numerator = 2.0 * (flat_p * flat_t).sum(dim=1) + smooth
    denominator = flat_p.pow(2).sum(dim=1) + flat_t.pow(2).sum(dim=1) + smooth
    return (1.0 - numerator / denominator).mean()


def _compute_seg_loss(outputs: Any, gt_mask: torch.Tensor) -> torch.Tensor:
    """L = L_BCE + L_Dice  (paper Section 3.3)."""
    logits = normalize_pred_masks_to_4d(outputs.pred_masks)
    target = gt_mask.unsqueeze(1)
    target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    l_bce = F.binary_cross_entropy_with_logits(logits, target)
    l_dice = _dice_loss(torch.sigmoid(logits), target)
    return l_bce + l_dice


def _compute_batch_dice(outputs: Any, gt_mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Hard Dice at threshold 0.5 for logging/monitoring."""
    logits = normalize_pred_masks_to_4d(outputs.pred_masks)
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).to(dtype=torch.float32)
    target = gt_mask.unsqueeze(1)
    target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    target = (target >= 0.5).to(dtype=torch.float32)

    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


def _build_adamw_param_groups(model: torch.nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_norm_or_scale = p.ndim <= 1
        if is_bias or is_norm_or_scale:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def _build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    epoch: int,
    best_val_loss: float,
    wait: int,    history: Dict[str, List[float]],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "wait": int(wait),
        "history": history,
    }


def maybe_finetune(model: Any, processor: Any, config: Dict[str, Any], profiler: Optional[Any] = None) -> Any:
    def _get_bool(key: str, default: bool = False) -> bool:
        """從config字典讀取布林值"""
        val = config.get(key, default)
        if isinstance(val, bool):
            return val
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

    skip = _get_bool("skip_finetune", True)
    if skip:
        print("⏭️ 跳過 fine-tune（skip_finetune 啟用）")
        return model

    device = str(config["device"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tensor Core / cuDNN optimizations ─────────────────────────────────
    if device == "cuda":
        _setup_cuda_backends()
    amp_dtype = _choose_amp_dtype() if device == "cuda" else torch.float32
    use_scaler = (device == "cuda") and (amp_dtype == torch.float16)
    # ──────────────────────────────────────────────────────────────────────

    train_backbone = _get_bool("finetune_train_backbone", False)
    epochs = int(config.get("finetune_epochs", 100))
    batch_size = int(config.get("finetune_batch", 8))
    lr = float(config.get("finetune_lr", 1e-4))
    weight_decay = float(config.get("finetune_weight_decay", 1e-3))
    adamw_beta1 = float(config.get("finetune_adamw_beta1", 0.9))
    adamw_beta2 = float(config.get("finetune_adamw_beta2", 0.999))
    adamw_eps = float(config.get("finetune_adamw_eps", 1e-8))
    patience = int(config.get("finetune_patience", 20))
    min_delta = float(config.get("finetune_min_delta", 1e-4))
    grad_accum = max(1, int(config.get("finetune_grad_accum", 2)))
    grad_clip = float(config.get("finetune_grad_clip", 1.0))
    min_epochs = int(config.get("finetune_min_epochs", 30))
    raw_workers = int(config.get("finetune_workers", 0))
    num_workers = _auto_train_workers(device) if raw_workers <= 0 else raw_workers
    max_samples = int(config.get("finetune_max_samples", 0))
    use_fused_adamw = _get_bool("finetune_use_fused_adamw", True)
    use_plateau_scheduler = _get_bool("finetune_use_plateau_scheduler", True)
    plateau_factor = float(config.get("finetune_plateau_factor", 0.5))
    plateau_patience = int(config.get("finetune_plateau_patience", 5))
    plateau_cooldown = int(config.get("finetune_plateau_cooldown", 2))
    plateau_min_lr = float(config.get("finetune_plateau_min_lr", 1e-6))
    early_stop_require_min_lr = _get_bool("finetune_early_stop_require_min_lr", True)
    resume_weight_path_raw = str(config.get("resume_weight_path", "")).strip()
    resume_weight_path = Path(resume_weight_path_raw) if resume_weight_path_raw else None
    precompute_batch = int(os.getenv("MEDSAM_PRECOMPUTE_BATCH", ENV_DEFAULTS["MEDSAM_PRECOMPUTE_BATCH"]))
    precompute_workers = int(os.getenv("MEDSAM_PRECOMPUTE_WORKERS", ENV_DEFAULTS["MEDSAM_PRECOMPUTE_WORKERS"]))
    cuda_mem_gb = _cuda_total_memory_gb() if device == "cuda" else None
    low_vram_mode = bool(cuda_mem_gb is not None and cuda_mem_gb <= 12.5)
    use_emb_cache = device == "cuda" and not train_backbone and _env_bool("MEDSAM_PRECOMPUTE_EMBEDDINGS", True)

    if use_emb_cache and precompute_batch <= 0:
        if cuda_mem_gb is None:
            precompute_batch = 4
        elif cuda_mem_gb <= 12.5:
            precompute_batch = 2
        elif cuda_mem_gb <= 24.5:
            precompute_batch = 16
        else:
            precompute_batch = 24

    if precompute_workers <= 0:
        precompute_workers = num_workers
    if low_vram_mode and use_emb_cache:
        # Embedding cache removes ViT from the forward pass → decoder-only VRAM → larger batch
        safe_batch_emb = int(config.get("finetune_safe_batch_emb", 2))
        safe_batch_emb = max(1, safe_batch_emb)
        if batch_size > safe_batch_emb:
            print(f"⚠️ Low-VRAM mode ({cuda_mem_gb:.1f}GB): batch size {batch_size} -> {safe_batch_emb}")
            batch_size = safe_batch_emb
        elif batch_size <= 1 and safe_batch_emb >= 2:
            batch_size = 2
            print("  Embedding cache: batch size raised to 2 (safe low-VRAM setting)")
    elif low_vram_mode:
        safe_batch_12gb = int(config.get("finetune_safe_batch_12gb", 2))
        safe_batch_12gb = max(1, safe_batch_12gb)
        if batch_size > safe_batch_12gb:
            print(f"⚠️ Low-VRAM mode ({cuda_mem_gb:.1f}GB): batch size {batch_size} -> {safe_batch_12gb}")
            batch_size = safe_batch_12gb

    if low_vram_mode and precompute_batch > 2:
        precompute_batch = 2

    ft_total_start = time.perf_counter()
    train_data_move_total = 0.0
    train_forward_total = 0.0
    train_backward_total = 0.0
    train_optimizer_total = 0.0
    val_forward_total = 0.0

    print("=" * 80)
    print("[1/4] 準備資料集 ...")
    print("=" * 80)
    t0 = time.time()

    train_dataset, val_dataset = _build_finetune_datasets(config=config, processor=processor)

    if max_samples > 0 and len(train_dataset) > max_samples:
        train_dataset, _ = random_split(
            train_dataset,
            [max_samples, len(train_dataset) - max_samples],
            generator=torch.Generator().manual_seed(42),
        )

    print(f"  train samples : {len(train_dataset)}")
    print(f"  val   samples : {len(val_dataset)}")
    print(f"  batch size    : {batch_size}")
    print(f"  epochs        : {epochs}  (patience={patience})")
    print(f"  min epochs    : {min_epochs}")
    print(f"  lr            : {lr}")
    print(f"  adamw         : wd={weight_decay}, betas=({adamw_beta1},{adamw_beta2}), eps={adamw_eps}")
    print(f"  grad_accum    : {grad_accum}")
    print(f"  train backbone: {train_backbone}")
    print(f"  amp dtype     : {amp_dtype}")
    print(f"  emb cache     : {use_emb_cache}")
    print(f"  scheduler     : {'ReduceLROnPlateau' if use_plateau_scheduler else 'off'}")
    if use_emb_cache:
        print(f"  precompute    : batch={precompute_batch}, workers={precompute_workers}")
    print(f"  資料準備耗時  : {time.time() - t0:.1f}s")

    val_batch_size = batch_size if low_vram_mode else min(batch_size * 4, max(1, len(val_dataset) // 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=_finetune_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=_finetune_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    compiled_wrapped = hasattr(model, "_orig_mod") and getattr(model, "_orig_mod") is not None
    if compiled_wrapped:
        base_model = model._orig_mod
        # Fine-tune 時解除 compile 包裝，避免額外顯存佔用
        model = base_model
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        base_model = model

    _configure_trainable_params(base_model, train_backbone=train_backbone)

    # ── Pre-compute image embeddings (skip frozen ViT every epoch) ────────
    if use_emb_cache:
        print("  🚀 Pre-computing image embeddings for train + val sets ...")
        t_emb = time.time()
        train_embs = _precompute_image_embeddings(
            base_model,
            train_dataset,
            device,
            batch_size=precompute_batch,
            amp_dtype=amp_dtype,
            num_workers=precompute_workers,
        )
        val_embs = _precompute_image_embeddings(
            base_model,
            val_dataset,
            device,
            batch_size=precompute_batch,
            amp_dtype=amp_dtype,
            num_workers=precompute_workers,
        )
        train_dataset = FinetuneEmbeddingDataset(train_dataset, train_embs)
        val_dataset   = FinetuneEmbeddingDataset(val_dataset,   val_embs)
        del train_embs, val_embs
        if device == "cuda":
            torch.cuda.empty_cache()
        emb_mem_gb = len(train_dataset) * 256 * 64 * 64 * 2 / 1024**3
        print(f"  Pre-compute done ({time.time()-t_emb:.1f}s) | ~{emb_mem_gb:.1f}GB CPU RAM used")
        # Rebuild DataLoaders with updated (wrapped) datasets
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device == "cuda"),
            drop_last=False, collate_fn=_finetune_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"),
            drop_last=False, collate_fn=_finetune_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
    # ──────────────────────────────────────────────────────────────────────

    base_model.train()

    params = [p for p in base_model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in params)
    total_count = sum(p.numel() for p in base_model.parameters())
    print(f"\n[2/4] 設定優化器 ...")
    print(f"  可訓練參數: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)")

    param_groups = _build_adamw_param_groups(base_model, weight_decay=weight_decay)
    optimizer_kwargs = {
        "lr": lr,
        "betas": (adamw_beta1, adamw_beta2),
        "eps": adamw_eps,
    }
    if device == "cuda":
        optimizer_kwargs["fused"] = use_fused_adamw
    optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
    scheduler = None
    if use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=min_delta,
            threshold_mode="abs",
            cooldown=plateau_cooldown,
            min_lr=plateau_min_lr,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val_loss = float("inf")
    weight_prefix = str(config.get("finetune_weight_prefix", "medsam_finetuned")).strip() or "medsam_finetuned"
    stats_prefix = str(config.get("finetune_stats_prefix", "finetune")).strip() or "finetune"
    best_path = output_dir / f"{weight_prefix}_best.pth"
    last_path = output_dir / f"{weight_prefix}_last.pth"
    async_saver = _AsyncCheckpointSaver()

    wait = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "lr": [],
    }
    epoch_durations_sec: List[float] = []
    start_epoch = 1

    if resume_weight_path is not None and resume_weight_path.exists():
        ckpt = torch.load(resume_weight_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "best_val_loss" in ckpt:
                try:
                    best_val_loss = float(ckpt["best_val_loss"])
                except (TypeError, ValueError):
                    best_val_loss = float("inf")
            if "wait" in ckpt:
                try:
                    wait = int(ckpt["wait"])
                except (TypeError, ValueError):
                    wait = 0
            if "epoch" in ckpt:
                try:
                    start_epoch = int(ckpt["epoch"]) + 1
                except (TypeError, ValueError):
                    start_epoch = 1
            if isinstance(ckpt.get("history"), dict):
                hist = ckpt["history"]
                for k in ["train_loss", "val_loss", "train_dice", "val_dice", "lr"]:
                    if isinstance(hist.get(k), list):
                        history[k] = list(hist[k])

        load_state_dict_compat(base_model, resume_weight_path, map_location=device)
        print(
            f"  🔁 Resume checkpoint: {resume_weight_path} | "
            f"start_epoch={start_epoch} best_val={best_val_loss:.6f} wait={wait}"
        )

    print(f"\n[3/4] 開始訓練 (共 {epochs} epochs) ...")
    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="Epoch", unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:
        base_model.train()
        train_losses: List[float] = []
        train_dices: List[float] = []
        optimizer.zero_grad(set_to_none=True)
        epoch_t0 = time.time()

        train_bar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"  Train",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
        )
        for step, batch in train_bar:
            t_data = time.perf_counter()
            batch = _move_batch_to_device(batch, device)
            train_data_move_total += (time.perf_counter() - t_data)
            if "image_embedding" in batch:
                model_inputs = {
                    "image_embeddings": batch["image_embedding"],
                    "input_boxes": batch["input_boxes"],
                    "original_sizes": batch["original_sizes"],
                    "reshaped_input_sizes": batch["reshaped_input_sizes"],
                }
            else:
                model_inputs = {
                    "pixel_values": batch["pixel_values"],
                    "input_boxes": batch["input_boxes"],
                    "original_sizes": batch["original_sizes"],
                    "reshaped_input_sizes": batch["reshaped_input_sizes"],
                }

            t_forward = time.perf_counter()
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = base_model(**model_inputs)
                    loss = _compute_seg_loss(outputs, batch["gt_mask"]) / grad_accum
            else:
                outputs = base_model(**model_inputs)
                loss = _compute_seg_loss(outputs, batch["gt_mask"]) / grad_accum
            batch_dice = _compute_batch_dice(outputs, batch["gt_mask"])
            train_forward_total += (time.perf_counter() - t_forward)

            t_backward = time.perf_counter()
            scaler.scale(loss).backward()
            train_backward_total += (time.perf_counter() - t_backward)
            cur_loss = float(loss.item() * grad_accum)
            train_losses.append(cur_loss)
            train_dices.append(batch_dice)
            train_bar.set_postfix(loss=f"{cur_loss:.4f}", dice=f"{batch_dice:.4f}")

            if step % grad_accum == 0 or step == len(train_loader):
                t_opt = time.perf_counter()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                train_optimizer_total += (time.perf_counter() - t_opt)

        base_model.eval()
        val_losses: List[float] = []
        val_dices: List[float] = []
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"  Val  ",
                leave=False,
                dynamic_ncols=True,
                unit="batch",
            )
            for batch in val_bar:
                batch = _move_batch_to_device(batch, device)
                if "image_embedding" in batch:
                    model_inputs = {
                        "image_embeddings": batch["image_embedding"],
                        "input_boxes": batch["input_boxes"],
                        "original_sizes": batch["original_sizes"],
                        "reshaped_input_sizes": batch["reshaped_input_sizes"],
                    }
                else:
                    model_inputs = {
                        "pixel_values": batch["pixel_values"],
                        "input_boxes": batch["input_boxes"],
                        "original_sizes": batch["original_sizes"],
                        "reshaped_input_sizes": batch["reshaped_input_sizes"],
                    }

                t_val_forward = time.perf_counter()
                if device == "cuda":
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        outputs = base_model(**model_inputs)
                        val_loss = _compute_seg_loss(outputs, batch["gt_mask"])
                else:
                    outputs = base_model(**model_inputs)
                    val_loss = _compute_seg_loss(outputs, batch["gt_mask"])
                val_dice = _compute_batch_dice(outputs, batch["gt_mask"])
                val_forward_total += (time.perf_counter() - t_val_forward)

                val_bar.set_postfix(loss=f"{float(val_loss.item()):.4f}", dice=f"{val_dice:.4f}")
                val_losses.append(float(val_loss.item()))
                val_dices.append(val_dice)

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        train_dice = float(np.mean(train_dices)) if train_dices else 0.0
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

        if low_vram_mode and device == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - epoch_t0
        epoch_durations_sec.append(float(elapsed))
        improved_mark = "★" if val_loss < (best_val_loss - min_delta) else " "
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            tDice=f"{train_dice:.4f}",
            vDice=f"{val_dice:.4f}",
            best=f"{best_val_loss:.4f}",
            wait=wait,
            lr=f"{current_lr:.2e}",
        )
        tqdm.write(
            f"Epoch {epoch:03d}/{epochs} {improved_mark} | "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"train_dice={train_dice:.4f}  val_dice={val_dice:.4f}  "
            f"best={best_val_loss:.6f}  lr={current_lr:.2e}  "
            f"wait={wait}/{patience}  ({elapsed:.1f}s)"
        )

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            wait = 0
            async_saver.wait_for_save()
            best_payload = _build_checkpoint_payload(
                model=base_model,
                epoch=epoch,
                best_val_loss=best_val_loss,
                wait=wait,
                history=history,
            )
            # Save best synchronously for stronger recovery guarantees.
            torch.save(best_payload, best_path)
            tqdm.write(f"  ✅ New best checkpoint saved  (val={val_loss:.6f})")
        else:
            wait += 1
            reached_patience = wait >= patience and epoch >= min_epochs
            lr_at_floor = current_lr <= (plateau_min_lr + 1e-12)
            can_early_stop = reached_patience and (
                (not early_stop_require_min_lr) or (scheduler is None) or lr_at_floor
            )
            if can_early_stop:
                async_saver.wait_for_save()
                reason = "patience reached"
                if early_stop_require_min_lr and scheduler is not None:
                    reason += f", lr floor reached ({current_lr:.2e})"
                tqdm.write(f"  ⏹️ Early stopping @ epoch {epoch}  ({reason})")
                break

        # 後台保存 last checkpoint（含更新後的 best loss / wait 狀態）
        last_payload = _build_checkpoint_payload(
            model=base_model,
            epoch=epoch,
            best_val_loss=best_val_loss,
            wait=wait,
            history=history,
        )
        async_saver.save_async(last_payload, last_path)

        if low_vram_mode and device == "cuda":
            torch.cuda.empty_cache()

    epoch_bar.close()
    print(f"\n[4/4] 載入最佳權重 ...")
    async_saver.wait_for_save()
    if best_path.exists():
        load_state_dict_compat(base_model, best_path, map_location=device)
        print(f"  ✅ Loaded best weights: {best_path}  (best_val={best_val_loss:.6f})")
    elif last_path.exists():
        load_state_dict_compat(base_model, last_path, map_location=device)
        print(f"  ⚠️ Best checkpoint missing, fallback to last weights: {last_path}")
    else:
        print("  ⚠️ No checkpoint found to reload; keeping current in-memory weights.")

    epochs_ran = int(len(history.get("val_loss", [])))
    best_epoch = 0
    val_losses = history.get("val_loss", [])
    if val_losses:
        try:
            best_epoch = int(np.argmin(np.asarray(val_losses, dtype=np.float64))) + 1
        except Exception:
            best_epoch = 0

    ft_total_sec = float(time.perf_counter() - ft_total_start)
    avg_epoch_sec = float(np.mean(epoch_durations_sec)) if epoch_durations_sec else 0.0

    stats_payload = {
        "history": history,
        "best_val_loss": best_val_loss,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_config": epochs,
        "epochs_ran": epochs_ran,
        "convergence_epoch": best_epoch,
        "epoch_durations_sec": epoch_durations_sec,
        "avg_epoch_sec": avg_epoch_sec,
        "total_finetune_sec": ft_total_sec,
        "batch_size": batch_size,
        "lr": lr,
    }

    stats_path = output_dir / f"{stats_prefix}_stats.json"
    t_save_json = time.perf_counter()
    stats_path.write_text(
        json.dumps(stats_payload, indent=2),
        encoding="utf-8",
    )
    save_json_total = time.perf_counter() - t_save_json
    t_save_pt = time.perf_counter()
    torch.save(stats_payload, output_dir / f"{stats_prefix}_stats.pt")
    save_pt_total = time.perf_counter() - t_save_pt

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    if profiler is not None and profiler.enabled:
        profiler.record_duration("finetune.total", ft_total_sec)
        profiler.record_duration("finetune.data_move", train_data_move_total)
        profiler.record_duration("finetune.train_forward", train_forward_total)
        profiler.record_duration("finetune.train_backward", train_backward_total)
        profiler.record_duration("finetune.optimizer", train_optimizer_total)
        profiler.record_duration("finetune.val_forward", val_forward_total)
        profiler.record_duration("finetune.save_json", save_json_total)
        profiler.record_duration("finetune.save_pt", save_pt_total)
        profiler.add_counter("finetune.train_samples", float(len(train_dataset)))
        profiler.add_counter("finetune.val_samples", float(len(val_dataset)))
        profiler.add_counter("finetune.epochs_config", float(epochs))
        profiler.add_counter("finetune.batch_size", float(batch_size))
        profiler.flush()

    print("=" * 80)
    print("Fine-tune completed")
    print("=" * 80)
    return model
