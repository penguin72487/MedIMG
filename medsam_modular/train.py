import json
import hashlib
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
from medsam_modular.eval.evaluate import _compute_model_hash_tag
from medsam_modular.model import load_state_dict_compat, normalize_pred_masks_to_4d, resolve_amp_dtype


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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _cuda_total_memory_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        return float(props.total_memory) / (1024.0 ** 3)
    except Exception:
        return None


def _is_low_vram_cuda(device: str, cuda_mem_gb: Optional[float] = None) -> bool:
    if device != "cuda":
        return False
    limit_gb = max(0.0, _env_float("MEDSAM_VRAM_LIMIT_GB", 0.0))
    if limit_gb <= 0:
        return False
    total_gb = _cuda_total_memory_gb() if cuda_mem_gb is None else cuda_mem_gb
    return bool(total_gb is not None and total_gb <= limit_gb)


def _cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _auto_train_workers(device: str) -> int:
    cpu_count = _cpu_count()
    if device == "cuda":
        return max(2, min(8, cpu_count // 2))
    return cpu_count


def _setup_cuda_backends() -> None:
    """Enable Tensor Core optimizations (TF32 + cuDNN benchmark)."""
    torch.set_float32_matmul_precision(os.getenv("MEDSAM_CUDA_MATMUL_PRECISION", "high"))
    allow_tf32 = _env_bool("MEDSAM_CUDA_ALLOW_TF32", True)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = _env_bool("MEDSAM_CUDA_CUDNN_BENCHMARK", True)


def _build_optimizer(
    param_groups: List[Dict[str, Any]],
    *,
    device: str,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    use_fused_adamw: bool,
) -> Tuple[torch.optim.Optimizer, str]:
    optimizer_name = os.getenv("MEDSAM_FINETUNE_OPTIMIZER", "adamw").strip().lower()
    if device == "cuda" and optimizer_name in {"bnb_adamw8bit", "adamw8bit", "8bit_adamw"}:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(param_groups, lr=lr, betas=betas, eps=eps)
            return optimizer, "bitsandbytes.AdamW8bit"
        except Exception as exc:
            print(
                f"  [optimizer] bitsandbytes AdamW8bit unavailable ({type(exc).__name__}: {str(exc)[:160]}), fallback to torch AdamW",
                flush=True,
            )

    optimizer_kwargs: Dict[str, Any] = {
        "lr": lr,
        "betas": betas,
        "eps": eps,
    }
    if device == "cuda":
        optimizer_kwargs["fused"] = use_fused_adamw
    return torch.optim.AdamW(param_groups, **optimizer_kwargs), "torch.AdamW(fused)" if optimizer_kwargs.get("fused") else "torch.AdamW"


class EmbeddingDiskCache:
    def __init__(self, cache_dir: Path, model_hash: str, image_size: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_hash = str(model_hash)
        self.image_size = int(image_size)
        self.manifest_path = self.cache_dir / "manifest.jsonl"
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _digest(self, name: str, bbox: torch.Tensor, idx: int, scope: str) -> str:
        bbox_vals = [int(round(float(v))) for v in bbox.reshape(-1).detach().cpu().tolist()]
        payload = json.dumps(
            {
                "version": "emb_v2",
                "model_hash": self.model_hash,
                "image_size": self.image_size,
                "scope": str(scope),
                "idx": int(idx),
                "name": str(name),
                "bbox": bbox_vals,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def path_for(self, name: str, bbox: torch.Tensor, idx: int, scope: str) -> Path:
        return self.cache_dir / f"{self._digest(name=name, bbox=bbox, idx=idx, scope=scope)}.pt"

    def path_for_identity(self, identity: Dict[str, Any], idx: int, scope: str) -> Path:
        payload = json.dumps(
            {
                "version": "emb_v3",
                "model_hash": self.model_hash,
                "image_size": self.image_size,
                "scope": str(scope),
                "idx": int(idx),
                "identity": identity,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def get(self, path: Path) -> Optional[Dict[str, torch.Tensor]]:
        if not path.exists():
            self.misses += 1
            return None
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                payload = torch.load(path, map_location="cpu")
            required = {
                "image_embedding",
                "input_boxes",
                "original_sizes",
                "reshaped_input_sizes",
                "gt_mask",
                "name",
            }
            if not isinstance(payload, dict) or not required.issubset(payload.keys()):
                self.misses += 1
                return None
            self.hits += 1
            return payload
        except Exception:
            self.misses += 1
            return None

    def put(self, path: Path, payload: Dict[str, torch.Tensor], name: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        self.writes += 1
        row = {
            "version": "emb_v2",
            "model_hash": self.model_hash,
            "image_size": self.image_size,
            "name": str(name),
            "file": path.name,
        }
        with self.manifest_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


class ProcessorDiskCache:
    def __init__(self, cache_dir: Path, image_size: int, processor_tag: str = "sam_processor") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = int(image_size)
        self.processor_tag = str(processor_tag)
        self.manifest_path = self.cache_dir / "manifest.jsonl"
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def path_for(self, identity: Dict[str, Any], scope: str) -> Path:
        payload = json.dumps(
            {
                "version": "processor_v1",
                "processor": self.processor_tag,
                "image_size": self.image_size,
                "scope": str(scope),
                "identity": identity,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def get(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            self.misses += 1
            return None
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                payload = torch.load(path, map_location="cpu")
            required = {
                "pixel_values",
                "input_boxes",
                "original_sizes",
                "reshaped_input_sizes",
                "gt_mask",
                "name",
            }
            if not isinstance(payload, dict) or not required.issubset(payload.keys()):
                self.misses += 1
                return None
            self.hits += 1
            return payload
        except Exception:
            self.misses += 1
            return None

    def put(self, path: Path, payload: Dict[str, Any], name: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        self.writes += 1
        row = {
            "version": "processor_v1",
            "processor": self.processor_tag,
            "image_size": self.image_size,
            "name": str(name),
            "file": path.name,
        }
        with self.manifest_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _choose_amp_dtype() -> torch.dtype:
    """Project-wide AMP dtype resolver (BF16 preferred)."""
    return resolve_amp_dtype("cuda")


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _encode_vision_embeddings_cpu(
    *,
    vision_encoder: Any,
    pixel_values: torch.Tensor,
    device: str,
    amp_dtype: torch.dtype,
    initial_chunk: int,
) -> torch.Tensor:
    """Encode pixel_values with adaptive CUDA chunks and return AMP dtype CPU embeddings."""
    total = int(pixel_values.shape[0])
    if total <= 0:
        empty_dtype = amp_dtype if amp_dtype in {torch.float16, torch.bfloat16} else torch.float32
        return torch.empty((0,), dtype=empty_dtype)

    chunk = max(1, min(int(initial_chunk), total))
    outputs: List[torch.Tensor] = []
    start = 0
    non_blocking = device == "cuda"

    while start < total:
        cur = min(chunk, total - start)
        try:
            pv_chunk = pixel_values[start : start + cur].to(device, non_blocking=non_blocking)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                emb = vision_encoder(pv_chunk)[0]
            cache_dtype = amp_dtype if amp_dtype in {torch.float16, torch.bfloat16} else torch.float32
            outputs.append(emb.detach().to(dtype=cache_dtype).cpu())
            del emb, pv_chunk
            start += cur
        except RuntimeError as exc:
            if device == "cuda" and _is_cuda_oom(exc):
                try:
                    del pv_chunk
                except UnboundLocalError:
                    pass
                torch.cuda.empty_cache()
                if cur > 1:
                    chunk = max(1, cur // 2)
                    print(f"  [precompute-oom] reduce GPU chunk to {chunk}", flush=True)
                    continue
                raise RuntimeError(
                    "CUDA OOM during embedding precompute even with GPU chunk=1. "
                    "Try MEDSAM_PRECOMPUTE_BATCH=1 and MEDSAM_FINETUNE_BATCH=1."
                ) from exc
            raise

    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def _precompute_image_embeddings(
    model: Any,
    dataset: Dataset,
    device: str,
    batch_size: int = 4,
    amp_dtype: torch.dtype = torch.bfloat16,
    num_workers: int = 0,
    disk_cache: Optional[EmbeddingDiskCache] = None,
    cache_label: str = "dataset",
) -> Dict[int, Any]:
    """Run the frozen ViT once for every sample and cache FP16 embeddings.

    Returns dict {dataset_index -> payload_or_path}. By default this keeps only
    disk paths in memory to avoid multi-GB RAM growth on large/TTA-expanded sets.
    Calling this before the training loop removes ~95% of per-epoch GPU compute
    (the frozen vision encoder accounts for ~95% of SAM-ViT-B's FLOPs).
    """
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is None:
        return {}

    was_training = model.training
    model.eval()

    loader = DataLoader(
        _EmbeddingPrecomputeDataset(dataset, disk_cache=disk_cache, cache_label=cache_label),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=_embedding_precompute_collate,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=max(1, _env_int("MEDSAM_PRECOMPUTE_PREFETCH", 2)) if num_workers > 0 else None,
    )

    keep_ram_embeddings = _env_bool("MEDSAM_EMB_CACHE_KEEP_RAM", False)
    embeddings: Dict[int, Any] = {}
    emb_fast_hits = 0
    emb_misses = 0
    gpu_chunk = _env_int("MEDSAM_PRECOMPUTE_GPU_CHUNK", 0)
    if gpu_chunk <= 0:
        gpu_chunk = min(max(1, int(batch_size)), 4)
    empty_cache_every = max(1, _env_int("MEDSAM_PRECOMPUTE_EMPTY_CACHE_EVERY", 25))
    batch_counter = 0
    with torch.inference_mode():
        desc = f"  Pre-computing ViT embeddings ({cache_label})"
        progress_enabled = _env_bool("MEDSAM_PROGRESS", True)
        progress_interval = max(1.0, _env_float("MEDSAM_PROGRESS_INTERVAL", 1.0))
        for batch in tqdm(
            loader,
            desc=desc,
            leave=False,
            unit="batch",
            dynamic_ncols=False,
            mininterval=progress_interval,
            disable=not progress_enabled,
        ):
            batch_counter += 1
            for sample_idx, cached_payload, cached_path in batch["cached"]:
                embeddings[int(sample_idx)] = cached_payload if keep_ram_embeddings else Path(cached_path)
            emb_fast_hits += len(batch["cached"])

            if batch["pixel_values"] is None:
                if device == "cuda" and batch_counter % empty_cache_every == 0:
                    torch.cuda.empty_cache()
                continue

            emb_misses += int(batch["pixel_values"].shape[0])
            emb_cpu = _encode_vision_embeddings_cpu(
                vision_encoder=vision_encoder,
                pixel_values=batch["pixel_values"],
                device=device,
                amp_dtype=amp_dtype,
                initial_chunk=gpu_chunk,
            )
            boxes_cpu = batch["input_boxes"].to(dtype=torch.float32).cpu()
            orig_cpu = batch["original_sizes"].cpu()
            reshaped_cpu = batch["reshaped_input_sizes"].cpu()
            gt_cpu = batch["gt_mask"].to(dtype=torch.float32).cpu()
            names = batch["name"]
            sample_indices = batch["idx"]
            cache_paths = batch["cache_path"]

            for out_i, sample_idx in enumerate(sample_indices):
                sample_idx = int(sample_idx)
                sample_name = str(names[out_i] or f"sample_{sample_idx}")
                payload = {
                    "image_embedding": emb_cpu[out_i],
                    "input_boxes": boxes_cpu[out_i],
                    "original_sizes": orig_cpu[out_i],
                    "reshaped_input_sizes": reshaped_cpu[out_i],
                    "gt_mask": gt_cpu[out_i],
                    "name": sample_name,
                }
                embeddings[sample_idx] = payload
                if disk_cache is not None and cache_paths[out_i] is not None:
                    path = Path(cache_paths[out_i])
                    disk_cache.put(path, payload, sample_name)
                    if not keep_ram_embeddings:
                        embeddings[sample_idx] = path
            del emb_cpu, boxes_cpu, orig_cpu, reshaped_cpu, gt_cpu
            if device == "cuda" and batch_counter % empty_cache_every == 0:
                torch.cuda.empty_cache()

    if was_training:
        model.train()
    if device == "cuda":
        torch.cuda.empty_cache()
    if disk_cache is not None:
        print(
            f"  [emb-cache:{cache_label}] fast_hits={emb_fast_hits} "
            f"misses={emb_misses} writes={disk_cache.writes} dir={disk_cache.cache_dir}",
            flush=True,
        )
    return embeddings


class FinetuneEmbeddingDataset(Dataset):
    """Wraps FinetuneProcessorDataset; substitutes pixel_values with pre-computed embedding."""

    def __init__(self, base: Dataset, embeddings: Dict[int, Any]) -> None:
        self.base = base
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cached = self.embeddings.get(idx)
        if cached is None:
            # Fallback path (should be rare): use original processor-based sample.
            return dict(self.base[idx])
        if isinstance(cached, (str, Path)):
            try:
                cached = torch.load(Path(cached), map_location="cpu", weights_only=True)
            except Exception:
                cached = torch.load(Path(cached), map_location="cpu")

        return {
            "image_embedding": cached["image_embedding"],
            "input_boxes": cached["input_boxes"],
            "original_sizes": cached["original_sizes"],
            "reshaped_input_sizes": cached["reshaped_input_sizes"],
            "gt_mask": cached["gt_mask"],
            "name": str(cached.get("name", f"sample_{idx}")),
        }


class FinetuneProcessorDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        processor: Any,
        processor_cache: Optional[ProcessorDiskCache] = None,
        cache_scope: str = "dataset",
    ):
        self.base_dataset = base_dataset
        self.processor = processor
        self.processor_cache = processor_cache
        self.cache_scope = str(cache_scope)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def cache_identity(self, idx: int) -> Dict[str, Any]:
        return {
            "wrapper": "FinetuneProcessorDataset",
            "processor": self.processor.__class__.__name__,
            "scope": self.cache_scope,
            "base": _dataset_cache_identity(self.base_dataset, idx),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache_path: Optional[Path] = None
        if self.processor_cache is not None:
            identity = _dataset_cache_identity(self.base_dataset, idx)
            cache_path = self.processor_cache.path_for(identity, self.cache_scope)
            cached = self.processor_cache.get(cache_path)
            if cached is not None:
                return cached

        sample = self.base_dataset[idx]
        image = sample["image"]
        bbox = sample["bbox"]
        mask = sample["mask"]

        inputs = self.processor(images=[image], input_boxes=[[bbox]], return_tensors="pt")
        packed = {k: v.squeeze(0) for k, v in inputs.items()}
        packed["gt_mask"] = mask.float()
        packed["name"] = str(sample.get("name", f"sample_{idx}"))
        if self.processor_cache is not None and cache_path is not None:
            self.processor_cache.put(cache_path, packed, packed["name"])
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

    def cache_identity(self, idx: int) -> Dict[str, Any]:
        base_idx = int(self.indices[idx])
        return {
            "wrapper": "NameFilteredDataset",
            "base_idx": base_idx,
            "base": _dataset_cache_identity(self.base, base_idx),
        }

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

    def cache_identity(self, idx: int) -> Dict[str, Any]:
        base_idx = int(idx % self.base_len)
        aug_idx = int(idx // self.base_len)
        aug = str(self.augmentations[aug_idx])
        return {
            "wrapper": "TTAAugmentedRawDataset",
            "base_idx": base_idx,
            "aug": aug,
            "base": _dataset_cache_identity(self.base, base_idx),
        }

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


def _path_signature(path_like: Any) -> Dict[str, Any]:
    path = Path(path_like)
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    except Exception:
        return {"path": str(path)}


def _sample_entry_signature(entry: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in [
        "name",
        "image_id",
        "component_index",
        "bbox",
        "boxes",
        "source_size",
        "width",
        "height",
        "xml_width",
        "xml_height",
    ]:
        if key in entry:
            out[key] = entry[key]
    for key in ["image_path", "mask_path", "xml_path", "annotation_path"]:
        if key in entry:
            out[key] = _path_signature(entry[key])
    if "svg" in entry:
        svg = str(entry.get("svg", ""))
        out["svg_md5"] = hashlib.md5(svg.encode("utf-8", errors="ignore")).hexdigest()
    return out


def _concat_child_for_index(dataset: ConcatDataset, idx: int) -> Tuple[Dataset, int, int]:
    child_idx = 0
    prev_size = 0
    for child_idx, cumulative_size in enumerate(dataset.cumulative_sizes):
        if idx < cumulative_size:
            return dataset.datasets[child_idx], int(idx - prev_size), child_idx
        prev_size = cumulative_size
    raise IndexError(idx)


def _dataset_cache_identity(dataset: Dataset, idx: int) -> Dict[str, Any]:
    if hasattr(dataset, "cache_identity"):
        return getattr(dataset, "cache_identity")(idx)

    if isinstance(dataset, ConcatDataset):
        child, child_idx, child_pos = _concat_child_for_index(dataset, int(idx))
        return {
            "wrapper": "ConcatDataset",
            "child": child_pos,
            "child_idx": child_idx,
            "base": _dataset_cache_identity(child, child_idx),
        }

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        indices = getattr(dataset, "indices")
        base = getattr(dataset, "dataset")
        base_idx = int(indices[idx])
        return {
            "wrapper": dataset.__class__.__name__,
            "base_idx": base_idx,
            "base": _dataset_cache_identity(base, base_idx),
        }

    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list) and 0 <= int(idx) < len(samples) and isinstance(samples[int(idx)], dict):
        return {
            "dataset": dataset.__class__.__name__,
            "idx": int(idx),
            "sample": _sample_entry_signature(samples[int(idx)]),
        }

    return {
        "dataset": dataset.__class__.__name__,
        "idx": int(idx),
        "length": int(len(dataset)),
    }


class _EmbeddingPrecomputeDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        disk_cache: Optional[EmbeddingDiskCache],
        cache_label: str,
    ) -> None:
        self.base = base
        self.disk_cache = disk_cache
        self.cache_label = str(cache_label)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache_path: Optional[Path] = None
        if self.disk_cache is not None:
            identity = _dataset_cache_identity(self.base, idx)
            cache_path = self.disk_cache.path_for_identity(identity, idx=idx, scope=self.cache_label)
            cached = self.disk_cache.get(cache_path)
            if cached is not None:
                return {
                    "__emb_cached__": True,
                    "__idx__": int(idx),
                    "__emb_cache_path__": str(cache_path),
                    "payload": cached,
                }

        item = dict(self.base[idx])
        item["__emb_cached__"] = False
        item["__idx__"] = int(idx)
        item["__emb_cache_path__"] = str(cache_path) if cache_path is not None else ""
        return item


def _embedding_precompute_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    cached: List[Tuple[int, Dict[str, torch.Tensor], str]] = []
    missing: List[Dict[str, Any]] = []

    for item in batch:
        if bool(item.get("__emb_cached__", False)):
            cached.append((int(item["__idx__"]), item["payload"], str(item.get("__emb_cache_path__", ""))))
        else:
            missing.append(item)

    out: Dict[str, Any] = {"cached": cached}
    if not missing:
        out.update(
            {
                "pixel_values": None,
                "input_boxes": None,
                "original_sizes": None,
                "reshaped_input_sizes": None,
                "gt_mask": None,
                "name": [],
                "idx": [],
                "cache_path": [],
            }
        )
        return out

    out.update(
        {
            "pixel_values": torch.stack([item["pixel_values"] for item in missing], dim=0),
            "input_boxes": torch.stack([item["input_boxes"] for item in missing], dim=0),
            "original_sizes": torch.stack([item["original_sizes"] for item in missing], dim=0),
            "reshaped_input_sizes": torch.stack([item["reshaped_input_sizes"] for item in missing], dim=0),
            "gt_mask": torch.stack([item["gt_mask"] for item in missing], dim=0),
            "name": [str(item.get("name", "")) for item in missing],
            "idx": [int(item["__idx__"]) for item in missing],
            "cache_path": [str(item.get("__emb_cache_path__", "")) for item in missing],
        }
    )
    return out


def _iter_processor_caches(dataset: Dataset) -> List[ProcessorDiskCache]:
    found: List[ProcessorDiskCache] = []
    cache = getattr(dataset, "processor_cache", None)
    if isinstance(cache, ProcessorDiskCache):
        found.append(cache)

    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            found.extend(_iter_processor_caches(child))
    elif hasattr(dataset, "dataset"):
        found.extend(_iter_processor_caches(getattr(dataset, "dataset")))
    elif hasattr(dataset, "base"):
        found.extend(_iter_processor_caches(getattr(dataset, "base")))
    elif hasattr(dataset, "base_dataset"):
        found.extend(_iter_processor_caches(getattr(dataset, "base_dataset")))
    return found


def _print_processor_cache_summary(label: str, dataset: Dataset) -> None:
    seen: Set[int] = set()
    for cache in _iter_processor_caches(dataset):
        ident = id(cache)
        if ident in seen:
            continue
        seen.add(ident)
        print(
            f"  [processor-cache:{label}] hits={cache.hits} "
            f"misses={cache.misses} writes={cache.writes} dir={cache.cache_dir}",
            flush=True,
        )


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
    output_dir = Path(config["output_dir"])

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

    processor_cache_enabled = _env_bool("MEDSAM_PROCESSOR_CACHE", True)
    train_processor_cache: Optional[ProcessorDiskCache] = None
    val_processor_cache: Optional[ProcessorDiskCache] = None
    if processor_cache_enabled:
        processor_tag = processor.__class__.__name__
        processor_cache_root = output_dir / "processor_cache" / f"image_size_{image_size}" / processor_tag
        train_processor_cache = ProcessorDiskCache(
            cache_dir=processor_cache_root / "train",
            image_size=image_size,
            processor_tag=processor_tag,
        )
        val_processor_cache = ProcessorDiskCache(
            cache_dir=processor_cache_root / "val",
            image_size=image_size,
            processor_tag=processor_tag,
        )
        print(f"  [processor-cache] disk cache: {processor_cache_root}", flush=True)

    return (
        FinetuneProcessorDataset(
            train_raw,
            processor,
            processor_cache=train_processor_cache,
            cache_scope="train",
        ),
        FinetuneProcessorDataset(
            val_concat,
            processor,
            processor_cache=val_processor_cache,
            cache_scope="val",
        ),
    )


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
            if k == "pixel_values" and v.dim() == 4:
                moved[k] = v.to(device, non_blocking=non_blocking, memory_format=torch.channels_last)
            else:
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


def _prepare_logits_target(outputs: Any, gt_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = normalize_pred_masks_to_4d(outputs.pred_masks)
    target = gt_mask.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    target = target.to(dtype=logits.dtype)
    return logits, target


def _compute_seg_loss_from_logits_target(logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return loss and sigmoid probabilities from already-normalized CUDA tensors."""
    probs = torch.sigmoid(logits)
    l_bce = F.binary_cross_entropy_with_logits(logits, target)
    l_dice = _dice_loss(probs, target)
    return l_bce + l_dice, probs


def _compute_batch_dice_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (probs.detach() >= 0.5).to(dtype=torch.float32)
    target_f = (target.detach() >= 0.5).to(dtype=torch.float32)
    inter = (pred * target_f).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target_f.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


def _compute_seg_loss(outputs: Any, gt_mask: torch.Tensor) -> torch.Tensor:
    """L = L_BCE + L_Dice  (paper Section 3.3)."""
    logits, target = _prepare_logits_target(outputs, gt_mask)
    loss, _ = _compute_seg_loss_from_logits_target(logits, target)
    return loss


def _compute_batch_dice(outputs: Any, gt_mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Hard Dice at threshold 0.5 for logging/monitoring."""
    logits, target = _prepare_logits_target(outputs, gt_mask)
    probs = torch.sigmoid(logits)
    return _compute_batch_dice_from_probs(probs, target, eps=eps)


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
    train_prefetch = max(1, _env_int("MEDSAM_FINETUNE_PREFETCH", 2))
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
    low_vram_mode = _is_low_vram_cuda(device, cuda_mem_gb)
    low_vram_empty_cache_every = max(
        0,
        _env_int("MEDSAM_LOW_VRAM_EMPTY_CACHE_EVERY", 1 if low_vram_mode else 0),
    )
    use_emb_cache = device == "cuda" and not train_backbone and _env_bool("MEDSAM_PRECOMPUTE_EMBEDDINGS", True)

    if use_emb_cache and precompute_batch <= 0:
        if cuda_mem_gb is None:
            precompute_batch = 4
        elif low_vram_mode:
            precompute_batch = 2
        elif cuda_mem_gb <= 24.5:
            precompute_batch = 8
        else:
            precompute_batch = 12

    if precompute_workers <= 0:
        precompute_workers = num_workers
    if low_vram_mode and use_emb_cache:
        # Embedding cache removes ViT from the forward pass → decoder-only VRAM → larger batch
        safe_batch_emb = _env_int("MEDSAM_FINETUNE_SAFE_BATCH_EMB", int(config.get("finetune_safe_batch_emb", 2)))
        safe_batch_emb = max(1, safe_batch_emb)
        if batch_size > safe_batch_emb:
            print(f"⚠️ Low-VRAM mode ({cuda_mem_gb:.1f}GB): batch size {batch_size} -> {safe_batch_emb}")
            batch_size = safe_batch_emb
    elif low_vram_mode:
        safe_batch_12gb = _env_int("MEDSAM_FINETUNE_SAFE_BATCH_12GB", int(config.get("finetune_safe_batch_12gb", 1)))
        safe_batch_12gb = max(1, safe_batch_12gb)
        if batch_size > safe_batch_12gb:
            print(f"⚠️ Low-VRAM mode ({cuda_mem_gb:.1f}GB): batch size {batch_size} -> {safe_batch_12gb}")
            batch_size = safe_batch_12gb

    precompute_gpu_chunk = _env_int("MEDSAM_PRECOMPUTE_GPU_CHUNK", 0)
    if precompute_gpu_chunk <= 0:
        if cuda_mem_gb is None:
            precompute_gpu_chunk = min(precompute_batch, 4)
        elif low_vram_mode:
            precompute_gpu_chunk = min(precompute_batch, 4)
        elif cuda_mem_gb <= 24.5:
            precompute_gpu_chunk = min(precompute_batch, 2)
        else:
            precompute_gpu_chunk = min(precompute_batch, 4)
        os.environ["MEDSAM_PRECOMPUTE_GPU_CHUNK"] = str(precompute_gpu_chunk)

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
    print(f"  workers       : train={num_workers}, prefetch={train_prefetch}")
    print(f"  train backbone: {train_backbone}")
    print(f"  amp dtype     : {amp_dtype}")
    print(f"  emb cache     : {use_emb_cache}")
    if device == "cuda" and low_vram_empty_cache_every > 0:
        print(f"  low-vram clear: every {low_vram_empty_cache_every} batch(es)")
    print(f"  scheduler     : {'ReduceLROnPlateau' if use_plateau_scheduler else 'off'}")
    if use_emb_cache:
        print(f"  precompute    : batch={precompute_batch}, gpu_chunk={precompute_gpu_chunk}, workers={precompute_workers}")
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
        prefetch_factor=train_prefetch if num_workers > 0 else None,
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
        prefetch_factor=train_prefetch if num_workers > 0 else None,
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
        emb_model_hash = _compute_model_hash_tag(base_model)
        emb_cache_root = output_dir / "emb_cache" / f"image_size_{int(config['image_size'])}" / emb_model_hash
        print(f"  [emb-cache] disk cache: {emb_cache_root}", flush=True)
        train_disk_cache = EmbeddingDiskCache(
            cache_dir=emb_cache_root / "train",
            model_hash=emb_model_hash,
            image_size=int(config["image_size"]),
        )
        val_disk_cache = EmbeddingDiskCache(
            cache_dir=emb_cache_root / "val",
            model_hash=emb_model_hash,
            image_size=int(config["image_size"]),
        )
        train_embs = _precompute_image_embeddings(
            base_model,
            train_dataset,
            device,
            batch_size=precompute_batch,
            amp_dtype=amp_dtype,
            num_workers=precompute_workers,
            disk_cache=train_disk_cache,
            cache_label="train",
        )
        val_embs = _precompute_image_embeddings(
            base_model,
            val_dataset,
            device,
            batch_size=precompute_batch,
            amp_dtype=amp_dtype,
            num_workers=precompute_workers,
            disk_cache=val_disk_cache,
            cache_label="val",
        )
        _print_processor_cache_summary("train", train_dataset)
        _print_processor_cache_summary("val", val_dataset)
        train_dataset = FinetuneEmbeddingDataset(train_dataset, train_embs)
        val_dataset   = FinetuneEmbeddingDataset(val_dataset,   val_embs)
        del train_embs, val_embs
        if device == "cuda":
            torch.cuda.empty_cache()
        if _env_bool("MEDSAM_EMB_CACHE_KEEP_RAM", False):
            emb_mem_gb = len(train_dataset) * 256 * 64 * 64 * 2 / 1024**3
            mem_note = f"~{emb_mem_gb:.1f}GB CPU RAM used"
        else:
            mem_note = "disk-backed embeddings; RAM only keeps paths"
        print(f"  Pre-compute done ({time.time()-t_emb:.1f}s) | {mem_note}")
        # Rebuild DataLoaders with updated (wrapped) datasets
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device == "cuda"),
            drop_last=False, collate_fn=_finetune_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=train_prefetch if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"),
            drop_last=False, collate_fn=_finetune_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=train_prefetch if num_workers > 0 else None,
        )
    # ──────────────────────────────────────────────────────────────────────

    base_model.train()

    params = [p for p in base_model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in params)
    total_count = sum(p.numel() for p in base_model.parameters())
    print(f"\n[2/4] 設定優化器 ...")
    setup_t0 = time.perf_counter()
    print(f"  可訓練參數: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)")

    param_groups = _build_adamw_param_groups(base_model, weight_decay=weight_decay)
    optimizer, optimizer_label = _build_optimizer(
        param_groups,
        device=device,
        lr=lr,
        betas=(adamw_beta1, adamw_beta2),
        eps=adamw_eps,
        use_fused_adamw=use_fused_adamw,
    )
    print(f"  optimizer: {optimizer_label}", flush=True)
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
    print(f"  優化器/排程器設定耗時: {time.perf_counter() - setup_t0:.1f}s")

    print(f"\n[3/4] 開始訓練 (共 {epochs} epochs) ...")
    progress_enabled = _env_bool("MEDSAM_PROGRESS", True)
    progress_interval = max(1.0, _env_float("MEDSAM_PROGRESS_INTERVAL", 1.0))
    epoch_bar = tqdm(
        range(start_epoch, epochs + 1),
        desc="Epoch",
        unit="ep",
        dynamic_ncols=False,
        mininterval=progress_interval,
        disable=not progress_enabled,
    )

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
            dynamic_ncols=False,
            unit="batch",
            mininterval=progress_interval,
            disable=not progress_enabled,
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
                    logits, target = _prepare_logits_target(outputs, batch["gt_mask"])
                    raw_loss, probs = _compute_seg_loss_from_logits_target(logits, target)
                    loss = raw_loss / grad_accum
            else:
                outputs = base_model(**model_inputs)
                logits, target = _prepare_logits_target(outputs, batch["gt_mask"])
                raw_loss, probs = _compute_seg_loss_from_logits_target(logits, target)
                loss = raw_loss / grad_accum
            batch_dice = _compute_batch_dice_from_probs(probs, target)
            train_forward_total += (time.perf_counter() - t_forward)

            t_backward = time.perf_counter()
            scaler.scale(loss).backward()
            train_backward_total += (time.perf_counter() - t_backward)
            cur_loss = float(loss.item() * grad_accum)
            train_losses.append(cur_loss)
            train_dices.append(batch_dice)
            if progress_enabled:
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

            del loss, raw_loss, outputs, logits, target, probs, model_inputs
            if device == "cuda" and low_vram_empty_cache_every > 0 and step % low_vram_empty_cache_every == 0:
                torch.cuda.empty_cache()

        base_model.eval()
        val_losses: List[float] = []
        val_dices: List[float] = []
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"  Val  ",
                leave=False,
                dynamic_ncols=False,
                unit="batch",
                mininterval=progress_interval,
                disable=not progress_enabled,
            )
            for val_step, batch in enumerate(val_bar, start=1):
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
                        logits, target = _prepare_logits_target(outputs, batch["gt_mask"])
                        val_loss, probs = _compute_seg_loss_from_logits_target(logits, target)
                else:
                    outputs = base_model(**model_inputs)
                    logits, target = _prepare_logits_target(outputs, batch["gt_mask"])
                    val_loss, probs = _compute_seg_loss_from_logits_target(logits, target)
                val_dice = _compute_batch_dice_from_probs(probs, target)
                val_forward_total += (time.perf_counter() - t_val_forward)
                val_loss_float = float(val_loss.item())

                if progress_enabled:
                    val_bar.set_postfix(loss=f"{val_loss_float:.4f}", dice=f"{val_dice:.4f}")
                val_losses.append(val_loss_float)
                val_dices.append(val_dice)
                del val_loss, outputs, logits, target, probs, model_inputs
                if device == "cuda" and low_vram_empty_cache_every > 0 and val_step % low_vram_empty_cache_every == 0:
                    torch.cuda.empty_cache()

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
        if progress_enabled:
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
        else:
            print(
                f"Epoch {epoch:03d}/{epochs} {improved_mark} | "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"train_dice={train_dice:.4f}  val_dice={val_dice:.4f}  "
                f"best={best_val_loss:.6f}  lr={current_lr:.2e}  "
                f"wait={wait}/{patience}  ({elapsed:.1f}s)",
                flush=True,
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
    reload_t0 = time.perf_counter()
    async_saver.wait_for_save()
    if best_path.exists():
        load_state_dict_compat(base_model, best_path, map_location=device)
        print(f"  ✅ Loaded best weights: {best_path}  (best_val={best_val_loss:.6f})")
    elif last_path.exists():
        load_state_dict_compat(base_model, last_path, map_location=device)
        print(f"  ⚠️ Best checkpoint missing, fallback to last weights: {last_path}")
    else:
        print("  ⚠️ No checkpoint found to reload; keeping current in-memory weights.")
    print(f"  載入/等待 checkpoint 耗時: {time.perf_counter() - reload_t0:.1f}s")

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
    print(f"  stats JSON 保存耗時: {save_json_total:.3f}s")
    print(f"  stats PT 保存耗時  : {save_pt_total:.3f}s")

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
    print(f"  total_finetune: {ft_total_sec:.1f}s")
    print(f"  avg_epoch     : {avg_epoch_sec:.1f}s")
    print(f"  train_forward : {train_forward_total:.1f}s")
    print(f"  train_backward: {train_backward_total:.1f}s")
    print(f"  val_forward   : {val_forward_total:.1f}s")
    print(f"  optimizer     : {train_optimizer_total:.1f}s")
    print(f"  data_move     : {train_data_move_total:.1f}s")
    print("=" * 80)
    return model
