import time
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scipy import ndimage
except Exception:
    ndimage = None

from medsam_modular.cache import PredictionCache, make_cache_key
from medsam_modular.model import build_inputs_batch, predict_prob_masks_from_inputs
from medsam_modular.profiler import PerformanceProfiler, get_active_profiler


def _build_stats_from_store(
    *,
    dataset_name: str,
    results: List[Dict[str, Any]],
    metrics_store: Dict[str, List[float]],
    inference_times: List[float],
    data_times: List[float],
    ood_times: List[float],
    metrics_times: List[float],
    post_times: List[float],
    total_time: float,
    ood_scores: Optional[List[float]] = None,
    uncertainties: Optional[List[float]] = None,
    ood_eval_scores: Optional[List[float]] = None,
    ood_eval_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    mean_dice, std_dice = _mean_std(metrics_store["dice"])
    mean_jaccard, std_jaccard = _mean_std(metrics_store["jaccard"])
    mean_precision, std_precision = _mean_std(metrics_store["precision"])
    mean_recall, std_recall = _mean_std(metrics_store["recall"])
    mean_sensitivity, std_sensitivity = _mean_std(metrics_store.get("sensitivity", metrics_store["recall"]))
    mean_f1, std_f1 = _mean_std(metrics_store["f1"])
    mean_bce, std_bce = _mean_std(metrics_store.get("bce", [0.0]))
    mean_hd95, std_hd95 = _mean_std(metrics_store.get("hd95", [float("nan")]))
    mean_assd, std_assd = _mean_std(metrics_store.get("assd", [float("nan")]))
    mean_ece, std_ece = _mean_std(metrics_store.get("ece", [float("nan")]))

    stats: Dict[str, Any] = {
        "dataset": dataset_name,
        "num_samples": int(len(results)),
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_jaccard": mean_jaccard,
        "std_jaccard": std_jaccard,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "mean_sensitivity": mean_sensitivity,
        "std_sensitivity": std_sensitivity,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_bce": mean_bce,
        "std_bce": std_bce,
        "mean_hd95": mean_hd95,
        "std_hd95": std_hd95,
        "mean_assd": mean_assd,
        "std_assd": std_assd,
        "mean_ece": mean_ece,
        "std_ece": std_ece,
        "total_time_sec": float(total_time),
        "avg_inference_time_ms": float(np.mean(inference_times) * 1000.0),
        "avg_data_time_ms": float(np.mean(data_times) * 1000.0),
        "avg_ood_time_ms": float(np.mean(ood_times) * 1000.0),
        "avg_metrics_time_ms": float(np.mean(metrics_times) * 1000.0),
        "avg_post_time_ms": float(np.mean(post_times) * 1000.0),
        "throughput_samples_per_sec": float(len(results) / total_time if total_time > 0 else 0.0),
    }

    if ood_scores:
        stats["mean_ood_score"] = float(np.mean(ood_scores))
        stats["std_ood_score"] = float(np.std(ood_scores))
        stats["num_ood_detected"] = int(sum(1 for r in results if r.get("is_ood", False)))
        stats["ood_ratio"] = float(stats["num_ood_detected"] / max(1, len(results)))

    if ood_eval_scores and ood_eval_labels:
        det_metrics = _compute_ood_detection_stats(ood_eval_scores, ood_eval_labels)
        stats.update(det_metrics)

    if uncertainties:
        stats["mean_uncertainty"] = float(np.mean(uncertainties))
        stats["std_uncertainty"] = float(np.std(uncertainties))

    component_totals = {
        "data": float(np.sum(data_times)),
        "inference": float(np.sum(inference_times)),
        "ood": float(np.sum(ood_times)),
        "metrics": float(np.sum(metrics_times)),
        "post": float(np.sum(post_times)),
    }
    bottleneck_name, bottleneck_total = max(component_totals.items(), key=lambda kv: kv[1])
    stats["bottleneck_component"] = bottleneck_name
    stats["bottleneck_component_ratio"] = float(bottleneck_total / max(1e-8, total_time))
    return stats


def _maybe_warm_dataset_cache(dataset: Any, dataset_name: str, profiler: Optional[PerformanceProfiler], profile_prefix: str) -> None:
    warm_enabled = os.getenv("MEDSAM_EVAL_WARM_CACHE", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    if not warm_enabled:
        return
    if not hasattr(dataset, "__len__"):
        return

    warm_samples = max(0, int(os.getenv("MEDSAM_EVAL_WARM_SAMPLES", "16")))
    if warm_samples <= 0:
        return

    n = min(int(len(dataset)), warm_samples)
    t0 = time.perf_counter()
    for idx in range(n):
        _ = dataset[idx]
    if profiler is not None and profiler.enabled:
        profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.warm_cache", time.perf_counter() - t0)


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


def _auto_eval_workers(device: str) -> int:
    cores = _cpu_count()
    if device == "cuda":
        return max(2, min(16, cores // 2))
    return max(1, min(8, cores - 1))


class OODDetector:
    def __init__(self, threshold: float = 0.5, method: str = "entropy"):
        self.threshold = threshold
        self.method = method
        self.max_side = max(8, int(os.getenv("MEDSAM_OOD_MAX_SIDE", "64")))

    def _score_from_tensor(self, p: torch.Tensor) -> Tuple[float, float]:
        p = p.to(dtype=torch.float32)
        if p.numel() == 0:
            return 0.0, 1.0

        if p.dim() == 2:
            h, w = int(p.shape[0]), int(p.shape[1])
            max_dim = max(h, w)
            if max_dim > self.max_side:
                scale = float(self.max_side) / float(max_dim)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                p = F.interpolate(
                    p.unsqueeze(0).unsqueeze(0),
                    size=(new_h, new_w),
                    mode="area",
                ).squeeze(0).squeeze(0)

        p = p.reshape(-1)
        hard_binary = torch.all((p <= 0.0) | (p >= 1.0))
        if bool(hard_binary.item()):
            if self.method == "confidence":
                score_t = p.new_tensor(-1.0)
                confidence_t = p.new_tensor(1.0)
            elif self.method == "variance":
                score_t = torch.var(p)
                confidence_t = torch.clamp(1.0 - score_t, min=0.0)
            else:
                score_t = p.new_tensor(0.0)
                confidence_t = p.new_tensor(1.0)
            return float(score_t.item()), float(confidence_t.item())

        p = p.clamp(1e-6, 1.0 - 1e-6)
        if self.method == "confidence":
            score_t = -(torch.abs(p - 0.5) * 2.0).mean()
        elif self.method == "variance":
            score_t = torch.var(p)
        else:
            score_t = -(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean()

        confidence_t = torch.clamp(1.0 - score_t, min=0.0)
        return float(score_t.item()), float(confidence_t.item())

    def _score_batch_from_tensor(self, mask_prob_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_prob_batch.dim() == 2:
            mask_prob_batch = mask_prob_batch.unsqueeze(0)
        if mask_prob_batch.dim() != 3:
            raise ValueError(f"Expected [B,H,W] tensor, got shape {tuple(mask_prob_batch.shape)}")

        p = mask_prob_batch.to(dtype=torch.float32)
        if p.numel() == 0:
            zeros = torch.zeros((p.shape[0],), dtype=torch.float32, device=p.device)
            ones = torch.ones((p.shape[0],), dtype=torch.float32, device=p.device)
            return zeros, ones

        h, w = int(p.shape[-2]), int(p.shape[-1])
        max_dim = max(h, w)
        if max_dim > self.max_side:
            scale = float(self.max_side) / float(max_dim)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            p = F.interpolate(p.unsqueeze(1), size=(new_h, new_w), mode="area").squeeze(1)

        flat = p.reshape(p.shape[0], -1)
        hard_binary = torch.all((flat <= 0.0) | (flat >= 1.0), dim=1)
        clamped = flat.clamp(1e-6, 1.0 - 1e-6)

        if self.method == "confidence":
            soft_score = -(torch.abs(clamped - 0.5) * 2.0).mean(dim=1)
            hard_score = torch.full_like(soft_score, -1.0)
            hard_conf = torch.ones_like(soft_score)
        elif self.method == "variance":
            soft_score = torch.var(clamped, dim=1)
            hard_score = torch.var(flat, dim=1)
            hard_conf = torch.clamp(1.0 - hard_score, min=0.0)
        else:
            soft_score = -(clamped * clamped.log() + (1.0 - clamped) * (1.0 - clamped).log()).mean(dim=1)
            hard_score = torch.zeros_like(soft_score)
            hard_conf = torch.ones_like(soft_score)

        confidence = torch.clamp(1.0 - soft_score, min=0.0)
        score = torch.where(hard_binary, hard_score, soft_score)
        confidence = torch.where(hard_binary, hard_conf, confidence)
        return score, confidence

    def detect_tensor(self, mask_prob: torch.Tensor) -> Dict[str, Any]:
        p = mask_prob
        if p.dim() > 2:
            p = p.squeeze()
        score, confidence = self._score_from_tensor(p)
        return {
            "ood_score": score,
            "is_ood": bool(score > self.threshold),
            "confidence": confidence,
        }

    def detect_batch_tensor(self, mask_prob_batch: torch.Tensor) -> List[Dict[str, Any]]:
        scores, confidences = self._score_batch_from_tensor(mask_prob_batch)
        return [
            {
                "ood_score": float(scores[i].item()),
                "is_ood": bool(scores[i].item() > self.threshold),
                "confidence": float(confidences[i].item()),
            }
            for i in range(scores.shape[0])
        ]


class TTAPredictor:
    """Test Time Augmentation predictor with multiple augmentation strategies."""
    
    def __init__(
        self,
        augmentations: Optional[List[str]] = None,
        fusion_mode: str = "entropy_weighted",
        use_fast_mode: bool = False,
    ):
        """
        Args:
            augmentations: List of augmentations to apply. If None, uses defaults.
            fusion_mode: "mean", "median", or "entropy_weighted"
            use_fast_mode: If True, uses only flip augmentations (faster)
        """
        if use_fast_mode:
            # Fast mode: only essential augmentations
            raw_augmentations = augmentations or ["none", "hflip", "vflip", "hvflip"]
        else:
            # Full mode: fixed augmentations (elastic_deform removed for speed).
            base_augs = [
                "none",
                "hflip",
                "vflip",
                "hvflip",
                "rotate_90",
                "rotate_270",
            ]
            raw_augmentations = augmentations or base_augs

        self.augmentations = self._canonicalize_augmentations(raw_augmentations)
        
        self.fusion_mode = fusion_mode
        env_fixed_batch = int(os.getenv("MEDSAM_TTA_FIXED_BATCH", "0"))
        self.fixed_batch_size = max(0, env_fixed_batch)
        cuda_mem_gb = _cuda_total_memory_gb()
        default_chunk = 4 if (cuda_mem_gb is not None and cuda_mem_gb <= 12.5) else 8
        chunk_env_raw = os.getenv("MEDSAM_TTA_CHUNK_SIZE", "").strip()
        if chunk_env_raw:
            self.infer_chunk_size = max(1, int(chunk_env_raw))
        else:
            self.infer_chunk_size = max(1, int(default_chunk))

        autotune_raw = os.getenv("MEDSAM_TTA_AUTOTUNE", "1").strip().lower()
        self._autotune_enabled = autotune_raw in {"1", "true", "yes", "y", "on"}

        # If user fixed chunk size or disabled autotune, skip first-sample tuner.
        self._chunk_size_tuned = bool(chunk_env_raw) or (not self._autotune_enabled)
        self._norm_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._aug_to_id = {name: idx for idx, name in enumerate(self.augmentations)}
        assert fusion_mode in ["mean", "median", "entropy_weighted"], \
            f"fusion_mode must be 'mean', 'median', or 'entropy_weighted', got {fusion_mode}"

    def _canonicalize_augmentations(self, augmentations: List[str]) -> List[str]:
        """Deduplicate equivalent augmentations to avoid redundant compute."""
        canonical_map = {
            "rotate_180": "hvflip",  # exactly equivalent spatial transform
        }
        removed_aliases = {"elastic_deform"}
        normalized: List[str] = []
        seen = set()
        for name in augmentations:
            aug = canonical_map.get(str(name).strip(), str(name).strip())
            if aug in removed_aliases:
                continue
            if not aug:
                continue
            if aug not in seen:
                normalized.append(aug)
                seen.add(aug)
        return normalized or ["none"]

    def _get_sam_target_edge(self, processor: Any) -> int:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            return 1024
        size_cfg = getattr(image_processor, "size", None)
        if isinstance(size_cfg, dict):
            if "longest_edge" in size_cfg:
                return int(size_cfg["longest_edge"])
            if "height" in size_cfg and "width" in size_cfg:
                return int(max(size_cfg["height"], size_cfg["width"]))
        if isinstance(size_cfg, int):
            return int(size_cfg)
        return 1024

    def _get_sam_norm_tensors(self, processor: Any, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(processor)
        cached = self._norm_cache.get(key)
        if cached is not None:
            return cached[0].to(device), cached[1].to(device)

        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        else:
            image_mean = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
            image_std = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])

        mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self._norm_cache[key] = (mean, std)
        return mean.to(device), std.to(device)

    def _preprocess_image_tensor(self, image_np: np.ndarray, processor: Any, device: str) -> Tuple[torch.Tensor, int]:
        target_edge = self._get_sam_target_edge(processor)
        mean, std = self._get_sam_norm_tensors(processor, device)

        image_t = torch.from_numpy(np.ascontiguousarray(image_np)).to(device)
        image_t = image_t.permute(2, 0, 1).to(torch.float32).div_(255.0).unsqueeze(0)
        if image_t.shape[-2] != target_edge or image_t.shape[-1] != target_edge:
            image_t = F.interpolate(image_t, size=(target_edge, target_edge), mode="bilinear", align_corners=False)
        image_t = (image_t - mean.unsqueeze(0)) / std.unsqueeze(0)
        return image_t[0].contiguous(), target_edge

    def _apply_tensor_aug(self, image_tensor: torch.Tensor, aug_name: str) -> torch.Tensor:
        """Apply augmentation to a preprocessed tensor."""
        if aug_name == "none":
            return image_tensor
        if aug_name == "hflip":
            return image_tensor.flip(-1).contiguous()
        if aug_name == "vflip":
            return image_tensor.flip(-2).contiguous()
        if aug_name == "hvflip":
            return image_tensor.flip(-2).flip(-1).contiguous()
        if aug_name == "rotate_90":
            return torch.rot90(image_tensor, k=1, dims=(-2, -1)).contiguous()
        if aug_name == "rotate_180":
            return torch.rot90(image_tensor, k=2, dims=(-2, -1)).contiguous()
        if aug_name == "rotate_270":
            return torch.rot90(image_tensor, k=3, dims=(-2, -1)).contiguous()
        raise ValueError(f"Unsupported tensor augmentation: {aug_name}")

    def _augment_square_bbox(self, bbox: torch.Tensor, aug_name: str, size: int) -> torch.Tensor:
        """Transform a bbox already scaled to the square preprocessing size."""
        x1, y1, x2, y2 = bbox.to(torch.float32).unbind(-1)
        max_coord = bbox.new_tensor(float(size - 1), dtype=torch.float32)

        if aug_name == "none":
            return torch.stack([x1, y1, x2, y2])
        if aug_name == "hflip":
            return torch.stack([max_coord - x2, y1, max_coord - x1, y2])
        if aug_name == "vflip":
            return torch.stack([x1, max_coord - y2, x2, max_coord - y1])
        if aug_name == "hvflip":
            return torch.stack([max_coord - x2, max_coord - y2, max_coord - x1, max_coord - y1])
        if aug_name == "rotate_90":
            # k=1 (CCW 90): (x, y) -> (y, M-x)
            return torch.stack([y1, max_coord - x2, y2, max_coord - x1])
        if aug_name == "rotate_180":
            return torch.stack([max_coord - x2, max_coord - y2, max_coord - x1, max_coord - y1])
        if aug_name == "rotate_270":
            # k=3 (CCW 270 / CW 90): (x, y) -> (M-y, x)
            return torch.stack([max_coord - y2, x1, max_coord - y1, x2])
        raise ValueError(f"Unsupported square augmentation: {aug_name}")

    def _build_tta_inputs_batch(
        self,
        processor: Any,
        images: List[Image.Image],
        bboxes: List[List[int]],
        device: str,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Tuple[int, int]], List[int]]:
        """
        Build batched inputs for TTA with multiple samples.
        
        Returns:
            (inputs_dict, aug_info_list, output_sizes_list, true_aug_counts_list)
            where aug_info_list = [(aug_name, sample_idx), ...]
        """
        profiler = get_active_profiler()
        t0 = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
        
        n_samples = len(images)
        all_pixel_values: List[torch.Tensor] = []
        all_input_boxes: List[torch.Tensor] = []
        all_original_sizes: List[torch.Tensor] = []
        all_reshaped_sizes: List[torch.Tensor] = []
        aug_ids: List[int] = []
        output_sizes_list: List[Tuple[int, int]] = []
        true_aug_counts: List[int] = []
        
        for sample_idx, (image, bbox) in enumerate(zip(images, bboxes)):
            image_np = np.array(image.convert("RGB"))
            h, w = image_np.shape[:2]
            
            base_tensor, target_edge = self._preprocess_image_tensor(image_np=image_np, processor=processor, device=device)
            # _preprocess_image_tensor uses direct resize to (target_edge, target_edge),
            # so bbox scaling must follow per-axis resize factors.
            sx = float(target_edge) / float(max(w, 1))
            sy = float(target_edge) / float(max(h, 1))
            x1, y1, x2, y2 = [float(v) for v in bbox]
            base_box = torch.tensor(
                [
                    float(np.clip(x1 * sx, 0.0, float(target_edge - 1))),
                    float(np.clip(y1 * sy, 0.0, float(target_edge - 1))),
                    float(np.clip(x2 * sx, 0.0, float(target_edge - 1))),
                    float(np.clip(y2 * sy, 0.0, float(target_edge - 1))),
                ],
                device=device,
                dtype=torch.float32,
            )
            target_edge = int(base_tensor.shape[-1])
            
            output_sizes_list.append((h, w))
            aug_count = 0
            
            for aug_name in self.augmentations:
                all_pixel_values.append(self._apply_tensor_aug(base_tensor, aug_name))
                all_input_boxes.append(self._augment_square_bbox(base_box, aug_name, target_edge))
                
                all_original_sizes.append(torch.tensor([h, w], dtype=torch.int64, device=device))
                all_reshaped_sizes.append(torch.tensor([target_edge, target_edge], dtype=torch.int64, device=device))
                aug_ids.append(int(self._aug_to_id[aug_name]))
                aug_count += 1
            
            true_aug_counts.append(aug_count)
        
        stacked_pixels = torch.stack(all_pixel_values, dim=0).contiguous()
        if stacked_pixels.dim() == 4:
            stacked_pixels = stacked_pixels.contiguous(memory_format=torch.channels_last)
        
        total_augs = len(aug_ids)
        target_batch = self.fixed_batch_size if self.fixed_batch_size > 0 else total_augs
        if target_batch > total_augs:
            pad_count = target_batch - total_augs
            stacked_pixels = torch.cat([stacked_pixels, stacked_pixels[-1:].repeat(pad_count, 1, 1, 1)], dim=0)
            all_input_boxes.extend([all_input_boxes[-1].clone() for _ in range(pad_count)])
            all_original_sizes.extend([all_original_sizes[-1].clone() for _ in range(pad_count)])
            all_reshaped_sizes.extend([all_reshaped_sizes[-1].clone() for _ in range(pad_count)])
            aug_ids.extend([aug_ids[-1]] * pad_count)
        
        inputs = {
            "pixel_values": stacked_pixels,
            "input_boxes": torch.stack(all_input_boxes, dim=0).unsqueeze(1),
            "original_sizes": torch.stack(all_original_sizes, dim=0),
            "reshaped_input_sizes": torch.stack(all_reshaped_sizes, dim=0),
        }
        
        if profiler is not None and profiler.enabled:
            profiler.record_duration("tta.build_inputs", time.perf_counter() - t0)

        return inputs, torch.tensor(aug_ids, device=device, dtype=torch.int64), output_sizes_list, true_aug_counts

    def _deaugment_grouped_batch(self, preds_t: torch.Tensor, aug_ids: torch.Tensor) -> torch.Tensor:
        if preds_t.dim() != 3:
            raise ValueError(f"Expected [N,H,W] predictions, got shape {tuple(preds_t.shape)}")
        out = preds_t.clone()

        for aug_name, aug_id in self._aug_to_id.items():
            idx = torch.where(aug_ids == int(aug_id))[0]
            if idx.numel() == 0:
                continue
            src = preds_t.index_select(0, idx)
            if aug_name == "none":
                mapped = src
            elif aug_name == "hflip":
                mapped = src.flip(-1)
            elif aug_name == "vflip":
                mapped = src.flip(-2)
            elif aug_name == "hvflip":
                mapped = src.flip(-2).flip(-1)
            elif aug_name == "rotate_90":
                mapped = torch.rot90(src, k=3, dims=(-2, -1))
            elif aug_name == "rotate_180":
                mapped = torch.rot90(src, k=2, dims=(-2, -1))
            elif aug_name == "rotate_270":
                mapped = torch.rot90(src, k=1, dims=(-2, -1))
            else:
                raise ValueError(f"Unsupported augmentation: {aug_name}")
            out.index_copy_(0, idx, mapped)
        return out

    def _deaugment_ordered_tensor(self, preds_t: torch.Tensor) -> torch.Tensor:
        """Fast deaugment path for ordered augment dimension [B, A, H, W]."""
        if preds_t.dim() != 4:
            raise ValueError(f"Expected [B,A,H,W], got shape {tuple(preds_t.shape)}")

        bsz, aug_n = int(preds_t.shape[0]), int(preds_t.shape[1])
        if aug_n <= 0:
            return preds_t

        out = preds_t.clone()
        ordered_augs = self.augmentations[:aug_n]
        for aug_pos, aug_name in enumerate(ordered_augs):
            src = preds_t[:, aug_pos]
            if aug_name == "none":
                mapped = src
            elif aug_name == "hflip":
                mapped = src.flip(-1)
            elif aug_name == "vflip":
                mapped = src.flip(-2)
            elif aug_name == "hvflip":
                mapped = src.flip(-2).flip(-1)
            elif aug_name == "rotate_90":
                mapped = torch.rot90(src, k=3, dims=(-2, -1))
            elif aug_name == "rotate_180":
                mapped = torch.rot90(src, k=2, dims=(-2, -1))
            elif aug_name == "rotate_270":
                mapped = torch.rot90(src, k=1, dims=(-2, -1))
            else:
                raise ValueError(f"Unsupported augmentation: {aug_name}")
            out[:, aug_pos].copy_(mapped)
        return out

    def _combine_fusion(
        self,
        preds: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse predictions along the augmentation dimension.

        Supports either [A, H, W] for one sample or [B, A, H, W] for a batch.
        """
        profiler = get_active_profiler()
        t0 = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
        if preds.dim() == 3:
            aug_dim = 0
        elif preds.dim() == 4:
            aug_dim = 1
        else:
            raise ValueError(f"Expected [A,H,W] or [B,A,H,W], got shape {tuple(preds.shape)}")

        stacked_t = preds.to(torch.float32)
        uncertainties_t = torch.nan_to_num(
            uncertainties.to(device=stacked_t.device, dtype=torch.float32),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        avg_uncertainty = uncertainties_t.mean(dim=aug_dim)

        if self.fusion_mode == "mean":
            fused_t = stacked_t.mean(dim=aug_dim)
        elif self.fusion_mode == "median":
            fused_t = torch.median(stacked_t, dim=aug_dim).values
        elif self.fusion_mode == "entropy_weighted":
            if stacked_t.shape[aug_dim] == 1:
                fused_t = stacked_t.select(aug_dim, 0)
            else:
                weights = torch.softmax(-uncertainties_t, dim=aug_dim)
                if aug_dim == 0:
                    fused_t = torch.sum(stacked_t * weights[:, None, None], dim=0)
                else:
                    fused_t = torch.sum(stacked_t * weights[:, :, None, None], dim=1)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        if profiler is not None and profiler.enabled:
            profiler.record_duration(f"tta.fuse.{self.fusion_mode}", time.perf_counter() - t0)
        return fused_t, avg_uncertainty

    def _fuse_predictions(
        self,
        preds: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Fuse multiple predictions using specified strategy."""
        fused_t, avg_uncertainty = self._combine_fusion(preds, uncertainties)
        return fused_t, float(avg_uncertainty.item())

    def _resize_batch_grouped(
        self,
        fused_batch: torch.Tensor,
        output_sizes_list: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """Resize predictions by grouping samples with same target size to reduce kernel launches."""
        if fused_batch.dim() != 3:
            raise ValueError(f"Expected [B,H,W], got shape {tuple(fused_batch.shape)}")
        if int(fused_batch.shape[0]) != len(output_sizes_list):
            raise ValueError("Batch size and output_sizes_list length mismatch")

        grouped: Dict[Tuple[int, int], List[int]] = {}
        for idx, size in enumerate(output_sizes_list):
            grouped.setdefault((int(size[0]), int(size[1])), []).append(idx)

        out: List[Optional[torch.Tensor]] = [None] * int(fused_batch.shape[0])
        for (out_h, out_w), idxs in grouped.items():
            idx_t = torch.as_tensor(idxs, device=fused_batch.device, dtype=torch.long)
            chunk = fused_batch.index_select(0, idx_t)
            if int(chunk.shape[-2]) != out_h or int(chunk.shape[-1]) != out_w:
                chunk = F.interpolate(
                    chunk.unsqueeze(1),
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            for local_i, sample_i in enumerate(idxs):
                out[sample_i] = chunk[local_i]

        return [t for t in out if t is not None]

    def _auto_tune_chunk_size(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        bbox: List[int],
        device: str,
    ) -> None:
        """Auto-tune chunk_size on first sample to find optimal value for this GPU."""
        if self._chunk_size_tuned:
            return
        if not self._autotune_enabled:
            self._chunk_size_tuned = True
            return

        cuda_mem_gb = _cuda_total_memory_gb() if device == "cuda" else None
        if device != "cuda":
            test_sizes = [1, 2, 4, 8]
        elif cuda_mem_gb is not None and cuda_mem_gb <= 12.5:
            test_sizes = [4, 8, 12, 16, 24, 32]
        else:
            test_sizes = [8, 12, 16, 24, 32, 48, 64]
        best_size = self.infer_chunk_size
        best_speed = 0.0
        
        print(f"[TTA Tuner] Searching optimal chunk_size on first sample (sizes: {test_sizes})", flush=True)
        
        for test_chunk_size in test_sizes:
            # Build inputs once
            inputs, aug_names, output_size, true_aug_count = self._build_tta_inputs_batch(
                processor=processor,
                images=[image],
                bboxes=[bbox],
                device=device,
            )
            
            pixel_values = inputs["pixel_values"]
            input_boxes = inputs["input_boxes"]
            original_sizes = inputs["original_sizes"]
            reshaped_input_sizes = inputs["reshaped_input_sizes"]
            
            total_count = int(pixel_values.shape[0])
            
            # Warm up
            try:
                torch.cuda.synchronize() if device == "cuda" else None
                t_start = time.perf_counter()
                
                pred_chunks: List[torch.Tensor] = []
                for start in range(0, total_count, test_chunk_size):
                    end = min(start + test_chunk_size, total_count)
                    chunk_inputs = {
                        "pixel_values": pixel_values[start:end],
                        "input_boxes": input_boxes[start:end],
                        "original_sizes": original_sizes[start:end],
                        "reshaped_input_sizes": reshaped_input_sizes[start:end],
                    }
                    pred_chunk = predict_prob_masks_from_inputs(
                        model=model,
                        inputs=chunk_inputs,
                        device=device,
                        output_size=None,
                        use_amp=True,
                        # NOTE:
                        # TTA long runs may hit non-finite outputs when reusing CUDA graph replay.
                        # Force eager path here for numerical stability.
                        inputs_already_on_device=False,
                    )[:, 0]
                    pred_chunks.append(pred_chunk)
                
                torch.cuda.synchronize() if device == "cuda" else None
                elapsed = time.perf_counter() - t_start
                throughput = total_count / elapsed if elapsed > 0 else 0.0
                
                print(f"[TTA Tuner]   chunk_size={test_chunk_size}: {elapsed:.3f}s, throughput={throughput:.1f} augs/s", flush=True)
                
                if throughput > best_speed:
                    best_speed = throughput
                    best_size = test_chunk_size
                
                del pred_chunks
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                    print(f"[TTA Tuner]   chunk_size={test_chunk_size}: OOM", flush=True)
                    if device == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise
        
        self.infer_chunk_size = best_size
        self._chunk_size_tuned = True
        print(f"[TTA Tuner] Selected chunk_size={best_size} (throughput={best_speed:.1f} augs/s)", flush=True)

    def predict(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        bbox: List[int],
        device: str,
    ) -> Tuple[torch.Tensor, float]:
        """
        Predict with test-time augmentation (single image).

        Returns:
            (prob_mask_tensor, avg_uncertainty)
        """
        probs, uncertainties = self.predict_batch(
            model=model,
            processor=processor,
            images=[image],
            bboxes=[bbox],
            device=device,
        )
        return probs[0], float(uncertainties[0])

    def predict_batch(
        self,
        model: Any,
        processor: Any,
        images: List[Image.Image],
        bboxes: List[List[int]],
        device: str,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Predict with TTA for multiple images in a single batched forward pass.
        
        Args:
            model: The segmentation model
            processor: The image processor
            images: List of PIL images
            bboxes: List of bboxes corresponding to images
            device: Device to run on ("cuda" or "cpu")
        
        Returns:
            (probability_masks, avg_uncertainties)
            probability_masks: list of [H,W] probability tensors
            avg_uncertainties: list of float uncertainties
        """
        if not images:
            return [], []
        
        # Auto-tune on first call
        if not self._chunk_size_tuned:
            self._auto_tune_chunk_size(
                model=model,
                processor=processor,
                image=images[0],
                bbox=bboxes[0],
                device=device,
            )
        
        profiler = get_active_profiler()
        t_predict_total = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
        
        # Build all augmentations for all samples in single batch
        inputs, aug_ids, output_sizes_list, true_aug_counts = self._build_tta_inputs_batch(
            processor=processor,
            images=images,
            bboxes=bboxes,
            device=device,
        )

        pixel_values = inputs["pixel_values"]
        input_boxes = inputs["input_boxes"]
        original_sizes = inputs["original_sizes"]
        reshaped_input_sizes = inputs["reshaped_input_sizes"]

        total_count = int(pixel_values.shape[0])
        chunk_size = max(1, self.infer_chunk_size)

        # Chunk-wise inference with dynamic OOM recovery
        while True:
            pred_chunks: List[torch.Tensor] = []
            try:
                for start in range(0, total_count, chunk_size):
                    end = min(start + chunk_size, total_count)
                    chunk_inputs = {
                        "pixel_values": pixel_values[start:end],
                        "input_boxes": input_boxes[start:end],
                        "original_sizes": original_sizes[start:end],
                        "reshaped_input_sizes": reshaped_input_sizes[start:end],
                    }
                    t_chunk = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
                    pred_chunk = predict_prob_masks_from_inputs(
                        model=model,
                        inputs=chunk_inputs,
                        device=device,
                        output_size=None,
                        use_amp=True,
                        # Keep autotune behavior aligned with predict_batch inference path.
                        inputs_already_on_device=False,
                    )[:, 0]
                    if profiler is not None and profiler.enabled:
                        profiler.record_duration("tta.chunk_inference", time.perf_counter() - t_chunk)
                    pred_chunks.append(pred_chunk)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                is_oom = "out of memory" in msg or "cuda" in msg and "memory" in msg
                if not is_oom or chunk_size <= 1:
                    raise
                if device == "cuda":
                    torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)

        pred_batch_t = torch.cat(pred_chunks, dim=0)

        # Ignore optional pad entries; only true augmentations should contribute.
        valid_total = int(sum(int(c) for c in true_aug_counts))
        pred_batch_t = pred_batch_t[:valid_total]
        aug_ids = aug_ids[:valid_total]

        n_samples = len(images)
        result_probs: List[torch.Tensor] = []
        result_uncertainties: List[float] = []

        # Fast path: all samples share same augmentation count and ordered augmentation ids.
        aug_count = int(true_aug_counts[0]) if (n_samples > 0 and true_aug_counts) else 0
        can_vectorize = (
            n_samples > 0
            and aug_count > 0
            and all(int(c) == aug_count for c in true_aug_counts)
            and valid_total == n_samples * aug_count
        )

        if can_vectorize:
            pred_4d = pred_batch_t.view(n_samples, aug_count, pred_batch_t.shape[-2], pred_batch_t.shape[-1])
            aug_2d = aug_ids.view(n_samples, aug_count)
            expected_aug_ids = torch.arange(aug_count, device=aug_ids.device, dtype=aug_ids.dtype).unsqueeze(0).expand(n_samples, -1)
            ordered = bool(torch.equal(aug_2d, expected_aug_ids))

            if ordered:
                t_deaug = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
                deaug_4d = self._deaugment_ordered_tensor(pred_4d)
                if profiler is not None and profiler.enabled:
                    profiler.record_duration("tta.deaugment", time.perf_counter() - t_deaug)

                prob_4d = deaug_4d.to(torch.float32).clamp(1e-6, 1.0 - 1e-6)
                entropy_4d = -(prob_4d * torch.log(prob_4d) + (1.0 - prob_4d) * torch.log1p(-prob_4d))
                uncertainties_ba = entropy_4d.flatten(2).mean(dim=2).to(torch.float32)

                fused_batch, uncertainty_mean_batch = self._combine_fusion(deaug_4d, uncertainties_ba)

                resized_probs = self._resize_batch_grouped(fused_batch, output_sizes_list)
                for sample_idx in range(n_samples):
                    result_probs.append(resized_probs[sample_idx])
                    result_uncertainties.append(float(uncertainty_mean_batch[sample_idx].item()))
            else:
                can_vectorize = False

        if not can_vectorize:
            aug_offsets: List[int] = []
            offset = 0
            for count in true_aug_counts:
                aug_offsets.append(offset)
                offset += int(count)

            fused_list: List[torch.Tensor] = []
            for sample_idx in range(n_samples):
                true_aug_count = true_aug_counts[sample_idx]
                start = aug_offsets[sample_idx]
                end = start + int(true_aug_count)

                t_deaug = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
                sample_pred_t = pred_batch_t[start:end]
                sample_aug_ids = aug_ids[start:end]
                stacked_t = self._deaugment_grouped_batch(sample_pred_t, sample_aug_ids)
                if profiler is not None and profiler.enabled:
                    profiler.record_duration("tta.deaugment", time.perf_counter() - t_deaug)

                prob_t = stacked_t.to(torch.float32).clamp(1e-6, 1.0 - 1e-6)
                entropy_t = -(prob_t * torch.log(prob_t) + (1.0 - prob_t) * torch.log1p(-prob_t))
                uncertainties_t = entropy_t.reshape(entropy_t.shape[0], -1).mean(dim=1).to(torch.float32)

                fused_t, uncertainties_mean = self._fuse_predictions(stacked_t, uncertainties_t)
                fused_list.append(fused_t)
                result_uncertainties.append(float(uncertainties_mean))

            fused_batch = torch.stack(fused_list, dim=0)
            result_probs.extend(self._resize_batch_grouped(fused_batch, output_sizes_list))

        if profiler is not None and profiler.enabled:
            profiler.record_duration("tta.predict_total", time.perf_counter() - t_predict_total)

        return result_probs, result_uncertainties


def compute_metrics_tensor(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:
    pred = pred_mask.to(dtype=torch.bool).reshape(-1)
    gt = gt_mask.to(dtype=torch.bool).reshape(-1)

    tp = torch.logical_and(pred, gt).sum().to(torch.float32)
    fp = torch.logical_and(pred, torch.logical_not(gt)).sum().to(torch.float32)
    fn = torch.logical_and(torch.logical_not(pred), gt).sum().to(torch.float32)

    eps = pred.new_tensor(1e-8, dtype=torch.float32)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    jaccard = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    sensitivity = recall
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    return {
        "dice": float(dice.item()),
        "jaccard": float(jaccard.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "sensitivity": float(sensitivity.item()),
        "f1": float(f1.item()),
        "tp": int(tp.item()),
        "fp": int(fp.item()),
        "fn": int(fn.item()),
    }


def compute_metrics_batch_tensor(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred = pred_masks.to(dtype=torch.bool)
    gt = gt_masks.to(dtype=torch.bool)

    if pred.dim() != gt.dim():
        raise ValueError(f"pred and gt must have same rank, got {pred.dim()} vs {gt.dim()}")

    reduce_dims = tuple(range(1, pred.dim())) if pred.dim() > 1 else (0,)
    tp = torch.logical_and(pred, gt).sum(dim=reduce_dims).to(torch.float32)
    fp = torch.logical_and(pred, torch.logical_not(gt)).sum(dim=reduce_dims).to(torch.float32)
    fn = torch.logical_and(torch.logical_not(pred), gt).sum(dim=reduce_dims).to(torch.float32)

    eps = pred.new_tensor(1e-8, dtype=torch.float32)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    jaccard = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    sensitivity = recall
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    return {
        "dice": dice,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "f1": f1,
        "tp": tp.to(torch.int64),
        "fp": fp.to(torch.int64),
        "fn": fn.to(torch.int64),
    }


def compute_bce_batch_tensor(prob_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Binary Cross-Entropy (paper Section 3.3).

    L_BCE = -(1/N) * Σ [g * log(p) + (1-g) * log(1-p)]

    Args:
        prob_masks: [B, H, W] float32 probability values in [0, 1]
        gt_masks:   [B, H, W] float32 binary ground-truth masks
    Returns:
        [B] tensor of per-sample BCE values
    """
    p = prob_masks.to(torch.float32).clamp(1e-7, 1.0 - 1e-7)
    g = gt_masks.to(torch.float32)
    reduce_dims = tuple(range(1, p.dim()))
    return -(g * p.log() + (1.0 - g) * (1.0 - p).log()).mean(dim=reduce_dims)


def _surface_boundary(mask: np.ndarray) -> np.ndarray:
    if ndimage is None:
        return mask
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = np.logical_xor(mask, eroded)
    if not boundary.any():
        return mask
    return boundary


def _compute_surface_metrics_np(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if not pred.any() and not gt.any():
        return 0.0, 0.0

    fallback = float(np.hypot(pred.shape[0], pred.shape[1]))
    if (not pred.any()) or (not gt.any()):
        return fallback, fallback

    if ndimage is None:
        return float("nan"), float("nan")

    pred_b = _surface_boundary(pred)
    gt_b = _surface_boundary(gt)

    dt_gt = ndimage.distance_transform_edt(~gt_b)
    dt_pred = ndimage.distance_transform_edt(~pred_b)
    d_pred_to_gt = dt_gt[pred_b]
    d_gt_to_pred = dt_pred[gt_b]

    if d_pred_to_gt.size == 0 and d_gt_to_pred.size == 0:
        return 0.0, 0.0

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)
    hd95 = float(np.percentile(all_d, 95)) if all_d.size > 0 else 0.0

    mean_a = float(d_pred_to_gt.mean()) if d_pred_to_gt.size > 0 else 0.0
    mean_b = float(d_gt_to_pred.mean()) if d_gt_to_pred.size > 0 else 0.0
    assd = 0.5 * (mean_a + mean_b)
    return hd95, assd


def compute_surface_metrics_batch_tensor(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
    if pred_masks.shape != gt_masks.shape:
        raise ValueError(f"pred_masks and gt_masks shape mismatch: {tuple(pred_masks.shape)} vs {tuple(gt_masks.shape)}")

    pred_np = pred_masks.detach().to(device="cpu", dtype=torch.bool).numpy()
    gt_np = gt_masks.detach().to(device="cpu", dtype=torch.bool).numpy()

    hd95_vals: List[float] = []
    assd_vals: List[float] = []
    for i in range(pred_np.shape[0]):
        hd95, assd = _compute_surface_metrics_np(pred_np[i], gt_np[i])
        hd95_vals.append(float(hd95))
        assd_vals.append(float(assd))

    return {
        "hd95": torch.tensor(hd95_vals, device=pred_masks.device, dtype=torch.float32),
        "assd": torch.tensor(assd_vals, device=pred_masks.device, dtype=torch.float32),
    }


def compute_ece_batch_tensor(prob_masks: torch.Tensor, gt_masks: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    p = prob_masks.to(torch.float32).clamp(1e-7, 1.0 - 1e-7)
    g = gt_masks.to(torch.float32)

    if p.dim() != 3 or g.dim() != 3:
        raise ValueError(f"Expected [B,H,W] tensors for ECE, got {tuple(p.shape)} and {tuple(g.shape)}")

    p_flat = p.reshape(p.shape[0], -1)
    g_flat = g.reshape(g.shape[0], -1)

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device, dtype=torch.float32)
    ece = torch.zeros((p.shape[0],), device=p.device, dtype=torch.float32)

    for b in range(n_bins):
        left = edges[b]
        right = edges[b + 1]
        if b == 0:
            in_bin = (p_flat >= left) & (p_flat <= right)
        else:
            in_bin = (p_flat > left) & (p_flat <= right)

        bin_count = in_bin.sum(dim=1).to(torch.float32)
        nonzero = bin_count > 0
        if not bool(nonzero.any().item()):
            continue

        bin_prob = torch.where(in_bin, p_flat, torch.zeros_like(p_flat)).sum(dim=1)
        bin_acc = torch.where(in_bin, g_flat, torch.zeros_like(g_flat)).sum(dim=1)
        conf = torch.zeros_like(bin_count)
        acc = torch.zeros_like(bin_count)
        conf[nonzero] = bin_prob[nonzero] / bin_count[nonzero]
        acc[nonzero] = bin_acc[nonzero] / bin_count[nonzero]

        weight = bin_count / float(p_flat.shape[1])
        ece += torch.abs(acc - conf) * weight

    return ece


def _compute_ood_detection_stats(ood_scores: List[float], ood_labels: List[int]) -> Dict[str, Any]:
    scores = np.asarray(ood_scores, dtype=np.float64)
    labels = np.asarray(ood_labels, dtype=np.int64)

    if scores.shape[0] != labels.shape[0] or scores.size == 0:
        return {
            "ood_eval_num_samples": int(0),
            "ood_eval_num_positive": int(0),
            "ood_eval_num_negative": int(0),
            "ood_auroc": float("nan"),
            "ood_fpr95": float("nan"),
        }

    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    if n_pos == 0 or n_neg == 0:
        return {
            "ood_eval_num_samples": int(scores.size),
            "ood_eval_num_positive": n_pos,
            "ood_eval_num_negative": n_neg,
            "ood_auroc": float("nan"),
            "ood_fpr95": float("nan"),
        }

    order = np.argsort(scores)
    scores_sorted = scores[order]
    ranks_sorted = np.arange(1, scores.size + 1, dtype=np.float64)

    tie_start = 0
    while tie_start < scores.size:
        tie_end = tie_start
        while tie_end + 1 < scores.size and scores_sorted[tie_end + 1] == scores_sorted[tie_start]:
            tie_end += 1
        if tie_end > tie_start:
            avg_rank = 0.5 * (tie_start + 1 + tie_end + 1)
            ranks_sorted[tie_start : tie_end + 1] = avg_rank
        tie_start = tie_end + 1

    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    sum_ranks_pos = float(ranks[pos].sum())
    auroc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)

    pos_scores = scores[pos]
    neg_scores = scores[neg]
    threshold = float(np.percentile(pos_scores, 5.0))
    fpr95 = float(np.mean(neg_scores >= threshold))

    return {
        "ood_eval_num_samples": int(scores.size),
        "ood_eval_num_positive": n_pos,
        "ood_eval_num_negative": n_neg,
        "ood_auroc": float(auroc),
        "ood_fpr95": float(fpr95),
    }


def _tta_predict_with_oom_recovery(
    *,
    tta_predictor: TTAPredictor,
    model: Any,
    processor: Any,
    images: List[Image.Image],
    bboxes: List[List[int]],
    device: str,
) -> Tuple[List[torch.Tensor], List[float]]:
    """Run TTA with adaptive sample micro-batching to avoid CUDA OOM."""
    n = len(images)
    if n == 0:
        return [], []

    micro_bs = n
    while True:
        probs_all: List[torch.Tensor] = []
        uncs_all: List[float] = []
        try:
            for start in range(0, n, micro_bs):
                end = min(start + micro_bs, n)
                probs, uncs = tta_predictor.predict_batch(
                    model=model,
                    processor=processor,
                    images=images[start:end],
                    bboxes=bboxes[start:end],
                    device=device,
                )
                probs_all.extend(probs)
                uncs_all.extend([float(u) for u in uncs])
            return probs_all, uncs_all
        except RuntimeError as e:
            msg = str(e).lower()
            is_oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg)
            if not is_oom or micro_bs <= 1:
                raise
            if device == "cuda":
                torch.cuda.empty_cache()
            micro_bs = max(1, micro_bs // 2)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def _predict_baseline_batch_tensor(
    *,
    model: Any,
    processor: Any,
    images: List[Image.Image],
    bboxes: List[List[int]],
    dataset_name: str,
    sample_names: List[str],
    device: str,
    pred_cache: Optional[PredictionCache],
    profiler: Optional[PerformanceProfiler],
    profile_prefix: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    pred_masks_t: List[Optional[torch.Tensor]] = [None] * len(images)
    prob_for_ood_t: List[Optional[torch.Tensor]] = [None] * len(images)

    miss_indices: List[int] = []
    miss_images: List[Image.Image] = []
    miss_boxes: List[List[int]] = []
    miss_keys: List[str] = []

    for i, (sample_name, bbox) in enumerate(zip(sample_names, bboxes)):
        t_cache = time.perf_counter()
        cache_key = make_cache_key(dataset_name, sample_name, bbox, mode="baseline")
        cached = pred_cache.get(cache_key) if pred_cache is not None else None
        if profiler is not None and profiler.enabled:
            profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.cache_lookup", time.perf_counter() - t_cache)

        if cached is None:
            miss_indices.append(i)
            miss_images.append(images[i])
            miss_boxes.append(bbox)
            miss_keys.append(cache_key)
            continue

        cached_t = torch.from_numpy(cached)
        if cached_t.dtype in (torch.uint8, torch.bool):
            prob_t = cached_t.to(torch.float32)
        else:
            prob_t = cached_t.to(torch.float32)
        pred_t = (prob_t > 0.5).to(torch.uint8)
        if device == "cuda":
            pred_t = pred_t.to(device=device, non_blocking=True)
            prob_t = prob_t.to(device=device, non_blocking=True)
        else:
            pred_t = pred_t.to(device=device)
            prob_t = prob_t.to(device=device)
        pred_masks_t[i] = pred_t
        prob_for_ood_t[i] = prob_t

    if miss_indices:
        packed_boxes = [[box] for box in miss_boxes]
        output_w, output_h = miss_images[0].size

        t_pred = time.perf_counter()
        batch_inputs = build_inputs_batch(processor=processor, images=miss_images, input_boxes=packed_boxes)
        prob_batch = predict_prob_masks_from_inputs(
            model=model,
            inputs=batch_inputs,
            device=device,
            output_size=(output_h, output_w),
            use_amp=True,
            inputs_already_on_device=False,
        )[:, 0]
        pred_batch = (prob_batch > 0.5).to(torch.uint8)
        if profiler is not None and profiler.enabled:
            profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.predict_binary_mask", time.perf_counter() - t_pred)

        for local_idx, global_idx in enumerate(miss_indices):
            prob_t = prob_batch[local_idx]
            pred_t = pred_batch[local_idx]
            pred_masks_t[global_idx] = pred_t
            prob_for_ood_t[global_idx] = prob_t
            if pred_cache is not None:
                t_put = time.perf_counter()
                pred_cache.put(miss_keys[local_idx], prob_t.detach().to(torch.float16).cpu().numpy())
                if profiler is not None and profiler.enabled:
                    profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.cache_store", time.perf_counter() - t_put)

    return [p for p in pred_masks_t if p is not None], [p for p in prob_for_ood_t if p is not None]


def evaluate_dataset(
    dataset: Any,
    dataset_name: str,
    model: Any,
    processor: Any,
    device: str,
    use_ood: bool,
    use_tta: bool,
    ood_detector: Optional[OODDetector],
    tta_predictor: Optional[TTAPredictor],
    pred_cache: Optional[PredictionCache] = None,
    profiler: Optional[PerformanceProfiler] = None,
    profile_prefix: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics_keys = ["dice", "jaccard", "precision", "recall", "sensitivity", "f1", "bce", "hd95", "assd", "ece"]
    metrics_store: Dict[str, List[float]] = {k: [] for k in metrics_keys}

    results: List[Dict[str, Any]] = []
    ood_scores: List[float] = []
    ood_eval_scores: List[float] = []
    ood_eval_labels: List[int] = []
    uncertainties: List[float] = []
    inference_times: List[float] = []
    data_times: List[float] = []
    ood_times: List[float] = []
    metrics_times: List[float] = []
    post_times: List[float] = []

    start = time.perf_counter()

    default_eval_workers = str(_auto_eval_workers(device))
    # TTA path benefits from small batching; keep baseline higher and TTA moderate to avoid OOM.
    if device == "cuda":
        cuda_mem_gb = _cuda_total_memory_gb()
        if not use_tta:
            default_eval_batch = "8"
        else:
            default_eval_batch = "2" if (cuda_mem_gb is not None and cuda_mem_gb <= 12.5) else "4"
    else:
        default_eval_batch = "1"
    eval_workers = max(0, int(os.getenv("MEDSAM_EVAL_WORKERS", default_eval_workers)))
    eval_batch_size = max(1, int(os.getenv("MEDSAM_EVAL_BATCH", default_eval_batch)))
    eval_prefetch = max(2, int(os.getenv("MEDSAM_EVAL_PREFETCH", "4")))
    pin_memory = device == "cuda"

    _maybe_warm_dataset_cache(dataset=dataset, dataset_name=dataset_name, profiler=profiler, profile_prefix=profile_prefix)

    if eval_workers > 0:
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=eval_prefetch,
            collate_fn=lambda b: b,
        )
        iterable = loader
        total = len(loader)
    else:
        iterable = ([dataset[idx]] for idx in range(len(dataset)))
        total = len(dataset)

    sample_index = 0
    for batch_samples in tqdm(iterable, total=total, desc=f"Evaluating {dataset_name}"):
        batch_start = time.perf_counter()
        images = [s["image"] for s in batch_samples]
        bboxes = [s["bbox"] for s in batch_samples]
        sample_names = [str(s.get("name", f"sample_{sample_index + i}")) for i, s in enumerate(batch_samples)]
        batch_ood_labels: List[Optional[int]] = []
        for s in batch_samples:
            raw_label = s.get("ood_label", s.get("is_ood_gt", s.get("is_ood", None)))
            if raw_label is None:
                batch_ood_labels.append(None)
            else:
                batch_ood_labels.append(int(bool(raw_label)))
        gt_masks_t = [
            (s["mask"] if isinstance(s["mask"], torch.Tensor) else torch.as_tensor(s["mask"]))
            .to(device=device, dtype=torch.float32, non_blocking=(device == "cuda"))
            for s in batch_samples
        ]
        per_sample_data_time = (time.perf_counter() - batch_start) / max(1, len(batch_samples))
        data_times.extend([per_sample_data_time] * len(batch_samples))

        t0 = time.perf_counter()
        batch_uncertainties: List[float] = [0.0] * len(batch_samples)
        if use_tta and tta_predictor is not None:
            # Batch TTA: process all samples + augmentations in single forward
            pred_masks_t: List[torch.Tensor] = []
            prob_for_ood_t: List[torch.Tensor] = []
            t_tta = time.perf_counter()
            tta_prob_t, tta_uncertainties = _tta_predict_with_oom_recovery(
                tta_predictor=tta_predictor,
                model=model,
                processor=processor,
                images=images,
                bboxes=bboxes,
                device=device,
            )
            for i, prob_t in enumerate(tta_prob_t):
                pred_masks_t.append((prob_t > 0.5).to(torch.uint8))
                prob_for_ood_t.append(prob_t)
                u = float(tta_uncertainties[i])
                batch_uncertainties[i] = u
                uncertainties.append(u)
            if profiler is not None and profiler.enabled:
                profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.tta_predict", time.perf_counter() - t_tta)
        else:
            pred_masks_t, prob_for_ood_t = _predict_baseline_batch_tensor(
                model=model,
                processor=processor,
                images=images,
                bboxes=bboxes,
                dataset_name=dataset_name,
                sample_names=sample_names,
                device=device,
                pred_cache=pred_cache,
                profiler=profiler,
                profile_prefix=profile_prefix,
            )

        per_sample_infer_time = (time.perf_counter() - t0) / max(1, len(batch_samples))
        inference_times.extend([per_sample_infer_time] * len(batch_samples))

        pred_batch_t = torch.stack(pred_masks_t, dim=0)
        gt_batch_t = torch.stack(gt_masks_t, dim=0)
        prob_batch_t = torch.stack(prob_for_ood_t, dim=0)

        t_metric = time.perf_counter()
        batch_metrics = compute_metrics_batch_tensor(pred_batch_t, gt_batch_t)
        batch_metrics["bce"] = compute_bce_batch_tensor(prob_batch_t, gt_batch_t)
        surface_metrics = compute_surface_metrics_batch_tensor(pred_batch_t, gt_batch_t)
        batch_metrics["hd95"] = surface_metrics["hd95"]
        batch_metrics["assd"] = surface_metrics["assd"]
        batch_metrics["ece"] = compute_ece_batch_tensor(prob_batch_t, gt_batch_t)
        metric_elapsed = time.perf_counter() - t_metric
        metrics_times.extend([metric_elapsed / max(1, len(batch_samples))] * len(batch_samples))
        if profiler is not None and profiler.enabled:
            profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.metrics_batch", metric_elapsed)

        t_ood = time.perf_counter()
        if use_ood and ood_detector is not None:
            ood_batch = ood_detector.detect_batch_tensor(prob_batch_t)
        else:
            ood_batch = [
                {"ood_score": 0.0, "is_ood": False, "confidence": 0.0}
                for _ in range(len(batch_samples))
            ]
        ood_elapsed = time.perf_counter() - t_ood
        ood_times.extend([ood_elapsed / max(1, len(batch_samples))] * len(batch_samples))
        if profiler is not None and profiler.enabled:
            profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.ood_batch", ood_elapsed)

        t_post = time.perf_counter()
        for i in range(len(batch_samples)):
            ood = ood_batch[i]
            ood_score = float(ood["ood_score"])
            is_ood = bool(ood["is_ood"])
            confidence = float(ood["confidence"])
            if use_ood and ood_detector is not None:
                ood_scores.append(ood_score)
                ood_label = batch_ood_labels[i]
                if ood_label is not None:
                    ood_eval_scores.append(ood_score)
                    ood_eval_labels.append(int(ood_label))

            m = {k: float(batch_metrics[k][i].item()) for k in metrics_keys}
            for k in metrics_keys:
                metrics_store[k].append(float(m[k]))

            results.append(
                {
                    "index": int(sample_index),
                    "name": sample_names[i],
                    "dice": float(m["dice"]),
                    "jaccard": float(m["jaccard"]),
                    "precision": float(m["precision"]),
                    "recall": float(m["recall"]),
                    "sensitivity": float(m["sensitivity"]),
                    "f1": float(m["f1"]),
                    "bce": float(m["bce"]),
                    "hd95": float(m["hd95"]),
                    "assd": float(m["assd"]),
                    "ece": float(m["ece"]),
                    "ood_score": ood_score,
                    "is_ood": is_ood,
                    "ood_label": batch_ood_labels[i],
                    "confidence": confidence,
                    "uncertainty": float(batch_uncertainties[i]),
                }
            )
            if profiler is not None and profiler.enabled:
                profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.sample_total", time.perf_counter() - batch_start)
            sample_index += 1
        post_elapsed = time.perf_counter() - t_post
        post_times.extend([post_elapsed / max(1, len(batch_samples))] * len(batch_samples))

    total_time = time.perf_counter() - start

    stats = _build_stats_from_store(
        dataset_name=dataset_name,
        results=results,
        metrics_store=metrics_store,
        inference_times=inference_times,
        data_times=data_times,
        ood_times=ood_times,
        metrics_times=metrics_times,
        post_times=post_times,
        total_time=total_time,
        ood_scores=ood_scores,
        uncertainties=uncertainties,
        ood_eval_scores=ood_eval_scores,
        ood_eval_labels=ood_eval_labels,
    )

    if profiler is not None and profiler.enabled:
        prefix = profile_prefix or f"eval.{dataset_name}"
        component_totals = {
            "data": float(np.sum(data_times)),
            "inference": float(np.sum(inference_times)),
            "ood": float(np.sum(ood_times)),
            "metrics": float(np.sum(metrics_times)),
            "post": float(np.sum(post_times)),
        }
        profiler.record_duration(f"{prefix}.data", component_totals["data"], count=max(1, len(data_times)))
        profiler.record_duration(f"{prefix}.inference", component_totals["inference"], count=max(1, len(inference_times)))
        profiler.record_duration(f"{prefix}.ood", component_totals["ood"], count=max(1, len(ood_times)))
        profiler.record_duration(f"{prefix}.metrics", component_totals["metrics"], count=max(1, len(metrics_times)))
        profiler.record_duration(f"{prefix}.post", component_totals["post"], count=max(1, len(post_times)))
        profiler.record_duration(f"{prefix}.total", total_time, count=max(1, len(results)))
        profiler.flush()

    return results, stats


def evaluate_dataset_ood_tta(
    dataset: Any,
    dataset_name: str,
    model: Any,
    processor: Any,
    device: str,
    ood_detector: OODDetector,
    tta_predictor: TTAPredictor,
    pred_cache: Optional[PredictionCache] = None,
    profiler: Optional[PerformanceProfiler] = None,
    profile_prefix: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    metrics_keys = ["dice", "jaccard", "precision", "recall", "sensitivity", "f1", "bce", "hd95", "assd", "ece"]

    ood_metrics_store: Dict[str, List[float]] = {k: [] for k in metrics_keys}
    tta_metrics_store: Dict[str, List[float]] = {k: [] for k in metrics_keys}
    ood_results: List[Dict[str, Any]] = []
    tta_results: List[Dict[str, Any]] = []
    ood_scores: List[float] = []
    ood_eval_scores: List[float] = []
    ood_eval_labels: List[int] = []
    uncertainties: List[float] = []

    inference_times_ood: List[float] = []
    inference_times_tta: List[float] = []
    data_times: List[float] = []
    ood_times: List[float] = []
    metrics_times: List[float] = []
    post_times: List[float] = []

    start = time.perf_counter()
    default_eval_workers = str(_auto_eval_workers(device))
    if device == "cuda":
        cuda_mem_gb = _cuda_total_memory_gb()
        default_eval_batch = "2" if (cuda_mem_gb is not None and cuda_mem_gb <= 12.5) else "4"
    else:
        default_eval_batch = "1"
    eval_workers = max(0, int(os.getenv("MEDSAM_EVAL_WORKERS", default_eval_workers)))
    eval_batch_size = max(1, int(os.getenv("MEDSAM_EVAL_BATCH", default_eval_batch)))
    eval_prefetch = max(2, int(os.getenv("MEDSAM_EVAL_PREFETCH", "4")))
    pin_memory = device == "cuda"

    _maybe_warm_dataset_cache(dataset=dataset, dataset_name=dataset_name, profiler=profiler, profile_prefix=profile_prefix)

    if eval_workers > 0:
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=eval_prefetch,
            collate_fn=lambda b: b,
        )
        iterable = loader
        total = len(loader)
    else:
        iterable = ([dataset[idx]] for idx in range(len(dataset)))
        total = len(dataset)

    sample_index = 0
    for batch_samples in tqdm(iterable, total=total, desc=f"Evaluating {dataset_name} (ood+tta)"):
        batch_start = time.perf_counter()
        images = [s["image"] for s in batch_samples]
        bboxes = [s["bbox"] for s in batch_samples]
        sample_names = [str(s.get("name", f"sample_{sample_index + i}")) for i, s in enumerate(batch_samples)]
        batch_ood_labels: List[Optional[int]] = []
        for s in batch_samples:
            raw_label = s.get("ood_label", s.get("is_ood_gt", s.get("is_ood", None)))
            if raw_label is None:
                batch_ood_labels.append(None)
            else:
                batch_ood_labels.append(int(bool(raw_label)))
        gt_masks_t = [
            (s["mask"] if isinstance(s["mask"], torch.Tensor) else torch.as_tensor(s["mask"]))
            .to(device=device, dtype=torch.float32, non_blocking=(device == "cuda"))
            for s in batch_samples
        ]
        per_sample_data_time = (time.perf_counter() - batch_start) / max(1, len(batch_samples))
        data_times.extend([per_sample_data_time] * len(batch_samples))

        t_ood_inf = time.perf_counter()
        ood_pred_masks_t, ood_prob_t = _predict_baseline_batch_tensor(
            model=model,
            processor=processor,
            images=images,
            bboxes=bboxes,
            dataset_name=dataset_name,
            sample_names=sample_names,
            device=device,
            pred_cache=pred_cache,
            profiler=profiler,
            profile_prefix=f"{profile_prefix}.ood" if profile_prefix else f"eval.{dataset_name}.ood",
        )
        inference_times_ood.extend([(time.perf_counter() - t_ood_inf) / max(1, len(batch_samples))] * len(batch_samples))

        t_tta_inf = time.perf_counter()
        tta_prob_t, tta_batch_uncertainties = _tta_predict_with_oom_recovery(
            tta_predictor=tta_predictor,
            model=model,
            processor=processor,
            images=images,
            bboxes=bboxes,
            device=device,
        )
        tta_pred_masks_t = [(prob_t > 0.5).to(torch.uint8) for prob_t in tta_prob_t]
        uncertainties.extend([float(u) for u in tta_batch_uncertainties])
        inference_times_tta.extend([(time.perf_counter() - t_tta_inf) / max(1, len(batch_samples))] * len(batch_samples))

        t_ood = time.perf_counter()
        ood_prob_stack = torch.stack(ood_prob_t, dim=0)
        tta_prob_stack = torch.stack(tta_prob_t, dim=0)
        ood_pred_stack = torch.stack(ood_pred_masks_t, dim=0)
        tta_pred_stack = torch.stack(tta_pred_masks_t, dim=0)
        ood_batch = ood_detector.detect_batch_tensor(ood_prob_stack)
        ood_elapsed = time.perf_counter() - t_ood
        ood_times.extend([ood_elapsed / max(1, len(batch_samples))] * len(batch_samples))

        t_metric = time.perf_counter()
        gt_stack = torch.stack(gt_masks_t, dim=0)
        ood_batch_metrics = compute_metrics_batch_tensor(ood_pred_stack, gt_stack)
        ood_batch_metrics["bce"] = compute_bce_batch_tensor(ood_prob_stack, gt_stack)
        ood_surface_metrics = compute_surface_metrics_batch_tensor(ood_pred_stack, gt_stack)
        ood_batch_metrics["hd95"] = ood_surface_metrics["hd95"]
        ood_batch_metrics["assd"] = ood_surface_metrics["assd"]
        ood_batch_metrics["ece"] = compute_ece_batch_tensor(ood_prob_stack, gt_stack)
        tta_batch_metrics = compute_metrics_batch_tensor(tta_pred_stack, gt_stack)
        tta_batch_metrics["bce"] = compute_bce_batch_tensor(tta_prob_stack, gt_stack)
        tta_surface_metrics = compute_surface_metrics_batch_tensor(tta_pred_stack, gt_stack)
        tta_batch_metrics["hd95"] = tta_surface_metrics["hd95"]
        tta_batch_metrics["assd"] = tta_surface_metrics["assd"]
        tta_batch_metrics["ece"] = compute_ece_batch_tensor(tta_prob_stack, gt_stack)
        metric_elapsed = time.perf_counter() - t_metric
        metrics_times.extend([metric_elapsed / max(1, len(batch_samples))] * len(batch_samples))

        t_post = time.perf_counter()
        for i in range(len(batch_samples)):
            ood_info = ood_batch[i]
            ood_score = float(ood_info["ood_score"])
            is_ood = bool(ood_info["is_ood"])
            confidence = float(ood_info["confidence"])
            ood_scores.append(ood_score)
            if batch_ood_labels[i] is not None:
                ood_eval_scores.append(ood_score)
                ood_eval_labels.append(int(batch_ood_labels[i]))

            ood_m = {k: float(ood_batch_metrics[k][i].item()) for k in metrics_keys}
            tta_m = {k: float(tta_batch_metrics[k][i].item()) for k in metrics_keys}
            for k in metrics_keys:
                ood_metrics_store[k].append(ood_m[k])
                tta_metrics_store[k].append(tta_m[k])

            ood_results.append(
                {
                    "index": int(sample_index),
                    "name": sample_names[i],
                    "dice": float(ood_m["dice"]),
                    "jaccard": float(ood_m["jaccard"]),
                    "precision": float(ood_m["precision"]),
                    "recall": float(ood_m["recall"]),
                    "sensitivity": float(ood_m["sensitivity"]),
                    "f1": float(ood_m["f1"]),
                    "bce": float(ood_m["bce"]),
                    "hd95": float(ood_m["hd95"]),
                    "assd": float(ood_m["assd"]),
                    "ece": float(ood_m["ece"]),
                    "ood_score": ood_score,
                    "is_ood": is_ood,
                    "ood_label": batch_ood_labels[i],
                    "confidence": confidence,
                    "uncertainty": 0.0,
                }
            )
            tta_results.append(
                {
                    "index": int(sample_index),
                    "name": sample_names[i],
                    "dice": float(tta_m["dice"]),
                    "jaccard": float(tta_m["jaccard"]),
                    "precision": float(tta_m["precision"]),
                    "recall": float(tta_m["recall"]),
                    "sensitivity": float(tta_m["sensitivity"]),
                    "f1": float(tta_m["f1"]),
                    "bce": float(tta_m["bce"]),
                    "hd95": float(tta_m["hd95"]),
                    "assd": float(tta_m["assd"]),
                    "ece": float(tta_m["ece"]),
                    "ood_score": 0.0,
                    "is_ood": False,
                    "ood_label": batch_ood_labels[i],
                    "confidence": 0.0,
                    "uncertainty": float(tta_batch_uncertainties[i]),
                }
            )
            sample_index += 1
        post_elapsed = time.perf_counter() - t_post
        post_times.extend([post_elapsed / max(1, len(batch_samples))] * len(batch_samples))

    total_time = time.perf_counter() - start
    ood_stats = _build_stats_from_store(
        dataset_name=dataset_name,
        results=ood_results,
        metrics_store=ood_metrics_store,
        inference_times=inference_times_ood,
        data_times=data_times,
        ood_times=ood_times,
        metrics_times=metrics_times,
        post_times=post_times,
        total_time=total_time,
        ood_scores=ood_scores,
        uncertainties=None,
        ood_eval_scores=ood_eval_scores,
        ood_eval_labels=ood_eval_labels,
    )
    tta_stats = _build_stats_from_store(
        dataset_name=dataset_name,
        results=tta_results,
        metrics_store=tta_metrics_store,
        inference_times=inference_times_tta,
        data_times=data_times,
        ood_times=ood_times,
        metrics_times=metrics_times,
        post_times=post_times,
        total_time=total_time,
        ood_scores=None,
        uncertainties=uncertainties,
    )

    if profiler is not None and profiler.enabled:
        prefix = profile_prefix or f"eval.{dataset_name}"
        profiler.record_duration(f"{prefix}.ood_tta.total", total_time, count=max(1, len(ood_results)))
        profiler.flush()

    return ood_results, ood_stats, tta_results, tta_stats
