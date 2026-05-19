import time
import os
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
) -> Dict[str, Any]:
    mean_dice, std_dice = _mean_std(metrics_store["dice"])
    mean_jaccard, std_jaccard = _mean_std(metrics_store["jaccard"])
    mean_precision, std_precision = _mean_std(metrics_store["precision"])
    mean_recall, std_recall = _mean_std(metrics_store["recall"])
    mean_f1, std_f1 = _mean_std(metrics_store["f1"])

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
        "mean_f1": mean_f1,
        "std_f1": std_f1,
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
            self.augmentations = augmentations or ["none", "hflip", "vflip", "hvflip"]
        else:
            # Full mode: fixed 8 augmentations.
            base_augs = [
                "none",
                "hflip",
                "vflip",
                "hvflip",
                "rotate_90",
                "rotate_180",
                "rotate_270",
                "elastic_deform",
            ]
            self.augmentations = augmentations or base_augs
        
        self.fusion_mode = fusion_mode
        env_fixed_batch = int(os.getenv("MEDSAM_TTA_FIXED_BATCH", "0"))
        self.fixed_batch_size = max(0, env_fixed_batch)
        cuda_mem_gb = _cuda_total_memory_gb()
        default_chunk = 4 if (cuda_mem_gb is not None and cuda_mem_gb <= 12.5) else 8
        self.infer_chunk_size = max(1, int(os.getenv("MEDSAM_TTA_CHUNK_SIZE", str(default_chunk))))
        self._chunk_size_tuned = False
        self._elastic_disp_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._elastic_kernel_cache: Dict[Tuple[float], torch.Tensor] = {}
        self._elastic_grid_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._norm_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._aug_to_id = {name: idx for idx, name in enumerate(self.augmentations)}
        assert fusion_mode in ["mean", "median", "entropy_weighted"], \
            f"fusion_mode must be 'mean', 'median', or 'entropy_weighted', got {fusion_mode}"

    def _get_gaussian_kernel_2d(self, sigma: float) -> torch.Tensor:
        sigma = float(max(0.1, sigma))
        key = (round(sigma, 4),)
        cached = self._elastic_kernel_cache.get(key)
        if cached is not None:
            return cached

        radius = max(1, int(math.ceil(3.0 * sigma)))
        size = radius * 2 + 1
        x = torch.arange(size, dtype=torch.float32) - float(radius)
        kernel_1d = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.view(1, 1, size, size).contiguous()
        self._elastic_kernel_cache[key] = kernel_2d
        return kernel_2d

    def _get_base_grid(self, h: int, w: int, device: str) -> torch.Tensor:
        key = (h, w, device)
        cached = self._elastic_grid_cache.get(key)
        if cached is not None:
            return cached

        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).contiguous()
        self._elastic_grid_cache[key] = base_grid
        return base_grid

    def _get_elastic_displacement(self, h: int, w: int, alpha: float, sigma: float) -> torch.Tensor:
        key = (h, w)
        cached = self._elastic_disp_cache.get(key)
        if cached is not None:
            return cached

        gen = torch.Generator(device="cpu")
        gen.manual_seed(42 + h * 1009 + w * 9176)

        dx = torch.randn((1, 1, h, w), generator=gen, dtype=torch.float32)
        dy = torch.randn((1, 1, h, w), generator=gen, dtype=torch.float32)

        kernel = self._get_gaussian_kernel_2d(sigma)
        pad = int((kernel.shape[-1] - 1) // 2)
        dx = F.conv2d(dx, kernel, padding=pad)
        dy = F.conv2d(dy, kernel, padding=pad)

        dx = (dx / (dx.abs().amax() + 1e-6)) * float(alpha)
        dy = (dy / (dy.abs().amax() + 1e-6)) * float(alpha)

        scale_x = 2.0 / float(max(1, w - 1))
        scale_y = 2.0 / float(max(1, h - 1))
        dx_norm = dx * scale_x
        dy_norm = dy * scale_y
        disp = torch.cat([dx_norm, dy_norm], dim=1).permute(0, 2, 3, 1).contiguous()
        self._elastic_disp_cache[key] = disp
        return disp

    def _elastic_deform_tensor(self, image_t: torch.Tensor, alpha: float = 30.0, sigma: float = 4.0) -> torch.Tensor:
        """Apply elastic deformation via torch.grid_sample on GPU/CPU tensor path."""
        if image_t.dim() != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(image_t.shape)}")

        _, h, w = image_t.shape
        device = str(image_t.device)
        base_grid = self._get_base_grid(h, w, device=device)
        disp = self._get_elastic_displacement(h, w, alpha=alpha, sigma=sigma).to(image_t.device, dtype=torch.float32)
        grid = (base_grid + disp).clamp(-1.0, 1.0)

        src = image_t.unsqueeze(0).to(torch.float32)
        warped = F.grid_sample(
            src,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )
        return warped.squeeze(0).to(dtype=image_t.dtype).contiguous()

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
                if aug_name == "elastic_deform":
                    all_pixel_values.append(self._elastic_deform_tensor(base_tensor))
                    all_input_boxes.append(base_box)
                else:
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
            if aug_name == "none" or aug_name == "elastic_deform":
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

    def _fuse_predictions(
        self,
        preds: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Fuse multiple predictions using specified strategy."""
        profiler = get_active_profiler()
        t0 = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
        stacked_t = preds.to(torch.float32)
        uncertainties_t = torch.nan_to_num(
            uncertainties.to(device=stacked_t.device, dtype=torch.float32),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        avg_uncertainty = float(uncertainties_t.mean().item())

        if self.fusion_mode == "mean":
            fused_t = stacked_t.mean(dim=0)
        elif self.fusion_mode == "median":
            fused_t = torch.median(stacked_t, dim=0).values
        elif self.fusion_mode == "entropy_weighted":
            if uncertainties_t.numel() == 1:
                fused_t = stacked_t[0]
            else:
                weights = torch.softmax(-uncertainties_t, dim=0)
                fused_t = torch.sum(stacked_t * weights[:, None, None], dim=0)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        if profiler is not None and profiler.enabled:
            profiler.record_duration(f"tta.fuse.{self.fusion_mode}", time.perf_counter() - t0)
        return fused_t, avg_uncertainty

    def _deaugment_mask_tensor(self, mask_t: torch.Tensor, aug_name: str) -> torch.Tensor:
        if aug_name == "none":
            return mask_t
        if aug_name == "hflip":
            return mask_t.flip(-1)
        if aug_name == "vflip":
            return mask_t.flip(-2)
        if aug_name == "hvflip":
            return mask_t.flip(-2).flip(-1)
        if aug_name == "rotate_90":
            return torch.rot90(mask_t, k=3, dims=(-2, -1))
        if aug_name == "rotate_180":
            return torch.rot90(mask_t, k=2, dims=(-2, -1))
        if aug_name == "rotate_270":
            return torch.rot90(mask_t, k=1, dims=(-2, -1))
        if aug_name == "elastic_deform":
            return mask_t
        raise ValueError(f"Unsupported augmentation: {aug_name}")

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
        
        test_sizes = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
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
                        inputs_already_on_device=True,
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
        results = self.predict_batch(
            model=model,
            processor=processor,
            images=[image],
            bboxes=[bbox],
            device=device,
        )
        return results[0]

    def predict_batch(
        self,
        model: Any,
        processor: Any,
        images: List[Image.Image],
        bboxes: List[List[int]],
        device: str,
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Predict with TTA for multiple images in a single batched forward pass.
        
        Args:
            model: The segmentation model
            processor: The image processor
            images: List of PIL images
            bboxes: List of bboxes corresponding to images
            device: Device to run on ("cuda" or "cpu")
        
        Returns:
            List of (prob_mask_tensor, avg_uncertainty) tuples, one per image
        """
        if not images:
            return []
        
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
                        inputs_already_on_device=True,
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
        
        # Collect predictions per sample and fuse
        n_samples = len(images)
        results: List[Tuple[torch.Tensor, float]] = []

        aug_offsets: List[int] = []
        offset = 0
        for count in true_aug_counts:
            aug_offsets.append(offset)
            offset += int(count)

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

            # Compute uncertainties
            prob_t = stacked_t.to(torch.float32).clamp(1e-6, 1.0 - 1e-6)
            entropy_t = -(prob_t * torch.log(prob_t) + (1.0 - prob_t) * torch.log1p(-prob_t))
            uncertainties_t = entropy_t.reshape(entropy_t.shape[0], -1).mean(dim=1).to(torch.float32)

            # Fuse predictions
            fused_t, uncertainties_mean = self._fuse_predictions(stacked_t, uncertainties_t)
            
            # Interpolate to original size
            out_h, out_w = output_sizes_list[sample_idx]
            if int(fused_t.shape[-2]) != out_h or int(fused_t.shape[-1]) != out_w:
                fused_t = F.interpolate(
                    fused_t.unsqueeze(0).unsqueeze(0),
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            results.append((fused_t, uncertainties_mean))

        if profiler is not None and profiler.enabled:
            profiler.record_duration("tta.predict_total", time.perf_counter() - t_predict_total)

        return results


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
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    return {
        "dice": float(dice.item()),
        "jaccard": float(jaccard.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
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
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    return {
        "dice": dice,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp.to(torch.int64),
        "fp": fp.to(torch.int64),
        "fn": fn.to(torch.int64),
    }


def _mean_std(values: List[float]) -> Tuple[float, float]:
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
    metrics_keys = ["dice", "jaccard", "precision", "recall", "f1"]
    metrics_store: Dict[str, List[float]] = {k: [] for k in metrics_keys}

    results: List[Dict[str, Any]] = []
    ood_scores: List[float] = []
    uncertainties: List[float] = []
    inference_times: List[float] = []
    data_times: List[float] = []
    ood_times: List[float] = []
    metrics_times: List[float] = []
    post_times: List[float] = []

    start = time.perf_counter()

    default_eval_workers = "16"
    default_eval_batch = "8" if (device == "cuda" and not use_tta) else "1"
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
            tta_results = tta_predictor.predict_batch(
                model=model,
                processor=processor,
                images=images,
                bboxes=bboxes,
                device=device,
            )
            for i, (prob_t, uncertainty) in enumerate(tta_results):
                pred_masks_t.append((prob_t > 0.5).to(torch.uint8))
                prob_for_ood_t.append(prob_t)
                batch_uncertainties[i] = float(uncertainty)
                uncertainties.append(float(uncertainty))
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

        t_metric = time.perf_counter()
        batch_metrics = compute_metrics_batch_tensor(pred_batch_t, gt_batch_t)
        metric_elapsed = time.perf_counter() - t_metric
        metrics_times.extend([metric_elapsed / max(1, len(batch_samples))] * len(batch_samples))
        if profiler is not None and profiler.enabled:
            profiler.record_duration(f"{profile_prefix or f'eval.{dataset_name}'}.metrics_batch", metric_elapsed)

        t_ood = time.perf_counter()
        if use_ood and ood_detector is not None:
            ood_batch = ood_detector.detect_batch_tensor(torch.stack(prob_for_ood_t, dim=0))
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
                    "f1": float(m["f1"]),
                    "ood_score": ood_score,
                    "is_ood": is_ood,
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
    metrics_keys = ["dice", "jaccard", "precision", "recall", "f1"]

    ood_metrics_store: Dict[str, List[float]] = {k: [] for k in metrics_keys}
    tta_metrics_store: Dict[str, List[float]] = {k: [] for k in metrics_keys}
    ood_results: List[Dict[str, Any]] = []
    tta_results: List[Dict[str, Any]] = []
    ood_scores: List[float] = []
    uncertainties: List[float] = []

    inference_times_ood: List[float] = []
    inference_times_tta: List[float] = []
    data_times: List[float] = []
    ood_times: List[float] = []
    metrics_times: List[float] = []
    post_times: List[float] = []

    start = time.perf_counter()
    default_eval_workers = "16"
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
        tta_batch_uncertainties: List[float] = [0.0] * len(batch_samples)
        tta_pred_masks_t: List[torch.Tensor] = []
        tta_prob_t: List[torch.Tensor] = []
        tta_batch = tta_predictor.predict_batch(
            model=model,
            processor=processor,
            images=images,
            bboxes=bboxes,
            device=device,
        )
        for i, (prob_t, uncertainty) in enumerate(tta_batch):
            tta_pred_masks_t.append((prob_t > 0.5).to(torch.uint8))
            tta_prob_t.append(prob_t)
            tta_batch_uncertainties[i] = float(uncertainty)
            uncertainties.append(float(uncertainty))
        inference_times_tta.extend([(time.perf_counter() - t_tta_inf) / max(1, len(batch_samples))] * len(batch_samples))

        t_ood = time.perf_counter()
        ood_batch = ood_detector.detect_batch_tensor(torch.stack(ood_prob_t, dim=0))
        ood_elapsed = time.perf_counter() - t_ood
        ood_times.extend([ood_elapsed / max(1, len(batch_samples))] * len(batch_samples))

        t_metric = time.perf_counter()
        ood_batch_metrics = compute_metrics_batch_tensor(torch.stack(ood_pred_masks_t, dim=0), torch.stack(gt_masks_t, dim=0))
        tta_batch_metrics = compute_metrics_batch_tensor(torch.stack(tta_pred_masks_t, dim=0), torch.stack(gt_masks_t, dim=0))
        metric_elapsed = time.perf_counter() - t_metric
        metrics_times.extend([metric_elapsed / max(1, len(batch_samples))] * len(batch_samples))

        t_post = time.perf_counter()
        for i in range(len(batch_samples)):
            ood_info = ood_batch[i]
            ood_score = float(ood_info["ood_score"])
            is_ood = bool(ood_info["is_ood"])
            confidence = float(ood_info["confidence"])
            ood_scores.append(ood_score)

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
                    "f1": float(ood_m["f1"]),
                    "ood_score": ood_score,
                    "is_ood": is_ood,
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
                    "f1": float(tta_m["f1"]),
                    "ood_score": 0.0,
                    "is_ood": False,
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
