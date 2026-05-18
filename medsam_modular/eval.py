import time
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from medsam_modular.cache import PredictionCache, make_cache_key
from medsam_modular.model import build_inputs_batch, predict_binary_mask, predict_prob_masks_from_inputs


def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax normalization."""
    x = np.asarray(x, dtype=np.float32)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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

    def detect(self, mask_prob: np.ndarray) -> Dict[str, Any]:
        p = np.asarray(mask_prob, dtype=np.float64).reshape(-1)
        p = np.clip(p, 1e-10, 1 - 1e-10)

        if self.method == "confidence":
            score = -float(np.mean(np.abs(p - 0.5) * 2.0))
        elif self.method == "variance":
            score = float(np.var(p))
        else:
            score = float(-np.mean(p * np.log(p) + (1 - p) * np.log(1 - p)))

        is_ood = score > self.threshold
        confidence = float(max(0.0, 1.0 - score))
        return {"ood_score": score, "is_ood": is_ood, "confidence": confidence}


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
            # Full mode: comprehensive augmentations
            self.augmentations = augmentations or [
                "none",
                "hflip",
                "vflip",
                "hvflip",
                "rotate_90",
                "rotate_180",
                "rotate_270",
                "elastic_deform",
            ]
        
        self.fusion_mode = fusion_mode
        env_fixed_batch = int(os.getenv("MEDSAM_TTA_FIXED_BATCH", "0"))
        self.fixed_batch_size = max(0, env_fixed_batch)
        cuda_mem_gb = _cuda_total_memory_gb()
        default_chunk = 2 if (cuda_mem_gb is not None and cuda_mem_gb <= 12.5) else 8
        self.infer_chunk_size = max(1, int(os.getenv("MEDSAM_TTA_CHUNK_SIZE", str(default_chunk))))
        self._elastic_map_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        assert fusion_mode in ["mean", "median", "entropy_weighted"], \
            f"fusion_mode must be 'mean', 'median', or 'entropy_weighted', got {fusion_mode}"

    def _apply_aug(self, image_np: np.ndarray, aug_name: str) -> np.ndarray:
        """Apply augmentation to image."""
        if aug_name == "none":
            return image_np
        elif aug_name == "hflip":
            return np.ascontiguousarray(np.flip(image_np, axis=1))
        elif aug_name == "vflip":
            return np.ascontiguousarray(np.flip(image_np, axis=0))
        elif aug_name == "hvflip":
            return np.ascontiguousarray(np.flip(np.flip(image_np, axis=0), axis=1))
        elif aug_name == "rotate_90":
            return np.ascontiguousarray(np.rot90(image_np, k=1))
        elif aug_name == "rotate_180":
            return np.ascontiguousarray(np.rot90(image_np, k=2))
        elif aug_name == "rotate_270":
            return np.ascontiguousarray(np.rot90(image_np, k=3))
        elif aug_name == "elastic_deform":
            # Simple elastic deformation for medical images
            return self._elastic_deform(image_np)
        else:
            raise ValueError(f"Unsupported augmentation: {aug_name}")

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

    def _get_elastic_maps(self, h: int, w: int, alpha: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        key = (h, w)
        cached = self._elastic_map_cache.get(key)
        if cached is not None:
            return cached

        # Deterministic per-size deformation map for speed and reproducibility.
        random_state = np.random.RandomState(42)
        dx = random_state.randn(h, w).astype(np.float32)
        dy = random_state.randn(h, w).astype(np.float32)

        dx = cv2.GaussianBlur(dx, (5, 5), sigma)
        dy = cv2.GaussianBlur(dy, (5, 5), sigma)

        dx_scale = max(float(np.max(np.abs(dx))), 1e-6)
        dy_scale = max(float(np.max(np.abs(dy))), 1e-6)
        dx = (dx / dx_scale) * alpha
        dy = (dy / dy_scale) * alpha

        x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = x + dx
        map_y = y + dy
        self._elastic_map_cache[key] = (map_x, map_y)
        return map_x, map_y

    def _elastic_deform(self, image_np: np.ndarray, alpha: float = 30.0, sigma: float = 4.0) -> np.ndarray:
        """Apply elastic deformation to image (medical imaging augmentation)."""
        h, w = image_np.shape[:2]
        map_x, map_y = self._get_elastic_maps(h, w, alpha=alpha, sigma=sigma)
        
        # Warp image
        if len(image_np.shape) == 3:
            # For RGB images, process each channel
            result = np.zeros_like(image_np)
            for c in range(image_np.shape[2]):
                result[:, :, c] = cv2.remap(
                    image_np[:, :, c], map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
            return result
        else:
            return cv2.remap(
                image_np, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

    def _deaugment_mask(self, mask_np: np.ndarray, aug_name: str) -> np.ndarray:
        """Reverse augmentation on mask."""
        if aug_name == "none":
            return mask_np
        elif aug_name == "hflip":
            return np.ascontiguousarray(np.flip(mask_np, axis=1))
        elif aug_name == "vflip":
            return np.ascontiguousarray(np.flip(mask_np, axis=0))
        elif aug_name == "hvflip":
            return np.ascontiguousarray(np.flip(np.flip(mask_np, axis=0), axis=1))
        elif aug_name == "rotate_90":
            return np.ascontiguousarray(np.rot90(mask_np, k=3))  # Reverse rotation
        elif aug_name == "rotate_180":
            return np.ascontiguousarray(np.rot90(mask_np, k=2))
        elif aug_name == "rotate_270":
            return np.ascontiguousarray(np.rot90(mask_np, k=1))
        elif aug_name == "elastic_deform":
            # Cannot perfectly reverse elastic deformation, return as is
            return mask_np.copy()
        else:
            raise ValueError(f"Unsupported augmentation: {aug_name}")

    def _augment_bbox(self, bbox: List[int], aug_name: str, h: int, w: int) -> List[int]:
        """Transform bounding box according to augmentation."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        max_x = w - 1
        max_y = h - 1
        
        if aug_name == "none":
            return [x1, y1, x2, y2]
        elif aug_name == "hflip":
            return [max_x - x2, y1, max_x - x1, y2]
        elif aug_name == "vflip":
            return [x1, max_y - y2, x2, max_y - y1]
        elif aug_name == "hvflip":
            return [max_x - x2, max_y - y2, max_x - x1, max_y - y1]
        elif aug_name == "rotate_90":
            # After 90° rotation: (x,y) -> (h-1-y, x)
            new_x1, new_y1 = max_y - y2, x1
            new_x2, new_y2 = max_y - y1, x2
            return [new_x1, new_y1, new_x2, new_y2]
        elif aug_name == "rotate_180":
            # After 180° rotation: (x,y) -> (w-1-x, h-1-y)
            new_x1 = max_x - x2
            new_y1 = max_y - y2
            new_x2 = max_x - x1
            new_y2 = max_y - y1
            return [new_x1, new_y1, new_x2, new_y2]
        elif aug_name == "rotate_270":
            # After 270° rotation: (x,y) -> (y, w-1-x)
            new_x1, new_y1 = y1, max_x - x2
            new_x2, new_y2 = y2, max_x - x1
            return [new_x1, new_y1, new_x2, new_y2]
        elif aug_name == "elastic_deform":
            # Cannot transform bbox for elastic deformation, return original
            return [x1, y1, x2, y2]
        else:
            raise ValueError(f"Unsupported augmentation: {aug_name}")

    def _augment_square_bbox(self, bbox: torch.Tensor, aug_name: str, size: int) -> List[float]:
        """Transform a bbox already scaled to the square preprocessing size."""
        x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
        max_coord = float(size - 1)

        if aug_name == "none":
            return [x1, y1, x2, y2]
        if aug_name == "hflip":
            return [max_coord - x2, y1, max_coord - x1, y2]
        if aug_name == "vflip":
            return [x1, max_coord - y2, x2, max_coord - y1]
        if aug_name == "hvflip":
            return [max_coord - x2, max_coord - y2, max_coord - x1, max_coord - y1]
        if aug_name == "rotate_90":
            return [max_coord - y2, x1, max_coord - y1, x2]
        if aug_name == "rotate_180":
            return [max_coord - x2, max_coord - y2, max_coord - x1, max_coord - y1]
        if aug_name == "rotate_270":
            return [y1, max_coord - x2, y2, max_coord - x1]
        raise ValueError(f"Unsupported square augmentation: {aug_name}")

    def _build_tta_inputs(
        self,
        processor: Any,
        image: Image.Image,
        bbox: List[int],
    ) -> Tuple[Dict[str, torch.Tensor], List[str], Tuple[int, int], int]:
        """Build a single batched input for TTA with one preprocess pass."""
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]

        base_inputs = build_inputs_batch(
            processor=processor,
            images=[image_np],
            input_boxes=[[bbox]],
        )
        base_tensor = base_inputs["pixel_values"][0]
        base_box = base_inputs["input_boxes"][0, 0]
        target_edge = int(base_tensor.shape[-1])

        pixel_values: List[torch.Tensor] = []
        input_boxes: List[torch.Tensor] = []
        aug_names: List[str] = []

        for aug_name in self.augmentations:
            if aug_name == "elastic_deform":
                aug_image_np = self._elastic_deform(image_np)
                aug_inputs = build_inputs_batch(
                    processor=processor,
                    images=[aug_image_np],
                    input_boxes=[[bbox]],
                )
                pixel_values.append(aug_inputs["pixel_values"][0])
                input_boxes.append(aug_inputs["input_boxes"][0, 0])
            else:
                pixel_values.append(self._apply_tensor_aug(base_tensor, aug_name))
                input_boxes.append(torch.tensor(self._augment_square_bbox(base_box, aug_name, target_edge), dtype=torch.float32))
            aug_names.append(aug_name)

        stacked_pixels = torch.stack(pixel_values, dim=0).contiguous()
        if stacked_pixels.dim() == 4:
            stacked_pixels = stacked_pixels.contiguous(memory_format=torch.channels_last)

        true_aug_count = len(aug_names)
        target_batch = self.fixed_batch_size if self.fixed_batch_size > 0 else true_aug_count
        if target_batch > len(aug_names):
            pad_count = target_batch - len(aug_names)
            stacked_pixels = torch.cat([stacked_pixels, stacked_pixels[-1:].repeat(pad_count, 1, 1, 1)], dim=0)
            input_boxes.extend([input_boxes[-1].clone() for _ in range(pad_count)])
            aug_names.extend([aug_names[-1]] * pad_count)

        inputs = {
            "pixel_values": stacked_pixels,
            "input_boxes": torch.stack(input_boxes, dim=0).unsqueeze(1),
            "original_sizes": base_inputs["original_sizes"].repeat(len(aug_names), 1),
            "reshaped_input_sizes": base_inputs["reshaped_input_sizes"].repeat(len(aug_names), 1),
        }
        return inputs, aug_names, (h, w), true_aug_count

    def _fuse_predictions(
        self,
        preds: Any,
        uncertainties: List[float],
    ) -> Tuple[np.ndarray, float]:
        """Fuse multiple predictions using specified strategy."""
        if isinstance(preds, list):
            stacked = np.stack(preds, axis=0)
        else:
            stacked = np.asarray(preds)
        
        if self.fusion_mode == "mean":
            fused = stacked.mean(axis=0)
            avg_uncertainty = float(np.mean(uncertainties))
        
        elif self.fusion_mode == "median":
            fused = np.median(stacked, axis=0)
            avg_uncertainty = float(np.mean(uncertainties))
        
        elif self.fusion_mode == "entropy_weighted":
            # Entropy-based weighting
            uncertainties_arr = np.array(uncertainties, dtype=np.float32)
            # Lower uncertainty -> higher weight
            weights = _softmax(1.0 - uncertainties_arr).astype(np.float32)
            fused = np.sum(stacked * weights[:, None, None], axis=0)
            avg_uncertainty = float(np.mean(uncertainties))
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        return fused, avg_uncertainty

    def predict(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        bbox: List[int],
        device: str,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict with test-time augmentation.
        
        Returns:
            (prob_mask, avg_uncertainty)
        """
        inputs, aug_names, output_size, true_aug_count = self._build_tta_inputs(
            processor=processor,
            image=image,
            bbox=bbox,
        )

        pixel_values = inputs["pixel_values"]
        input_boxes = inputs["input_boxes"]
        original_sizes = inputs["original_sizes"]
        reshaped_input_sizes = inputs["reshaped_input_sizes"]

        total_count = int(pixel_values.shape[0])
        chunk_size = max(1, self.infer_chunk_size)

        while True:
            pred_chunks: List[np.ndarray] = []
            try:
                for start in range(0, total_count, chunk_size):
                    end = min(start + chunk_size, total_count)
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
                        output_size=output_size,
                        use_amp=True,
                    ).detach().cpu().numpy()[:, 0]
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

        pred_batch = np.concatenate(pred_chunks, axis=0)[:true_aug_count]

        # Reverse augmentations on predictions
        preds = [
            self._deaugment_mask(pred, aug_name)
            for pred, aug_name in zip(pred_batch, aug_names)
        ]

        # Vectorized per-augmentation uncertainty (mean entropy).
        stacked = np.stack(preds, axis=0).astype(np.float32, copy=False)
        prob = np.clip(stacked, 1e-6, 1.0 - 1e-6)
        entropy = -(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob))
        uncertainties = entropy.mean(axis=(1, 2)).astype(np.float32).tolist()

        # Fuse predictions
        fused_mask, uncertainties_mean = self._fuse_predictions(stacked, uncertainties)
        
        return fused_mask, uncertainties_mean


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask.astype(bool).reshape(-1)
    gt = gt_mask.astype(bool).reshape(-1)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    dice = float((2 * tp) / (2 * tp + fp + fn + 1e-8))
    jaccard = float(tp / (tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float((2 * precision * recall) / (precision + recall + 1e-8))

    return {
        "dice": dice,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _mean_std(values: List[float]) -> Tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics_keys = ["dice", "jaccard", "precision", "recall", "f1"]
    metrics_store: Dict[str, List[float]] = {k: [] for k in metrics_keys}

    results: List[Dict[str, Any]] = []
    ood_scores: List[float] = []
    uncertainties: List[float] = []
    inference_times: List[float] = []

    start = time.perf_counter()

    default_eval_workers = "16" if device == "cuda" else "0"
    eval_workers = max(0, int(os.getenv("MEDSAM_EVAL_WORKERS", default_eval_workers)))
    pin_memory = device == "cuda"

    if eval_workers > 0:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            collate_fn=lambda b: b[0],
        )
        iterable = enumerate(loader)
        total = len(loader)
    else:
        iterable = ((idx, dataset[idx]) for idx in range(len(dataset)))
        total = len(dataset)

    for idx, sample in tqdm(iterable, total=total, desc=f"Evaluating {dataset_name}"):
        image = sample["image"]
        bbox = sample["bbox"]
        sample_name = str(sample.get("name", f"sample_{idx}"))

        gt_mask = sample["mask"]
        gt_mask_np = gt_mask.numpy() if hasattr(gt_mask, "numpy") else np.asarray(gt_mask)

        t0 = time.perf_counter()
        if use_tta and tta_predictor is not None:
            prob_mask, uncertainty = tta_predictor.predict(
                model=model,
                processor=processor,
                image=image,
                bbox=bbox,
                device=device,
            )
            pred_mask = (prob_mask > 0.5).astype(np.uint8)
            uncertainties.append(float(uncertainty))
            prob_for_ood = prob_mask
        else:
            cache_key = make_cache_key(dataset_name, sample_name, bbox, mode="baseline")
            pred_mask = pred_cache.get(cache_key) if pred_cache is not None else None
            if pred_mask is None:
                pred_mask = predict_binary_mask(model, processor, image, bbox, device=device, use_amp=True)
                if pred_cache is not None:
                    pred_cache.put(cache_key, pred_mask)
            prob_for_ood = pred_mask.astype(np.float32)

        inference_times.append(time.perf_counter() - t0)

        ood_score = 0.0
        is_ood = False
        confidence = 0.0
        if use_ood and ood_detector is not None:
            ood = ood_detector.detect(prob_for_ood)
            ood_score = float(ood["ood_score"])
            is_ood = bool(ood["is_ood"])
            confidence = float(ood["confidence"])
            ood_scores.append(ood_score)

        m = compute_metrics(pred_mask, gt_mask_np)
        for k in metrics_keys:
            metrics_store[k].append(float(m[k]))

        results.append(
            {
                "index": int(idx),
                "name": sample_name,
                "dice": float(m["dice"]),
                "jaccard": float(m["jaccard"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
                "ood_score": ood_score,
                "is_ood": is_ood,
                "confidence": confidence,
                "uncertainty": float(uncertainties[-1]) if uncertainties else 0.0,
            }
        )

    total_time = time.perf_counter() - start

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

    return results, stats
