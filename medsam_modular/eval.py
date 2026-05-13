import time
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from medsam_modular.cache import PredictionCache, make_cache_key
from medsam_modular.model import predict_binary_mask, predict_prob_masks_batch


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
    def __init__(self, augmentations: Optional[List[str]] = None):
        self.augmentations = augmentations or ["none", "hflip", "vflip", "hvflip"]

    def _apply_aug(self, image_np: np.ndarray, aug_name: str) -> np.ndarray:
        if aug_name == "none":
            return image_np
        if aug_name == "hflip":
            return np.ascontiguousarray(np.flip(image_np, axis=1))
        if aug_name == "vflip":
            return np.ascontiguousarray(np.flip(image_np, axis=0))
        if aug_name == "hvflip":
            return np.ascontiguousarray(np.flip(np.flip(image_np, axis=0), axis=1))
        raise ValueError(f"Unsupported augmentation: {aug_name}")

    def _deaugment_mask(self, mask_np: np.ndarray, aug_name: str) -> np.ndarray:
        if aug_name == "none":
            return mask_np
        if aug_name == "hflip":
            return np.ascontiguousarray(np.flip(mask_np, axis=1))
        if aug_name == "vflip":
            return np.ascontiguousarray(np.flip(mask_np, axis=0))
        if aug_name == "hvflip":
            return np.ascontiguousarray(np.flip(np.flip(mask_np, axis=0), axis=1))
        raise ValueError(f"Unsupported augmentation: {aug_name}")

    def _augment_bbox(self, bbox: List[int], aug_name: str, h: int, w: int) -> List[int]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        max_x = w - 1
        max_y = h - 1
        if aug_name == "none":
            return [x1, y1, x2, y2]
        if aug_name == "hflip":
            return [max_x - x2, y1, max_x - x1, y2]
        if aug_name == "vflip":
            return [x1, max_y - y2, x2, max_y - y1]
        if aug_name == "hvflip":
            return [max_x - x2, max_y - y2, max_x - x1, max_y - y1]
        raise ValueError(f"Unsupported augmentation: {aug_name}")

    def predict(self, model: Any, processor: Any, image: Image.Image, bbox: List[int], device: str) -> Tuple[np.ndarray, np.ndarray]:
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]

        aug_images: List[Image.Image] = []
        aug_boxes: List[List[int]] = []
        aug_names: List[str] = []
        for aug_name in self.augmentations:
            aug_img_np = self._apply_aug(image_np, aug_name)
            aug_images.append(Image.fromarray(aug_img_np))
            aug_boxes.append(self._augment_bbox(bbox, aug_name, h, w))
            aug_names.append(aug_name)

        pred_batch = predict_prob_masks_batch(
            model=model,
            processor=processor,
            images=aug_images,
            input_boxes=aug_boxes,
            device=device,
            use_amp=True,
        )

        preds = [self._deaugment_mask(pred, aug_name) for pred, aug_name in zip(pred_batch, aug_names)]

        stack = np.stack(preds, axis=0)
        return stack.mean(axis=0), stack.std(axis=0)


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

    eval_workers = max(0, int(os.getenv("MEDSAM_EVAL_WORKERS", "0")))
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
            uncertainties.append(float(np.mean(uncertainty)))
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
