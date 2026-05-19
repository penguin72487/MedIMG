import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torch.nn.functional as F
from tqdm import tqdm

from medsam_modular.data import prepare_datasets_by_split
from medsam_modular.model import load_state_dict_compat, normalize_pred_masks_to_4d
from medsam_modular.profiler import PerformanceProfiler


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
    raw = os.getenv(name, "1" if default else "0").strip().lower()
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


def _finetune_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["pixel_values", "input_boxes", "original_sizes", "reshaped_input_sizes"]
    collated: Dict[str, Any] = {}
    for k in keys:
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

    train_concat = ConcatDataset([ds for ds in train_sets.values() if len(ds) > 0])
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

    return FinetuneProcessorDataset(train_concat, processor), FinetuneProcessorDataset(val_concat, processor)


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


def _compute_seg_loss(outputs: Any, gt_mask: torch.Tensor) -> torch.Tensor:
    logits = normalize_pred_masks_to_4d(outputs.pred_masks)
    target = gt_mask.unsqueeze(1)
    target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    return F.binary_cross_entropy_with_logits(logits, target)


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
    wait: int,
    history: Dict[str, List[float]],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "wait": int(wait),
        "history": history,
    }


def maybe_finetune(model: Any, processor: Any, config: Dict[str, Any], profiler: Optional[PerformanceProfiler] = None) -> Any:
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
    num_workers = int(config.get("finetune_workers", 4))
    max_samples = int(config.get("finetune_max_samples", 0))
    use_fused_adamw = _get_bool("finetune_use_fused_adamw", True)
    resume_weight_path_raw = str(config.get("resume_weight_path", "")).strip()
    resume_weight_path = Path(resume_weight_path_raw) if resume_weight_path_raw else None
    cuda_mem_gb = _cuda_total_memory_gb() if device == "cuda" else None
    low_vram_mode = bool(cuda_mem_gb is not None and cuda_mem_gb <= 12.5)
    if low_vram_mode:
        safe_batch_12gb = int(config.get("finetune_safe_batch_12gb", 2))
        safe_batch_12gb = max(1, safe_batch_12gb)
        if batch_size > safe_batch_12gb:
            print(f"⚠️ Low-VRAM mode ({cuda_mem_gb:.1f}GB): batch size {batch_size} -> {safe_batch_12gb}")
            batch_size = safe_batch_12gb

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
    print(f"  lr            : {lr}")
    print(f"  adamw         : wd={weight_decay}, betas=({adamw_beta1},{adamw_beta2}), eps={adamw_eps}")
    print(f"  grad_accum    : {grad_accum}")
    print(f"  train backbone: {train_backbone}")
    print(f"  資料準備耗時  : {time.time() - t0:.1f}s")

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
        batch_size=batch_size if low_vram_mode else min(batch_size * 4, max(1, len(val_dataset) // 4)),
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

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best_val_loss = float("inf")
    best_path = output_dir / "medsam_finetuned_best.pth"
    last_path = output_dir / "medsam_finetuned_last.pth"
    async_saver = _AsyncCheckpointSaver()

    wait = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
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
                if isinstance(hist.get("train_loss"), list) and isinstance(hist.get("val_loss"), list):
                    history = {"train_loss": list(hist["train_loss"]), "val_loss": list(hist["val_loss"])}

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
            model_inputs = {
                "pixel_values": batch["pixel_values"],
                "input_boxes": batch["input_boxes"],
                "original_sizes": batch["original_sizes"],
                "reshaped_input_sizes": batch["reshaped_input_sizes"],
            }

            t_forward = time.perf_counter()
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = base_model(**model_inputs)
                    loss = _compute_seg_loss(outputs, batch["gt_mask"]) / grad_accum
            else:
                outputs = base_model(**model_inputs)
                loss = _compute_seg_loss(outputs, batch["gt_mask"]) / grad_accum
            train_forward_total += (time.perf_counter() - t_forward)

            t_backward = time.perf_counter()
            scaler.scale(loss).backward()
            train_backward_total += (time.perf_counter() - t_backward)
            cur_loss = float(loss.item() * grad_accum)
            train_losses.append(cur_loss)
            train_bar.set_postfix(loss=f"{cur_loss:.4f}")

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
                model_inputs = {
                    "pixel_values": batch["pixel_values"],
                    "input_boxes": batch["input_boxes"],
                    "original_sizes": batch["original_sizes"],
                    "reshaped_input_sizes": batch["reshaped_input_sizes"],
                }

                t_val_forward = time.perf_counter()
                if device == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = base_model(**model_inputs)
                        val_loss = _compute_seg_loss(outputs, batch["gt_mask"])
                else:
                    outputs = base_model(**model_inputs)
                    val_loss = _compute_seg_loss(outputs, batch["gt_mask"])
                val_forward_total += (time.perf_counter() - t_val_forward)

                val_bar.set_postfix(loss=f"{float(val_loss.item()):.4f}")
                val_losses.append(float(val_loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if low_vram_mode and device == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - epoch_t0
        improved_mark = "★" if val_loss < (best_val_loss - min_delta) else " "
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            best=f"{best_val_loss:.4f}",
            wait=wait,
        )
        tqdm.write(
            f"Epoch {epoch:03d}/{epochs} {improved_mark} | "
            f"train={train_loss:.6f}  val={val_loss:.6f}  best={best_val_loss:.6f}  "
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
            async_saver.save_async(best_payload, best_path)
            tqdm.write(f"  ✅ New best checkpoint saved  (val={val_loss:.6f})")
        else:
            wait += 1
            if wait >= patience:
                async_saver.wait_for_save()
                tqdm.write(f"  ⏹️ Early stopping @ epoch {epoch}  (wait={wait}/{patience})")
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

    stats_path = output_dir / "finetune_stats.json"
    t_save_json = time.perf_counter()
    stats_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_val_loss": best_val_loss,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "epochs_config": epochs,
                "batch_size": batch_size,
                "lr": lr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_json_total = time.perf_counter() - t_save_json
    t_save_pt = time.perf_counter()
    torch.save(
        {
            "history": history,
            "best_val_loss": best_val_loss,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "epochs_config": epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        output_dir / "finetune_stats.pt",
    )
    save_pt_total = time.perf_counter() - t_save_pt

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    if profiler is not None and profiler.enabled:
        ft_total_sec = time.perf_counter() - ft_total_start
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
