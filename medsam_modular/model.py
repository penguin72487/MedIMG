import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SamModel, SamProcessor


_SAM_NORM_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _prepend_env_path(var_name: str, new_path: str) -> None:
    if not new_path or not os.path.isdir(new_path):
        return
    current = os.environ.get(var_name, "")
    parts = [p for p in current.split(":") if p]
    if new_path not in parts:
        os.environ[var_name] = f"{new_path}:{current}" if current else new_path


def _candidate_prefixes() -> List[str]:
    prefixes: List[str] = []
    for p in [
        os.environ.get("CONDA_PREFIX", ""),
        os.path.dirname(os.path.dirname(sys.executable)),
        sys.prefix,
        "/home/penguin72487/miniforge3/envs/medsam",
    ]:
        if p and os.path.isdir(p) and p not in prefixes:
            prefixes.append(p)
    return prefixes


def _parse_cuda_release_from_ptxas(ptxas_bin: str) -> Tuple[Optional[Tuple[int, int]], str]:
    try:
        out = subprocess.check_output([ptxas_bin, "--version"], text=True, stderr=subprocess.STDOUT)
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return (int(m.group(1)), int(m.group(2))), out
        return None, out
    except Exception as e:
        return None, str(e)


def _find_best_ptxas(prefixes: List[str]) -> Tuple[Optional[str], Optional[Tuple[int, int]], List[str]]:
    candidates: List[str] = []

    for p in prefixes:
        cand = os.path.join(p, "bin", "ptxas")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            candidates.append(cand)

    which_ptxas = shutil.which("ptxas")
    if which_ptxas:
        candidates.append(which_ptxas)

    scan_roots = [
        os.path.expanduser("~/.local/share/mamba/pkgs"),
        os.path.expanduser("~/miniforge3/pkgs"),
        os.path.expanduser("~/mambaforge/pkgs"),
        os.path.expanduser("~/anaconda3/pkgs"),
        os.path.expanduser("~/miniconda3/pkgs"),
    ]
    for root in scan_roots:
        if not os.path.isdir(root):
            continue
        patterns = [
            os.path.join(root, "**", "cuda-nvcc-*", "bin", "ptxas"),
            os.path.join(root, "**", "cuda-compiler-*", "bin", "ptxas"),
        ]
        for pat in patterns:
            for cand in glob.glob(pat, recursive=True):
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    candidates.append(cand)

    uniq: List[str] = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)

    best_ptxas = None
    best_ver = None
    for c in uniq:
        ver, _ = _parse_cuda_release_from_ptxas(c)
        if ver is None:
            continue
        if best_ver is None or ver > best_ver:
            best_ver = ver
            best_ptxas = c

    return best_ptxas, best_ver, uniq


def _target_ptx_from_cuda_release(release: Optional[Tuple[int, int]]) -> Optional[int]:
    if release is None:
        return None
    major, minor = release
    # 目標修復：CUDA 13.2 對齊 PTX 9.2
    if (major, minor) >= (13, 2):
        return 92
    # CUDA 13.0/13.1 先維持 9.0，避免超前 ptxas
    if major >= 13:
        return 90
    if (major, minor) >= (12, 4):
        return 84
    return None


def _patch_triton_ptx_cap(target_ptx_version: int) -> bool:
    try:
        import triton.backends.nvidia.compiler as triton_nvidia_compiler
    except Exception:
        return False

    sentinel = "_medsam_ptx_patch_orig"
    if not hasattr(triton_nvidia_compiler, sentinel):
        setattr(triton_nvidia_compiler, sentinel, triton_nvidia_compiler.get_ptx_version_from_options)

    orig_fn = getattr(triton_nvidia_compiler, sentinel)

    def _patched_get_ptx_version_from_options(options, arch):
        ptx_version = orig_fn(options, arch)
        try:
            ptx_version = int(ptx_version)
        except Exception:
            pass
        return min(ptx_version, target_ptx_version)

    triton_nvidia_compiler.get_ptx_version_from_options = _patched_get_ptx_version_from_options
    return True


def _normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(str(k).startswith("_orig_mod.") for k in sd.keys()):
        return sd
    return {
        (k[len("_orig_mod.") :] if str(k).startswith("_orig_mod.") else k): v
        for k, v in sd.items()
    }


def load_state_dict_compat(target_model: torch.nn.Module, path: Path, map_location: str) -> None:
    sd = torch.load(path, map_location=map_location)
    if isinstance(sd, dict):
        sd = _normalize_state_dict_keys(sd)
    base = target_model._orig_mod if hasattr(target_model, "_orig_mod") and getattr(target_model, "_orig_mod") is not None else target_model
    base.load_state_dict(sd, strict=False)


def _build_compile_warmup_inputs(processor: SamProcessor, device: str, image_size: int) -> Dict[str, torch.Tensor]:
    test_img = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    inputs = processor(
        images=[test_img],
        input_boxes=[[[0, 0, image_size - 1, image_size - 1]]],
        return_tensors="pt",
    )
    return _move_inputs_to_device(inputs, device)


def _try_compile_model(model: SamModel, processor: SamProcessor, device: str, image_size: int) -> Tuple[SamModel, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "compiled": False,
        "backend": None,
        "error": "",
        "ptxas_path": None,
        "ptxas_version": None,
        "target_ptx": None,
        "triton_patch": False,
    }

    enable_compile = _env_bool("MEDSAM_ENABLE_COMPILE", True)
    if not enable_compile:
        report["error"] = "CompileDisabledByEnv"
        return model, report

    try:
        torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
        has_torch_compile = torch_version >= (2, 0)
    except Exception:
        has_torch_compile = False

    if not has_torch_compile:
        report["error"] = "torch.compile not available"
        return model, report

    try:
        import triton  # noqa: F401
        has_triton = True
    except Exception:
        has_triton = False

    if not has_triton:
        report["error"] = "Triton not available"
        return model, report

    prefixes = _candidate_prefixes()
    for prefix in prefixes:
        _prepend_env_path("PATH", os.path.join(prefix, "bin"))
        _prepend_env_path("CPATH", os.path.join(prefix, "include"))
        _prepend_env_path("LIBRARY_PATH", os.path.join(prefix, "lib"))
        _prepend_env_path("LIBRARY_PATH", os.path.join(prefix, "targets", "x86_64-linux", "lib"))
        _prepend_env_path("LD_LIBRARY_PATH", os.path.join(prefix, "lib"))
        _prepend_env_path("LD_LIBRARY_PATH", os.path.join(prefix, "targets", "x86_64-linux", "lib"))

    best_ptxas, best_release, _all_candidates = _find_best_ptxas(prefixes)
    report["ptxas_path"] = best_ptxas
    report["ptxas_version"] = best_release

    if best_ptxas:
        os.environ["TRITON_PTXAS_PATH"] = best_ptxas
        _prepend_env_path("PATH", os.path.dirname(best_ptxas))

    target_ptx = _target_ptx_from_cuda_release(best_release)
    report["target_ptx"] = target_ptx

    if target_ptx is None:
        report["error"] = f"No compatible PTX target for ptxas release={best_release}"
        return model, report

    strict_132 = _env_bool("MEDSAM_STRICT_PTXAS_13_2", False)
    if strict_132 and (best_release is None or best_release < (13, 2)):
        report["error"] = f"STRICT 13.2 enabled, found ptxas={best_release}"
        return model, report

    report["triton_patch"] = _patch_triton_ptx_cap(target_ptx)

    compile_mode = os.getenv("MEDSAM_COMPILE_MODE", "max-autotune")
    try:
        compiled = torch.compile(
            model,
            backend="inductor",
            mode=compile_mode,
            fullgraph=False,
            dynamic=True,
        )
        warmup_inputs = _build_compile_warmup_inputs(processor, device=device, image_size=image_size)
        with torch.no_grad():
            _ = compiled(**warmup_inputs)
        report["compiled"] = True
        report["backend"] = "inductor"
        return compiled, report
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        return model, report


def load_medsam(model_id: str, device: str, image_size: int, local_weight_path: str = "") -> Tuple[SamModel, SamProcessor, Dict[str, Any]]:
    model = SamModel.from_pretrained(model_id)
    processor = SamProcessor.from_pretrained(model_id)

    if local_weight_path and Path(local_weight_path).exists():
        load_state_dict_compat(model, Path(local_weight_path), map_location=device)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model, compile_report = _try_compile_model(model=model, processor=processor, device=device, image_size=image_size)
    return model, processor, compile_report


def _move_inputs_to_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    moved = {}
    non_blocking = device == "cuda"
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "input_boxes" and v.dtype == torch.float64:
                v = v.to(torch.float32)
            moved[k] = v.to(device, non_blocking=non_blocking)
        else:
            moved[k] = v
    return moved


def _normalize_masks_to_4d(masks: torch.Tensor) -> torch.Tensor:
    if masks.dim() == 5:
        return masks[:, 0, 0, :, :].unsqueeze(1)
    if masks.dim() == 4:
        return masks
    if masks.dim() == 3:
        return masks.unsqueeze(1)
    raise ValueError(f"Unexpected pred_masks dims: {masks.dim()}")


def normalize_pred_masks_to_4d(masks: torch.Tensor) -> torch.Tensor:
    return _normalize_masks_to_4d(masks)


def build_inputs(processor: SamProcessor, image: Image.Image, input_box: List[int]) -> Dict[str, torch.Tensor]:
    return build_inputs_batch(processor=processor, images=[image], input_boxes=[[input_box]])


def _get_sam_norm_tensors(processor: SamProcessor) -> Tuple[torch.Tensor, torch.Tensor]:
    key = id(processor)
    cached = _SAM_NORM_CACHE.get(key)
    if cached is not None:
        return cached

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        _SAM_NORM_CACHE[key] = (mean, std)
        return mean, std

    image_mean = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
    image_std = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])
    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
    _SAM_NORM_CACHE[key] = (mean, std)
    return mean, std


def _get_sam_target_edge(processor: SamProcessor) -> int:
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


def _to_rgb_uint8_np(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _build_inputs_fast(
    processor: SamProcessor,
    images: List[Any],
    input_boxes: List[List[List[int]]],
) -> Dict[str, torch.Tensor]:
    target_edge = _get_sam_target_edge(processor)
    mean, std = _get_sam_norm_tensors(processor)

    pixel_values = []
    scaled_boxes_all: List[List[List[float]]] = []
    original_sizes: List[List[int]] = []
    reshaped_input_sizes: List[List[int]] = []

    for image, boxes_per_image in zip(images, input_boxes):
        arr = _to_rgb_uint8_np(image)
        orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])
        sx = float(target_edge) / float(max(orig_w, 1))
        sy = float(target_edge) / float(max(orig_h, 1))

        if orig_h != target_edge or orig_w != target_edge:
            import cv2

            arr = cv2.resize(arr, (target_edge, target_edge), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32).div_(255.0)
        tensor = (tensor - mean) / std
        pixel_values.append(tensor)

        scaled_boxes: List[List[float]] = []
        for box in boxes_per_image:
            x1, y1, x2, y2 = [float(v) for v in box]
            nx1 = float(np.clip(x1 * sx, 0.0, float(target_edge - 1)))
            ny1 = float(np.clip(y1 * sy, 0.0, float(target_edge - 1)))
            nx2 = float(np.clip(x2 * sx, 0.0, float(target_edge - 1)))
            ny2 = float(np.clip(y2 * sy, 0.0, float(target_edge - 1)))
            scaled_boxes.append([nx1, ny1, nx2, ny2])

        scaled_boxes_all.append(scaled_boxes)
        original_sizes.append([orig_h, orig_w])
        reshaped_input_sizes.append([target_edge, target_edge])

    return {
        "pixel_values": torch.stack(pixel_values, dim=0),
        "input_boxes": torch.tensor(scaled_boxes_all, dtype=torch.float32),
        "original_sizes": torch.tensor(original_sizes, dtype=torch.int64),
        "reshaped_input_sizes": torch.tensor(reshaped_input_sizes, dtype=torch.int64),
    }


def build_inputs_batch(
    processor: SamProcessor,
    images: List[Any],
    input_boxes: List[List[List[int]]],
) -> Dict[str, torch.Tensor]:
    use_fast = _env_bool("MEDSAM_USE_FAST_PREPROCESS", True)
    if use_fast:
        try:
            return _build_inputs_fast(processor=processor, images=images, input_boxes=input_boxes)
        except Exception:
            pass
    return processor(images=images, input_boxes=input_boxes, return_tensors="pt")


def predict_prob_mask(
    model: SamModel,
    processor: SamProcessor,
    image: Image.Image,
    input_box: List[int],
    device: str,
    use_amp: bool = True,
) -> np.ndarray:
    w, h = image.size
    inputs = build_inputs(processor, image, input_box)
    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.pred_masks)
        else:
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.pred_masks)

        probs = _normalize_masks_to_4d(probs)
        probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)

    return probs[0, 0].detach().cpu().numpy()


def predict_prob_masks_batch(
    model: SamModel,
    processor: SamProcessor,
    images: List[Image.Image],
    input_boxes: List[List[int]],
    device: str,
    use_amp: bool = True,
) -> np.ndarray:
    if not images:
        return np.empty((0, 0, 0), dtype=np.float32)

    w, h = images[0].size
    packed_boxes = [[box] for box in input_boxes]
    inputs = build_inputs_batch(processor=processor, images=images, input_boxes=packed_boxes)
    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.pred_masks)
        else:
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.pred_masks)

        probs = _normalize_masks_to_4d(probs)
        probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)

    return probs[:, 0].detach().cpu().numpy()


def predict_binary_mask(
    model: SamModel,
    processor: SamProcessor,
    image: Image.Image,
    input_box: List[int],
    device: str,
    use_amp: bool = True,
    threshold: float = 0.5,
) -> np.ndarray:
    prob = predict_prob_mask(
        model=model,
        processor=processor,
        image=image,
        input_box=input_box,
        device=device,
        use_amp=use_amp,
    )
    return (prob > threshold).astype(np.uint8)
