import glob
import os
import re
import shutil
import subprocess
import sys
import importlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure Triton discovers in-tree backends before any compile-related import paths run.
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SamModel, SamProcessor


def get_active_profiler() -> Optional[Any]:
    return None


_SAM_NORM_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
_CUDA_GRAPH_RUNNERS: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], bool], "_CudaGraphRunner"] = {}


class _CudaGraphRunner:
    def __init__(self, model: SamModel, sample_inputs: Dict[str, torch.Tensor], use_amp: bool):
        self.model = model
        self.use_amp = use_amp
        self.device = str(sample_inputs["pixel_values"].device)
        self.stream = torch.cuda.Stream(device=sample_inputs["pixel_values"].device)
        self.graph = torch.cuda.CUDAGraph()
        self.static_inputs = {
            "pixel_values": sample_inputs["pixel_values"].clone(),
            "input_boxes": sample_inputs["input_boxes"].clone(),
            "original_sizes": sample_inputs["original_sizes"].clone(),
            "reshaped_input_sizes": sample_inputs["reshaped_input_sizes"].clone(),
        }
        self.static_out: Optional[torch.Tensor] = None
        self._captured = False
        self._capture()

    def _forward(self) -> torch.Tensor:
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = self.model(**self.static_inputs)
                probs = torch.sigmoid(outputs.pred_masks)
        else:
            outputs = self.model(**self.static_inputs)
            probs = torch.sigmoid(outputs.pred_masks)
        return _normalize_masks_to_4d(probs)

    def _capture(self) -> None:
        # Warmup on side stream.
        with torch.cuda.stream(self.stream):
            for _ in range(2):
                _ = self._forward()
        torch.cuda.current_stream().wait_stream(self.stream)

        with torch.cuda.graph(self.graph):
            self.static_out = self._forward()
        self._captured = True

    def run(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self._captured or self.static_out is None:
            raise RuntimeError("CUDA graph is not captured")
        self.static_inputs["pixel_values"].copy_(inputs["pixel_values"])
        self.static_inputs["input_boxes"].copy_(inputs["input_boxes"])
        self.static_inputs["original_sizes"].copy_(inputs["original_sizes"])
        self.static_inputs["reshaped_input_sizes"].copy_(inputs["reshaped_input_sizes"])
        self.graph.replay()
        return self.static_out


def _ensure_triton_in_tree_backends() -> None:
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    try:
        importlib.import_module("triton.backends")
    except Exception:
        pass


def _env_bool(name: str, default: bool = False) -> bool:
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
        "/root/miniforge3/envs/medsam",
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
        # 支援封裝 checkpoint（含 metadata）與純 state_dict 兩種格式。
        if "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
            sd = sd["model_state_dict"]
        elif "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        sd = _normalize_state_dict_keys(sd)
    base = target_model._orig_mod if hasattr(target_model, "_orig_mod") and getattr(target_model, "_orig_mod") is not None else target_model
    base.load_state_dict(sd, strict=False)


def _build_compile_warmup_inputs(
    processor: SamProcessor,
    device: str,
    image_size: int,
    batch_size: int = 1,
) -> Dict[str, torch.Tensor]:
    bs = max(1, int(batch_size))
    images = [Image.new("RGB", (image_size, image_size), color=(0, 0, 0)) for _ in range(bs)]
    boxes = [[[0, 0, image_size - 1, image_size - 1]] for _ in range(bs)]
    inputs = processor(images=images, input_boxes=boxes, return_tensors="pt")
    return _move_inputs_to_device(inputs, device)


def _parse_warmup_batches(raw: str, default_batches: List[int]) -> List[int]:
    if not raw.strip():
        return default_batches

    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            n = int(token)
        except ValueError:
            continue
        if n > 0 and n not in values:
            values.append(n)

    return values or default_batches


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

    _ensure_triton_in_tree_backends()

    # Only use inductor for compilation per user preference.
    backend_candidates = ["inductor"]
    mode_candidates = ["reduce-overhead"]

    default_dynamic = False if device == "cuda" else True
    compile_dynamic = _env_bool("MEDSAM_COMPILE_DYNAMIC", default_dynamic)
    cuda_total_gb = _cuda_total_memory_gb() if device == "cuda" else None
    warmup_batches = _parse_warmup_batches(
        os.getenv("MEDSAM_COMPILE_WARMUP_BATCHES", ""),
        [1] if (device == "cuda" and cuda_total_gb is not None and cuda_total_gb <= 12.5) else ([1, 8] if device == "cuda" else [1]),
    )
    if device == "cuda" and cuda_total_gb is not None and cuda_total_gb <= 12.5:
        warmup_batches = [b for b in warmup_batches if b <= 1] or [1]

    last_error = ""

    for compile_backend in backend_candidates:
        for compile_mode in mode_candidates:
            try:
                compiled = torch.compile(
                    model,
                    backend=compile_backend,
                    mode=compile_mode,
                    fullgraph=False,
                    dynamic=compile_dynamic,
                )
                with torch.no_grad():
                    for bs in warmup_batches:
                        warmup_inputs = _build_compile_warmup_inputs(
                            processor,
                            device=device,
                            image_size=image_size,
                            batch_size=bs,
                        )
                        _ = compiled(**warmup_inputs)
                report["compiled"] = True
                report["backend"] = compile_backend
                report["error"] = ""
                report["compile_mode"] = compile_mode
                report["compile_dynamic"] = compile_dynamic
                report["warmup_batches"] = warmup_batches
                return compiled, report
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                break

    report["error"] = last_error
    return model, report


def load_medsam(model_id: str, device: str, image_size: int, local_weight_path: str = "") -> Tuple[SamModel, SamProcessor, Dict[str, Any]]:
    profiler = get_active_profiler()
    t_load = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        model = SamModel.from_pretrained(model_id, local_files_only=True)
        processor = SamProcessor.from_pretrained(model_id, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            "Local-only model loading failed. Please provide a local model directory via "
            "--model-id or ensure the model is already cached locally."
        ) from e

    if local_weight_path and Path(local_weight_path).exists():
        load_state_dict_compat(model, Path(local_weight_path), map_location=device)

    model = model.to(device)

    # Enable TensorFloat32 on CUDA and set matmul precision to avoid runtime warnings
    # while keeping performance characteristics suitable for training/eval.
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model, compile_report = _try_compile_model(model=model, processor=processor, device=device, image_size=image_size)
    if profiler is not None and profiler.enabled:
        profiler.record_duration("model.load_medsam", time.perf_counter() - t_load)
    return model, processor, compile_report


def _move_inputs_to_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    profiler = get_active_profiler()
    t0 = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
    moved = {}
    non_blocking = device == "cuda"
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "input_boxes" and v.dtype == torch.float64:
                v = v.to(torch.float32)
            if non_blocking and v.device.type == "cpu":
                try:
                    v = v.pin_memory()
                except Exception:
                    pass
            moved[k] = v.to(device, non_blocking=non_blocking)
        else:
            moved[k] = v
    if profiler is not None and profiler.enabled:
        profiler.record_duration("model.device_move", time.perf_counter() - t0)
    return moved


def move_inputs_to_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return _move_inputs_to_device(inputs, device)


def _run_prob_inference(
    model: SamModel,
    inputs: Dict[str, torch.Tensor],
    output_size: Optional[Tuple[int, int]],
    device: str,
    use_amp: bool = True,
) -> torch.Tensor:
    profiler = get_active_profiler()
    t0 = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
    with torch.inference_mode():
        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.pred_masks)
        else:
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.pred_masks)

        probs = _normalize_masks_to_4d(probs)
        if output_size is not None:
            probs = F.interpolate(probs, size=output_size, mode="bilinear", align_corners=False)
    if profiler is not None and profiler.enabled:
        profiler.record_duration("model.prob_inference", time.perf_counter() - t0)
    return probs


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
    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


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
    profiler = get_active_profiler()
    t0 = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
    use_fast = _env_bool("MEDSAM_USE_FAST_PREPROCESS", True)
    if use_fast:
        try:
            outputs = _build_inputs_fast(processor=processor, images=images, input_boxes=input_boxes)
            if profiler is not None and profiler.enabled:
                profiler.record_duration("model.build_inputs_fast", time.perf_counter() - t0)
            return outputs
        except Exception:
            pass
    outputs = processor(images=images, input_boxes=input_boxes, return_tensors="pt")
    if profiler is not None and profiler.enabled:
        profiler.record_duration("model.build_inputs_processor", time.perf_counter() - t0)
    return outputs


def predict_prob_masks_from_inputs(
    model: SamModel,
    inputs: Dict[str, torch.Tensor],
    device: str,
    output_size: Optional[Tuple[int, int]],
    use_amp: bool = True,
    inputs_already_on_device: bool = False,
) -> torch.Tensor:
    moved_inputs = inputs if inputs_already_on_device else _move_inputs_to_device(inputs, device)
    enable_cuda_graph = _env_bool("MEDSAM_ENABLE_CUDA_GRAPH", True)
    use_cuda_graph = (
        enable_cuda_graph
        and device == "cuda"
        and output_size is None
        and inputs_already_on_device
        and all(k in moved_inputs for k in ("pixel_values", "input_boxes", "original_sizes", "reshaped_input_sizes"))
    )

    if use_cuda_graph:
        try:
            key = (
                id(model),
                tuple(moved_inputs["pixel_values"].shape),
                tuple(moved_inputs["input_boxes"].shape),
                tuple(moved_inputs["original_sizes"].shape),
                tuple(moved_inputs["reshaped_input_sizes"].shape),
                bool(use_amp),
            )
            runner = _CUDA_GRAPH_RUNNERS.get(key)
            if runner is None:
                runner = _CudaGraphRunner(model=model, sample_inputs=moved_inputs, use_amp=use_amp)
                _CUDA_GRAPH_RUNNERS[key] = runner
            return runner.run(moved_inputs)
        except Exception:
            # Fallback to regular eager path when graph capture/replay is unsupported.
            pass

    return _run_prob_inference(
        model=model,
        inputs=moved_inputs,
        output_size=output_size,
        device=device,
        use_amp=use_amp,
    )
