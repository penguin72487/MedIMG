# %% [markdown]
# # MedSAM with OOD Detection and Test Time Augmentation (TTA)
# ## 在 TN3K 和 DDTI 甲狀腺超聲數據集上的評估框架
# 
# 這個筆記本實現了一個完整的醫學影像分割框架，集成了：
# - **MedSAM (Segment Anything in Medical Images)**: 基於 SAM 的醫學影像分割模型
# - **OOD Detection**: Out-of-Distribution 檢測，識別異常或分布外樣本
# - **TTA (Test Time Augmentation)**: 測試時增強，通過多種增強策略提升預測穩健性
# - **評估指標**: Dice、Jaccard、F1-Score 等多維度指標評估

# %% [markdown]
# ## 1. 環境與套件匯入
# 
# 匯入 PyTorch、NumPy、OpenCV 等核心科學計算與深度學習庫。

# %%
# 基礎科學計算與深度學習庫
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# PyTorch 和模型相關
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info

# 圖像處理與計算機視覺
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers 和 SAM 模型
from transformers import SamModel, SamProcessor

# XML 處理
from xml.etree import ElementTree as ET

# 進度條
from tqdm import tqdm

# 顯示設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass

print("✅ 所有套件已成功匯入")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== 統一環境參數預設值區 ====================
# 所有可透過環境變數覆寫的預設值都集中於此，避免分散在各處。
ENV_DEFAULTS: Dict[str, str] = {
    "MEDSAM_DATA_ROOT": "",
    "MEDSAM_DATASET_PATH_FALLBACK": "",
    "MEDSAM_WEIGHT_PATH": "/mnt/c/gitproject/0302/medsam_vit_b.pth",
    "MEDSAM_COMPILE_MODE": "max-autotune",
    "MEDSAM_PRED_CACHE_MAX": "3000",
    "MEDSAM_EVAL_MICROBATCH": "4",
    "MEDSAM_EMBED_BATCH": "8",
    "MEDSAM_EMBED_WORKERS": "auto",
    "MEDSAM_USE_FAST_PREPROCESS": "1",
    "MEDSAM_FINETUNE_COMPILE_PHASE2": "0",
    "MEDSAM_SKIP_FINETUNE": "0",
    "MEDSAM_FINETUNE_MAX_SAMPLES": "0",
    "MEDSAM_FINETUNE_EPOCHS": "100",
    "MEDSAM_FINETUNE_BATCH": "8",
    "MEDSAM_FINETUNE_LR": "1e-4",
    "MEDSAM_FINETUNE_VAL_RATIO": "0.1",
    "MEDSAM_FINETUNE_PATIENCE": "20",
    "MEDSAM_FINETUNE_MIN_DELTA": "1e-4",
    "MEDSAM_FINETUNE_GRAD_ACCUM": "2",
    "MEDSAM_FINETUNE_GRAD_CLIP": "1.0",
    "MEDSAM_AUTOTUNE_MICROBATCH": "1",
    "MEDSAM_TARGET_VRAM_UTIL": "0.95",
    "MEDSAM_VRAM_LIMIT_GB": "12",
    "MEDSAM_EVAL_MICROBATCH_MAX": "64",
    "MEDSAM_EVAL_MICROBATCH_MIN": "1",
    "MEDSAM_SPLIT_ROOT": "",
}


def env_default(name: str, fallback: str = "") -> str:
    return ENV_DEFAULTS.get(name, fallback)


def env_get(name: str, fallback: Optional[str] = None) -> str:
    _fallback = env_default(name) if fallback is None else fallback
    return os.getenv(name, _fallback)


_TRUE_SET = {"1", "true", "yes", "y", "on"}
_FALSE_SET = {"0", "false", "no", "n", "off", ""}


def parse_bool_str(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUE_SET:
        return True
    if normalized in _FALSE_SET:
        return False
    return default


def env_get_bool(name: str, fallback: bool = False) -> bool:
    raw = env_get(name, "1" if fallback else "0")
    return parse_bool_str(raw, default=fallback)


USE_FAST_PREPROCESS = env_get_bool("MEDSAM_USE_FAST_PREPROCESS", True)

# %%
# ==================== 性能優化配置 ====================
# Tensor Core 和 cuDNN 優化
import shutil
import subprocess
import torch.backends.cudnn as cudnn

def _prepend_env_path(var_name: str, new_path: str):
    if not new_path or not os.path.isdir(new_path):
        return
    current = os.environ.get(var_name, "")
    parts = [p for p in current.split(":") if p]
    if new_path not in parts:
        os.environ[var_name] = f"{new_path}:{current}" if current else new_path

# 依多來源建立候選前綴（Notebook 的 sys.executable 可能不是 conda env python）
candidate_prefixes = []
for p in [
    os.environ.get("CONDA_PREFIX", ""),
    os.path.dirname(os.path.dirname(sys.executable)),
    sys.prefix,
    "/home/penguin72487/miniforge3/envs/medsam"
 ]:
    if p and os.path.isdir(p) and p not in candidate_prefixes:
        candidate_prefixes.append(p)

for env_prefix in candidate_prefixes:
    _prepend_env_path("PATH", os.path.join(env_prefix, "bin"))
    _prepend_env_path("CPATH", os.path.join(env_prefix, "include"))
    _prepend_env_path("LIBRARY_PATH", os.path.join(env_prefix, "lib"))
    _prepend_env_path("LIBRARY_PATH", os.path.join(env_prefix, "targets", "x86_64-linux", "lib"))
    _prepend_env_path("LD_LIBRARY_PATH", os.path.join(env_prefix, "lib"))
    _prepend_env_path("LD_LIBRARY_PATH", os.path.join(env_prefix, "targets", "x86_64-linux", "lib"))

cudnn.enabled = True
cudnn.benchmark = True  # 自動優化卷積算法
cudnn.deterministic = False  # 關閉確定性以獲得更快速度

# 啟用混合精度優化
torch.set_float32_matmul_precision('high')

# 檢查 torch.compile / triton 可用性
try:
    _torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    HAS_TORCH_COMPILE = _torch_version >= (2, 0)
except Exception:
    HAS_TORCH_COMPILE = False

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

# 針對目前環境：使用可取得的 12.4 ptxas，並強制 Triton 產生 PTX 8.4
TRITON_FORCE_PTX_VERSION = 84  # PTX 8.4
TRITON_PTXAS_FALLBACK = "/home/penguin72487/.local/share/mamba/pkgs/https/conda.anaconda.org/nvidia/linux-64/cuda-nvcc-12.4.131-0/bin/ptxas"
TRITON_PTXAS_FALLBACK_DIR = os.path.dirname(TRITON_PTXAS_FALLBACK)
if os.path.isfile(TRITON_PTXAS_FALLBACK) and os.access(TRITON_PTXAS_FALLBACK, os.X_OK):
    os.environ["TRITON_PTXAS_PATH"] = TRITON_PTXAS_FALLBACK
    _prepend_env_path("PATH", TRITON_PTXAS_FALLBACK_DIR)

# 檢查 ptxas 版本（避免 PTX 版本不相容）
PTXAS_PATH = shutil.which("ptxas") or os.environ.get("TRITON_PTXAS_PATH")
if not PTXAS_PATH:
    for prefix in candidate_prefixes:
        cand = os.path.join(prefix, "bin", "ptxas")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            PTXAS_PATH = cand
            break

if PTXAS_PATH:
    os.environ["TRITON_PTXAS_PATH"] = PTXAS_PATH
    _prepend_env_path("PATH", os.path.dirname(PTXAS_PATH))

PTXAS_VERSION = None
PTXAS_OK_FOR_TRITON = False
if PTXAS_PATH:
    try:
        ptxas_out = subprocess.check_output([PTXAS_PATH, "--version"], text=True, stderr=subprocess.STDOUT)
        # 例如: release 12.4, V12.4.131
        import re
        m = re.search(r"release\s+(\d+)\.(\d+)", ptxas_out)
        if m:
            PTXAS_VERSION = (int(m.group(1)), int(m.group(2)))
            # 只要 ptxas >= 12.4，就能配合 PTX 8.4 的降版輸出
            PTXAS_OK_FOR_TRITON = PTXAS_VERSION >= (12, 4)
    except Exception:
        pass

# 使用者偏好：強制 torch.compile 使用 Triton(Inductor)
FORCE_TRITON_INDUCTOR = True

# 讓 Triton/Inductor 生成較低 PTX，對齊現有 ptxas
# 使用 sentinel attribute 確保 patch 可重複執行而不造成遞迴
TRITON_PTX_VERSION_PATCHED = False
if HAS_TORCH_COMPILE and HAS_TRITON:
    try:
        import triton.backends.nvidia.compiler as triton_nvidia_compiler
        _PTX_PATCH_SENTINEL = "_medsam_ptx_patch_orig"
        # 只在第一次執行時儲存真正的 original（存在 module 上，重跑不影響）
        if not hasattr(triton_nvidia_compiler, _PTX_PATCH_SENTINEL):
            setattr(triton_nvidia_compiler, _PTX_PATCH_SENTINEL,
                    triton_nvidia_compiler.get_ptx_version_from_options)
        _orig_get_ptx_version_from_options = getattr(triton_nvidia_compiler, _PTX_PATCH_SENTINEL)

        def _patched_get_ptx_version_from_options(options, arch):
            ptx_version = _orig_get_ptx_version_from_options(options, arch)
            try:
                ptx_version = int(ptx_version)
            except Exception:
                pass
            return min(ptx_version, TRITON_FORCE_PTX_VERSION)

        triton_nvidia_compiler.get_ptx_version_from_options = _patched_get_ptx_version_from_options
        TRITON_PTX_VERSION_PATCHED = True
    except Exception as e:
        print(f"⚠️  Triton PTX version patch 失敗: {e}")

if HAS_TORCH_COMPILE and HAS_TRITON and PTXAS_OK_FOR_TRITON:
    COMPILE_BACKEND = "inductor"
else:
    COMPILE_BACKEND = None

ENABLE_MODEL_COMPILE = COMPILE_BACKEND is not None

print("✅ Tensor Core 優化已啟用")
print(f"   候選前綴: {candidate_prefixes}")
print(f"   cuDNN Benchmark: {cudnn.benchmark}")
print(f"   cuDNN Deterministic: {cudnn.deterministic}")
print(f"   torch.compile 可用: {HAS_TORCH_COMPILE}")
print(f"   Triton 可用: {HAS_TRITON}")
print(f"   ptxas 路徑: {PTXAS_PATH}")
print(f"   ptxas 版本: {PTXAS_VERSION}")
print(f"   ptxas 可用於 Triton: {PTXAS_OK_FOR_TRITON}")
print(f"   Triton PTX version patch: {TRITON_PTX_VERSION_PATCHED}")
print(f"   Triton 目標 PTX 版本: {TRITON_FORCE_PTX_VERSION // 10}.{TRITON_FORCE_PTX_VERSION % 10}")
print(f"   torch.compile backend: {COMPILE_BACKEND}")
print(f"   強制 Triton/Inductor: {FORCE_TRITON_INDUCTOR}")
print("   Float32 MatMul Precision: high")

if FORCE_TRITON_INDUCTOR and HAS_TORCH_COMPILE:
    if not HAS_TRITON:
        print("\n⚠️  你要求 torch.compile + Triton，但目前環境缺少 Triton。")
        print("   請在 mamba 環境安裝相容版本後重啟 kernel。")
        print("   建議: pip install -U triton")
    elif not PTXAS_OK_FOR_TRITON:
        print("\n⚠️  你要求 Triton/Inductor，但 ptxas 版本不符。")
        print("   目前會安全降級 eager；若要強制 inductor，請補齊 ptxas。")


# %%
# ==================== Full-Throttle Runtime（無降級） ====================
# 目標：最大化吞吐，不允許自動保守降級
MAX_THROUGHPUT_MODE = True
STRICT_NO_FALLBACK = True

if torch.cuda.is_available():
    # 允許 TensorFloat-32，通常可提升 matmul/conv 吞吐
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 盡量提高 CPU 端供料能力（可依實機微調，過高可能反而抖動）
try:
    _logical_threads = os.cpu_count() or 16
    torch.set_num_threads(max(1, _logical_threads - 1))
    torch.set_num_interop_threads(max(1, min(8, _logical_threads // 2)))
except Exception as e:
    print(f"⚠️ 無法調整 PyTorch thread pools: {e}")

print("✅ Full-Throttle Runtime 已啟用")
print(f"   STRICT_NO_FALLBACK: {STRICT_NO_FALLBACK}")
print(f"   TF32 matmul: {getattr(torch.backends.cuda.matmul, 'allow_tf32', None)}")
print(f"   TF32 cudnn: {getattr(torch.backends.cudnn, 'allow_tf32', None)}")

# %%
# ptxas 路徑診斷（若 compile 無法啟用請先檢查）
import os, shutil, sys
_ptxas_abs = "/home/penguin72487/miniforge3/envs/medsam/bin/ptxas"
print("sys.executable:", sys.executable)
print("ptxas abs exists:", os.path.exists(_ptxas_abs))
print("ptxas abs isfile:", os.path.isfile(_ptxas_abs))
print("which ptxas:", shutil.which("ptxas"))

# %% [markdown]
# ## 2. 資料路徑與實驗設定
# 
# 定義 TN3K 與 DDTI 測試數據路徑、模型權重路徑、輸出目錄和實驗超參數。

# %%
# ==================== 實驗配置 ====================
# 數據路徑
DATA_PATHS = {
    "TN3K": "/mnt/c/gitproject/0302/TN3K",
    "DDTI": "/mnt/c/gitproject/0302/DDTI",
    "TN5000": "/mnt/c/gitproject/0302/TN5000/TN5000_forReview",
}


def _resolve_data_paths(paths: Dict[str, str]) -> Dict[str, str]:
    """解析資料路徑，優先使用使用者提供的高速路徑覆寫。"""
    resolved = dict(paths)
    data_root = env_get("MEDSAM_DATA_ROOT").strip()

    for name, default_path in paths.items():
        specific = os.getenv(f"MEDSAM_{name}_PATH", env_default("MEDSAM_DATASET_PATH_FALLBACK")).strip()
        if specific and Path(specific).exists():
            resolved[name] = specific
            continue

        if data_root:
            default_rel = Path(default_path).relative_to("/mnt/c/gitproject/0302")
            candidate = Path(data_root) / default_rel
            if candidate.exists():
                resolved[name] = str(candidate)

    return resolved


DATA_PATHS = _resolve_data_paths(DATA_PATHS)


def _resolve_split_root() -> Path:
    configured = env_get("MEDSAM_SPLIT_ROOT").strip()
    if configured:
        return Path(configured)
    try:
        return Path(__file__).resolve().parent / "splits"
    except Exception:
        return Path.cwd() / "splits"


SPLIT_ROOT = _resolve_split_root()


def _split_file(dataset_name: str, split_name: str) -> Optional[Path]:
    path = SPLIT_ROOT / dataset_name / f"{split_name}.txt"
    return path if path.exists() else None


def _read_split_ids(split_file: Optional[Path]) -> Optional[set]:
    if split_file is None:
        return None
    try:
        lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()]
        return {line for line in lines if line}
    except Exception:
        return None

# 預訓練權重路徑（可由 MEDSAM_WEIGHT_PATH 覆寫）
LOCAL_WEIGHT_PATH = env_get("MEDSAM_WEIGHT_PATH")
USE_LOCAL_WEIGHT = True  # 優先使用本地權重

# CPU / DataLoader 優化設定（5800X3D / 16 threads）
CPU_LOGICAL_THREADS = os.cpu_count() or 16
RECOMMENDED_DATALOADER_WORKERS = os.cpu_count()
RECOMMENDED_PREFETCH_FACTOR = 4

# 模型配置 - 強制使用 CUDA
MODEL_CONFIG = {
    "model_id": "facebook/sam-vit-base",  # SAM Base 模型
    "local_weight_path": LOCAL_WEIGHT_PATH if Path(LOCAL_WEIGHT_PATH).exists() else None,
    "image_size": 512,  # 輸入圖像大小
    "device": "cuda",  # 強制使用 CUDA
}

# 實驗參數
EXPERIMENT_CONFIG = {
    "batch_size": 1,
    "num_workers": RECOMMENDED_DATALOADER_WORKERS,
    "prefetch_factor": RECOMMENDED_PREFETCH_FACTOR,
    "persistent_workers": True,
    "pin_memory": True,
    "random_seed": 42,
    "num_tta_augmentations": 4,  # TTA 增強次數
    "tta_fast_mode": True,  # True: 使用較輕量增強以提高評估速度
    "ood_threshold": 0.5,  # OOD 檢測閾值
    "ood_method": "entropy",  # OOD 方法: entropy, confidence, msp, energy
}

# 輸出目錄
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 固定輸入 shape（對 compile / kernel fusion 友善）
FIXED_IMAGE_SIZE = (MODEL_CONFIG["image_size"], MODEL_CONFIG["image_size"])

# 預先計算 bbox：減少評估時的 Python 迴圈工作量
def compute_bbox_from_mask_np(mask_np: np.ndarray, jitter: int = 10) -> List[int]:
    y_idx, x_idx = np.where(mask_np > 0)
    if len(x_idx) == 0 or len(y_idx) == 0:
        return [0, 0, mask_np.shape[1], mask_np.shape[0]]
    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    h, w = mask_np.shape
    x_min = max(0, int(x_min) - jitter)
    x_max = min(w - 1, int(x_max) + jitter)
    y_min = max(0, int(y_min) - jitter)
    y_max = min(h - 1, int(y_max) + jitter)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def get_sample_identifier(sample: Dict[str, Any], idx: int) -> str:
    """產生穩定且唯一的樣本識別字串，避免不同資料集快取鍵衝突。"""
    if "name" in sample and sample["name"]:
        return str(sample["name"])
    if "image_id" in sample and sample["image_id"]:
        return str(sample["image_id"])
    if "case_id" in sample and "img_idx" in sample:
        return f"{sample['case_id']}_{sample['img_idx']}"
    if "case_id" in sample and sample["case_id"] is not None:
        return str(sample["case_id"])
    return f"sample_{idx}"

# 設置隨機種子
np.random.seed(EXPERIMENT_CONFIG["random_seed"])
torch.manual_seed(EXPERIMENT_CONFIG["random_seed"])
torch.cuda.manual_seed_all(EXPERIMENT_CONFIG["random_seed"])

print("=" * 60)
print("📊 實驗配置")
print("=" * 60)
print(f"設備: {MODEL_CONFIG['device']}")
print(f"模型: {MODEL_CONFIG['model_id']}")
print(f"圖像大小: {MODEL_CONFIG['image_size']}")
print(f"固定輸入 shape: {FIXED_IMAGE_SIZE}")
print(f"CPU logical threads: {CPU_LOGICAL_THREADS}")
print(f"建議 DataLoader workers: {RECOMMENDED_DATALOADER_WORKERS}")
if MODEL_CONFIG['local_weight_path']:
    print(f"本地權重: {MODEL_CONFIG['local_weight_path']}")
    print(f"權重文件存在: {Path(MODEL_CONFIG['local_weight_path']).exists()}")
print(f"TTA 增強次數: {EXPERIMENT_CONFIG['num_tta_augmentations']}")
print(f"TTA 快速模式: {EXPERIMENT_CONFIG['tta_fast_mode']}")
print(f"OOD 檢測方法: {EXPERIMENT_CONFIG['ood_method']}")
print(f"資料路徑 TN3K: {DATA_PATHS['TN3K']}")
print(f"資料路徑 DDTI: {DATA_PATHS['DDTI']}")
print(f"資料路徑 TN5000: {DATA_PATHS['TN5000']}")
print(f"隨機種子: {EXPERIMENT_CONFIG['random_seed']}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# %%
# ==================== Full-Throttle Data Pipeline 覆寫 ====================
# 以高吞吐優先覆寫評估設定
if 'EXPERIMENT_CONFIG' not in globals():
    raise RuntimeError("EXPERIMENT_CONFIG 尚未定義，請先執行上一格")

_cpu_threads = os.cpu_count() or 16
_aggressive_workers = os.cpu_count()
_aggressive_prefetch = 8

EXPERIMENT_CONFIG.update({
    "batch_size": 1,  # 目前流程以單樣本 bbox 推論為主，保持 1 避免形狀/對齊副作用
    "num_workers": _aggressive_workers,
    "prefetch_factor": _aggressive_prefetch,
    "persistent_workers": True,
    "pin_memory": True,
})

# 強制高負載模式
EXPERIMENT_CONFIG["tta_fast_mode"] = True
EXPERIMENT_CONFIG["num_tta_augmentations"] = max(4, EXPERIMENT_CONFIG.get("num_tta_augmentations", 4))

print("✅ Data Pipeline 已切換 Full-Throttle")
print(f"   CPU threads: {_cpu_threads}")
print(f"   num_workers: {EXPERIMENT_CONFIG['num_workers']}")
print(f"   prefetch_factor: {EXPERIMENT_CONFIG['prefetch_factor']}")
print(f"   persistent_workers: {EXPERIMENT_CONFIG['persistent_workers']}")
print(f"   pin_memory: {EXPERIMENT_CONFIG['pin_memory']}")

# %% [markdown]
# ## 3. 載入 TN3K 與 DDTI 測試資料
# 
# 建立資料讀取與前處理流程，將影像與標註 mask 轉換為可供 MedSAM 推論與評估的格式。

# %%
# TN3K 數據集類
class TN3KDataset(Dataset):
    """TN3K 甲狀腺超聲數據集"""

    def __init__(self, root_dir: str, split: str = "test", image_size: int = 512, split_file: Optional[Path] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.split_file = split_file
        self.samples = []
        self._load_samples()

    def _resolve_tn3k_pair(self, sample_id: str) -> Optional[Tuple[Path, Path]]:
        candidates = [
            (self.root_dir / f"{self.split}-image", self.root_dir / f"{self.split}-mask"),
            (self.root_dir / "train-image", self.root_dir / "train-mask"),
            (self.root_dir / "trainval-image", self.root_dir / "trainval-mask"),
            (self.root_dir / "test-image", self.root_dir / "test-mask"),
        ]
        for image_dir, mask_dir in candidates:
            img_path = image_dir / f"{sample_id}.jpg"
            mask_path = mask_dir / f"{sample_id}.jpg"
            if img_path.exists() and mask_path.exists():
                return img_path, mask_path
        return None

    def _load_samples(self):
        """載入所有樣本"""
        image_dir = self.root_dir / f"{self.split}-image"
        mask_dir = self.root_dir / f"{self.split}-mask"
        split_ids = _read_split_ids(self.split_file)

        if split_ids is not None:
            for sample_id in sorted(split_ids):
                pair = self._resolve_tn3k_pair(sample_id)
                if pair is None:
                    continue
                img_path, mask_path = pair
                self.samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "name": sample_id,
                })
            return

        if image_dir.exists():
            image_files = sorted(image_dir.glob("*.jpg"))
            for img_file in image_files:
                mask_file = mask_dir / img_file.name
                if mask_file.exists():
                    self.samples.append({
                        "image_path": img_file,
                        "mask_path": mask_file,
                        "name": img_file.stem,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 載入圖像和遮罩
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        # 調整大小
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        mask_np = np.array(mask) > 127
        bbox = compute_bbox_from_mask_np(mask_np.astype(np.uint8))

        return {
            "image": image,
            "image_np": np.array(image),
            "mask": torch.tensor(mask_np.astype(np.float32), dtype=torch.float32),
            "bbox": bbox,
            "name": sample["name"],
        }


# DDTI 數據集類
class DDTIDataset(Dataset):
    """DDTI 甲狀腺超聲數據集 - XML 標註格式"""

    def __init__(self, root_dir: str, image_size: int = 512, split_file: Optional[Path] = None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.split_file = split_file
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """載入所有樣本"""
        xml_files = sorted(self.root_dir.glob("*.xml"))
        split_ids = _read_split_ids(self.split_file)

        for xml_file in xml_files:
            case_id = int(xml_file.stem)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 查找所有標註
            marks = root.findall(".//mark")
            for mark in marks:
                img_elem = mark.find("image")
                svg_elem = mark.find("svg")

                if img_elem is not None and svg_elem is not None:
                    img_idx = int(img_elem.text)
                    sample_name = f"{case_id}_{img_idx}"
                    if split_ids is not None and sample_name not in split_ids:
                        continue
                    img_path = self.root_dir / f"{case_id}_{img_idx}.jpg"

                    if img_path.exists():
                        self.samples.append({
                            "image_path": img_path,
                            "case_id": case_id,
                            "img_idx": img_idx,
                            "svg": svg_elem.text,
                            "name": sample_name,
                        })

    def _svg_to_mask(self, svg_str: str, h: int, w: int) -> np.ndarray:
        """將 SVG 多邊形轉換為二值遮罩"""
        mask = np.zeros((h, w), dtype=np.uint8)

        try:
            annotations = json.loads(svg_str)

            for annotation in annotations:
                if annotation.get("regionType") == "freehand":
                    points = annotation.get("points", [])
                    if len(points) > 2:
                        pts = np.array(
                            [[p["x"], p["y"]] for p in points],
                            dtype=np.int32,
                        )
                        cv2.fillPoly(mask, [pts], 1)
        except:
            pass

        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 載入圖像
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image)

        # 生成遮罩
        mask = self._svg_to_mask(sample["svg"], self.image_size, self.image_size)
        bbox = compute_bbox_from_mask_np(mask)

        return {
            "image": image,
            "image_np": image_np,
            "mask": torch.tensor(mask.astype(np.float32), dtype=torch.float32),
            "bbox": bbox,
            "case_id": sample["case_id"],
            "img_idx": sample["img_idx"],
            "name": sample["name"],
        }


class TN5000Dataset(Dataset):
    """TN5000 資料集（VOC XML bbox，轉換為矩形 mask）"""

    def __init__(self, root_dir: str, split: str = "test", image_size: int = 512, split_file: Optional[Path] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.split_file = split_file
        self.samples = []
        self._load_samples()

    def _parse_boxes(self, xml_root: ET.Element) -> List[List[int]]:
        boxes = []
        for obj in xml_root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                xmin = int(float(bnd.findtext("xmin", default="0")))
                ymin = int(float(bnd.findtext("ymin", default="0")))
                xmax = int(float(bnd.findtext("xmax", default="0")))
                ymax = int(float(bnd.findtext("ymax", default="0")))
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
            except Exception:
                continue
        return boxes

    def _load_samples(self):
        ann_dir = self.root_dir / "Annotations"
        image_dir = self.root_dir / "JPEGImages"
        split_file = self.root_dir / "ImageSets" / "Main" / f"{self.split}.txt"
        split_ids = _read_split_ids(self.split_file)

        if split_ids is not None:
            image_ids = sorted(split_ids)
        elif split_file.exists():
            image_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        else:
            image_ids = [p.stem for p in sorted(ann_dir.glob("*.xml"))]

        for image_id in image_ids:
            xml_path = ann_dir / f"{image_id}.xml"
            img_path = image_dir / f"{image_id}.jpg"
            if not xml_path.exists() or not img_path.exists():
                continue

            try:
                root = ET.parse(xml_path).getroot()
                width = int(root.findtext("size/width", default="0"))
                height = int(root.findtext("size/height", default="0"))
                boxes = self._parse_boxes(root)
                if width > 0 and height > 0 and len(boxes) > 0:
                    self.samples.append({
                        "image_id": image_id,
                        "image_path": img_path,
                        "width": width,
                        "height": height,
                        "boxes": boxes,
                    })
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image)

        # 以標註 bbox 生成矩形 mask，讓既有 segmentation 評估流程可直接沿用
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        sx = self.image_size / max(sample["width"], 1)
        sy = self.image_size / max(sample["height"], 1)

        for xmin, ymin, xmax, ymax in sample["boxes"]:
            x1 = int(np.clip(round((xmin - 1) * sx), 0, self.image_size - 1))
            y1 = int(np.clip(round((ymin - 1) * sy), 0, self.image_size - 1))
            x2 = int(np.clip(round((xmax - 1) * sx), 0, self.image_size - 1))
            y2 = int(np.clip(round((ymax - 1) * sy), 0, self.image_size - 1))
            if x2 >= x1 and y2 >= y1:
                mask[y1:y2 + 1, x1:x2 + 1] = 1

        bbox = compute_bbox_from_mask_np(mask)

        return {
            "image": image,
            "image_np": image_np,
            "mask": torch.tensor(mask.astype(np.float32), dtype=torch.float32),
            "bbox": bbox,
            "name": sample["image_id"],
        }


# 載入數據集
print("🔄 載入 TN3K 測試集...")
tn3k_test_split_file = _split_file("TN3K", "test")
tn3k_dataset = TN3KDataset(
    DATA_PATHS["TN3K"],
    split="test",
    image_size=MODEL_CONFIG["image_size"],
    split_file=tn3k_test_split_file,
)
print(f"✅ TN3K 測試集: {len(tn3k_dataset)} 個樣本")

print("\n🔄 載入 DDTI 數據集...")
ddti_test_split_file = _split_file("DDTI", "test")
ddti_dataset = DDTIDataset(
    DATA_PATHS["DDTI"],
    image_size=MODEL_CONFIG["image_size"],
    split_file=ddti_test_split_file,
)
print(f"✅ DDTI 數據集: {len(ddti_dataset)} 個樣本")

print("\n🔄 載入 TN5000 測試集...")
tn5000_test_split_file = _split_file("TN5000", "test")
tn5000_dataset = TN5000Dataset(
    DATA_PATHS["TN5000"],
    split="test",
    image_size=MODEL_CONFIG["image_size"],
    split_file=tn5000_test_split_file,
)
print(f"✅ TN5000 測試集: {len(tn5000_dataset)} 個樣本")

# %% [markdown]
# ## 4. MedSAM 模型載入與推論流程
# 
# 載入預訓練 SAM（Segment Anything Model）權重，作為 MedSAM 的基礎。建立影像編碼、提示生成與分割輸出流程。

# %%
def _move_inputs_to_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    use_cuda = (device == "cuda")
    moved = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            # input_boxes 必須是 float32；processor() 預設回傳 float64 會導致 dynamo 反覆重編譯
            if k == "input_boxes" and v.dtype == torch.float64:
                v = v.to(torch.float32)
            moved[k] = v.to(device, non_blocking=use_cuda)
        else:
            moved[k] = v
    return moved


def _run_sam_forward_prob_masks(
    model: SamModel,
    inputs: Dict[str, torch.Tensor],
    device: str,
    use_amp: bool,
) -> torch.Tensor:
    """執行 SAM forward，回傳 sigmoid 後的 mask 機率張量。"""
    def _forward_once(run_model):
        if use_amp and device == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _outputs = run_model(**inputs)
                return torch.sigmoid(_outputs.pred_masks)
        _outputs = run_model(**inputs)
        return torch.sigmoid(_outputs.pred_masks)

    try:
        return _forward_once(model)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        is_inductor_ptxas_err = (
            "InductorError" in err or
            "PTXAS" in err or
            "Unsupported .version" in err
        )
        if is_inductor_ptxas_err and globals().get("EAGER_MODEL_FALLBACK") is not None:
            if not globals().get("_COMPILE_RUNTIME_FALLBACK_WARNED", False):
                print("⚠️  偵測到 Triton/ptxas 編譯錯誤，已自動回退 eager 以避免評估中斷。")
                print(f"   原因: {err}")
                globals()["_COMPILE_RUNTIME_FALLBACK_WARNED"] = True
            globals()["MODEL_COMPILED"] = False
            globals()["MODEL_COMPILE_ERROR"] = err
            globals()["model"] = globals()["EAGER_MODEL_FALLBACK"]
            return _forward_once(globals()["EAGER_MODEL_FALLBACK"])
        raise


def _normalize_masks_to_4d(masks: torch.Tensor) -> torch.Tensor:
    """將不同形狀的 pred_masks 統一轉成 (B, 1, H, W)。"""
    if masks.dim() == 5:
        return masks[:, 0, 0, :, :].unsqueeze(1)
    if masks.dim() == 4:
        return masks
    if masks.dim() == 3:
        return masks.unsqueeze(1)
    raise ValueError(f"Unexpected mask dims: {masks.dim()}")


_FAST_PREPROCESS_WARNED = False
_SAM_NORM_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _should_pin_fast_preprocess_outputs() -> bool:
    """只在主程序 pin memory，避免 DataLoader worker 觸發 CUDA 初始化錯誤。"""
    if get_worker_info() is not None:
        return False
    return torch.cuda.is_available()


def _get_sam_norm_tensors(processor: SamProcessor) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_key = id(processor)
    cached = _SAM_NORM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("SamProcessor 缺少 image_processor，無法使用快速前處理")

    image_mean = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
    image_std = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])
    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
    _SAM_NORM_CACHE[cache_key] = (mean, std)
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
        if "shortest_edge" in size_cfg:
            return int(size_cfg["shortest_edge"])
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


def _build_sam_inputs_fast(
    processor: SamProcessor,
    images: List[Any],
    input_boxes: List[List[List[int]]],
) -> Dict[str, torch.Tensor]:
    if len(images) != len(input_boxes):
        raise ValueError("images 與 input_boxes 長度不一致")

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
            arr = cv2.resize(arr, (target_edge, target_edge), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32).div_(255.0)
        tensor = (tensor - mean) / std
        pixel_values.append(tensor)

        scaled_boxes = []
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

    pixel_values_t = torch.stack(pixel_values, dim=0)
    input_boxes_t = torch.tensor(scaled_boxes_all, dtype=torch.float32)
    original_sizes_t = torch.tensor(original_sizes, dtype=torch.int64)
    reshaped_input_sizes_t = torch.tensor(reshaped_input_sizes, dtype=torch.int64)

    if _should_pin_fast_preprocess_outputs():
        pixel_values_t = pixel_values_t.pin_memory()
        input_boxes_t = input_boxes_t.pin_memory()
        original_sizes_t = original_sizes_t.pin_memory()
        reshaped_input_sizes_t = reshaped_input_sizes_t.pin_memory()

    return {
        "pixel_values": pixel_values_t,
        "input_boxes": input_boxes_t,
        "original_sizes": original_sizes_t,
        "reshaped_input_sizes": reshaped_input_sizes_t,
    }


def build_sam_inputs(
    processor: SamProcessor,
    images: List[Any],
    input_boxes: List[List[List[int]]],
) -> Dict[str, torch.Tensor]:
    global _FAST_PREPROCESS_WARNED

    if USE_FAST_PREPROCESS:
        try:
            return _build_sam_inputs_fast(processor, images, input_boxes)
        except Exception as e:
            if not _FAST_PREPROCESS_WARNED:
                print(f"⚠️ 快速前處理失敗，回退 processor(): {e}")
                _FAST_PREPROCESS_WARNED = True

    return processor(images=images, input_boxes=input_boxes, return_tensors="pt")


def _predict_from_processor_inputs(
    model: SamModel,
    inputs: Dict[str, torch.Tensor],
    image_hw: Tuple[int, int],
    device: str,
    use_amp: bool,
) -> np.ndarray:
    """共用推論路徑：inputs -> model -> resize -> binary mask。"""
    h, w = image_hw
    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        masks = _run_sam_forward_prob_masks(model, inputs, device, use_amp)
        masks = _normalize_masks_to_4d(masks)
        masks = F.interpolate(
            masks,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

    return (masks[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def _predict_from_processor_inputs_batch(
    model: SamModel,
    inputs: Dict[str, torch.Tensor],
    image_hw: Tuple[int, int],
    device: str,
    use_amp: bool,
) -> np.ndarray:
    """批次推論路徑：一次 forward 回傳 (B, H, W) 二值遮罩。"""
    h, w = image_hw
    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        masks = _run_sam_forward_prob_masks(model, inputs, device, use_amp)
        masks = _normalize_masks_to_4d(masks)
        masks = F.interpolate(
            masks,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

    return (masks[:, 0].cpu().numpy() > 0.5).astype(np.uint8)


def predict_medsam(
    model: SamModel,
    processor: SamProcessor,
    image: Image.Image,
    input_box: List[int],
    device: str = "cpu",
    use_amp: bool = True
) -> np.ndarray:
    """
    使用 MedSAM 進行預測 (優化版本)
    
    Args:
        model: SAM 模型
        processor: SAM 處理器
        image: 輸入圖像
        input_box: 邊界框 [x_min, y_min, x_max, y_max]
        device: 計算設備
        use_amp: 是否使用混合精度
        
    Returns:
        預測遮罩 (H, W)
    """
    w, h = image.size
    assert (h, w) == FIXED_IMAGE_SIZE, f"固定輸入 shape 應為 {FIXED_IMAGE_SIZE}，目前為 {(h, w)}"

    inputs = build_sam_inputs(
        processor=processor,
        images=[image],
        input_boxes=[[input_box]],
    )
    return _predict_from_processor_inputs(
        model=model,
        inputs=inputs,
        image_hw=(h, w),
        device=device,
        use_amp=use_amp,
    )

# %%
# ==================== 載入 MedSAM 模型 (優化版本) ====================
print("🔄 載入 MedSAM 模型...")

device = MODEL_CONFIG["device"]
print(f"設備: {device}")

# 載入預訓練模型
print("載入 SamModel...")
model = SamModel.from_pretrained(MODEL_CONFIG["model_id"])

print("載入 SamProcessor...")
processor = SamProcessor.from_pretrained(MODEL_CONFIG["model_id"])


# 載入本地權重
if MODEL_CONFIG["local_weight_path"] and Path(MODEL_CONFIG["local_weight_path"]).exists():
    print(f"🔄 載入本地權重: {MODEL_CONFIG['local_weight_path']}")
    try:
        state_dict = torch.load(MODEL_CONFIG["local_weight_path"], map_location=device)
        if isinstance(state_dict, dict) and any(
            isinstance(k, str) and k.startswith("_orig_mod.") for k in state_dict.keys()
        ):
            state_dict = {
                (k[len("_orig_mod."):] if isinstance(k, str) and k.startswith("_orig_mod.") else k): v
                for k, v in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=False)
        print("✅ 本地權重載入成功")
    except Exception as e:
        print(f"⚠️  本地權重載入失敗: {e}")

# 移至設備並設置優化
model = model.to(device)
model.eval()

# 禁用梯度計算 (推論時不需要)
for param in model.parameters():
    param.requires_grad = False

# 追蹤實際編譯狀態
MODEL_COMPILED = False
MODEL_COMPILE_ERROR = ""
MODEL_COMPILE_BACKEND = COMPILE_BACKEND
_COMPILE_RUNTIME_FALLBACK_WARNED = False

# 保存 eager 版本，供 runtime fallback 使用
EAGER_MODEL_FALLBACK = model

# 小型 warmup 測試輸入
def _build_test_inputs():
    test_img = Image.new("RGB", FIXED_IMAGE_SIZE, color=(0, 0, 0))
    _inputs = build_sam_inputs(
        processor=processor,
        images=[test_img],
        input_boxes=[[[0, 0, FIXED_IMAGE_SIZE[0] - 1, FIXED_IMAGE_SIZE[1] - 1]]],
    )
    return {k: v.to(device, non_blocking=(device == 'cuda')) for k, v in _inputs.items()}


def _state_dict_without_compile_prefix(m: torch.nn.Module) -> Dict[str, torch.Tensor]:
    base = m
    if hasattr(m, "_orig_mod") and getattr(m, "_orig_mod") is not None:
        base = m._orig_mod
    return base.state_dict()


def _load_state_dict_compat(target_model: torch.nn.Module, path: Path, map_location: str) -> None:
    sd = torch.load(path, map_location=map_location)
    if isinstance(sd, dict) and any(isinstance(k, str) and k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {
            (k[len("_orig_mod."):] if isinstance(k, str) and k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()
        }

    # 若 target 是 compiled OptimizedModule，直接載入到底層 eager model，
    # 避免 key 前綴（_orig_mod.）與 wrapper key-space 不一致。
    base = target_model
    if hasattr(target_model, "_orig_mod") and getattr(target_model, "_orig_mod") is not None:
        base = target_model._orig_mod
    base.load_state_dict(sd, strict=False)

# 安全編譯：使用者要求時，僅接受 Triton/Inductor
if HAS_TORCH_COMPILE:
    eager_model = model
    backends_to_try = []

    if COMPILE_BACKEND is not None:
        backends_to_try.append(COMPILE_BACKEND)

    if not backends_to_try and FORCE_TRITON_INDUCTOR:
        MODEL_COMPILE_ERROR = (
            "CompilePrecheckFailed: Triton/ptxas 條件不滿足（可能是 ptxas 過舊）。"
            "請先升級 ptxas 到 CUDA 13.x 後再執行。"
        )
        print(f"⚠️  {MODEL_COMPILE_ERROR}")

    for backend_name in backends_to_try:
        compile_mode = env_get("MEDSAM_COMPILE_MODE")
        print(f"🔄 嘗試編譯模型 (torch.compile + {backend_name})...")
        try:
            compiled_model = torch.compile(
                eager_model,
                backend=backend_name,
                mode=compile_mode,
                fullgraph=False,
                dynamic=True,
            )

            test_inputs = _build_test_inputs()
            with torch.no_grad():
                _ = compiled_model(**test_inputs)

            model = compiled_model
            MODEL_COMPILED = True
            MODEL_COMPILE_BACKEND = backend_name
            MODEL_COMPILE_ERROR = ""
            print(f"✅ 模型編譯成功並通過 warmup (backend={backend_name}, mode={compile_mode})")
            break
        except Exception as e:
            MODEL_COMPILE_ERROR = f"{type(e).__name__}: {e}"
            model = eager_model
            print(f"⚠️  backend={backend_name} 編譯失敗: {MODEL_COMPILE_ERROR}")

    if not MODEL_COMPILED:
        if FORCE_TRITON_INDUCTOR:
            print("⚠️  你要求 Triton/Inductor；目前未成功編譯，保留 eager 供流程可執行。")
        else:
            print("⚠️  編譯未成功，回退 eager")
else:
    print("ℹ️  跳過模型編譯（torch.compile 不可用）")

# 設置模型內存優化
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print(f"✅ 模型已優化並移至 {device}")
print(f"   模型: {MODEL_CONFIG['model_id']}")
print(f"   參數數量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print(f"   可訓練參數: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
print(f"   實際編譯狀態: {'compiled' if MODEL_COMPILED else 'eager'}")
print(f"   編譯 backend: {MODEL_COMPILE_BACKEND}")
if MODEL_COMPILE_ERROR:
    print(f"   編譯訊息: {MODEL_COMPILE_ERROR}")

# ==================== 定義輔助函數 ====================
def get_bbox_from_mask(mask: np.ndarray, jitter: int = 10) -> List[int]:
    """
    從遮罩提取邊界框 (優化版本)
    """
    return compute_bbox_from_mask_np(mask.astype(np.uint8), jitter=jitter)

print("✅ 輔助函數已定義")

# %%
# ==================== 嚴格全速守門（禁止降級 eager） ====================
if globals().get("STRICT_NO_FALLBACK", False):
    # 關閉 predict_medsam 內的 runtime eager fallback 條件
    globals()["EAGER_MODEL_FALLBACK"] = None

    _compiled_ok = bool(globals().get("MODEL_COMPILED", False))
    _backend = str(globals().get("MODEL_COMPILE_BACKEND", ""))
    if (not _compiled_ok) or (_backend != "inductor"):
        raise RuntimeError(
            "STRICT_NO_FALLBACK 啟用中：模型未處於 torch.compile(inductor) 狀態，已中止以避免降速。\n"
            f"MODEL_COMPILED={_compiled_ok}, MODEL_COMPILE_BACKEND={_backend}, "
            f"MODEL_COMPILE_ERROR={globals().get('MODEL_COMPILE_ERROR', '')}"
        )

print("✅ 嚴格全速守門通過：僅允許 compiled + inductor")

# %%
# ==================== Root-Cause 自動修復 + 強制重編譯（不降級） ====================
import re, glob, shutil, subprocess

def _parse_cuda_release_from_ptxas(ptxas_bin: str):
    try:
        out = subprocess.check_output([ptxas_bin, "--version"], text=True, stderr=subprocess.STDOUT)
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return (int(m.group(1)), int(m.group(2))), out
    except Exception as e:
        return None, str(e)
    return None, ""

def _find_best_ptxas(candidate_prefixes):
    candidates = []
    # 1) 既有環境與 PATH
    for p in candidate_prefixes:
        c = os.path.join(p, "bin", "ptxas")
        if os.path.isfile(c) and os.access(c, os.X_OK):
            candidates.append(c)
    which_ptxas = shutil.which("ptxas")
    if which_ptxas:
        candidates.append(which_ptxas)

    # 2) 掃描常見 conda/mamba 套件快取（找最高版本）
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
            for c in glob.glob(pat, recursive=True):
                if os.path.isfile(c) and os.access(c, os.X_OK):
                    candidates.append(c)

    # 去重
    uniq = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)

    best = None
    best_ver = (-1, -1)
    for c in uniq:
        ver, _ = _parse_cuda_release_from_ptxas(c)
        if ver and ver > best_ver:
            best_ver = ver
            best = c
    return best, best_ver, uniq

def _derive_target_ptx(cuda_release):
    # 保守對齊：12.4 對應 PTX 8.4；13.x 先給 9.0，避免超前 ptxas
    if not cuda_release:
        return 84
    major, minor = cuda_release
    if major >= 13:
        return 90
    if major == 12 and minor >= 4:
        return 84
    # 低於 12.4 不接受，直接報錯
    return None

if not globals().get("STRICT_NO_FALLBACK", False):
    raise RuntimeError("必須先啟用 STRICT_NO_FALLBACK，否則不允許執行此格")

best_ptxas, best_cuda_rel, all_ptxas = _find_best_ptxas(candidate_prefixes)
if not best_ptxas:
    raise RuntimeError(
        "找不到可執行的 ptxas。請先安裝 CUDA NVCC（建議 12.4+ 或 13.x）再重跑。"
    )

target_ptx = _derive_target_ptx(best_cuda_rel)
if target_ptx is None:
    raise RuntimeError(
        f"找到的最高 ptxas 版本為 {best_cuda_rel}，低於 12.4，不符合 Triton/Inductor 高速需求。"
    )

os.environ["TRITON_PTXAS_PATH"] = best_ptxas
_prepend_env_path("PATH", os.path.dirname(best_ptxas))
TRITON_FORCE_PTX_VERSION = target_ptx

# 重新 patch Triton PTX 版本上限
try:
    import triton.backends.nvidia.compiler as triton_nvidia_compiler
    _PTX_PATCH_SENTINEL = "_medsam_ptx_patch_orig"
    if not hasattr(triton_nvidia_compiler, _PTX_PATCH_SENTINEL):
        setattr(
            triton_nvidia_compiler,
            _PTX_PATCH_SENTINEL,
            triton_nvidia_compiler.get_ptx_version_from_options
        )
    _orig_get_ptx_version_from_options = getattr(triton_nvidia_compiler, _PTX_PATCH_SENTINEL)

    def _patched_get_ptx_version_from_options(options, arch):
        ptx_version = _orig_get_ptx_version_from_options(options, arch)
        try:
            ptx_version = int(ptx_version)
        except Exception:
            pass
        return min(ptx_version, TRITON_FORCE_PTX_VERSION)

    triton_nvidia_compiler.get_ptx_version_from_options = _patched_get_ptx_version_from_options
    TRITON_PTX_VERSION_PATCHED = True
except Exception as e:
    raise RuntimeError(f"Triton PTX patch 失敗: {e}")

# 強制重編譯（根因修復後重建）
if not HAS_TORCH_COMPILE:
    raise RuntimeError("當前 torch 不支援 torch.compile，無法進入最高速路徑")
if not HAS_TRITON:
    raise RuntimeError("當前環境缺少 Triton，無法使用 inductor 高速路徑")

try:
    eager_model = model
    if hasattr(model, "_orig_mod") and model._orig_mod is not None:
        eager_model = model._orig_mod
except Exception:
    eager_model = model

compiled_model = torch.compile(
    eager_model,
    backend="inductor",
    mode="max-autotune",
    fullgraph=False,
    dynamic=True,
)

# warmup 驗證
test_inputs = _build_test_inputs()
with torch.no_grad():
    _ = compiled_model(**test_inputs)

model = compiled_model
globals()["model"] = compiled_model
MODEL_COMPILED = True
MODEL_COMPILE_BACKEND = "inductor"
MODEL_COMPILE_ERROR = ""
EAGER_MODEL_FALLBACK = None

print("✅ Root-Cause 修復完成，已強制 compiled + inductor")
print(f"   可用 ptxas 候選數: {len(all_ptxas)}")
print(f"   選用 ptxas: {best_ptxas}")
print(f"   ptxas CUDA release: {best_cuda_rel}")
print(f"   目標 PTX: {TRITON_FORCE_PTX_VERSION // 10}.{TRITON_FORCE_PTX_VERSION % 10}")
print(f"   MODEL_COMPILED: {MODEL_COMPILED}")
print(f"   MODEL_COMPILE_BACKEND: {MODEL_COMPILE_BACKEND}")

# %%
# 快速煙霧測試：避免一口氣跑完整資料才發現 compile/runtime 問題
import time

def _mini_dataset(dataset, n=2):
    class _Mini:
        def __len__(self):
            return min(n, len(dataset))
        def __getitem__(self, idx):
            return dataset[idx]
    return _Mini()

def _dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = (pred & gt).sum()
    return float((2.0 * inter) / (pred.sum() + gt.sum() + 1e-8))

mini_ds = _mini_dataset(tn3k_dataset, n=2)
mini_dice = []
mini_times = []

for idx in range(len(mini_ds)):
    sample = mini_ds[idx]
    image = sample["image"]
    gt_mask = sample["mask"].numpy()
    bbox = sample.get("bbox") or get_bbox_from_mask(gt_mask)

    t0 = time.perf_counter()
    pred_mask = predict_medsam(model, processor, image, bbox, device, use_amp=True)
    dt = time.perf_counter() - t0

    mini_dice.append(_dice_score(pred_mask, gt_mask))
    mini_times.append(dt)

mini_stats = {
    "num_samples": int(len(mini_ds)),
    "mean_dice": float(np.mean(mini_dice)),
    "std_dice": float(np.std(mini_dice)),
    "avg_inference_time_ms": float(np.mean(mini_times) * 1000.0),
}

print("✅ mini smoke test passed")
print(mini_stats)

# %%
# ==================== 調試：檢查模型輸出形狀 ====================
print("🔍 調試模型輸出形狀...")

# 取得第一個樣本進行測試
test_sample = tn3k_dataset[0]
test_image = test_sample["image"]
test_mask = test_sample["mask"].numpy()
test_bbox = get_bbox_from_mask(test_mask)

print(f"測試圖像大小: {np.array(test_image).shape}")
print(f"測試邊界框: {test_bbox}")

# 執行推論獲得原始輸出
inputs = build_sam_inputs(
    processor=processor,
    images=[test_image],
    input_boxes=[[test_bbox]],
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    print(f"\n原始 pred_masks 形狀: {outputs.pred_masks.shape}")
    print(f"原始 pred_masks dtype: {outputs.pred_masks.dtype}")
    
    masks = torch.sigmoid(outputs.pred_masks)
    print(f"Sigmoid 後形狀: {masks.shape}")
    print(f"Sigmoid 後 min/max: {masks.min():.4f} / {masks.max():.4f}")

print("\n✅ 模型輸出檢查完成")


# %% [markdown]
# ## 5. OOD 檢測分支設計
# 
# 實作 OOD（Out-of-Distribution）偵測邏輯，利用特徵距離、信心分數或不確定性分數，將可疑樣本標記並分流處理。

# %%
class OODDetector:
    """Out-of-Distribution 檢測器"""
    
    def __init__(self, threshold: float = 0.5, method: str = "entropy"):
        self.threshold = threshold
        self.method = method
    
    def compute_entropy(self, mask_prob: np.ndarray) -> float:
        """計算遮罩概率的熵"""
        p = mask_prob.flatten()
        p = np.clip(p, 1e-10, 1 - 1e-10)
        entropy = -np.mean(p * np.log(p) + (1 - p) * np.log(1 - p))
        return entropy
    
    def compute_confidence(self, mask_prob: np.ndarray) -> float:
        """計算遮罩區域的平均信心"""
        return np.mean(np.abs(mask_prob - 0.5) * 2)
    
    def compute_variance(self, mask_prob: np.ndarray) -> float:
        """計算遮罩概率的方差"""
        return np.var(mask_prob.flatten())
    
    def detect(
        self,
        mask_logits: np.ndarray,
        mask_prob: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        檢測 OOD 樣本
        
        Args:
            mask_logits: 原始模型輸出（未應用 sigmoid）
            mask_prob: 概率化遮罩 [0, 1]
            
        Returns:
            包含 OOD 分數和預測的字典
        """
        if mask_prob is None:
            mask_prob = 1 / (1 + np.exp(-mask_logits))
        
        if self.method == "entropy":
            score = self.compute_entropy(mask_prob)
        elif self.method == "confidence":
            score = -self.compute_confidence(mask_prob)
        elif self.method == "variance":
            score = self.compute_variance(mask_prob)
        else:
            score = self.compute_entropy(mask_prob)
        
        is_ood = score > self.threshold
        
        return {
            "ood_score": score,
            "is_ood": is_ood,
            "confidence": 1 - score if score <= 1 else 0
        }


# 初始化 OOD 檢測器
ood_detector = OODDetector(
    threshold=EXPERIMENT_CONFIG["ood_threshold"],
    method=EXPERIMENT_CONFIG["ood_method"]
)

print(f"✅ OOD 檢測器初始化完成 (方法: {EXPERIMENT_CONFIG['ood_method']})")

# %% [markdown]
# ## 6. TTA 測試時增強分支設計
# 
# 對測試影像套用多種增強策略（旋轉、翻轉、噪聲），聚合多次推論結果，以提升對不同測試分佈的穩健性。

# %%
class TTAPredictor:
    """Test Time Augmentation 預測器 (優化版本)"""
    
    def __init__(self, num_augmentations: int = 4, fast_mode: bool = True):
        self.num_augmentations = num_augmentations
        self.fast_mode = fast_mode
        self.augmentations = self._create_augmentations()
    
    def _create_augmentations(self):
        """建立可逆幾何增強，確保聚合前可對齊回原座標。"""
        base = ["none", "hflip", "vflip", "hvflip"]
        if self.fast_mode:
            return base[:self.num_augmentations]
        # 非 fast 模式補充亮度/對比擾動（幾何不變，不需逆變換）
        extended = base + ["brightness", "contrast"]
        return extended[:self.num_augmentations]

    def _apply_aug(self, image_np: np.ndarray, aug_name: str) -> np.ndarray:
        if aug_name == "none":
            return image_np
        if aug_name == "hflip":
            return np.ascontiguousarray(np.flip(image_np, axis=1))
        if aug_name == "vflip":
            return np.ascontiguousarray(np.flip(image_np, axis=0))
        if aug_name == "hvflip":
            return np.ascontiguousarray(np.flip(np.flip(image_np, axis=0), axis=1))
        if aug_name == "brightness":
            out = image_np.astype(np.float32) * 1.1
            return np.clip(out, 0, 255).astype(np.uint8)
        if aug_name == "contrast":
            mean = image_np.mean(axis=(0, 1), keepdims=True)
            out = (image_np.astype(np.float32) - mean) * 1.1 + mean
            return np.clip(out, 0, 255).astype(np.uint8)
        raise ValueError(f"Unsupported augmentation: {aug_name}")

    def _deaugment_mask_np(self, mask_np: np.ndarray, aug_name: str) -> np.ndarray:
        if aug_name in ("none", "brightness", "contrast"):
            return mask_np
        if aug_name == "hflip":
            return np.ascontiguousarray(np.flip(mask_np, axis=1))
        if aug_name == "vflip":
            return np.ascontiguousarray(np.flip(mask_np, axis=0))
        if aug_name == "hvflip":
            return np.ascontiguousarray(np.flip(np.flip(mask_np, axis=0), axis=1))
        raise ValueError(f"Unsupported augmentation: {aug_name}")

    def _augment_bbox(self, input_box: List[int], aug_name: str, h: int, w: int) -> List[int]:
        x1, y1, x2, y2 = [int(v) for v in input_box]
        max_x = w - 1
        max_y = h - 1

        if aug_name in ("none", "brightness", "contrast"):
            return [x1, y1, x2, y2]
        if aug_name == "hflip":
            return [max_x - x2, y1, max_x - x1, y2]
        if aug_name == "vflip":
            return [x1, max_y - y2, x2, max_y - y1]
        if aug_name == "hvflip":
            return [max_x - x2, max_y - y2, max_x - x1, max_y - y1]
        raise ValueError(f"Unsupported augmentation: {aug_name}")
    
    def predict_tta(
        self,
        model: SamModel,
        processor: SamProcessor,
        image: Image.Image,
        input_box: List[int],
        device: str = "cpu",
        use_amp: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 TTA 進行預測 (批次優化版本)
        
        同一張圖像的多個增強會合併成一次 batch forward，
        大幅降低 processor/model 的重複呼叫開銷。
        """
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        
        aug_images = []
        aug_boxes = []
        for aug_name in self.augmentations:
            aug_image = self._apply_aug(image_np, aug_name)
            aug_images.append(Image.fromarray(aug_image))
            aug_boxes.append([self._augment_bbox(input_box, aug_name, h, w)])

        inputs = build_sam_inputs(
            processor=processor,
            images=aug_images,
            input_boxes=aug_boxes,
        )
        inputs = {k: v.to(device, non_blocking=(device == "cuda")) for k, v in inputs.items()}
        
        with torch.no_grad():
            masks = _run_sam_forward_prob_masks(model, inputs, device, use_amp)
            masks = _normalize_masks_to_4d(masks)
            masks = F.interpolate(
                masks,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )

        predictions = masks.detach().cpu().numpy()[:, 0, :, :]
        # 將增強後預測逆變換回原圖座標，再做平均
        predictions = np.stack(
            [self._deaugment_mask_np(pred, aug_name) for pred, aug_name in zip(predictions, self.augmentations)],
            axis=0,
        )
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)

        return mean_pred, uncertainty


# 初始化
tta_predictor = TTAPredictor(
    num_augmentations=EXPERIMENT_CONFIG["num_tta_augmentations"],
    fast_mode=EXPERIMENT_CONFIG.get("tta_fast_mode", True),
)
print(
    f"✅ TTA 預測器初始化完成 (批次優化版本, "
    f"fast_mode={EXPERIMENT_CONFIG.get('tta_fast_mode', True)})"
)

# %% [markdown]
# ## 7. 評估指標與測試函數
# 
# 定義分割評估指標計算函數，包括 Dice、Jaccard、精度、召回率等。

# %%
# ==================== 評估指標與優化評估函數 ====================
import time

# JSON 安全序列化：處理 numpy 標量/陣列
def json_default_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _single_item_collate(batch):
    return batch[0]


def build_eval_dataloader(dataset):
    """建立適合醫療影像評估的 DataLoader。"""
    return DataLoader(
        dataset,
        batch_size=EXPERIMENT_CONFIG["batch_size"],
        shuffle=False,
        num_workers=EXPERIMENT_CONFIG["num_workers"],
        pin_memory=EXPERIMENT_CONFIG["pin_memory"] and torch.cuda.is_available(),
        persistent_workers=(EXPERIMENT_CONFIG["persistent_workers"] and EXPERIMENT_CONFIG["num_workers"] > 0),
        prefetch_factor=EXPERIMENT_CONFIG["prefetch_factor"] if EXPERIMENT_CONFIG["num_workers"] > 0 else None,
        drop_last=False,
        collate_fn=_single_item_collate if EXPERIMENT_CONFIG["batch_size"] == 1 else None,
    )


# 向量化指標計算 (比循環快 10 倍)
def compute_metrics_batch(pred_masks: np.ndarray, gt_masks: np.ndarray) -> Dict[str, np.ndarray]:
    """
    批量計算分割評估指標 (向量化版本)
    """
    pred_flat = pred_masks.astype(bool).reshape(len(pred_masks), -1)
    gt_flat = gt_masks.astype(bool).reshape(len(gt_masks), -1)

    tp = (pred_flat & gt_flat).sum(axis=1)
    fp = (pred_flat & ~gt_flat).sum(axis=1)
    fn = (~pred_flat & gt_flat).sum(axis=1)

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    jaccard = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "dice": dice,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """單個樣本的指標計算（回傳 Python 原生型別）"""
    pred_flat = pred_mask.astype(np.bool_).flatten()
    gt_flat = gt_mask.astype(np.bool_).flatten()

    tp = int((pred_flat & gt_flat).sum())
    fp = int((pred_flat & ~gt_flat).sum())
    fn = int((~pred_flat & gt_flat).sum())

    dice = float(2 * tp / (2 * tp + fp + fn + 1e-8))
    jaccard = float(tp / (tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    return {
        "dice": dice,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def evaluate_dataset(
    dataset,
    model,
    processor,
    ood_detector,
    tta_predictor,
    device,
    use_tta: bool = False,
    use_ood: bool = False,
    dataset_name: str = "Unknown",
    use_amp: bool = True,
    enable_timing: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    評估數據集 (優化版本 + 內嵌式瓶頸分析)
    
    在執行過程中自動計時各個環節，評估結束後輸出詳細的瓶頸診斷報告。
    """
    results = []
    ood_scores_list = []
    uncertainties_list = []

    metrics_list = {
        "dice": [],
        "jaccard": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    start_time = time.time()
    inference_times = []
    
    # ===== 瓶頸分析計時 =====
    _prof_timings = {
        "forward": [],
        "ood_detect": [],
        "metrics": []
    }

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    data_loader = build_eval_dataloader(dataset)

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}")):
            image = sample["image"]
            gt_mask = sample["mask"].numpy()
            bbox = sample.get("bbox") or compute_bbox_from_mask_np(gt_mask)

            iter_start = time.time()

            # ① 推論 (predict_medsam or TTA)
            _t0 = time.perf_counter()
            if use_tta:
                pred_mask_prob, uncertainty = tta_predictor.predict_tta(
                    model, processor, image, bbox, device, use_amp=use_amp
                )
                pred_mask = (pred_mask_prob > 0.5).astype(np.uint8)
                uncertainties_list.append(float(uncertainty.mean()))
            else:
                pred_mask = predict_medsam(model, processor, image, bbox, device, use_amp=use_amp)
            
            if enable_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            _forward_time = time.perf_counter() - _t0
            _prof_timings["forward"].append(_forward_time)

            iter_time = time.time() - iter_start
            inference_times.append(float(iter_time))

            # ② OOD 檢測
            _t0 = time.perf_counter()
            ood_result = {}
            if use_ood:
                ood_result = ood_detector.detect(pred_mask.astype(np.float32))
                # 確保 OOD 欄位可序列化
                ood_result = {
                    "ood_score": float(ood_result["ood_score"]),
                    "is_ood": bool(ood_result["is_ood"]),
                    "confidence": float(ood_result["confidence"])
                }
                ood_scores_list.append(float(ood_result["ood_score"]))
            _prof_timings["ood_detect"].append(time.perf_counter() - _t0)

            # ③ 指標計算
            _t0 = time.perf_counter()
            metrics = compute_metrics(pred_mask, gt_mask)
            _prof_timings["metrics"].append(time.perf_counter() - _t0)

            for key in metrics_list:
                metrics_list[key].append(float(metrics[key]))

            result = {
                "index": int(idx),
                "name": get_sample_identifier(sample, idx),
                **metrics,
                **ood_result
            }
            results.append(result)

    total_time = float(time.time() - start_time)

    stats = {
        "num_samples": int(len(results)),
        "mean_dice": float(np.mean(metrics_list["dice"])),
        "std_dice": float(np.std(metrics_list["dice"])),
        "mean_jaccard": float(np.mean(metrics_list["jaccard"])),
        "std_jaccard": float(np.std(metrics_list["jaccard"])),
        "mean_f1": float(np.mean(metrics_list["f1"])),
        "std_f1": float(np.std(metrics_list["f1"])),
        "mean_precision": float(np.mean(metrics_list["precision"])),
        "std_precision": float(np.std(metrics_list["precision"])),
        "mean_recall": float(np.mean(metrics_list["recall"])),
        "std_recall": float(np.std(metrics_list["recall"])),
        "total_time_sec": total_time,
        "avg_inference_time_ms": float(np.mean(inference_times) * 1000),
        "throughput_samples_per_sec": float(len(results) / total_time if total_time > 0 else 0.0)
    }

    if use_ood and ood_scores_list:
        stats["mean_ood_score"] = float(np.mean(ood_scores_list))
        stats["std_ood_score"] = float(np.std(ood_scores_list))
        stats["num_ood_detected"] = int(sum(1 for r in results if r.get("is_ood", False)))
        stats["ood_ratio"] = float(stats["num_ood_detected"] / len(results))

    if use_tta and uncertainties_list:
        stats["mean_uncertainty"] = float(np.mean(uncertainties_list))
        stats["std_uncertainty"] = float(np.std(uncertainties_list))

    if device == "cuda":
        stats["peak_gpu_memory_gb"] = float(torch.cuda.max_memory_allocated() / 1e9)
        stats["current_gpu_memory_gb"] = float(torch.cuda.memory_allocated() / 1e9)

    # ===== 輸出瓶頸分析報告 =====
    if enable_timing and _prof_timings["forward"]:
        print("\n" + "=" * 80)
        print(f"  🔍 瓶頸分析報告 [{dataset_name}]")
        print("=" * 80)
        
        _fwd_ms = np.mean(_prof_timings["forward"]) * 1000
        _ood_ms = np.mean(_prof_timings["ood_detect"]) * 1000
        _met_ms = np.mean(_prof_timings["metrics"]) * 1000
        _total_measure = _fwd_ms + _ood_ms + _met_ms
        
        _rows = [
            ("推論 (predict_medsam/TTA)", _fwd_ms),
            ("OOD 檢測", _ood_ms),
            ("指標計算", _met_ms),
        ]
        
        print(f"\n{'環節':<28}  {'ms/sample':>10}  {'占比':>8}  比例圖")
        print("-" * 75)
        for _name, _v in _rows:
            _pct = _v / (_total_measure + 1e-9) * 100
            _bar = "[" + "█" * int(_pct / 5) + "░" * (20 - int(_pct / 5)) + "]"
            print(f"{_name:<28}  {_v:>10.2f}  {_pct:>7.1f}%  {_bar}")
        
        _bottleneck = max(_rows, key=lambda x: x[1])
        print("-" * 75)
        print(f"{'合計':<28}  {_total_measure:>10.2f}  {'100.0%':>8}")
        print(f"\n🏆 主要瓶頸：【{_bottleneck[0]}】 {_bottleneck[1]:.2f} ms ({_bottleneck[1]/_total_measure*100:.1f}%)")
        print(f"   吞吐量：{len(results) / total_time:.1f} samples/sec")
        print("=" * 80 + "\n")

    return results, stats


print("✅ 優化評估函數已定義")

# %% [markdown]
# ## 8. TN3K 測試與指標評估
# 
# 在 TN3K 測試集上執行基線、OOD 與 TTA 推論，並計算各項分割指標。

# %%

# ============================================================================
# 🚀 瓶頸優化工具 - 預先計算 processor 輸出快取
# ============================================================================
#
# 優化策略：
# 1. 在 __getitem__ 前預先快取 processor 輸出（.pt 格式）
#    預期減少：~35-40 ms per sample（processor 耗時從 50ms 降至 5ms）
# 2. 支持增量更新（只處理未快取的樣本）
# 3. 與現有 DataLoader 無縫集成
#
# ============================================================================

import os
from pathlib import Path
from functools import lru_cache

print("\n" + "="*90)
print("  🚀 初始化 Processor 輸出快取系統")
print("="*90)

# 快取目錄配置
CACHE_CONFIG = {
    "enabled": True,  # 是否啟用快取
    "cache_dir": Path("./processor_cache"),  # 快取存儲位置
    "overwrite": False,  # 是否覆寫現有快取
    "compress": False,  # 是否壓縮快取文件（節省空間）
}

# 中期優化策略：先 warmup，再於評估前補齊快取覆蓋率
CACHE_POLICY = {
    "warmup_samples": 50,
    "auto_prepare_before_eval": True,
    "target_coverage": 1.0,  # 1.0 = 評估前補齊全量快取
}

CACHE_CONFIG["cache_dir"].mkdir(exist_ok=True)
_CACHE_PREPARED_DATASETS = set()

# ─────────────────────────────────────────────────────────────────────────
# 快取輔助函數
# ─────────────────────────────────────────────────────────────────────────

def _get_cache_path(dataset_name: str, sample_name: str) -> Path:
    """獲取快取文件路徑"""
    return CACHE_CONFIG["cache_dir"] / f"{dataset_name}_{sample_name}_processor.pt"

def _cache_processor_output(dataset_name: str, sample_name: str, processor_dict: dict):
    """快取 processor 輸出"""
    if not CACHE_CONFIG["enabled"]:
        return
    
    cache_path = _get_cache_path(dataset_name, sample_name)
    if cache_path.exists() and not CACHE_CONFIG["overwrite"]:
        return
    
    try:
        torch.save(processor_dict, cache_path)
    except Exception as e:
        print(f"  ⚠️  快取失敗 [{sample_name}]: {e}")

def _load_cached_processor_output(dataset_name: str, sample_name: str) -> dict:
    """加載快取的 processor 輸出"""
    if not CACHE_CONFIG["enabled"]:
        return None
    
    cache_path = _get_cache_path(dataset_name, sample_name)
    if cache_path.exists():
        try:
            cached = torch.load(cache_path, weights_only=False)
            # 修正舊快取可能保存 float64 的 input_boxes
            if isinstance(cached, dict) and "input_boxes" in cached:
                if isinstance(cached["input_boxes"], torch.Tensor) and cached["input_boxes"].dtype == torch.float64:
                    cached["input_boxes"] = cached["input_boxes"].to(torch.float32)
            return cached
        except:
            return None
    return None

# ─────────────────────────────────────────────────────────────────────────
# 預処理工具：為所有數據集批量生成快取
# ─────────────────────────────────────────────────────────────────────────

def precompute_processor_cache(dataset, dataset_name: str, num_samples: int = None):
    """
    為數據集預先計算所有 processor 輸出並快取
    
    Args:
        dataset: Dataset 實例
        dataset_name: 數據集名稱（用於快取標識）
        num_samples: 限制處理樣本數（None = 全部）
    """
    if not CACHE_CONFIG["enabled"]:
        print(f"  ⊘ 快取已禁用，跳過預計算")
        return
    
    total = min(num_samples, len(dataset)) if num_samples else len(dataset)
    cached = 0
    skipped = 0
    
    print(f"\n  📦 預計算 {dataset_name} processor 輸出快取...", end=" ", flush=True)
    _t0 = time.perf_counter()
    
    for i in tqdm(range(total), desc=f"Cache {dataset_name}", unit="sample", leave=False):
        sample = dataset[i]
        sample_name = get_sample_identifier(sample, i)
        
        cache_path = _get_cache_path(dataset_name, str(sample_name))
        if cache_path.exists() and not CACHE_CONFIG["overwrite"]:
            skipped += 1
            continue
        
        # 執行 processor
        image = sample["image"]
        bbox = sample["bbox"]
        try:
            processor_output = build_sam_inputs(
                processor=processor,
                images=[image],
                input_boxes=[[bbox]],
            )
            _cache_processor_output(dataset_name, str(sample_name), processor_output)
            cached += 1
        except Exception as e:
            print(f"\n    ⚠️  {sample_name}: {e}")
    
    elapsed = time.perf_counter() - _t0
    print(f"✅ 完成 ({cached} 新增, {skipped} 已存在, {elapsed:.1f}s)")
    return cached, skipped


def _count_cached_samples(dataset, dataset_name: str) -> int:
    cached = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_name = get_sample_identifier(sample, i)
        if _get_cache_path(dataset_name, str(sample_name)).exists():
            cached += 1
    return cached


def ensure_cache_coverage(dataset, dataset_name: str, target_coverage: float = 1.0):
    """確保快取覆蓋率達標；預設補齊到 100%。"""
    if not CACHE_CONFIG["enabled"] or len(dataset) == 0:
        return

    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
    total = len(dataset)
    current_cached = _count_cached_samples(dataset, dataset_name)
    target_cached = int(np.ceil(total * target_coverage))

    if current_cached >= target_cached:
        print(f"  ✅ {dataset_name} 快取覆蓋率已達標: {current_cached}/{total} ({current_cached/total:.1%})")
        return

    need = target_cached - current_cached
    print(f"  🔄 {dataset_name} 快取補齊中: 目前 {current_cached}/{total}，目標 {target_cached}/{total}")
    precompute_processor_cache(dataset, dataset_name, num_samples=target_cached)


def _canonical_dataset_name(dataset_name: str) -> str:
    return dataset_name.split("-")[0].strip().upper()


def prepare_dataset_cache_for_eval(dataset, dataset_name: str):
    """評估前一次性補齊快取，避免大量 processor miss。"""
    canonical = _canonical_dataset_name(dataset_name)
    if canonical in _CACHE_PREPARED_DATASETS:
        return
    if CACHE_POLICY.get("auto_prepare_before_eval", False):
        ensure_cache_coverage(
            dataset,
            canonical,
            target_coverage=CACHE_POLICY.get("target_coverage", 1.0),
        )
    _CACHE_PREPARED_DATASETS.add(canonical)

# ─────────────────────────────────────────────────────────────────────────
# 增強 Dataset 以支持快取讀取
# ─────────────────────────────────────────────────────────────────────────

class CachedDatasetMixin:
    """
    可混合至任何 Dataset，添加快取支持
    
    使用方式：
        class MyDataset(CachedDatasetMixin, Dataset):
            def __init__(self, ...):
                self.dataset_name = "my_dataset"
                super().__init__(...)
    """
    
    def _get_sample_name(self, idx: int) -> str:
        """返回樣本唯一名稱（子類應覆寫）"""
        return f"sample_{idx}"
    
    def _get_processor_from_cache_or_compute(self, image, bbox, sample_name: str) -> dict:
        """
        嘗試從快取讀取 processor 輸出，否則計算並快取
        """
        # 嘗試快取讀取
        cached = _load_cached_processor_output(self.dataset_name, str(sample_name))
        if cached is not None:
            return cached
        
        # 計算 processor
        processor_output = build_sam_inputs(
            processor=processor,
            images=[image],
            input_boxes=[[bbox]],
        )
        
        # 快取保存
        _cache_processor_output(self.dataset_name, str(sample_name), processor_output)
        
        return processor_output

# ─────────────────────────────────────────────────────────────────────────
# 執行快取預計算
# ─────────────────────────────────────────────────────────────────────────

try:
    # 若已加載數據集，進行快取預計算
    _warmup_n = CACHE_POLICY.get("warmup_samples", 50)
    if 'tn3k_dataset' in globals():
        precompute_processor_cache(tn3k_dataset, "TN3K", num_samples=_warmup_n)
    if 'ddti_dataset' in globals():
        precompute_processor_cache(ddti_dataset, "DDTI", num_samples=_warmup_n)
    if 'tn5000_dataset' in globals():
        precompute_processor_cache(tn5000_dataset, "TN5000", num_samples=_warmup_n)
except NameError:
    print("  ⊘ 尚未加載數據集，跳過預計算（在加載數據後重新執行此 cell）")

print("\n" + "="*90)
print(f"  快取配置：enabled={CACHE_CONFIG['enabled']}, dir={CACHE_CONFIG['cache_dir']}")
print(
    "  快取策略："
    f"warmup_samples={CACHE_POLICY['warmup_samples']}, "
    f"auto_prepare_before_eval={CACHE_POLICY['auto_prepare_before_eval']}, "
    f"target_coverage={CACHE_POLICY['target_coverage']}"
)
print("="*90)


# %%

# ============================================================================
# ⚡ 快取感知推論優化
# ============================================================================
# 
# 改進點：
# 1. predict_medsam_cached() - 智能使用快取的 processor 輸出
# 2. 在評估時自動檢查並使用已快取的 processor 輸出
# 3. 在 evaluate_dataset 中集成快取
# ============================================================================

print("\n" + "="*90)
print("  ⚡ 啟用快取感知推論優化")
print("="*90)

# 快取池（運行時快取，避免重複讀取同一樣本的快取文件）
_PROCESSOR_CACHE_POOL = {}
_PRED_MASK_CACHE_POOL = OrderedDict()
_PRED_MASK_CACHE_MAX = int(env_get("MEDSAM_PRED_CACHE_MAX"))


def _pred_cache_make_key(dataset_name: str, sample_name: str, input_box: List[int]) -> str:
    b = tuple(int(x) for x in input_box) if input_box is not None else ()
    return f"{dataset_name}|{sample_name}|{b}"


def _pred_cache_get(key: str) -> Optional[np.ndarray]:
    cached = _PRED_MASK_CACHE_POOL.get(key)
    if cached is None:
        return None
    _PRED_MASK_CACHE_POOL.move_to_end(key)
    return cached


def _pred_cache_put(key: str, pred_mask: np.ndarray) -> None:
    _PRED_MASK_CACHE_POOL[key] = pred_mask
    _PRED_MASK_CACHE_POOL.move_to_end(key)
    while len(_PRED_MASK_CACHE_POOL) > max(1, _PRED_MASK_CACHE_MAX):
        _PRED_MASK_CACHE_POOL.popitem(last=False)

def _load_processor_cache(dataset_name: str, sample_name: str) -> dict:
    """
    智能快取加載（支持運行時池）
    """
    cache_key = f"{dataset_name}_{sample_name}"
    
    # 檢查運行時池
    if cache_key in _PROCESSOR_CACHE_POOL:
        return _PROCESSOR_CACHE_POOL[cache_key]
    
    # 嘗試磁碟快取
    if CACHE_CONFIG["enabled"]:
        cached = _load_cached_processor_output(dataset_name, sample_name)
        if cached is not None:
            _PROCESSOR_CACHE_POOL[cache_key] = cached
            return cached
    
    return None

def predict_medsam_cached(
    model,
    processor,
    image,
    input_box,
    device="cuda",
    use_amp=True,
    dataset_name: str = None,
    sample_name: str = None,
    prefetched_inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> np.ndarray:
    """
    快取感知推論函數
    
    若快取存在，直接使用快取的 processor 輸出；否則計算並可選快取。
    預期減少 processor() 耗時 90%（從 50ms → 5ms）
    """
    w, h = image.size

    # 0) 先查推論結果快取：同 dataset+sample+bbox 可直接返回，
    #    可避免 Baseline/OOD 重複 forward。
    pred_cache_key = None
    if dataset_name and sample_name:
        pred_cache_key = _pred_cache_make_key(str(dataset_name), str(sample_name), input_box)
        pred_cached = _pred_cache_get(pred_cache_key)
        if pred_cached is not None:
            return pred_cached
    
    # ① 嘗試快取讀取
    inputs = prefetched_inputs
    if inputs is None and dataset_name and sample_name:
        inputs = _load_processor_cache(dataset_name, sample_name)
    
    # ② 若無快取，計算 processor
    if inputs is None:
        inputs = build_sam_inputs(
            processor=processor,
            images=[image],
            input_boxes=[[input_box]],
        )
        # 快取保存（若配置啟用）
        if CACHE_CONFIG["enabled"] and dataset_name and sample_name:
            _cache_processor_output(dataset_name, str(sample_name), inputs)
    
    pred_mask = _predict_from_processor_inputs(
        model=model,
        inputs=inputs,
        image_hw=(h, w),
        device=device,
        use_amp=use_amp,
    )

    if pred_cache_key is not None:
        _pred_cache_put(pred_cache_key, pred_mask)

    return pred_mask

# ─────────────────────────────────────────────────────────────────────────
# 替換現有評估函數中的 predict_medsam 調用
# ─────────────────────────────────────────────────────────────────────────

print("  ✅ 快取感知推論函數已啟用")
print(f"  ✅ 快取池大小：{len(_PROCESSOR_CACHE_POOL)}")
print(f"  ✅ 推論結果快取上限：{_PRED_MASK_CACHE_MAX}")
print("="*90 + "\n")


# %%

# ============================================================================
# 🎯 優化評估管道 - 集成快取和性能測量
# ============================================================================
#
# 與原有 evaluate_dataset 相比的改進：
# 1. 使用 predict_medsam_cached 替代 predict_medsam（智能快取）
# 2. 自動追蹤 processor 快取命中率
# 3. 詳細的分層性能分析（快取 vs 計算時間）
# ============================================================================

print("\n" + "="*90)
print("  🎯 初始化優化評估管道")
print("="*90)

def evaluate_dataset_optimized(
    model,
    processor,
    dataset,
    dataset_name: str = "unnamed",
    device: str = "cuda",
    use_ood: bool = True,
    use_tta: bool = True,
    ood_detector=None,
    tta_predictor=None,
    enable_timing: bool = True,
):
    """
    優化評估函數 - 使用快取感知推論
    
    與 evaluate_dataset 相比：
    - 自動使用快取 processor 輸出（若可用）
    - 追蹤快取命中率和性能改進
    - 詳細的快取 vs 計算時間分析
    """
    device = torch.device(device)
    cache_dataset_name = _canonical_dataset_name(dataset_name)
    prepare_dataset_cache_for_eval(dataset, dataset_name)

    eval_microbatch = max(1, int(env_get("MEDSAM_EVAL_MICROBATCH")))
    eval_microbatch_min = max(1, int(env_get("MEDSAM_EVAL_MICROBATCH_MIN")))
    eval_microbatch_max = max(eval_microbatch, int(env_get("MEDSAM_EVAL_MICROBATCH_MAX")))
    autotune_microbatch = env_get_bool("MEDSAM_AUTOTUNE_MICROBATCH", True)
    target_vram_util = float(np.clip(float(env_get("MEDSAM_TARGET_VRAM_UTIL")), 0.30, 0.99))
    vram_limit_gb = max(1.0, float(env_get("MEDSAM_VRAM_LIMIT_GB")))
    vram_limit_bytes = vram_limit_gb * 1e9
    gpu_total_bytes = 0.0
    effective_vram_bytes = 0.0
    if device.type == "cuda":
        gpu_total_bytes = float(torch.cuda.get_device_properties(device).total_memory)
        effective_vram_bytes = min(gpu_total_bytes, vram_limit_bytes)

    results = []
    metrics_list = {
        "dice": [], "jaccard": [], "f1": [],
        "precision": [], "recall": []
    }
    ood_scores_list = []
    uncertainties_list = []
    inference_times = []
    
    # ===== 新增：快取性能指標 =====
    cache_timings = {
        "processor_cache_hit": [],
        "processor_cache_miss": [],
        "processor_total": [],
    }
    cache_stats = {
        "hits": 0,
        "misses": 0,
    }
    
    # ===== 分層計時 =====
    _prof_timings = {
        "forward": [],
        "ood_detect": [],
        "metrics": [],
    }
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _safe_clear_cuda_memory() -> None:
        if device.type != "cuda":
            return
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    def _gpu_peak_gb() -> float:
        if device.type != "cuda":
            return 0.0
        peak = float(torch.cuda.max_memory_allocated(device))
        return peak / 1e9

    def _gpu_peak_ratio() -> float:
        if device.type != "cuda":
            return 0.0
        total = float(effective_vram_bytes if effective_vram_bytes > 0 else torch.cuda.get_device_properties(device).total_memory)
        peak = float(torch.cuda.max_memory_allocated(device))
        if total <= 0:
            return 0.0
        return peak / total

    def _autotune_eval_microbatch() -> None:
        nonlocal eval_microbatch
        if device.type != "cuda" or use_tta:
            return
        if len(dataset) == 0:
            return
        if not autotune_microbatch:
            return

        probe_cap = min(len(dataset), max(eval_microbatch_max, eval_microbatch))
        probe_images: List[Any] = []
        probe_boxes: List[List[List[int]]] = []
        probe_hw: Optional[Tuple[int, int]] = None
        for i in range(probe_cap):
            s = dataset[i]
            b = s.get("bbox")
            if b is None:
                continue
            img = s["image"]
            probe_images.append(img)
            probe_boxes.append([b])
            probe_hw = (img.size[1], img.size[0])

        if not probe_images or probe_hw is None:
            return

        candidate = max(eval_microbatch_min, min(eval_microbatch, eval_microbatch_max, len(probe_images)))
        best = candidate
        target_gb = vram_limit_gb * target_vram_util
        print(
            f"🔧 VRAM 自動微批次校準: target={target_vram_util*100:.0f}% of {vram_limit_gb:.1f}GB "
            f"(≈{target_gb:.2f}GB, 初始={candidate}, 上限={eval_microbatch_max})"
        )

        while candidate >= eval_microbatch_min:
            try:
                _safe_clear_cuda_memory()

                inputs = build_sam_inputs(
                    processor=processor,
                    images=probe_images[:candidate],
                    input_boxes=probe_boxes[:candidate],
                )
                _ = _predict_from_processor_inputs_batch(
                    model=model,
                    inputs=inputs,
                    image_hw=probe_hw,
                    device=str(device),
                    use_amp=True,
                )
                torch.cuda.synchronize(device)

                util = _gpu_peak_ratio()
                peak_gb = _gpu_peak_gb()
                # 以使用者設定的 VRAM 上限為硬邊界，超過即退縮
                if util > 1.0 or peak_gb >= vram_limit_gb * 0.995:
                    _safe_clear_cuda_memory()
                    next_candidate = max(eval_microbatch_min, candidate // 2)
                    if next_candidate == candidate:
                        break
                    candidate = next_candidate
                    continue
                best = candidate
                if util >= target_vram_util * 0.98:
                    break

                if candidate >= min(eval_microbatch_max, len(probe_images)):
                    break

                next_candidate = min(candidate * 2, eval_microbatch_max, len(probe_images))
                if next_candidate == candidate:
                    break
                candidate = next_candidate
            except Exception as e:
                err_msg = str(e).lower()
                if "out of memory" not in err_msg and "cuda error" not in err_msg:
                    raise
                _safe_clear_cuda_memory()
                next_candidate = max(eval_microbatch_min, candidate // 2)
                if next_candidate == candidate:
                    break
                candidate = next_candidate

        eval_microbatch = max(eval_microbatch_min, min(best, eval_microbatch_max))
        print(
            f"✅ VRAM 校準完成: eval_microbatch={eval_microbatch}, "
            f"peak≈{_gpu_peak_ratio()*100:.1f}% of {vram_limit_gb:.1f}GB (≈{_gpu_peak_gb():.2f}GB)"
        )
    
    print(f"\n📊 評估 {dataset_name} (優化版本，快取={'啟用' if CACHE_CONFIG['enabled'] else '禁用'})...")
    print("-" * 90)
    
    total_start = time.perf_counter()
    
    # 預熱：使用「實際 microbatch 大小」進行一次 batch forward，
    # 避免 baseline 首段被首次編譯/自動調參成本污染。
    if len(dataset) > 0:
        warm_images = []
        warm_boxes = []
        warm_hw = None
        warm_n = min(eval_microbatch, len(dataset))
        for wi in range(warm_n):
            warm_sample = dataset[wi]
            warm_image = warm_sample["image"]
            warm_bbox = warm_sample.get("bbox")
            if warm_bbox is None:
                continue
            warm_images.append(warm_image)
            warm_boxes.append([warm_bbox])
            warm_hw = (warm_image.size[1], warm_image.size[0])

        if warm_images and warm_hw is not None:
            warm_inputs = build_sam_inputs(
                processor=processor,
                images=warm_images,
                input_boxes=warm_boxes,
            )
            _ = _predict_from_processor_inputs_batch(
                model=model,
                inputs=warm_inputs,
                image_hw=warm_hw,
                device=str(device),
                use_amp=True,
            )

    _autotune_eval_microbatch()
    
    data_loader = build_eval_dataloader(dataset)
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=dataset_name, unit="sample")

    pending = []

    def _flush_pending() -> None:
        nonlocal pending
        if not pending:
            return

        batch_size_now = max(1, len(pending))

        if use_tta and tta_predictor is not None:
            for item in pending:
                t_fwd_start = time.perf_counter()
                pred_mask_prob, uncertainty = tta_predictor.predict_tta(
                    model,
                    processor,
                    item["image"],
                    item["bbox"],
                    device=str(device),
                    use_amp=True,
                )
                t_fwd_end = time.perf_counter()

                item["pred_mask"] = (pred_mask_prob > 0.5).astype(np.uint8)
                item["uncertainty_score"] = float(np.mean(uncertainty))
                uncertainties_list.append(item["uncertainty_score"])
                _prof_timings["forward"].append(t_fwd_end - t_fwd_start)
                inference_times.append(t_fwd_end - t_fwd_start)

            for item in pending:
                pred_mask = item["pred_mask"]
                gt_mask_np = item["gt_mask_np"]

                if use_ood and ood_detector is not None:
                    t_ood_start = time.perf_counter()
                    ood_result = ood_detector.detect(pred_mask.astype(np.float32))
                    ood_score = float(ood_result["ood_score"])
                    is_ood = bool(ood_result["is_ood"])
                    t_ood_end = time.perf_counter()
                    _prof_timings["ood_detect"].append(t_ood_end - t_ood_start)
                    ood_scores_list.append(ood_score)
                else:
                    is_ood = False
                    ood_score = 0.0

                t_met_start = time.perf_counter()
                metrics = compute_metrics(pred_mask, gt_mask_np)
                t_met_end = time.perf_counter()
                _prof_timings["metrics"].append(t_met_end - t_met_start)

                for metric_name, value in metrics.items():
                    if metric_name in metrics_list:
                        metrics_list[metric_name].append(value)

                results.append({
                    "sample_name": str(item["sample_name"]),
                    "dice": float(metrics.get("dice", 0)),
                    "jaccard": float(metrics.get("jaccard", 0)),
                    "f1": float(metrics.get("f1", 0)),
                    "precision": float(metrics.get("precision", 0)),
                    "recall": float(metrics.get("recall", 0)),
                    "ood_score": float(ood_score),
                    "is_ood": is_ood,
                    "uncertainty": float(item.get("uncertainty_score", 0.0)),
                })

            pending = []
            return

        t_fwd_start = time.perf_counter()

        to_compute = []
        batch_images = []
        batch_boxes = []

        for item in pending:
            pred_cache_key = item.get("pred_cache_key")
            pred_cached = _pred_cache_get(pred_cache_key) if pred_cache_key is not None else None
            if pred_cached is not None:
                item["pred_mask"] = pred_cached
                cache_stats["hits"] += 1
                continue

            to_compute.append(item)
            batch_images.append(item["image"])
            batch_boxes.append([item["bbox"]])
            cache_stats["misses"] += 1

        if to_compute:
            # 先嘗試命中磁碟 processor 快取；只對 miss 的樣本重算 preprocess。
            item_inputs: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(to_compute)
            miss_positions: List[int] = []
            miss_images: List[Any] = []
            miss_boxes: List[List[List[int]]] = []

            for pos, item in enumerate(to_compute):
                t_cache_start = time.perf_counter()
                cached_inputs = _load_processor_cache(cache_dataset_name, str(item["sample_name"]))
                t_cache_end = time.perf_counter()

                if cached_inputs is not None:
                    item_inputs[pos] = cached_inputs
                    cache_timings["processor_cache_hit"].append(t_cache_end - t_cache_start)
                    continue

                miss_positions.append(pos)
                miss_images.append(item["image"])
                miss_boxes.append([item["bbox"]])

            if miss_images:
                t_proc_start = time.perf_counter()
                miss_inputs = build_sam_inputs(
                    processor=processor,
                    images=miss_images,
                    input_boxes=miss_boxes,
                )
                t_proc_end = time.perf_counter()
                cache_timings["processor_cache_miss"].append((t_proc_end - t_proc_start) / max(1, len(miss_images)))

                for miss_idx, pos in enumerate(miss_positions):
                    single_inputs = {
                        k: v[miss_idx:miss_idx + 1]
                        for k, v in miss_inputs.items()
                    }
                    item_inputs[pos] = single_inputs
                    _cache_processor_output(cache_dataset_name, str(to_compute[pos]["sample_name"]), single_inputs)

            # 依原順序合併成一次 batch forward
            merged_inputs: Dict[str, torch.Tensor] = {}
            base_keys = list(item_inputs[0].keys()) if item_inputs and item_inputs[0] is not None else []
            for key in base_keys:
                merged_inputs[key] = torch.cat([inp[key] for inp in item_inputs if inp is not None], dim=0)

            h0, w0 = to_compute[0]["image_hw"]
            batch_pred_masks = _predict_from_processor_inputs_batch(
                model=model,
                inputs=merged_inputs,
                image_hw=(h0, w0),
                device=str(device),
                use_amp=True,
            )

            for item, pred_mask in zip(to_compute, batch_pred_masks):
                item["pred_mask"] = pred_mask
                if item.get("pred_cache_key") is not None:
                    _pred_cache_put(item["pred_cache_key"], pred_mask)

        t_fwd_end = time.perf_counter()
        per_sample_forward = (t_fwd_end - t_fwd_start) / batch_size_now
        _prof_timings["forward"].append(per_sample_forward)
        cache_timings["processor_total"].append(per_sample_forward)
        inference_times.append(per_sample_forward)

        for item in pending:
            pred_mask = item["pred_mask"]
            image = item["image"]
            bbox = item["bbox"]
            gt_mask_np = item["gt_mask_np"]
            sample_name = item["sample_name"]

            if use_ood and ood_detector is not None:
                t_ood_start = time.perf_counter()
                ood_result = ood_detector.detect(pred_mask.astype(np.float32))
                ood_score = float(ood_result["ood_score"])
                is_ood = bool(ood_result["is_ood"])
                t_ood_end = time.perf_counter()
                _prof_timings["ood_detect"].append(t_ood_end - t_ood_start)
                ood_scores_list.append(ood_score)
            else:
                is_ood = False
                ood_score = 0.0

            t_met_start = time.perf_counter()
            metrics = compute_metrics(pred_mask, gt_mask_np)
            t_met_end = time.perf_counter()
            _prof_timings["metrics"].append(t_met_end - t_met_start)

            for metric_name, value in metrics.items():
                if metric_name in metrics_list:
                    metrics_list[metric_name].append(value)

            uncertainty_score = float(item.get("uncertainty_score", 0.0))

            results.append({
                "sample_name": str(sample_name),
                "dice": float(metrics.get("dice", 0)),
                "jaccard": float(metrics.get("jaccard", 0)),
                "f1": float(metrics.get("f1", 0)),
                "precision": float(metrics.get("precision", 0)),
                "recall": float(metrics.get("recall", 0)),
                "ood_score": float(ood_score),
                "is_ood": is_ood,
                "uncertainty": uncertainty_score,
            })

        pending = []

    for idx, sample in pbar:
        image = sample["image"]
        gt_mask = sample["mask"]
        bbox = sample.get("bbox")
        sample_name = get_sample_identifier(sample, idx)

        if gt_mask.dim() > 2:
            gt_mask = gt_mask.squeeze()
        gt_mask_np = gt_mask.numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask

        pred_cache_key = _pred_cache_make_key(cache_dataset_name, str(sample_name), bbox)
        pending.append(
            {
                "image": image,
                "bbox": bbox,
                "gt_mask_np": gt_mask_np,
                "sample_name": str(sample_name),
                "pred_cache_key": pred_cache_key,
                "image_hw": (image.size[1], image.size[0]),
            }
        )

        if len(pending) >= eval_microbatch:
            _flush_pending()

    _flush_pending()
    
    pbar.close()
    total_time = time.perf_counter() - total_start
    
    # ===== 統計計算 =====
    stats = {
        "dataset": dataset_name,
        "num_samples": len(results),
        "total_time_sec": float(total_time),
        "mean_dice": float(np.mean(metrics_list["dice"])),
        "std_dice": float(np.std(metrics_list["dice"])),
        "mean_jaccard": float(np.mean(metrics_list["jaccard"])),
        "std_jaccard": float(np.std(metrics_list["jaccard"])),
        "mean_f1": float(np.mean(metrics_list["f1"])),
        "std_f1": float(np.std(metrics_list["f1"])),
        "mean_precision": float(np.mean(metrics_list["precision"])),
        "std_precision": float(np.std(metrics_list["precision"])),
        "mean_recall": float(np.mean(metrics_list["recall"])),
        "std_recall": float(np.std(metrics_list["recall"])),
        "avg_inference_time_ms": float(np.mean(inference_times) * 1000),
        "throughput_samples_per_sec": float(len(results) / total_time if total_time > 0 else 0.0),
        # 新增：快取性能指標
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "cache_hit_rate": float(cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"])) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0.0,
    }
    
    if cache_timings["processor_cache_hit"]:
        stats["avg_cache_hit_time_ms"] = float(np.mean(cache_timings["processor_cache_hit"]) * 1000)
    if cache_timings["processor_cache_miss"]:
        stats["avg_cache_miss_time_ms"] = float(np.mean(cache_timings["processor_cache_miss"]) * 1000)
    
    if use_ood and ood_scores_list:
        stats["mean_ood_score"] = float(np.mean(ood_scores_list))
        stats["std_ood_score"] = float(np.std(ood_scores_list))
        stats["num_ood_detected"] = int(sum(1 for r in results if r.get("is_ood", False)))
        stats["ood_ratio"] = float(stats["num_ood_detected"] / len(results))
    
    if use_tta and uncertainties_list:
        stats["mean_uncertainty"] = float(np.mean(uncertainties_list))
        stats["std_uncertainty"] = float(np.std(uncertainties_list))
    
    if device.type == "cuda":
        stats["peak_gpu_memory_gb"] = float(torch.cuda.max_memory_allocated(device) / 1e9)
        stats["current_gpu_memory_gb"] = float(torch.cuda.memory_allocated(device) / 1e9)
    
    # ===== 瓶頸分析報告（包括快取效果）=====
    if enable_timing and _prof_timings["forward"]:
        print("\n" + "="*80)
        print(f"  🔍 瓶頸分析報告 [{dataset_name}] - 快取優化版本")
        print("="*80)

        def _safe_mean_ms(values):
            if not values:
                return 0.0
            mean_value = float(np.mean(values))
            if not np.isfinite(mean_value):
                return 0.0
            return mean_value * 1000
        
        _fwd_ms = _safe_mean_ms(_prof_timings["forward"])
        _ood_ms = _safe_mean_ms(_prof_timings["ood_detect"])
        _met_ms = _safe_mean_ms(_prof_timings["metrics"])
        _total_measure = _fwd_ms + _ood_ms + _met_ms
        
        # 快取性能細分
        if cache_timings["processor_cache_hit"]:
            _cache_hit_ms = np.mean(cache_timings["processor_cache_hit"]) * 1000
        else:
            _cache_hit_ms = 0.0
        
        if cache_timings["processor_cache_miss"]:
            _cache_miss_ms = np.mean(cache_timings["processor_cache_miss"]) * 1000
        else:
            _cache_miss_ms = 0.0
        
        _rows = [
            ("推論 (predict_medsam_cached/TTA)", _fwd_ms),
            ("OOD 檢測", _ood_ms),
            ("指標計算", _met_ms),
        ]
        
        print(f"\n{'環節':<35}  {'ms/sample':>12}  {'占比':>8}  比例圖")
        print("-" * 80)
        for _name, _v in _rows:
            _pct = (_v / (_total_measure + 1e-9) * 100) if _total_measure > 0 else 0.0
            _pct = float(np.clip(_pct, 0.0, 100.0))
            _filled = int(np.clip(np.floor(_pct / 5), 0, 20))
            _bar = "[" + "█" * _filled + "░" * (20 - _filled) + "]"
            print(f"{_name:<35}  {_v:>12.2f}  {_pct:>7.1f}%  {_bar}")
        
        print("-" * 80)
        print(f"{'合計':<35}  {_total_measure:>12.2f}  {'100.0%':>8}")
        
        # 快取性能分析
        print(f"\n📊 快取性能分析：")
        print(f"  • 快取命中率: {stats['cache_hit_rate']*100:.1f}% ({cache_stats['hits']}/{cache_stats['hits']+cache_stats['misses']})")
        if _cache_hit_ms > 0:
            print(f"  • 快取命中平均: {_cache_hit_ms:.2f} ms/sample")
        if _cache_miss_ms > 0:
            print(f"  • 快取缺失平均: {_cache_miss_ms:.2f} ms/sample")
            if _cache_hit_ms > 0:
                speedup = _cache_miss_ms / _cache_hit_ms
                print(f"  • 加速倍數: {speedup:.1f}x (快取命中 vs 缺失)")
        
        _bottleneck = max(_rows, key=lambda x: x[1])
        _bottleneck_pct = (_bottleneck[1] / (_total_measure + 1e-9) * 100) if _total_measure > 0 else 0.0
        print(f"\n🏆 主要瓶頸：【{_bottleneck[0]}】 {_bottleneck[1]:.2f} ms ({_bottleneck_pct:.1f}%)")
        print(f"   吞吐量：{len(results) / total_time:.1f} samples/sec")
        print("="*80 + "\n")
    
    return results, stats

print("  ✅ 優化評估函數已定義")
print("="*90 + "\n")


# %%

# ============================================================================
# 📋 優化配置總結 + 預優化檢查清單
# ============================================================================

print("\n" + "="*90)
print("  📋 優化實施檢查清單")
print("="*90)

# 檢查清單
optimization_checklist = {
    "快取系統": {
        "已啟用": CACHE_CONFIG["enabled"],
        "快取目錄": str(CACHE_CONFIG["cache_dir"]),
        "狀態": "✅ 啟用" if CACHE_CONFIG["enabled"] else "❌ 禁用",
    },
    "推論優化": {
        "快取感知推論": "✅ 啟用 (predict_medsam_cached)",
        "混合精度 (AMP)": "✅ 啟用" if hasattr(torch.cuda, 'amp') else "⚠️ 未測試",
        "torch.compile": f"✅ 已編譯 ({MODEL_COMPILE_BACKEND})" if MODEL_COMPILED else "⚠️ 未編譯",
    },
    "DataLoader 配置": {
        "num_workers": EXPERIMENT_CONFIG.get("num_workers", "N/A"),
        "prefetch_factor": EXPERIMENT_CONFIG.get("prefetch_factor", "N/A"),
        "pin_memory": EXPERIMENT_CONFIG.get("pin_memory", False),
        "persistent_workers": EXPERIMENT_CONFIG.get("persistent_workers", False),
    },
    "硬體利用": {
        "CPU 邏輯核心": CPU_LOGICAL_THREADS,
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "CUDA 版本": torch.version.cuda,
    },
}

for category, items in optimization_checklist.items():
    print(f"\n🔹 {category}:")
    for key, value in items.items():
        if isinstance(value, dict):
            continue
        status = "  " if not any(c in str(value) for c in ["✅", "❌", "⚠️"]) else ""
        print(f"  • {key}: {status}{value}")

print("\n" + "-"*90)

# 優化預期改進
improvements = """
📊 預期性能改進（基於分析結果）：

瓶頸 #1: processor() 預處理 (40.2% → ~5-10%)
┌─ 原因: Python GIL 單線程 CPU 綁定（每次推論重複 normalize/resize/pad）
├─ 方案: 快取 processor 輸出 (.pt 格式)
└─ 預期: 50ms → 5ms，加速 10 倍 ✅

瓶頸 #2: DataLoader I/O (32.0% → ~8-15%)
┌─ 原因: WSL 9P NTFS 掛載只有 19 MB/s (vs Linux 500+ MB/s)
├─ 方案 A: 移至 Linux ext4 (/mnt/e 或 /home)
├─ 方案 B: 序列化至 LMDB/HDF5（預加載速度提升）
└─ 預期: 視移動目標而定，可達 10-20x

推論 (GPU) (22.9% → 20-25%, 相對提升)
┌─ 特點: RTX 3080 容量充足，已啟用 AMP + torch.compile
└─ 狀態: 已優化，GPU 利用率有上升空間（43% → 60%+）

目標: 從 8 samples/sec → 20-30 samples/sec (2.5-3.75 倍加速)
"""

print(improvements)

# 實施順序建議
implementation_guide = """
🎯 實施順序（優先級 HIGH → LOW）：

【立即可做】
1. ✅ [已完成] 啟用快取系統 + predict_medsam_cached
2. ✅ [已完成] 增加 num_workers 至激進配置
3. ⏳ [次步] 執行快取預計算（見下一格）

【中期優化】
4. 📂 移動數據至 Linux ext4 分區（若可用）
5. 🗂️ 序列化至 LMDB/HDF5 (可選)
6. 🔧 微調 batch_size（目前為 1，視 VRAM 提升）

【長期優化】
7. 🧠 量化模型（INT8 SAM）
8. 🔀 多 GPU 分散式評估
9. 🎛️ 超參數搜尋（num_workers, prefetch_factor）
"""

print(implementation_guide)

print("\n" + "="*90)


# %%

# ============================================================================
# 🔬 詳細多維度瓶頸分析工具
# ============================================================================
# 在實際評估前進行系統診斷，包括：
# - 硬體能力檢測（CPU、GPU、RAM、VRAM、PCIe、SSD）
# - 軟體棧分析（PyTorch、CUDA、cuDNN、Triton）
# - 模型複雜度分析（FLOPs、參數量、激活記憶體）
# - DataLoader 性能測試（吞吐量、延遲分佈）
# ============================================================================

import subprocess, json, time, threading
import psutil, numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

print("\n" + "="*90)
print(" 🔬 詳細多維度瓶頸分析（硬體、軟體、演算法、資料流）")
print("="*90)

# ── 1️⃣  硬體層診斷 ──────────────────────────────────────────────────
print("\n📊 [1] 硬體層診斷")
print("-" * 90)

# CPU 資訊
cpu_freq = psutil.cpu_freq()
cpu_count = psutil.cpu_count(logical=False)
cpu_logical = psutil.cpu_count(logical=True)
ram = psutil.virtual_memory()
print(f"  CPU：{cpu_count}P / {cpu_logical}L  |  頻率：{cpu_freq.current:.1f} MHz (基礎 {cpu_freq.min:.0f}, 睿頻 {cpu_freq.max:.0f})")
print(f"  RAM：{ram.total/1e9:.1f} GB  (可用 {ram.available/1e9:.1f} GB, 已用 {ram.percent:.1f}%)")

# GPU 資訊
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"  GPU：{gpu_name}  |  VRAM：{gpu_mem/1e9:.1f} GB")
    # nvidia-smi snapshot
    try:
        nv_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.current.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True
        ).strip().split(',')
        gpu_clock, mem_clock, mem_used, mem_total = [int(x.strip()) for x in nv_out]
        print(f"  GPU 時鐘：{gpu_clock} MHz  |  顯存時鐘：{mem_clock} MHz")
    except:
        pass
else:
    print("  ⚠️  GPU 不可用")

# PCIe 帶寬測試（256MB）
if torch.cuda.is_available():
    _bw_n = 256 * 1024 * 1024 // 4
    _src = torch.randn(_bw_n, dtype=torch.float32, pin_memory=True)
    _dst = torch.empty(_bw_n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    for _ in range(3):
        _dst.copy_(_src)
        torch.cuda.synchronize()
    _h2d_bw = (256 * 3) / (time.perf_counter() - _t0) / 1024
    del _src, _dst
    print(f"  PCIe H→D 帶寬：{_h2d_bw:.1f} GB/s  {'✅ 正常' if _h2d_bw >= 5 else '⚠️  偏低'}")

# SSD/磁碟速度測試（讀 100 個樣本）
if hasattr(tn3k_dataset, '__getitem__'):
    _ssd_times = []
    for i in range(min(10, len(tn3k_dataset))):
        _t0 = time.perf_counter()
        _ = tn3k_dataset[i]
        _ssd_times.append(time.perf_counter() - _t0)
    _ssd_ms = np.mean(_ssd_times) * 1000
    print(f"  SSD 讀取延遲：{_ssd_ms:.2f} ms/sample  (路徑：{DATA_PATHS.get('TN3K', 'unknown')})")

# ── 2️⃣  軟體棧診斷 ──────────────────────────────────────────────────
print("\n📦 [2] 軟體棧診斷")
print("-" * 90)

print(f"  PyTorch：{torch.__version__}")
print(f"  CUDA：{torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
try:
    cudnn_ver = torch.backends.cudnn.version()
    print(f"  cuDNN：{cudnn_ver}")
except:
    pass
print(f"  Triton：{'✅' if HAS_TRITON else '❌'} (torch.compile 依賴)")
print(f"  模型編譯：MODEL_COMPILED={MODEL_COMPILED}  backend={MODEL_COMPILE_BACKEND}")

# ── 3️⃣  模型複雜度分析 ──────────────────────────────────────────────
print("\n🧠 [3] 模型複雜度分析")
print("-" * 90)

# 參數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  總參數量：{total_params/1e6:.1f} M")
print(f"  可訓練參數：{trainable_params/1e6:.1f} M")

# 推論記憶體
torch.cuda.reset_peak_memory_stats()
_test_sample = tn3k_dataset[0]
_test_img = _test_sample["image"]
_test_bbox = _test_sample["bbox"]
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.float16):
        _test_out = predict_medsam(model, processor, _test_img, _test_bbox, device="cuda", use_amp=True)
torch.cuda.synchronize()
_peak_mem = torch.cuda.max_memory_allocated() / 1e6
print(f"  單樣本推論峰值 VRAM：{_peak_mem:.0f} MB (總 {torch.cuda.get_device_properties(0).total_memory/1e6:.0f} MB)")
print(f"  VRAM 利用率：{_peak_mem*100/torch.cuda.get_device_properties(0).total_memory:.1f}% → {'⚠️  接近瓶頸' if _peak_mem>5000 else '✅ 充足'}")

# FLOPs 估算（SAM Vision Transformer ~10-15 GFLOPs per inference）
print(f"  推論 FLOPs：~10-15 GFLOPS (SAM ViT baseline)")

# ── 4️⃣  DataLoader 性能測試 ──────────────────────────────────────────
print("\n⚡ [4] DataLoader 性能測試（含真實前處理）")
print("-" * 90)

_loader = build_eval_dataloader(tn3k_dataset)
_loader_times = []
_t0 = time.perf_counter()
for i, _samp in enumerate(_loader):
    if i >= min(30, len(tn3k_dataset)):
        break
    _loader_times.append(time.perf_counter() - _t0)
    _t0 = time.perf_counter()

_loader_ms = np.mean(_loader_times[1:]) * 1000  # 跳過第一個（冷啟動）
_loader_std = np.std(_loader_times[1:]) * 1000
print(f"  取樣延遲：{_loader_ms:.2f} ± {_loader_std:.2f} ms")
print(f"  吞吐量：{1000/_loader_ms:.1f} samples/sec")
print(f"  DataLoader 設定：num_workers={EXPERIMENT_CONFIG.get('num_workers',0)}, "
      f"pin_memory={EXPERIMENT_CONFIG.get('pin_memory',False)}, "
      f"prefetch_factor={EXPERIMENT_CONFIG.get('prefetch_factor',2)}")

# ── 5️⃣ 推論階段微觀計時 ──────────────────────────────────────────────
print("\n🎯 [5] 推論階段微觀計時（predict_medsam 拆解）")
print("-" * 90)

_test_img = _test_sample["image"]
_test_bbox = _test_sample["bbox"]

# 前處理（快速路徑，必要時自動回退 processor）
_proc_times = []
for i in range(min(10, len(tn3k_dataset))):
    _samp = tn3k_dataset[i]
    _t0 = time.perf_counter()
    _inp = build_sam_inputs(
        processor=processor,
        images=[_samp["image"]],
        input_boxes=[[_samp["bbox"]]],
    )
    _proc_times.append(time.perf_counter() - _t0)
_proc_ms = np.mean(_proc_times) * 1000
print(f"  preprocess() CPU 前處理：{_proc_ms:.2f} ms")

# H→D 傳輸
_h2d_times = []
for i in range(min(10, len(tn3k_dataset))):
    _samp = tn3k_dataset[i]
    _inp = build_sam_inputs(
        processor=processor,
        images=[_samp["image"]],
        input_boxes=[[_samp["bbox"]]],
    )
    _t0 = time.perf_counter()
    _inp_gpu = {k: v.to(device, non_blocking=False) for k, v in _inp.items()}
    torch.cuda.synchronize()
    _h2d_times.append(time.perf_counter() - _t0)
_h2d_ms = np.mean(_h2d_times) * 1000
print(f"  PCIe H→D 傳輸：{_h2d_ms:.2f} ms  ({sum(v.nbytes for v in _inp.values())/1024:.0f} KB)")

# GPU Forward
_fwd_times = []

# 先預熱，避免首次 compile/caching 污染微觀統計
_warm_inp = build_sam_inputs(
    processor=processor,
    images=[_test_img],
    input_boxes=[[_test_bbox]],
)
_warm_inp_gpu = {k: v.to(device, non_blocking=False) for k, v in _warm_inp.items()}
with torch.inference_mode():
    for _ in range(3):
        _ = _run_sam_forward_prob_masks(model, _warm_inp_gpu, device=device, use_amp=True)
if torch.cuda.is_available():
    torch.cuda.synchronize()

for i in range(min(10, len(tn3k_dataset))):
    _samp = tn3k_dataset[i]
    _inp = build_sam_inputs(
        processor=processor,
        images=[_samp["image"]],
        input_boxes=[[_samp["bbox"]]],
    )
    _inp_gpu = {k: v.to(device, non_blocking=False) for k, v in _inp.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    with torch.inference_mode():
        _ = _run_sam_forward_prob_masks(model, _inp_gpu, device=device, use_amp=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _fwd_times.append(time.perf_counter() - _t0)
_fwd_ms = np.mean(_fwd_times) * 1000
print(f"  GPU Forward (sync'd)：{_fwd_ms:.2f} ms")

# 後處理
_post_times = []
for i in range(min(10, len(tn3k_dataset))):
    _samp = tn3k_dataset[i]
    _h, _w = np.array(_samp["image"]).shape[:2]
    _m = torch.sigmoid(torch.randn(1, 1, 256, 256, device="cuda"))  # 模擬
    _t0 = time.perf_counter()
    _ = torch.nn.functional.interpolate(_m, size=(_h, _w), mode="bilinear", align_corners=False).cpu().numpy()
    torch.cuda.synchronize()
    _post_times.append(time.perf_counter() - _t0)
_post_ms = np.mean(_post_times) * 1000
print(f"  後處理 + D→H：{_post_ms:.2f} ms")

# 總計
_total_inf = _proc_ms + _h2d_ms + _fwd_ms + _post_ms
print(f"\n  📈 推論總耗時：{_total_inf:.2f} ms = {_proc_ms:.1f}(proc) + {_h2d_ms:.1f}(H2D) + {_fwd_ms:.1f}(fwd) + {_post_ms:.1f}(post)")
_bottleneck_stage = max([("processor", _proc_ms), ("H→D", _h2d_ms), ("forward", _fwd_ms), ("postproc", _post_ms)], key=lambda x: x[1])
print(f"  🏆 微觀瓶頸：【{_bottleneck_stage[0]}】{_bottleneck_stage[1]:.2f} ms ({_bottleneck_stage[1]/_total_inf*100:.1f}%)")

# ── 6️⃣  CPU vs GPU 負載分析 ──────────────────────────────────────────
print("\n⚙️  [6] CPU vs GPU 負載分析")
print("-" * 90)

_cpu_samples = []
_stop = threading.Event()

def _cpu_sampler():
    while not _stop.is_set():
        _cpu_samples.append(psutil.cpu_percent(percpu=False))
        time.sleep(0.01)

_samp = tn3k_dataset[0]
_inp = build_sam_inputs(
    processor=processor,
    images=[_samp["image"]],
    input_boxes=[[_samp["bbox"]]],
)
_inp_gpu = {k: v.to(device) for k, v in _inp.items()}
torch.cuda.synchronize()

_thr = threading.Thread(target=_cpu_sampler, daemon=True)
_thr.start()
for _ in range(10):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _ = model(**_inp_gpu)
torch.cuda.synchronize()
_stop.set()
_thr.join(timeout=1.0)

_cpu_mean = np.mean(_cpu_samples) if _cpu_samples else 0
print(f"  forward() 期間平均 CPU 利用率：{_cpu_mean:.1f}% {'✅ GPU主導' if _cpu_mean<50 else '⚠️  CPU偏高'}")

# ── 7️⃣  CUDA 核心層 Profile ──────────────────────────────────────────
print("\n🔧 [7] CUDA 核心層 Top 操作（torch.profiler）")
print("-" * 90)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True, with_stack=False, with_flops=True
) as _prof:
    for _ in range(5):
        with torch.no_grad():
            _samp = tn3k_dataset[_% len(tn3k_dataset)]
            predict_medsam(model, processor, _samp["image"], _samp["bbox"], device, use_amp=True)

_kavg = _prof.key_averages()
_cuda_ops = sorted([e for e in _kavg if any(attr in str(e.key) for attr in ["cuda", "Memcpy", "gemm"])], 
                    key=lambda e: getattr(e, 'cuda_time_total', getattr(e, 'device_time_total', 0)), reverse=True)
print(f"  {'排名':<4} {'操作':<45} {'CUDA(ms)':>10}")
print("  " + "-" * 65)
for rank, e in enumerate(_cuda_ops[:8], 1):
    _cuda_t = getattr(e, 'cuda_time_total', getattr(e, 'device_time_total', 0)) / 1000
    print(f"  {rank:<4} {e.key[:45]:<45} {_cuda_t:>10.2f}")

print("\n" + "="*90)
print(" 🎯 診斷摘要：")
print("="*90)
print(f"  ❌ 主要瓶頸定位：【{_bottleneck_stage[0].upper()}】 {_bottleneck_stage[1]:.2f} ms")
print(f"  💾 記憶體：VRAM {_peak_mem/1000:.1f}G / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}G (充足)" if _peak_mem < 5000 else f"  ⚠️  VRAM 接近上限")
print(f"  🚀 吞吐量預期：~{1000/_total_inf:.1f} samples/sec (單樣本 {_total_inf:.1f} ms)")
print("="*90 + "\n")


# %%
# ============================================================================
# 🎓 Fine-tune MedSAM 模型（三資料集合併訓練）
# ============================================================================
# 使用 TN3K、DDTI 和 TN5000 的訓練集對 MedSAM 進行 fine-tune，
# 然後評估 fine-tune 模型在三個測試集上的性能改進。
# ============================================================================

print("\n" + "="*90)
print("🎓 Fine-tune MedSAM 模型（三資料集合併訓練）")
print("="*90)

# ─────────────────────────────────────────────────────────────────────────
# 訓練數據集類
# ─────────────────────────────────────────────────────────────────────────

class TN3KTrainDataset(Dataset):
    """TN3K 訓練集"""
    def __init__(self, root_dir: str, image_size: int = 512, split_file: Optional[Path] = None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.split_file = split_file
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """載入訓練集樣本"""
        image_dir = self.root_dir / "train-image"
        mask_dir = self.root_dir / "train-mask"

        # TN3K 常見命名為 trainval-image/trainval-mask，若 train-* 不存在則自動回退
        if not image_dir.exists() or not mask_dir.exists():
            alt_image_dir = self.root_dir / "trainval-image"
            alt_mask_dir = self.root_dir / "trainval-mask"
            if alt_image_dir.exists() and alt_mask_dir.exists():
                image_dir = alt_image_dir
                mask_dir = alt_mask_dir
                print("ℹ️ TN3K 訓練資料夾使用 trainval-image/trainval-mask")
        
        if image_dir.exists():
            split_ids = _read_split_ids(self.split_file)
            image_files = sorted(image_dir.glob("*.jpg"))
            for img_file in image_files:
                if split_ids is not None and img_file.stem not in split_ids:
                    continue
                mask_file = mask_dir / img_file.name
                if mask_file.exists():
                    self.samples.append({
                        "image_path": img_file,
                        "mask_path": mask_file,
                        "name": img_file.stem,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")
        
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        mask_np = np.array(mask) > 127
        bbox = compute_bbox_from_mask_np(mask_np.astype(np.uint8))
        
        return {
            "image": image,
            "image_np": np.array(image),
            "mask": torch.tensor(mask_np.astype(np.float32), dtype=torch.float32),
            "bbox": bbox,
            "name": sample["name"],
        }


class CombinedTrainDataset(Dataset):
    """合併多個訓練數據集"""
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        cum = 0
        for ds in datasets:
            cum += len(ds)
            self.cumulative_sizes.append(cum)
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                dataset_idx = idx if i == 0 else idx - self.cumulative_sizes[i-1]
                return self.datasets[i][dataset_idx]
        raise IndexError(f"Index {idx} out of range")


# ─────────────────────────────────────────────────────────────────────────
# Worker 端預處理資料集（解決序列化問題）
# ─────────────────────────────────────────────────────────────────────────

class FinetuneProcessorDataset(Dataset):
    """
    在 DataLoader worker 端先完成 processor 預處理，
    回傳純 tensor，避免 PIL 物件在多進程間序列化。
    """

    def __init__(self, base_dataset: Dataset, processor: SamProcessor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        image = sample["image"]
        bbox = sample["bbox"]
        mask = sample["mask"]

        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"), dtype=np.uint8)
        else:
            image_np = np.asarray(sample.get("image_np", image), dtype=np.uint8)

        inputs = build_sam_inputs(
            processor=self.processor,
            images=[image_np],
            input_boxes=[[bbox]],
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_boxes": inputs["input_boxes"].squeeze(0),
            "original_sizes": inputs["original_sizes"].squeeze(0),
            "reshaped_input_sizes": inputs["reshaped_input_sizes"].squeeze(0),
            "mask": mask.to(torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────
# Fine-tune 訓練函數
# ─────────────────────────────────────────────────────────────────────────

def finetune_medsam(
    model: SamModel,
    processor: SamProcessor,
    train_dataset: Dataset,
    num_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: str = "medsam_finetuned.pth",
    val_ratio: float = 0.1,
    early_stop_patience: int = 4,
    min_delta: float = 1e-4,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 1.0,
) -> Dict[str, Any]:
    """
    加速版 Fine-tune MedSAM 模型：
      Phase 1 — 預計算全部圖像 embedding（no_grad，一次性，跳過 ViT 重複前向）
      Phase 2 — 只訓練 mask_decoder（~4M 參數），使用 GradScaler + batch_size=8

    相比原始「每 batch 跑完整 ViT forward+backward」方案，預估加速 20-30×。
    """
    # ── 取得 base（未 compile）模型 ──────────────────────────────────────
    base_model: SamModel = getattr(model, "_orig_mod", model)
    base_model = base_model.to(device)

    # ── 1. 冷凍 encoder，只訓練 mask_decoder ────────────────────────────
    for name, p in base_model.named_parameters():
        p.requires_grad = ("mask_decoder" in name)

    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Fine-tune: 沒有可訓練參數（mask_decoder 未找到）")

    trainable_count = sum(p.numel() for p in trainable_params)
    total_count     = sum(p.numel() for p in base_model.parameters())
    print(f"🔧 只訓練 mask_decoder: {trainable_count:,}/{total_count:,} 參數"
          f" ({100*trainable_count/total_count:.1f}%)")

    # ── 2. Phase 1：預計算全部圖像 embedding（支援磁碟快取）──────────────
    print("\n📦 Phase 1: 預計算圖像 embedding（compiled vision_encoder，ViT 只跑一次）...")
    n_expected = len(train_dataset)
    embed_cache_path = Path(save_path).with_suffix(".embeddings_cache.pt")

    all_emb: Optional[torch.Tensor] = None
    all_boxes: Optional[torch.Tensor] = None
    all_masks: Optional[torch.Tensor] = None

    if embed_cache_path.exists():
        try:
            cached = torch.load(embed_cache_path, map_location="cpu")
            cached_n = int(cached.get("n_samples", -1))
            if cached_n == n_expected:
                all_emb = cached["embeddings"].contiguous()
                all_boxes = cached["boxes"].contiguous()
                all_masks = cached["masks"].contiguous()
                print(f"✅ 已載入 embedding 快取: {embed_cache_path} (N={cached_n})")
            else:
                print(f"⚠️ embedding 快取樣本數不符（cache={cached_n}, expected={n_expected}），將重建")
        except Exception as e:
            print(f"⚠️ 載入 embedding 快取失敗，將重建: {e}")

    if all_emb is None or all_boxes is None or all_masks is None:
        worker_count = os.cpu_count()

        # Phase-1 可調參：優先用環境變數快速調參，不需改碼
        embed_batch = int(env_get("MEDSAM_EMBED_BATCH"))
        embed_workers_raw = env_get("MEDSAM_EMBED_WORKERS")
        embed_workers = worker_count if embed_workers_raw.strip().lower() == "auto" else int(embed_workers_raw)

        # 批次 collate 一次跑 processor，避免每個 sample 在 worker 端各自調用 processor。
        # 在 /mnt/c（WSL）常可明顯降低 Python 開銷與序列化成本。
        def _embed_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            images_np: List[np.ndarray] = []
            boxes: List[List[List[float]]] = []
            masks: List[torch.Tensor] = []

            for sample in samples:
                image = sample["image"]
                bbox = sample["bbox"]
                mask = sample["mask"]

                if isinstance(image, Image.Image):
                    image_np = np.array(image.convert("RGB"), dtype=np.uint8)
                else:
                    image_np = np.asarray(sample.get("image_np", image), dtype=np.uint8)

                images_np.append(image_np)
                boxes.append([bbox])
                masks.append(mask.to(torch.float32))

            inputs = build_sam_inputs(
                processor=processor,
                images=images_np,
                input_boxes=boxes,
            )

            return {
                "pixel_values": inputs["pixel_values"],
                "input_boxes": inputs["input_boxes"],
                "mask": torch.stack(masks, dim=0),
            }

        # 直接編譯 vision_encoder；固定 shape 下使用 dynamic=False 通常更快。
        compiled_ve = torch.compile(
            base_model.vision_encoder,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
        )
        embed_loader = DataLoader(
            train_dataset,
            batch_size=embed_batch,
            shuffle=False,
            num_workers=embed_workers,
            pin_memory=True,
            persistent_workers=(embed_workers > 0),
            prefetch_factor=(4 if embed_workers > 0 else None),
            collate_fn=_embed_collate_fn,
        )

        print(
            f"🚀 Phase-1 DataLoader: batch={embed_batch}, workers={embed_workers}, "
            f"prefetch=4, persistent_workers={embed_workers > 0}"
        )

        embed_list: List[torch.Tensor] = []
        boxes_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []

        base_model.eval()
        with torch.no_grad():
            for batch in tqdm(embed_loader, desc="預計算 embedding"):
                pv  = batch["pixel_values"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    emb = compiled_ve(pv)[0]                # (B, 256, 64, 64)
                embed_list.append(emb.half().cpu())         # float16 節省 RAM
                boxes_list.append(batch["input_boxes"].cpu())
                masks_list.append(batch["mask"].cpu())

        all_emb   = torch.cat(embed_list, dim=0)   # (N, 256, 64, 64) float16
        all_boxes = torch.cat(boxes_list, dim=0)   # (N, 1, 4)
        all_masks = torch.cat(masks_list, dim=0)   # (N, H, W) or (N, 1, H, W)
        del embed_list, boxes_list, masks_list
        del compiled_ve  # 釋放編譯快取，後續只用 mask_decoder

        try:
            torch.save(
                {
                    "n_samples": int(all_emb.shape[0]),
                    "embeddings": all_emb,
                    "boxes": all_boxes,
                    "masks": all_masks,
                },
                embed_cache_path,
            )
            print(f"✅ 已儲存 embedding 快取: {embed_cache_path}")
        except Exception as e:
            print(f"⚠️ 儲存 embedding 快取失敗（不影響本次訓練）: {e}")

    n_samples = len(all_emb)
    ram_gb = all_emb.element_size() * all_emb.numel() / 1e9
    print(f"✅ Embedding 準備完成: shape={tuple(all_emb.shape)}, RAM≈{ram_gb:.1f}GB")

    # ── 3. Phase-2 訓練圖（可選 compile）───────────────────────────────
    finetune_compile_phase2 = env_get_bool("MEDSAM_FINETUNE_COMPILE_PHASE2", False)
    base_model.train()
    if finetune_compile_phase2:
        train_model = torch.compile(
            base_model,
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
        print("✅ Phase-2 使用 torch.compile/inductor（dynamic=False）")
    else:
        train_model = base_model
        print("ℹ️  Phase-2 使用 eager（預設），避免 torch._dynamo recompile 抖動")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, early_stop_patience // 2),
        threshold=min_delta,
        min_lr=1e-6,
    )
    scaler    = torch.cuda.amp.GradScaler()
    loss_fn   = torch.nn.BCEWithLogitsLoss()

    training_stats: Dict[str, Any] = {
        "epochs": [],
        "total_loss": [],
        "avg_loss_per_epoch": [],
        "val_loss": [],
        "lr": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "stopped_early": False,
    }
    total_samples_processed = 0

    val_ratio = float(np.clip(val_ratio, 0.0, 0.5))
    if n_samples < 20:
        val_ratio = 0.0
    n_val = int(n_samples * val_ratio)
    if n_val > 0:
        shuffled = torch.randperm(n_samples)
        val_indices = shuffled[:n_val]
        train_indices = shuffled[n_val:]
    else:
        train_indices = torch.arange(n_samples)
        val_indices = torch.empty(0, dtype=torch.long)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve_epochs = 0

    print(
        f"📈 收斂控制: train={len(train_indices)}, val={len(val_indices)}, "
        f"patience={early_stop_patience}, min_delta={min_delta}, "
        f"grad_accum={grad_accum_steps}, grad_clip={grad_clip_norm}"
    )

    # ── 4. Phase 2：訓練 mask_decoder（embedding 已快取，無 ViT 開銷）──
    print(f"\n🎓 Phase 2: 訓練 mask_decoder — {num_epochs} epochs, batch_size={batch_size}")

    for epoch in range(num_epochs):
        epoch_losses: List[float] = []
        perm = train_indices[torch.randperm(len(train_indices))]

        pbar = tqdm(range(0, len(train_indices), batch_size),
                    desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        optimizer.zero_grad(set_to_none=True)
        grad_accum_steps = max(1, int(grad_accum_steps))

        for step_idx, start in enumerate(pbar):
            idx   = perm[start : start + batch_size]
            if len(idx) == 0:
                continue

            emb   = all_emb[idx].to(device, dtype=torch.float32, non_blocking=True)
            boxes = all_boxes[idx].to(device, non_blocking=True)
            masks = all_masks[idx].to(device, non_blocking=True)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1).float()   # (B, 1, H, W)
            elif masks.dim() == 4:
                masks = masks.float()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs    = train_model(image_embeddings=emb, input_boxes=boxes)
                pred_masks = _normalize_masks_to_4d(outputs.pred_masks)
                pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:],
                                           mode="bilinear", align_corners=False)
                loss = loss_fn(pred_masks, masks)
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()

            should_step = ((step_idx + 1) % grad_accum_steps == 0) or ((start + batch_size) >= len(perm))
            if should_step:
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(float(loss.item()))
            total_samples_processed += len(idx)
            pbar.set_postfix({"loss": f"{np.mean(epoch_losses):.4f}"})

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss = epoch_loss

        if len(val_indices) > 0:
            val_losses: List[float] = []
            base_model.eval()
            with torch.no_grad():
                for vstart in range(0, len(val_indices), batch_size):
                    vidx = val_indices[vstart : vstart + batch_size]
                    if len(vidx) == 0:
                        continue
                    vemb = all_emb[vidx].to(device, dtype=torch.float32, non_blocking=True)
                    vboxes = all_boxes[vidx].to(device, non_blocking=True)
                    vmasks = all_masks[vidx].to(device, non_blocking=True)
                    if vmasks.dim() == 3:
                        vmasks = vmasks.unsqueeze(1).float()
                    elif vmasks.dim() == 4:
                        vmasks = vmasks.float()

                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        vout = base_model(image_embeddings=vemb, input_boxes=vboxes)
                        vpred = _normalize_masks_to_4d(vout.pred_masks)
                        vpred = F.interpolate(vpred, size=vmasks.shape[-2:], mode="bilinear", align_corners=False)
                        vloss = loss_fn(vpred, vmasks)
                    val_losses.append(float(vloss.item()))
            base_model.train()
            val_loss = float(np.mean(val_losses)) if val_losses else epoch_loss

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        training_stats["epochs"].append(epoch + 1)
        training_stats["total_loss"].append(epoch_loss)
        training_stats["avg_loss_per_epoch"].append(epoch_loss)
        training_stats["val_loss"].append(val_loss)
        training_stats["lr"].append(current_lr)

        improved = (training_stats["best_val_loss"] - val_loss) > float(min_delta)
        if improved:
            training_stats["best_val_loss"] = val_loss
            training_stats["best_epoch"] = epoch + 1
            no_improve_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        else:
            no_improve_epochs += 1

        print(
            f"✅ Epoch {epoch+1} 完成 — train_loss: {epoch_loss:.4f}, "
            f"val_loss: {val_loss:.4f}, lr: {current_lr:.2e}, "
            f"best@{training_stats['best_epoch']}={training_stats['best_val_loss']:.4f}"
        )

        if no_improve_epochs >= max(1, early_stop_patience):
            training_stats["stopped_early"] = True
            print(f"⏹️  Early stopping 觸發：連續 {no_improve_epochs} 個 epoch 無改善")
            break

    if best_state is not None:
        base_model.load_state_dict(best_state, strict=True)
        print(
            f"✅ 已回復最佳權重：epoch {training_stats['best_epoch']} "
            f"(best_val_loss={training_stats['best_val_loss']:.4f})"
        )

    # ── 5. 儲存（base_model 包含訓練後權重）────────────────────────────
    base_model.eval()
    torch.save(base_model.state_dict(), save_path)
    print(f"\n✅ Fine-tune 模型已保存到 {save_path}")

    # 若原始推論 model 是 compiled wrapper，同步權重
    if hasattr(model, "_orig_mod") and model._orig_mod is not base_model:
        model._orig_mod.load_state_dict(base_model.state_dict())

    training_stats["total_samples_processed"] = total_samples_processed
    training_stats["save_path"] = save_path
    return training_stats


# ─────────────────────────────────────────────────────────────────────────
# 執行 Fine-tune
# ─────────────────────────────────────────────────────────────────────────

print("\n🔄 準備訓練數據集...")

skip_finetune_raw = str(env_get("MEDSAM_SKIP_FINETUNE")).strip()
if skip_finetune_raw not in {"0", "1"}:
    print(f"⚠️ MEDSAM_SKIP_FINETUNE 非法值: {skip_finetune_raw!r}，將視為 '0'（執行 fine-tune）")
    skip_finetune_raw = "0"
skip_finetune = (skip_finetune_raw == "1")
print(f"MEDSAM_SKIP_FINETUNE 原始值: {skip_finetune_raw!r} -> skip_finetune={skip_finetune}")
max_finetune_samples = int(env_get("MEDSAM_FINETUNE_MAX_SAMPLES"))

# 載入訓練集
tn3k_train_split_file = _split_file("TN3K", "train")
ddti_train_split_file = _split_file("DDTI", "train")
tn5000_train_split_file = _split_file("TN5000", "train")

tn3k_train_dataset = TN3KTrainDataset(
    DATA_PATHS["TN3K"],
    image_size=MODEL_CONFIG["image_size"],
    split_file=tn3k_train_split_file,
)
print(f"✅ TN3K 訓練集: {len(tn3k_train_dataset)} 個樣本")

ddti_train_dataset = DDTIDataset(
    DATA_PATHS["DDTI"],
    image_size=MODEL_CONFIG["image_size"],
    split_file=ddti_train_split_file,
)
print(f"✅ DDTI 訓練集: {len(ddti_train_dataset)} 個樣本")

tn5000_train_dataset = TN5000Dataset(
    DATA_PATHS["TN5000"],
    split="train",
    image_size=MODEL_CONFIG["image_size"],
    split_file=tn5000_train_split_file,
)
print(f"✅ TN5000 訓練集: {len(tn5000_train_dataset)} 個樣本")

# 合併訓練集
combined_train_dataset = CombinedTrainDataset([
    tn3k_train_dataset,
    ddti_train_dataset,
    tn5000_train_dataset
])

if max_finetune_samples > 0 and max_finetune_samples < len(combined_train_dataset):
    sampled_idx = torch.randperm(len(combined_train_dataset))[:max_finetune_samples].tolist()
    combined_train_dataset = torch.utils.data.Subset(combined_train_dataset, sampled_idx)
    print(f"⚡ 套用快速微調子集: {len(combined_train_dataset)} 個樣本")

print(f"✅ 合併訓練集: {len(combined_train_dataset)} 個樣本")

# 保存 fine-tune 前的原始模型（用於對比）
pretrained_model_path = OUTPUT_DIR / "medsam_pretrained.pth"
torch.save(_state_dict_without_compile_prefix(model), pretrained_model_path)
print(f"✅ 預訓練模型已保存到 {pretrained_model_path}")

# 執行 Fine-tune
print("\n" + "="*90)
print("🎓 開始 Fine-tune...")
print("="*90)

finetuned_model_path = OUTPUT_DIR / "medsam_finetuned.pth"
if skip_finetune:
    torch.save(_state_dict_without_compile_prefix(model), finetuned_model_path)
    finetune_stats = {
        "skipped": True,
        "total_samples_processed": 0,
        "epochs": [],
        "total_loss": [],
        "avg_loss_per_epoch": [],
        "save_path": str(finetuned_model_path),
    }
    print("⏭️ MEDSAM_SKIP_FINETUNE=1，已跳過 fine-tune，直接進入評估")
else:
    finetune_epochs = int(env_get("MEDSAM_FINETUNE_EPOCHS"))
    finetune_batch = int(env_get("MEDSAM_FINETUNE_BATCH"))
    finetune_lr = float(env_get("MEDSAM_FINETUNE_LR"))
    finetune_val_ratio = float(env_get("MEDSAM_FINETUNE_VAL_RATIO"))
    finetune_patience = int(env_get("MEDSAM_FINETUNE_PATIENCE"))
    finetune_min_delta = float(env_get("MEDSAM_FINETUNE_MIN_DELTA"))
    finetune_accum = int(env_get("MEDSAM_FINETUNE_GRAD_ACCUM"))
    finetune_clip = float(env_get("MEDSAM_FINETUNE_GRAD_CLIP"))

    finetune_stats = finetune_medsam(
        model=model,
        processor=processor,
        train_dataset=combined_train_dataset,
        num_epochs=finetune_epochs,
        batch_size=finetune_batch,
        lr=finetune_lr,
        device=device,
        save_path=str(finetuned_model_path),
        val_ratio=finetune_val_ratio,
        early_stop_patience=finetune_patience,
        min_delta=finetune_min_delta,
        grad_accum_steps=finetune_accum,
        grad_clip_norm=finetune_clip,
    )

    print(f"\n✅ Fine-tune 完成")
    print(f"   訓練樣本數: {finetune_stats['total_samples_processed']}")
    print(f"   訓練週期: {finetune_stats['epochs']}")
    print(f"   最終損失: {finetune_stats['total_loss'][-1]:.4f}")

# 保存訓練統計
finetune_stats_file = OUTPUT_DIR / "finetune_stats.json"
with open(finetune_stats_file, "w") as f:
    json.dump(finetune_stats, f, indent=2, default=json_default_serializer)
print(f"✅ Fine-tune 統計已保存到 {finetune_stats_file}")

# ─────────────────────────────────────────────────────────────────────────
# 加載 Fine-tune 後的模型進行評估
# ─────────────────────────────────────────────────────────────────────────

print("\n🔄 加載 Fine-tune 後的模型進行評估...")
_load_state_dict_compat(model, Path(finetuned_model_path), map_location=device)
model.eval()
print(f"✅ Fine-tune 模型已加載")


# %%
# ============================================================================
# 📊 TN3K 優化評估 - 預訓練 vs Fine-tune 對比
# ============================================================================
# 使用預處理後的快取系統進行三模式評估：
# 1. Baseline - MedSAM 原始推論
# 2. OOD Detection - 異常樣本檢測
# 3. TTA - 測試時增強
# ============================================================================

print("\n" + "="*90)
print("📊 評估 TN3K 數據集 - Fine-tune MedSAM 模型性能")
print("="*90)

try:
    if 'evaluate_dataset_optimized' in globals() and 'tn3k_dataset' in globals():
        
        # [1] 基線評估 - MedSAM 原始性能
        print("\n[1/3] 基線評估 (Baseline) - MedSAM 原始推論...")
        tn3k_baseline_results, tn3k_baseline_stats = evaluate_dataset_optimized(
            model=model,
            processor=processor,
            dataset=tn3k_dataset,
            dataset_name="TN3K-Baseline",
            device=device,
            use_ood=False,
            use_tta=False,
            ood_detector=ood_detector,
            tta_predictor=tta_predictor,
            enable_timing=True,
        )
        
        # [2] OOD 檢測模式 - 加入異常檢測
        print("\n[2/3] OOD 檢測評估...")
        tn3k_ood_results, tn3k_ood_stats = evaluate_dataset_optimized(
            model=model,
            processor=processor,
            dataset=tn3k_dataset,
            dataset_name="TN3K-OOD",
            device=device,
            use_ood=True,
            use_tta=False,
            ood_detector=ood_detector,
            tta_predictor=tta_predictor,
            enable_timing=True,
        )
        
        # [3] TTA 模式 - 測試時增強
        print("\n[3/3] TTA 增強評估...")
        tn3k_tta_results, tn3k_tta_stats = evaluate_dataset_optimized(
            model=model,
            processor=processor,
            dataset=tn3k_dataset,
            dataset_name="TN3K-TTA",
            device=device,
            use_ood=False,
            use_tta=True,
            ood_detector=ood_detector,
            tta_predictor=tta_predictor,
            enable_timing=True,
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 結果匯總
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        print("\n" + "="*90)
        print("📊 TN3K 評估結果 - 模型性能對比")
        print("="*90)
        
        print("\n✅ 基線 (Baseline - MedSAM 原始推論):")
        print(f"   Dice:      {tn3k_baseline_stats['mean_dice']:.4f} ± {tn3k_baseline_stats['std_dice']:.4f}")
        print(f"   Jaccard:   {tn3k_baseline_stats['mean_jaccard']:.4f} ± {tn3k_baseline_stats['std_jaccard']:.4f}")
        print(f"   F1-Score:  {tn3k_baseline_stats['mean_f1']:.4f} ± {tn3k_baseline_stats['std_f1']:.4f}")
        print(f"   精度:      {tn3k_baseline_stats['mean_precision']:.4f} ± {tn3k_baseline_stats['std_precision']:.4f}")
        print(f"   召回:      {tn3k_baseline_stats['mean_recall']:.4f} ± {tn3k_baseline_stats['std_recall']:.4f}")
        print(f"   樣本數:    {tn3k_baseline_stats['num_samples']}")
        
        print("\n✅ OOD 檢測 (Out-of-Distribution Detection):")
        print(f"   Dice:          {tn3k_ood_stats['mean_dice']:.4f} ± {tn3k_ood_stats['std_dice']:.4f}")
        print(f"   Jaccard:       {tn3k_ood_stats['mean_jaccard']:.4f} ± {tn3k_ood_stats['std_jaccard']:.4f}")
        print(f"   F1-Score:      {tn3k_ood_stats['mean_f1']:.4f} ± {tn3k_ood_stats['std_f1']:.4f}")
        print(f"   精度:          {tn3k_ood_stats['mean_precision']:.4f} ± {tn3k_ood_stats['std_precision']:.4f}")
        print(f"   召回:          {tn3k_ood_stats['mean_recall']:.4f} ± {tn3k_ood_stats['std_recall']:.4f}")
        ood_detected = tn3k_ood_stats.get('num_ood_detected', 0)
        ood_total = tn3k_ood_stats.get('num_samples', 1)
        print(f"   OOD 檢測:      {ood_detected}/{ood_total} ({ood_detected/ood_total*100:.1f}%)")
        
        print("\n✅ TTA 增強 (Test Time Augmentation):")
        print(f"   Dice:          {tn3k_tta_stats['mean_dice']:.4f} ± {tn3k_tta_stats['std_dice']:.4f}")
        print(f"   Jaccard:       {tn3k_tta_stats['mean_jaccard']:.4f} ± {tn3k_tta_stats['std_jaccard']:.4f}")
        print(f"   F1-Score:      {tn3k_tta_stats['mean_f1']:.4f} ± {tn3k_tta_stats['std_f1']:.4f}")
        print(f"   精度:          {tn3k_tta_stats['mean_precision']:.4f} ± {tn3k_tta_stats['std_precision']:.4f}")
        print(f"   召回:          {tn3k_tta_stats['mean_recall']:.4f} ± {tn3k_tta_stats['std_recall']:.4f}")
        print(f"   平均不確定性:  {tn3k_tta_stats.get('mean_uncertainty', 0):.4f}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 保存結果
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        tn3k_results_file = OUTPUT_DIR / "tn3k_results.json"
        with open(tn3k_results_file, "w") as f:
            json.dump(
                {
                    "baseline": tn3k_baseline_stats,
                    "ood": tn3k_ood_stats,
                    "tta": tn3k_tta_stats
                },
                f,
                indent=2,
                default=json_default_serializer
            )
        print(f"\n✅ TN3K 結果已保存到 {tn3k_results_file}")
        
    else:
        print("⚠️ 缺少必要的函數或數據集，請先執行第 1-29 單元格及優化初始化")
        
except Exception as e:
    print(f"❌ 評估出錯: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*90)

# %% [markdown]
# ## 9. DDTI 測試與指標評估
# 
# 在 DDTI 測試集上執行相同推論流程，輸出各項分割指標與樣本層級結果。

# %%
# 評估 DDTI 數據集
print("\n" + "="*70)
print("🔄 評估 DDTI 數據集...")
print("="*70)

# 1. 基線評估 (無 TTA，無 OOD)
print("\n[1/3] 基線評估 (Baseline)...")
ddti_baseline_results, ddti_baseline_stats = evaluate_dataset_optimized(
    model=model,
    processor=processor,
    dataset=ddti_dataset,
    dataset_name="DDTI-Baseline",
    device=device,
    use_ood=False,
    use_tta=False,
    ood_detector=ood_detector,
    tta_predictor=tta_predictor,
    enable_timing=True,
)

# 2. OOD 評估
print("\n[2/3] OOD 檢測評估...")
ddti_ood_results, ddti_ood_stats = evaluate_dataset_optimized(
    model=model,
    processor=processor,
    dataset=ddti_dataset,
    dataset_name="DDTI-OOD",
    device=device,
    use_ood=True,
    use_tta=False,
    ood_detector=ood_detector,
    tta_predictor=tta_predictor,
    enable_timing=True,
)

# 3. TTA 評估
print("\n[3/3] TTA 增強評估...")
ddti_tta_results, ddti_tta_stats = evaluate_dataset_optimized(
    model=model,
    processor=processor,
    dataset=ddti_dataset,
    dataset_name="DDTI-TTA",
    device=device,
    use_ood=False,
    use_tta=True,
    ood_detector=ood_detector,
    tta_predictor=tta_predictor,
    enable_timing=True,
)

# 列印 DDTI 結果
print("\n" + "="*70)
print("📊 DDTI 測試結果摘要")
print("="*70)

print("\n✅ 基線 (Baseline):")
print(f"   Dice:      {ddti_baseline_stats['mean_dice']:.4f} ± {ddti_baseline_stats['std_dice']:.4f}")
print(f"   Jaccard:   {ddti_baseline_stats['mean_jaccard']:.4f} ± {ddti_baseline_stats['std_jaccard']:.4f}")
print(f"   F1-Score:  {ddti_baseline_stats['mean_f1']:.4f} ± {ddti_baseline_stats['std_f1']:.4f}")
print(f"   Precision: {ddti_baseline_stats['mean_precision']:.4f} ± {ddti_baseline_stats['std_precision']:.4f}")
print(f"   Recall:    {ddti_baseline_stats['mean_recall']:.4f} ± {ddti_baseline_stats['std_recall']:.4f}")
print(f"   樣本數:    {ddti_baseline_stats['num_samples']}")

print("\n✅ OOD 檢測:")
print(f"   Dice:      {ddti_ood_stats['mean_dice']:.4f} ± {ddti_ood_stats['std_dice']:.4f}")
print(f"   Jaccard:   {ddti_ood_stats['mean_jaccard']:.4f} ± {ddti_ood_stats['std_jaccard']:.4f}")
print(f"   F1-Score:  {ddti_ood_stats['mean_f1']:.4f} ± {ddti_ood_stats['std_f1']:.4f}")
print(f"   OOD 檢測:  {ddti_ood_stats.get('num_ood_detected', 0)}/{ddti_ood_stats['num_samples']} ({ddti_ood_stats.get('ood_ratio', 0)*100:.1f}%)")

print("\n✅ TTA 增強:")
print(f"   Dice:      {ddti_tta_stats['mean_dice']:.4f} ± {ddti_tta_stats['std_dice']:.4f}")
print(f"   Jaccard:   {ddti_tta_stats['mean_jaccard']:.4f} ± {ddti_tta_stats['std_jaccard']:.4f}")
print(f"   F1-Score:  {ddti_tta_stats['mean_f1']:.4f} ± {ddti_tta_stats['std_f1']:.4f}")
print(f"   平均不確定性: {ddti_tta_stats.get('mean_uncertainty', 0):.4f}")

ddti_results_file = OUTPUT_DIR / "ddti_results.json"
with open(ddti_results_file, "w") as f:
    json.dump(
        {
            "baseline": ddti_baseline_stats,
            "ood": ddti_ood_stats,
            "tta": ddti_tta_stats
        },
        f,
        indent=2,
        default=json_default_serializer
    )
print(f"\n✅ DDTI 結果已保存到 {ddti_results_file}")

# %%
# 評估 TN5000 數據集（VOC bbox 轉 mask）
print("\n" + "="*70)
print("🔄 評估 TN5000 數據集...")
print("="*70)

# 1. 基線評估 (無 TTA，無 OOD)
print("\n[1/3] 基線評估 (Baseline)...")
tn5000_baseline_results, tn5000_baseline_stats = evaluate_dataset_optimized(
    model=model,
    processor=processor,
    dataset=tn5000_dataset,
    dataset_name="TN5000-Baseline",
    device=device,
    use_ood=False,
    use_tta=False,
    ood_detector=ood_detector,
    tta_predictor=tta_predictor,
    enable_timing=True,
)

# 2. OOD 評估
print("\n[2/3] OOD 檢測評估...")
tn5000_ood_results, tn5000_ood_stats = evaluate_dataset_optimized(
    model=model,
    processor=processor,
    dataset=tn5000_dataset,
    dataset_name="TN5000-OOD",
    device=device,
    use_ood=True,
    use_tta=False,
    ood_detector=ood_detector,
    tta_predictor=tta_predictor,
    enable_timing=True,
)

# 3. TTA 評估
print("\n[3/3] TTA 增強評估...")
tn5000_tta_results, tn5000_tta_stats = evaluate_dataset_optimized(
    model=model,
    processor=processor,
    dataset=tn5000_dataset,
    dataset_name="TN5000-TTA",
    device=device,
    use_ood=False,
    use_tta=True,
    ood_detector=ood_detector,
    tta_predictor=tta_predictor,
    enable_timing=True,
)

print("\n" + "="*70)
print("📊 TN5000 測試結果摘要")
print("="*70)

print("\n✅ 基線 (Baseline):")
print(f"   Dice:      {tn5000_baseline_stats['mean_dice']:.4f} ± {tn5000_baseline_stats['std_dice']:.4f}")
print(f"   Jaccard:   {tn5000_baseline_stats['mean_jaccard']:.4f} ± {tn5000_baseline_stats['std_jaccard']:.4f}")
print(f"   F1-Score:  {tn5000_baseline_stats['mean_f1']:.4f} ± {tn5000_baseline_stats['std_f1']:.4f}")
print(f"   Precision: {tn5000_baseline_stats['mean_precision']:.4f} ± {tn5000_baseline_stats['std_precision']:.4f}")
print(f"   Recall:    {tn5000_baseline_stats['mean_recall']:.4f} ± {tn5000_baseline_stats['std_recall']:.4f}")
print(f"   樣本數:    {tn5000_baseline_stats['num_samples']}")

print("\n✅ OOD 檢測:")
print(f"   Dice:      {tn5000_ood_stats['mean_dice']:.4f} ± {tn5000_ood_stats['std_dice']:.4f}")
print(f"   Jaccard:   {tn5000_ood_stats['mean_jaccard']:.4f} ± {tn5000_ood_stats['std_jaccard']:.4f}")
print(f"   F1-Score:  {tn5000_ood_stats['mean_f1']:.4f} ± {tn5000_ood_stats['std_f1']:.4f}")
print(f"   OOD 檢測:  {tn5000_ood_stats.get('num_ood_detected', 0)}/{tn5000_ood_stats['num_samples']} ({tn5000_ood_stats.get('ood_ratio', 0)*100:.1f}%)")

print("\n✅ TTA 增強:")
print(f"   Dice:      {tn5000_tta_stats['mean_dice']:.4f} ± {tn5000_tta_stats['std_dice']:.4f}")
print(f"   Jaccard:   {tn5000_tta_stats['mean_jaccard']:.4f} ± {tn5000_tta_stats['std_jaccard']:.4f}")
print(f"   F1-Score:  {tn5000_tta_stats['mean_f1']:.4f} ± {tn5000_tta_stats['std_f1']:.4f}")
print(f"   平均不確定性: {tn5000_tta_stats.get('mean_uncertainty', 0):.4f}")

tn5000_results_file = OUTPUT_DIR / "tn5000_results.json"
with open(tn5000_results_file, "w") as f:
    json.dump(
        {
            "baseline": tn5000_baseline_stats,
            "ood": tn5000_ood_stats,
            "tta": tn5000_tta_stats,
        },
        f,
        indent=2,
        default=json_default_serializer,
    )
print(f"\n✅ TN5000 結果已保存到 {tn5000_results_file}")

# %% [markdown]
# ## 10. Baseline / OOD / TTA Comparison and Visualization
# 
# Compare performance across TN3K, DDTI, and TN5000 under three settings, and generate summary tables and plots.

# %%
# Create comprehensive comparison table (including TN5000)
required_stats = [
    "tn3k_baseline_stats", "tn3k_ood_stats", "tn3k_tta_stats",
    "ddti_baseline_stats", "ddti_ood_stats", "ddti_tta_stats",
    "tn5000_baseline_stats", "tn5000_ood_stats", "tn5000_tta_stats"
]

missing_stats = [name for name in required_stats if name not in globals()]
if missing_stats:
    print("⚠️ Missing evaluation stats. Please run TN3K, DDTI, and TN5000 evaluation cells first.")
    print(f"Missing: {missing_stats}")
else:
    comparison_data = {
        "Method": ["Baseline", "OOD", "TTA"],
        "TN3K Dice": [
            f"{tn3k_baseline_stats['mean_dice']:.4f}",
            f"{tn3k_ood_stats['mean_dice']:.4f}",
            f"{tn3k_tta_stats['mean_dice']:.4f}"
        ],
        "TN3K Jaccard": [
            f"{tn3k_baseline_stats['mean_jaccard']:.4f}",
            f"{tn3k_ood_stats['mean_jaccard']:.4f}",
            f"{tn3k_tta_stats['mean_jaccard']:.4f}"
        ],
        "TN3K F1": [
            f"{tn3k_baseline_stats['mean_f1']:.4f}",
            f"{tn3k_ood_stats['mean_f1']:.4f}",
            f"{tn3k_tta_stats['mean_f1']:.4f}"
        ],
        "DDTI Dice": [
            f"{ddti_baseline_stats['mean_dice']:.4f}",
            f"{ddti_ood_stats['mean_dice']:.4f}",
            f"{ddti_tta_stats['mean_dice']:.4f}"
        ],
        "DDTI Jaccard": [
            f"{ddti_baseline_stats['mean_jaccard']:.4f}",
            f"{ddti_ood_stats['mean_jaccard']:.4f}",
            f"{ddti_tta_stats['mean_jaccard']:.4f}"
        ],
        "DDTI F1": [
            f"{ddti_baseline_stats['mean_f1']:.4f}",
            f"{ddti_ood_stats['mean_f1']:.4f}",
            f"{ddti_tta_stats['mean_f1']:.4f}"
        ],
        "TN5000 Dice": [
            f"{tn5000_baseline_stats['mean_dice']:.4f}",
            f"{tn5000_ood_stats['mean_dice']:.4f}",
            f"{tn5000_tta_stats['mean_dice']:.4f}"
        ],
        "TN5000 Jaccard": [
            f"{tn5000_baseline_stats['mean_jaccard']:.4f}",
            f"{tn5000_ood_stats['mean_jaccard']:.4f}",
            f"{tn5000_tta_stats['mean_jaccard']:.4f}"
        ],
        "TN5000 F1": [
            f"{tn5000_baseline_stats['mean_f1']:.4f}",
            f"{tn5000_ood_stats['mean_f1']:.4f}",
            f"{tn5000_tta_stats['mean_f1']:.4f}"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)

    print("\n" + "="*120)
    print("📊 Comprehensive Performance Comparison Table (TN3K + DDTI + TN5000)")
    print("="*120)
    print(comparison_df.to_string(index=False))
    print("="*120)

    # Save comparison table
    comparison_file = OUTPUT_DIR / "comparison_table.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✅ Comparison table saved to {comparison_file}")


# %%
# Visualization: bar chart comparison (TN3K + DDTI + TN5000)
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("MedSAM Performance Comparison: Baseline vs OOD vs TTA", fontsize=16, fontweight='bold')

methods = ["Baseline", "OOD", "TTA"]
colors = ["#3498db", "#e74c3c", "#2ecc71"]

# TN3K
tn3k_dice = [tn3k_baseline_stats['mean_dice'], tn3k_ood_stats['mean_dice'], tn3k_tta_stats['mean_dice']]
tn3k_jaccard = [tn3k_baseline_stats['mean_jaccard'], tn3k_ood_stats['mean_jaccard'], tn3k_tta_stats['mean_jaccard']]
tn3k_f1 = [tn3k_baseline_stats['mean_f1'], tn3k_ood_stats['mean_f1'], tn3k_tta_stats['mean_f1']]

# DDTI
ddti_dice = [ddti_baseline_stats['mean_dice'], ddti_ood_stats['mean_dice'], ddti_tta_stats['mean_dice']]
ddti_jaccard = [ddti_baseline_stats['mean_jaccard'], ddti_ood_stats['mean_jaccard'], ddti_tta_stats['mean_jaccard']]
ddti_f1 = [ddti_baseline_stats['mean_f1'], ddti_ood_stats['mean_f1'], ddti_tta_stats['mean_f1']]

# TN5000
tn5000_dice = [tn5000_baseline_stats['mean_dice'], tn5000_ood_stats['mean_dice'], tn5000_tta_stats['mean_dice']]
tn5000_jaccard = [tn5000_baseline_stats['mean_jaccard'], tn5000_ood_stats['mean_jaccard'], tn5000_tta_stats['mean_jaccard']]
tn5000_f1 = [tn5000_baseline_stats['mean_f1'], tn5000_ood_stats['mean_f1'], tn5000_tta_stats['mean_f1']]

# TN3K row
axes[0, 0].bar(methods, tn3k_dice, color=colors)
axes[0, 0].set_title("TN3K - Dice", fontweight='bold')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].set_ylabel("Score")
for i, v in enumerate(tn3k_dice):
    axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

axes[0, 1].bar(methods, tn3k_jaccard, color=colors)
axes[0, 1].set_title("TN3K - Jaccard", fontweight='bold')
axes[0, 1].set_ylim([0, 1])
for i, v in enumerate(tn3k_jaccard):
    axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

axes[0, 2].bar(methods, tn3k_f1, color=colors)
axes[0, 2].set_title("TN3K - F1", fontweight='bold')
axes[0, 2].set_ylim([0, 1])
for i, v in enumerate(tn3k_f1):
    axes[0, 2].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

# DDTI row
axes[1, 0].bar(methods, ddti_dice, color=colors)
axes[1, 0].set_title("DDTI - Dice", fontweight='bold')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].set_ylabel("Score")
for i, v in enumerate(ddti_dice):
    axes[1, 0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

axes[1, 1].bar(methods, ddti_jaccard, color=colors)
axes[1, 1].set_title("DDTI - Jaccard", fontweight='bold')
axes[1, 1].set_ylim([0, 1])
for i, v in enumerate(ddti_jaccard):
    axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

axes[1, 2].bar(methods, ddti_f1, color=colors)
axes[1, 2].set_title("DDTI - F1", fontweight='bold')
axes[1, 2].set_ylim([0, 1])
for i, v in enumerate(ddti_f1):
    axes[1, 2].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

# TN5000 row
axes[2, 0].bar(methods, tn5000_dice, color=colors)
axes[2, 0].set_title("TN5000 - Dice", fontweight='bold')
axes[2, 0].set_ylim([0, 1])
axes[2, 0].set_ylabel("Score")
for i, v in enumerate(tn5000_dice):
    axes[2, 0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

axes[2, 1].bar(methods, tn5000_jaccard, color=colors)
axes[2, 1].set_title("TN5000 - Jaccard", fontweight='bold')
axes[2, 1].set_ylim([0, 1])
for i, v in enumerate(tn5000_jaccard):
    axes[2, 1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

axes[2, 2].bar(methods, tn5000_f1, color=colors)
axes[2, 2].set_title("TN5000 - F1", fontweight='bold')
axes[2, 2].set_ylim([0, 1])
for i, v in enumerate(tn5000_f1):
    axes[2, 2].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

for ax in axes.flatten():
    ax.set_xlabel("Method")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "performance_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ Performance comparison figure saved to results/performance_comparison.png")

if skip_finetune:
    print("⏭️ MEDSAM_SKIP_FINETUNE=1：跳過『預訓練 vs Fine-tune』額外對比流程")
    raise SystemExit(0)


# %%
# ============================================================================
# 📈 預訓練 vs Fine-tune 性能對比分析
# ============================================================================

print("\n" + "="*90)
print("📈 預訓練 vs Fine-tune 性能對比")
print("="*90)

# 載入預訓練模型進行對比評估
print("\n🔄 載入預訓練模型進行對比評估...")
model_pretrained = SamModel.from_pretrained(MODEL_CONFIG["model_id"])
model_pretrained = model_pretrained.to(device)
model_pretrained.eval()

if Path(pretrained_model_path).exists():
    _load_state_dict_compat(model_pretrained, Path(pretrained_model_path), map_location=device)
    print(f"✅ 預訓練模型已加載")

# 評估預訓練模型在三個測試集上的性能
print("\n[預訓練模型] 評估 TN3K 數據集...")
tn3k_pretrained_baseline_results, tn3k_pretrained_baseline_stats = evaluate_dataset(
    tn3k_dataset, model_pretrained, processor, ood_detector, tta_predictor,
    device, use_tta=False, use_ood=False, dataset_name="TN3K-Pretrained-Baseline"
)

print("\n[預訓練模型] 評估 DDTI 數據集...")
ddti_pretrained_baseline_results, ddti_pretrained_baseline_stats = evaluate_dataset(
    ddti_dataset, model_pretrained, processor, ood_detector, tta_predictor,
    device, use_tta=False, use_ood=False, dataset_name="DDTI-Pretrained-Baseline"
)

print("\n[預訓練模型] 評估 TN5000 數據集...")
tn5000_pretrained_baseline_results, tn5000_pretrained_baseline_stats = evaluate_dataset(
    tn5000_dataset, model_pretrained, processor, ood_detector, tta_predictor,
    device, use_tta=False, use_ood=False, dataset_name="TN5000-Pretrained-Baseline"
)

# 重新加載 fine-tune 模型進行最終評估
print("\n🔄 重新加載 Fine-tune 模型...")
model_finetuned = SamModel.from_pretrained(MODEL_CONFIG["model_id"])
model_finetuned = model_finetuned.to(device)
model_finetuned.eval()
_load_state_dict_compat(model_finetuned, Path(finetuned_model_path), map_location=device)
print(f"✅ Fine-tune 模型已加載")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 性能對比表格
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "="*120)
print("📊 預訓練 vs Fine-tune 性能對比（Baseline - 無 TTA、無 OOD）")
print("="*120)

comparison_results = []

datasets_info = [
    ("TN3K", tn3k_pretrained_baseline_stats, tn3k_baseline_stats),
    ("DDTI", ddti_pretrained_baseline_stats, ddti_baseline_stats),
    ("TN5000", tn5000_pretrained_baseline_stats, tn5000_baseline_stats),
]

for dataset_name, pretrained_stats, finetuned_stats in datasets_info:
    print(f"\n{'='*50}")
    print(f"📊 {dataset_name} 數據集")
    print(f"{'='*50}")
    
    metrics = ["dice", "jaccard", "precision", "recall", "f1"]
    
    print(f"\n{'指標':<15} {'預訓練':<20} {'Fine-tune':<20} {'改進':<15}")
    print("-" * 70)
    
    for metric in metrics:
        pretrained_val = pretrained_stats.get(f"mean_{metric}", 0)
        finetuned_val = finetuned_stats.get(f"mean_{metric}", 0)
        improvement = finetuned_val - pretrained_val
        improvement_pct = (improvement / max(pretrained_val, 1e-8)) * 100
        
        improvement_str = f"+{improvement:.4f} ({improvement_pct:+.2f}%)" if improvement >= 0 else f"{improvement:.4f} ({improvement_pct:.2f}%)"
        
        print(f"{metric.upper():<15} {pretrained_val:<20.4f} {finetuned_val:<20.4f} {improvement_str:<15}")
        
        comparison_results.append({
            "dataset": dataset_name,
            "metric": metric.upper(),
            "pretrained": float(pretrained_val),
            "finetuned": float(finetuned_val),
            "improvement": float(improvement),
            "improvement_pct": float(improvement_pct),
        })

# 保存對比結果
comparison_results_file = OUTPUT_DIR / "pretrained_vs_finetuned_comparison.json"
with open(comparison_results_file, "w") as f:
    json.dump(comparison_results, f, indent=2, default=json_default_serializer)
print(f"\n✅ 對比結果已保存到 {comparison_results_file}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 性能對比可視化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("預訓練 vs Fine-tune 性能對比 (Dice Score)", fontsize=14, fontweight='bold')

datasets = ["TN3K", "DDTI", "TN5000"]
pretrained_dice = [
    tn3k_pretrained_baseline_stats['mean_dice'],
    ddti_pretrained_baseline_stats['mean_dice'],
    tn5000_pretrained_baseline_stats['mean_dice']
]
finetuned_dice = [
    tn3k_baseline_stats['mean_dice'],
    ddti_baseline_stats['mean_dice'],
    tn5000_baseline_stats['mean_dice']
]

colors_pretrained = "#3498db"
colors_finetuned = "#2ecc71"

for idx, dataset_name in enumerate(datasets):
    ax = axes[idx]
    
    categories = ["Pretrained", "Fine-tuned"]
    values = [pretrained_dice[idx], finetuned_dice[idx]]
    bars = ax.bar(categories, values, color=[colors_pretrained, colors_finetuned], alpha=0.8, edgecolor='black')
    
    # 添加數值標籤
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 添加改進百分比
    improvement_pct = ((finetuned_dice[idx] - pretrained_dice[idx]) / max(pretrained_dice[idx], 1e-8)) * 100
    ax.text(0.5, max(values) * 0.95, f'Improvement: {improvement_pct:+.2f}%',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_title(f"{dataset_name}", fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_ylabel("Dice Score")
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pretrained_vs_finetuned_dice.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ 對比圖已保存到 results/pretrained_vs_finetuned_dice.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 總結性能改進
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "="*90)
print("🎯 Fine-tune 效果總結")
print("="*90)

print("\n📊 各資料集的 Dice Score 改進：")
print("-" * 50)

total_improvement = 0
for idx, (dataset_name, pretrained_stats, finetuned_stats) in enumerate(datasets_info):
    pretrained_dice = pretrained_stats.get("mean_dice", 0)
    finetuned_dice = finetuned_stats.get("mean_dice", 0)
    improvement = finetuned_dice - pretrained_dice
    improvement_pct = (improvement / max(pretrained_dice, 1e-8)) * 100
    total_improvement += improvement_pct
    
    print(f"  {dataset_name:<10} {pretrained_dice:.4f} → {finetuned_dice:.4f}  "
          f"改進: {improvement:+.4f} ({improvement_pct:+.2f}%)")

avg_improvement = total_improvement / len(datasets_info)
print(f"\n  🏆 平均改進: {avg_improvement:+.2f}%")

print("\n✅ Fine-tune 流程完成！模型已保存並評估完畢。")
print("="*90 + "\n")



