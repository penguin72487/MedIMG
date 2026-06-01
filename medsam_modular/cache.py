import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from medsam_modular.config import ENV_DEFAULTS
from medsam_modular.io_async import get_global_async_writer


CACHE_VERSION = "v2"


class PredictionCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.jsonl"
        self._ram_max_entries = max(0, int(os.getenv("MEDSAM_CACHE_RAM_ENTRIES", ENV_DEFAULTS["MEDSAM_CACHE_RAM_ENTRIES"])))
        self._ram: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = Lock()
        self._async_disk_write = os.getenv("MEDSAM_CACHE_ASYNC_WRITE", ENV_DEFAULTS["MEDSAM_CACHE_ASYNC_WRITE"]).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            cached_ram = self._ram.get(key)
            if cached_ram is not None:
                self._ram.move_to_end(key)
                return cached_ram

        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            value = np.load(path)
            self._put_ram(key, value)
            return value
        except Exception:
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        self._put_ram(key, value)
        path = self._key_to_path(key)
        if self._async_disk_write:
            get_global_async_writer().submit_npy(path, value)
        else:
            np.save(path, value)
        self._append_manifest(key=key, path=path, value=value)

    def _put_ram(self, key: str, value: np.ndarray) -> None:
        if self._ram_max_entries <= 0:
            return
        with self._lock:
            self._ram[key] = value
            self._ram.move_to_end(key)
            while len(self._ram) > self._ram_max_entries:
                self._ram.popitem(last=False)

    def _append_manifest(self, key: str, path: Path, value: np.ndarray) -> None:
        payload = {
            "cache_version": CACHE_VERSION,
            "key": key,
            "file": path.name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        line = json.dumps(payload, ensure_ascii=False)
        if self._async_disk_write:
            get_global_async_writer().submit_text(self.manifest_path, line + "\n", encoding="utf-8", append=True)
        else:
            with self.manifest_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")


def make_cache_key(
    dataset_name: str,
    sample_name: str,
    bbox: list,
    mode: str,
    image_size: int = 0,
    model_hash: str = "",
    tta_aug_set: str = "",
    fusion: str = "",
) -> str:
    size_tag = f"|sz{image_size}" if image_size > 0 else ""
    model_tag = f"|mh:{model_hash}" if model_hash else ""
    aug_tag = f"|aug:{tta_aug_set}" if tta_aug_set else ""
    fusion_tag = f"|fuse:{fusion}" if fusion else ""
    return f"{CACHE_VERSION}|{dataset_name}|{sample_name}|{tuple(int(v) for v in bbox)}|{mode}{size_tag}{model_tag}{aug_tag}{fusion_tag}"
