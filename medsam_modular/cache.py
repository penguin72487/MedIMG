import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from medsam_modular.io_async import get_global_async_writer


class PredictionCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ram_max_entries = max(0, int(os.getenv("MEDSAM_CACHE_RAM_ENTRIES", "256")))
        self._ram: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = Lock()
        self._async_disk_write = os.getenv("MEDSAM_CACHE_ASYNC_WRITE", "1").strip().lower() in {"1", "true", "yes", "y", "on"}

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

    def _put_ram(self, key: str, value: np.ndarray) -> None:
        if self._ram_max_entries <= 0:
            return
        with self._lock:
            self._ram[key] = value
            self._ram.move_to_end(key)
            while len(self._ram) > self._ram_max_entries:
                self._ram.popitem(last=False)


def make_cache_key(dataset_name: str, sample_name: str, bbox: list, mode: str, image_size: int = 0) -> str:
    size_tag = f"|sz{image_size}" if image_size > 0 else ""
    return f"{dataset_name}|{sample_name}|{tuple(int(v) for v in bbox)}|{mode}{size_tag}"
