import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


class PredictionCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def get(self, key: str) -> Optional[np.ndarray]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        path = self._key_to_path(key)
        np.save(path, value)


def make_cache_key(dataset_name: str, sample_name: str, bbox: list, mode: str) -> str:
    return f"{dataset_name}|{sample_name}|{tuple(int(v) for v in bbox)}|{mode}"
