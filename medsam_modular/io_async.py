import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


class AsyncFileWriter:
    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="medsam-io")
        self._futures: List[Future] = []
        self._lock = threading.Lock()

    def submit_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        future = self._executor.submit(target.write_text, content, encoding=encoding)
        with self._lock:
            self._futures.append(future)

    def submit_npy(self, path: Path, value: np.ndarray) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        future = self._executor.submit(np.save, target, value)
        with self._lock:
            self._futures.append(future)

    def flush(self) -> None:
        with self._lock:
            futures = list(self._futures)
            self._futures.clear()
        for fut in futures:
            fut.result()

    def close(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)


_GLOBAL_WRITER: Optional[AsyncFileWriter] = None
_GLOBAL_LOCK = threading.Lock()


def get_global_async_writer() -> AsyncFileWriter:
    global _GLOBAL_WRITER
    with _GLOBAL_LOCK:
        if _GLOBAL_WRITER is None:
            _GLOBAL_WRITER = AsyncFileWriter(max_workers=1)
        return _GLOBAL_WRITER


def shutdown_global_async_writer() -> None:
    global _GLOBAL_WRITER
    with _GLOBAL_LOCK:
        writer = _GLOBAL_WRITER
        _GLOBAL_WRITER = None
    if writer is not None:
        writer.close()
