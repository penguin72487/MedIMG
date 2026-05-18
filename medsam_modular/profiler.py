import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch


_ACTIVE_PROFILER: Optional["PerformanceProfiler"] = None


def set_active_profiler(profiler: Optional["PerformanceProfiler"]) -> None:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = profiler


def get_active_profiler() -> Optional["PerformanceProfiler"]:
    return _ACTIVE_PROFILER


@dataclass
class _TimingStat:
    total_sec: float = 0.0
    exclusive_sec: float = 0.0
    count: int = 0
    min_sec: float = float("inf")
    max_sec: float = 0.0

    def add(self, elapsed_sec: float, n: int = 1, exclusive_sec: Optional[float] = None) -> None:
        e = float(max(0.0, elapsed_sec))
        exclusive = float(max(0.0, e if exclusive_sec is None else exclusive_sec))
        self.total_sec += e
        self.exclusive_sec += exclusive
        self.count += int(max(1, n))
        self.min_sec = min(self.min_sec, e)
        self.max_sec = max(self.max_sec, e)

    def to_dict(self, total_profiled_sec: float, total_exclusive_sec: float) -> Dict[str, float]:
        avg_sec = self.total_sec / max(1, self.count)
        exclusive_avg_sec = self.exclusive_sec / max(1, self.count)
        ratio = self.exclusive_sec / max(1e-12, total_exclusive_sec)
        return {
            "total_sec": float(self.total_sec),
            "exclusive_sec": float(self.exclusive_sec),
            "count": int(self.count),
            "avg_ms": float(avg_sec * 1000.0),
            "exclusive_avg_ms": float(exclusive_avg_sec * 1000.0),
            "min_ms": float((self.min_sec if np.isfinite(self.min_sec) else 0.0) * 1000.0),
            "max_ms": float(self.max_sec * 1000.0),
            "ratio": float(ratio),
        }


class PerformanceProfiler:
    def __init__(self, enabled: bool = True, run_name: str = "run"):
        self.enabled = bool(enabled)
        self.run_name = str(run_name)
        self.start_sec = time.perf_counter()
        self.timings: Dict[str, _TimingStat] = {}
        self.metadata: Dict[str, Any] = {}
        self.counters: Dict[str, float] = {}
        self.output_path: Optional[Path] = None
        self._section_stack: list[Dict[str, Any]] = []

    def set_metadata(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.metadata[str(key)] = value

    def add_counter(self, key: str, value: float) -> None:
        if not self.enabled:
            return
        self.counters[str(key)] = float(value)

    def inc_counter(self, key: str, delta: float = 1.0) -> None:
        if not self.enabled:
            return
        self.counters[str(key)] = float(self.counters.get(str(key), 0.0) + float(delta))

    def record_duration(self, section: str, elapsed_sec: float, count: int = 1, exclusive_sec: Optional[float] = None) -> None:
        if not self.enabled:
            return
        name = str(section)
        if name not in self.timings:
            self.timings[name] = _TimingStat()
        self.timings[name].add(elapsed_sec, n=count, exclusive_sec=exclusive_sec)

    def configure_output(self, output_path: Optional[Path]) -> None:
        self.output_path = output_path

    def flush(self) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.output_path is None:
            return None
        return self.save_json(self.output_path)

    @contextmanager
    def section(self, section: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        frame = {"section": str(section), "start": time.perf_counter(), "child_sec": 0.0}
        self._section_stack.append(frame)
        try:
            yield
        finally:
            finished = self._section_stack.pop()
            elapsed_sec = time.perf_counter() - float(finished["start"])
            child_sec = float(finished.get("child_sec", 0.0))
            exclusive_sec = max(0.0, elapsed_sec - child_sec)
            self.record_duration(section, elapsed_sec, exclusive_sec=exclusive_sec)
            if self._section_stack:
                self._section_stack[-1]["child_sec"] += elapsed_sec

    @contextmanager
    def section_and_flush(self, section: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with self.section(section):
            yield
        self.flush()

    def snapshot_cuda(self, tag: str) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        try:
            self.metadata[f"cuda_{tag}_allocated_mb"] = float(torch.cuda.memory_allocated() / (1024.0 ** 2))
            self.metadata[f"cuda_{tag}_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024.0 ** 2))
            self.metadata[f"cuda_{tag}_max_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
        except Exception:
            return

    def _recommendations(self, ordered_sections: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        rec: Dict[str, str] = {}
        for name in ordered_sections.keys():
            lname = name.lower()
            if "data" in lname or "loader" in lname:
                rec[name] = "資料載入占比高：可增加 MEDSAM_EVAL_WORKERS、檢查磁碟 I/O、開啟快取。"
            elif "tta" in lname or "augment" in lname:
                rec[name] = "TTA 占比高：可改 --tta-fast、減少 augmentations、降低 tta chunk size。"
            elif "inference" in lname or "forward" in lname:
                rec[name] = "模型推論占比高：可調整 batch/chunk、確認 compile 與 AMP 啟用、降低輸入尺寸。"
            elif "backward" in lname or "optimizer" in lname:
                rec[name] = "訓練反向傳播占比高：可調 grad_accum、凍結更多層、降低訓練解析度。"
            elif "metrics" in lname or "post" in lname:
                rec[name] = "後處理占比高：可向量化 metric/後處理，減少 Python 迴圈。"
        return rec

    def _analyze_headroom(self, top_bottlenecks: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not top_bottlenecks:
            return {
                "status": "unknown",
                "message": "沒有可分析的 bottleneck 資料。",
                "confidence": 0.0,
            }

        top = top_bottlenecks[0]
        top_ratio = float(top.get("ratio", 0.0))
        compute_ratio = sum(
            float(item.get("ratio", 0.0))
            for item in top_bottlenecks
            if any(token in str(item.get("section", "")).lower() for token in ("forward", "inference", "backward", "compile"))
        )
        overhead_ratio = sum(
            float(item.get("ratio", 0.0))
            for item in top_bottlenecks
            if any(token in str(item.get("section", "")).lower() for token in ("data", "loader", "post", "metrics", "save", "cache", "augment", "preprocess", "device_move"))
        )

        if compute_ratio >= 0.8 and top_ratio >= 0.35:
            return {
                "status": "near_limit",
                "message": "大部分時間已集中在核心計算路徑，剩餘優化空間有限，較可能需要更快硬體、較小模型或降低輸入規模。",
                "confidence": min(1.0, 0.6 + compute_ratio * 0.4),
            }
        if overhead_ratio >= 0.3:
            return {
                "status": "has_headroom",
                "message": "仍有明顯非核心計算開銷，可優先優化資料載入、前後處理、TTA 或存檔流程。",
                "confidence": min(1.0, 0.5 + overhead_ratio * 0.5),
            }
        return {
            "status": "mixed",
            "message": "目前瓶頸分散在多個子流程，還有優化空間，但需依 top bottlenecks 逐項處理。",
            "confidence": 0.55,
        }

    def to_dict(self) -> Dict[str, Any]:
        total_wall_sec = float(time.perf_counter() - self.start_sec)
        total_profiled_sec = float(sum(stat.total_sec for stat in self.timings.values()))
        total_exclusive_sec = float(sum(stat.exclusive_sec for stat in self.timings.values()))

        ordered = sorted(self.timings.items(), key=lambda kv: kv[1].exclusive_sec, reverse=True)
        sections = {
            name: stat.to_dict(
                total_profiled_sec=max(1e-12, total_profiled_sec),
                total_exclusive_sec=max(1e-12, total_exclusive_sec),
            )
            for name, stat in ordered
        }

        top_bottlenecks = []
        for name, payload in list(sections.items())[:10]:
            top_bottlenecks.append(
                {
                    "section": name,
                    "total_sec": payload["total_sec"],
                    "exclusive_sec": payload["exclusive_sec"],
                    "ratio": payload["ratio"],
                    "avg_ms": payload["avg_ms"],
                    "exclusive_avg_ms": payload["exclusive_avg_ms"],
                }
            )

        limit_analysis = self._analyze_headroom(top_bottlenecks)

        return {
            "run_name": self.run_name,
            "enabled": self.enabled,
            "total_wall_sec": total_wall_sec,
            "total_profiled_sec": total_profiled_sec,
            "total_exclusive_sec": total_exclusive_sec,
            "metadata": self.metadata,
            "counters": self.counters,
            "sections": sections,
            "top_bottlenecks": top_bottlenecks,
            "recommendations": self._recommendations(sections),
            "optimization_limit_analysis": limit_analysis,
        }

    def save_json(self, output_path: Path) -> Dict[str, Any]:
        payload = self.to_dict()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
