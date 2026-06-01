"""Shared data structures for MedSAM pipeline samples."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


@dataclass(frozen=True)
class SegSample:
    image: Image.Image
    mask: torch.Tensor
    bbox: List[int]
    name: str
    gt_boxes: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "image": self.image,
            "mask": self.mask,
            "bbox": self.bbox,
            "name": self.name,
        }
        if self.gt_boxes is not None:
            payload["gt_boxes"] = self.gt_boxes
        return payload


__all__ = ["SegSample"]
