import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".xml")


def compute_bbox_from_mask_np(mask_np: np.ndarray, jitter: int = 10) -> List[int]:
    y_idx, x_idx = np.where(mask_np > 0)
    if len(x_idx) == 0 or len(y_idx) == 0:
        return [0, 0, int(mask_np.shape[1]), int(mask_np.shape[0])]

    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    h, w = mask_np.shape

    x_min = max(0, int(x_min) - jitter)
    x_max = min(w - 1, int(x_max) + jitter)
    y_min = max(0, int(y_min) - jitter)
    y_max = min(h - 1, int(y_max) + jitter)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def canonical_split_id(raw_id: Any) -> str:
    sample_id = str(raw_id).strip()
    if not sample_id:
        return ""

    lower = sample_id.lower()
    for suffix in IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return sample_id[: -len(suffix)].strip()
    return sample_id


def read_split_ids(split_file: Optional[Path]) -> Optional[Set[str]]:
    if split_file is None or not split_file.exists():
        return None

    ids: Set[str] = set()
    for line in split_file.read_text(encoding="utf-8").splitlines():
        cid = canonical_split_id(line)
        if cid:
            ids.add(cid)
    return ids


def split_file(split_root: Path, dataset_name: str, split_name: str) -> Optional[Path]:
    path = split_root / dataset_name / f"{split_name}.txt"
    return path if path.exists() else None


class TN3KDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "test", image_size: int = 512, split_ids: Optional[Set[str]] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.split_ids = split_ids
        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    @staticmethod
    def _image_candidates(image_dir: Path) -> List[Path]:
        files: List[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            files.extend(sorted(image_dir.glob(pattern)))
        return files

    def _candidate_dirs(self) -> List[Tuple[Path, Path]]:
        return [
            (self.root_dir / f"{self.split}-image", self.root_dir / f"{self.split}-mask"),
            (self.root_dir / self.split / "images", self.root_dir / self.split / "masks"),
            (self.root_dir / "train-image", self.root_dir / "train-mask"),
            (self.root_dir / "trainval-image", self.root_dir / "trainval-mask"),
            (self.root_dir / "test-image", self.root_dir / "test-mask"),
            (self.root_dir / "test" / "images", self.root_dir / "test" / "masks"),
        ]

    @staticmethod
    def _resolve_mask_path(mask_dir: Path, stem: str) -> Optional[Path]:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _resolve_pair(self, sample_id: str) -> Optional[Tuple[Path, Path]]:
        for image_dir, mask_dir in self._candidate_dirs():
            if not image_dir.exists() or not mask_dir.exists():
                continue
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                img_path = image_dir / f"{sample_id}{ext}"
                if not img_path.exists():
                    continue
                mask_path = self._resolve_mask_path(mask_dir, sample_id)
                if mask_path is not None:
                    return img_path, mask_path
        return None

    def _load_samples(self) -> None:
        if self.split_ids is not None:
            for sid in sorted(self.split_ids):
                pair = self._resolve_pair(sid)
                if pair is None:
                    continue
                img_path, mask_path = pair
                self.samples.append({"image_path": img_path, "mask_path": mask_path, "name": sid})
            return

        for image_dir, mask_dir in self._candidate_dirs():
            if not image_dir.exists() or not mask_dir.exists():
                continue
            image_files = self._image_candidates(image_dir)
            if not image_files:
                continue
            for img_file in image_files:
                mask_file = self._resolve_mask_path(mask_dir, img_file.stem)
                if mask_file is None:
                    continue
                self.samples.append({"image_path": img_file, "mask_path": mask_file, "name": img_file.stem})
            if self.samples:
                return

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
            "mask": torch.tensor(mask_np.astype(np.float32), dtype=torch.float32),
            "bbox": bbox,
            "name": sample["name"],
        }


class DDTIDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "test", image_size: int = 512, split_ids: Optional[Set[str]] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.split_ids = split_ids
        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    @staticmethod
    def _resolve_image_path(image_root: Path, case_stem: str, img_idx: int, img_text: str = "") -> Optional[Path]:
        candidates: List[Path] = []
        if img_text:
            normalized = img_text.strip()
            if normalized:
                candidates.append(image_root / normalized)
                candidates.append(image_root / f"{normalized}.jpg")
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidates.append(image_root / f"{case_stem}_{img_idx}{ext}")
            candidates.append(image_root / f"{img_idx}{ext}")
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_samples(self) -> None:
        split_ann_root = self.root_dir / self.split / "annotations"
        split_img_root = self.root_dir / self.split / "images"

        if split_ann_root.exists() and split_img_root.exists():
            ann_root = split_ann_root
            image_root = split_img_root
        else:
            ann_root = self.root_dir
            image_root = self.root_dir

        for xml_file in sorted(ann_root.glob("*.xml")):
            case_stem = xml_file.stem
            try:
                case_id: Any = int(case_stem)
            except Exception:
                case_id = case_stem

            root = ET.parse(xml_file).getroot()
            for mark in root.findall(".//mark"):
                img_elem = mark.find("image")
                svg_elem = mark.find("svg")
                if img_elem is None or svg_elem is None:
                    continue

                try:
                    img_idx = int(str(img_elem.text).strip())
                except Exception:
                    continue

                sample_name = f"{case_stem}_{img_idx}"
                if self.split_ids is not None and sample_name not in self.split_ids:
                    continue

                img_path = self._resolve_image_path(image_root, case_stem, img_idx, str(img_elem.text or ""))
                if img_path.exists():
                    self.samples.append(
                        {
                            "image_path": img_path,
                            "case_id": case_id,
                            "img_idx": img_idx,
                            "svg": svg_elem.text,
                            "name": sample_name,
                        }
                    )

    def _svg_to_mask(self, svg_str: str, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            annotations = json.loads(svg_str)
            for ann in annotations:
                if ann.get("regionType") != "freehand":
                    continue
                points = ann.get("points", [])
                if len(points) <= 2:
                    continue
                pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
        except Exception:
            pass
        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        mask = self._svg_to_mask(sample["svg"], self.image_size, self.image_size)
        bbox = compute_bbox_from_mask_np(mask)
        return {
            "image": image,
            "mask": torch.tensor(mask.astype(np.float32), dtype=torch.float32),
            "bbox": bbox,
            "name": sample["name"],
        }


class TN5000Dataset(Dataset):
    def __init__(self, root_dir: str, split: str = "test", image_size: int = 512, split_ids: Optional[Set[str]] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.split_ids = split_ids
        self.samples: List[Dict[str, Any]] = []
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

    def _load_samples(self) -> None:
        voc_ann_dir = self.root_dir / "Annotations"
        voc_image_dir = self.root_dir / "JPEGImages"
        split_ann_dir = self.root_dir / self.split / "annotations"
        split_image_dir = self.root_dir / self.split / "images"

        if split_ann_dir.exists() and split_image_dir.exists():
            ann_dir = split_ann_dir
            image_dir = split_image_dir
            default_split_file = None
        else:
            ann_dir = voc_ann_dir
            image_dir = voc_image_dir
            default_split_file = self.root_dir / "ImageSets" / "Main" / f"{self.split}.txt"

        if self.split_ids is not None:
            image_ids = sorted(self.split_ids)
        elif default_split_file is not None and default_split_file.exists():
            image_ids = [line.strip() for line in default_split_file.read_text().splitlines() if line.strip()]
        else:
            image_ids = [p.stem for p in sorted(ann_dir.glob("*.xml"))]

        for image_id in image_ids:
            xml_path = ann_dir / f"{image_id}.xml"
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                cand = image_dir / f"{image_id}{ext}"
                if cand.exists():
                    img_path = cand
                    break

            if not xml_path.exists() or img_path is None:
                continue

            try:
                root = ET.parse(xml_path).getroot()
                width = int(float(root.findtext("size/width", default="0") or 0))
                height = int(float(root.findtext("size/height", default="0") or 0))
                if width <= 0 or height <= 0:
                    with Image.open(img_path) as im:
                        width, height = im.size
                boxes = self._parse_boxes(root)
                if width > 0 and height > 0 and boxes:
                    self.samples.append(
                        {
                            "image_id": image_id,
                            "image_path": img_path,
                            "width": width,
                            "height": height,
                            "boxes": boxes,
                        }
                    )
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        sx = self.image_size / max(sample["width"], 1)
        sy = self.image_size / max(sample["height"], 1)

        for xmin, ymin, xmax, ymax in sample["boxes"]:
            x1 = int(np.clip(round((xmin - 1) * sx), 0, self.image_size - 1))
            y1 = int(np.clip(round((ymin - 1) * sy), 0, self.image_size - 1))
            x2 = int(np.clip(round((xmax - 1) * sx), 0, self.image_size - 1))
            y2 = int(np.clip(round((ymax - 1) * sy), 0, self.image_size - 1))
            if x2 >= x1 and y2 >= y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 1

        bbox = compute_bbox_from_mask_np(mask)
        return {
            "image": image,
            "mask": torch.tensor(mask.astype(np.float32), dtype=torch.float32),
            "bbox": bbox,
            "name": sample["image_id"],
        }


def build_dataset(dataset_name: str, root_dir: str, split_name: str, image_size: int, split_ids: Optional[Set[str]]) -> Dataset:
    if dataset_name == "TN3K":
        return TN3KDataset(root_dir=root_dir, split=split_name, image_size=image_size, split_ids=split_ids)
    if dataset_name == "DDTI":
        return DDTIDataset(root_dir=root_dir, split=split_name, image_size=image_size, split_ids=split_ids)
    if dataset_name == "TN5000":
        return TN5000Dataset(root_dir=root_dir, split=split_name, image_size=image_size, split_ids=split_ids)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def prepare_datasets_by_split(
    data_paths: Dict[str, str],
    split_root: Path,
    split_name: str,
    image_size: int,
    dataset_names: Optional[List[str]] = None,
) -> Dict[str, Dataset]:
    names = dataset_names or ["TN3K", "DDTI", "TN5000"]
    prepared: Dict[str, Dataset] = {}

    for dataset_name in names:
        ids = read_split_ids(split_file(split_root, dataset_name, split_name))
        prepared[dataset_name] = build_dataset(
            dataset_name=dataset_name,
            root_dir=data_paths[dataset_name],
            split_name=split_name,
            image_size=image_size,
            split_ids=ids,
        )
    return prepared
