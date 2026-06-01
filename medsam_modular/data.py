import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from medsam_modular.types import SegSample


def get_active_profiler() -> None:
    return None


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


def split_connected_component_masks(mask_np: np.ndarray) -> List[np.ndarray]:
    mask_u8 = (mask_np > 0).astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components: List[Tuple[int, int, np.ndarray]] = []
    for label_idx in range(1, int(num_labels)):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        comp = (labels == label_idx).astype(np.uint8)
        components.append((y, x, comp))

    components.sort(key=lambda item: (item[0], item[1]))
    return [c for _, _, c in components]


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
    def __init__(self, root_dir: str, split: str = "test", image_size: int = 1024, split_ids: Optional[Set[str]] = None):
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
        def _append_expanded_samples(img_path: Path, mask_path: Path, base_name: str) -> None:
            try:
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
                mask_np = (np.array(mask) > 127).astype(np.uint8)
            except Exception:
                return

            components = split_connected_component_masks(mask_np)
            if not components:
                return

            multi = len(components) > 1
            for comp_idx, comp_mask in enumerate(components):
                comp_name = f"{base_name}#cc{comp_idx + 1}" if multi else base_name
                self.samples.append(
                    {
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "name": comp_name,
                        "component_index": comp_idx,
                        "bbox": compute_bbox_from_mask_np(comp_mask),
                    }
                )

        if self.split_ids is not None:
            for sid in sorted(self.split_ids):
                pair = self._resolve_pair(sid)
                if pair is None:
                    continue
                img_path, mask_path = pair
                _append_expanded_samples(img_path=img_path, mask_path=mask_path, base_name=sid)
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
                _append_expanded_samples(img_path=img_file, mask_path=mask_file, base_name=img_file.stem)
            if self.samples:
                return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        profiler = get_active_profiler()
        t0 = cv2.getTickCount() / cv2.getTickFrequency() if profiler is not None and profiler.enabled else 0.0
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        mask_np = (np.array(mask) > 127).astype(np.uint8)
        comp_masks = split_connected_component_masks(mask_np)
        comp_idx = int(sample.get("component_index", 0))
        if comp_masks and 0 <= comp_idx < len(comp_masks):
            mask_np = comp_masks[comp_idx]
        bbox = sample.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = compute_bbox_from_mask_np(mask_np)
        if profiler is not None and profiler.enabled:
            profiler.record_duration("data.TN3KDataset.__getitem__", (cv2.getTickCount() / cv2.getTickFrequency()) - t0)
        return SegSample(
            image=image,
            mask=torch.tensor(mask_np.astype(np.float32), dtype=torch.float32),
            bbox=bbox,
            name=sample["name"],
        ).to_dict()


class DDTIDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "test", image_size: int = 1024, split_ids: Optional[Set[str]] = None):
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
                if img_path is None or not img_path.exists():
                    continue

                try:
                    with Image.open(img_path) as im:
                        source_size = im.size
                except Exception:
                    source_size = None

                mask = self._svg_to_mask(
                    str(svg_elem.text or ""),
                    self.image_size,
                    self.image_size,
                    source_size=source_size,
                )
                components = split_connected_component_masks(mask)
                if not components:
                    continue

                multi = len(components) > 1
                for comp_idx, comp_mask in enumerate(components):
                    comp_name = f"{sample_name}#cc{comp_idx + 1}" if multi else sample_name
                    self.samples.append(
                        {
                            "image_path": img_path,
                            "case_id": case_id,
                            "img_idx": img_idx,
                            "svg": svg_elem.text,
                            "source_size": source_size,
                            "name": comp_name,
                            "component_index": comp_idx,
                            "bbox": compute_bbox_from_mask_np(comp_mask),
                        }
                    )

    def _svg_to_mask(self, svg_str: str, h: int, w: int, source_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            annotations = json.loads(svg_str)
            scale_x = 1.0
            scale_y = 1.0
            if source_size is not None:
                source_w, source_h = source_size
                if source_w > 0 and source_h > 0:
                    scale_x = float(w) / float(source_w)
                    scale_y = float(h) / float(source_h)
            for ann in annotations:
                if ann.get("regionType") != "freehand":
                    continue
                points = ann.get("points", [])
                if len(points) <= 2:
                    continue
                pts = np.array(
                    [
                        [
                            int(np.clip(round(float(p["x"]) * scale_x), 0, w - 1)),
                            int(np.clip(round(float(p["y"]) * scale_y), 0, h - 1)),
                        ]
                        for p in points
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts], 1)
        except Exception:
            pass
        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        profiler = get_active_profiler()
        t0 = cv2.getTickCount() / cv2.getTickFrequency() if profiler is not None and profiler.enabled else 0.0
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        source_size = sample.get("source_size")
        if source_size is None:
            source_size = getattr(image, "size", None)
        mask = self._svg_to_mask(sample["svg"], self.image_size, self.image_size, source_size=source_size)
        comp_masks = split_connected_component_masks(mask)
        comp_idx = int(sample.get("component_index", 0))
        if comp_masks and 0 <= comp_idx < len(comp_masks):
            mask = comp_masks[comp_idx]
        bbox = sample.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = compute_bbox_from_mask_np(mask)
        if profiler is not None and profiler.enabled:
            profiler.record_duration("data.DDTIDataset.__getitem__", (cv2.getTickCount() / cv2.getTickFrequency()) - t0)
        return SegSample(
            image=image,
            mask=torch.tensor(mask.astype(np.float32), dtype=torch.float32),
            bbox=bbox,
            name=sample["name"],
        ).to_dict()


class TN5000Dataset(Dataset):
    def __init__(self, root_dir: str, split: str = "test", image_size: int = 1024, split_ids: Optional[Set[str]] = None):
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
                xml_width = int(float(root.findtext("size/width", default="0") or 0))
                xml_height = int(float(root.findtext("size/height", default="0") or 0))
                with Image.open(img_path) as im:
                    image_width, image_height = im.size
                if image_width > 0 and image_height > 0:
                    width, height = image_width, image_height
                else:
                    width, height = xml_width, xml_height
                boxes = self._parse_boxes(root)
                if width > 0 and height > 0 and boxes:
                    self.samples.append(
                        {
                            "image_id": image_id,
                            "image_path": img_path,
                            "width": width,
                            "height": height,
                            "xml_width": xml_width,
                            "xml_height": xml_height,
                            "boxes": boxes,
                        }
                    )
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        profiler = get_active_profiler()
        t0 = cv2.getTickCount() / cv2.getTickFrequency() if profiler is not None and profiler.enabled else 0.0
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        gt_boxes: List[List[int]] = []
        sx = self.image_size / max(sample["width"], 1)
        sy = self.image_size / max(sample["height"], 1)

        for xmin, ymin, xmax, ymax in sample["boxes"]:
            x1 = int(np.clip(round((xmin - 1) * sx), 0, self.image_size - 1))
            y1 = int(np.clip(round((ymin - 1) * sy), 0, self.image_size - 1))
            x2 = int(np.clip(round((xmax - 1) * sx), 0, self.image_size - 1))
            y2 = int(np.clip(round((ymax - 1) * sy), 0, self.image_size - 1))
            if x2 >= x1 and y2 >= y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 1
                gt_boxes.append([x1, y1, x2, y2])

        bbox = compute_bbox_from_mask_np(mask)
        if profiler is not None and profiler.enabled:
            profiler.record_duration("data.TN5000Dataset.__getitem__", (cv2.getTickCount() / cv2.getTickFrequency()) - t0)
        return SegSample(
            image=image,
            mask=torch.tensor(mask.astype(np.float32), dtype=torch.float32),
            bbox=bbox,
            name=sample["image_id"],
            gt_boxes=gt_boxes,
        ).to_dict()


def build_dataset(dataset_name: str, root_dir: str, split_name: str, image_size: int, split_ids: Optional[Set[str]]) -> Dataset:
    if dataset_name in {"TN3K", "TG3K"}:
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
    names = dataset_names or ["TN3K", "TG3K", "DDTI", "TN5000"]
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
