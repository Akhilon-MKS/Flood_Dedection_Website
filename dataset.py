"""
dataset.py
----------
PyTorch Dataset for Sen1Floods11 (hand-labeled subset).

Expected folder layout (adjust ROOT_DIR if your download differs):

    data/
      S1Hand/            -> Sentinel-1 SAR chips, 2-band GeoTIFF (VV, VH)
      LabelHand/          -> Ground-truth masks, single-band GeoTIFF
                             (0 = no water, 1 = water/flood, -1 = no data)

Each SAR file and its matching label file share the same basename, e.g.:
    S1Hand/Bolivia_103757_S1Hand.tif
    LabelHand/Bolivia_103757_LabelHand.tif

If your Kaggle/GitHub download uses different folder names, just update
SAR_DIR / LABEL_DIR below.
"""

import os
import glob
import csv
import json
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A


def load_tif(path, bands=None):
    """Read a GeoTIFF and return as a numpy array (H, W, C)."""
    with rasterio.open(path) as src:
        arr = src.read(bands) if bands else src.read()
    return np.transpose(arr, (1, 2, 0))  # C,H,W -> H,W,C


def normalize_sar(img, min_db=-50, max_db=1):
    """Clip and scale SAR backscatter (dB) to [0, 1]."""
    img = np.nan_to_num(img, nan=min_db, posinf=max_db, neginf=min_db)
    img = np.clip(img, min_db, max_db)
    return (img - min_db) / (max_db - min_db)


def _chip_id(path):
    """Return the shared Sen1Floods11 identifier for SAR or label paths."""
    name = os.path.basename(path)
    for suffix in ("_S1Hand.tif", "_LabelHand.tif", "_S1Hand.tiff", "_LabelHand.tiff"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def _stac_pairs(root_dir):
    """Read local STAC item metadata when imagery is catalog-only.

    Sen1Floods11 catalog assets normally point to public GeoTIFF URLs.  If a
    matching GeoTIFF has been downloaded anywhere below ``root_dir``, prefer
    it; otherwise retain the catalog URL, which rasterio can open when GDAL's
    HTTP support is available.
    """
    item_files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    sources, labels = {}, {}
    for item_path in item_files:
        try:
            with open(item_path, encoding="utf-8") as handle:
                item = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        assets = item.get("assets", {})
        item_id = item.get("id", "")
        for key, asset in assets.items():
            href = asset.get("href")
            if not href:
                continue
            chip = _chip_id(href) if href.lower().endswith((".tif", ".tiff")) else item_id
            key_lower = key.lower()
            if "s1" in key_lower or "source" in key_lower:
                sources[chip] = href
            elif "label" in key_lower or "mask" in key_lower:
                labels[chip] = href
    return [(sources[key], labels[key]) for key in sorted(sources.keys() & labels.keys())]


class Sen1Floods11Dataset(Dataset):
    def __init__(self, root_dir, split_list=None, augment=False, img_size=256):
        """
        root_dir : path containing S1Hand/ and LabelHand/ folders
        split_list : optional .txt file with basenames to include
                      (use this for train/val/test splits)
        augment : whether to apply data augmentation (train only)
        """
        layout_candidates = [
            (os.path.join(root_dir, "S1Hand"), os.path.join(root_dir, "LabelHand")),
            (
                os.path.join(root_dir, "data", "flood_events", "HandLabeled", "S1Hand"),
                os.path.join(root_dir, "data", "flood_events", "HandLabeled", "LabelHand"),
            ),
        ]
        self.sar_dir = self.label_dir = None
        for sar_dir, label_dir in layout_candidates:
            if os.path.isdir(sar_dir) and os.path.isdir(label_dir):
                self.sar_dir = sar_dir
                self.label_dir = label_dir
                break

        self.img_size = img_size
        self.augment = augment

        requested_ids = None
        if split_list and os.path.exists(split_list):
            with open(split_list, newline="", encoding="utf-8") as split_file:
                rows = csv.reader(split_file)
                requested_ids = []
                for row in rows:
                    if not row or row[0] == "S1Hand":
                        continue
                    requested_ids.append(_chip_id(row[0].strip()))

        self.samples = []
        if self.sar_dir:
            label_by_id = {
                _chip_id(path): path
                for path in glob.glob(os.path.join(self.label_dir, "*_LabelHand.tif"))
                + glob.glob(os.path.join(self.label_dir, "*_LabelHand.tiff"))
            }
            for sar_path in sorted(
                glob.glob(os.path.join(self.sar_dir, "*_S1Hand.tif"))
                + glob.glob(os.path.join(self.sar_dir, "*_S1Hand.tiff"))
            ):
                chip = _chip_id(sar_path)
                if chip in label_by_id:
                    self.samples.append((sar_path, label_by_id[chip]))
        else:
            self.samples = _stac_pairs(root_dir)
            if not self.samples:
                expected = " or ".join(sar for sar, _ in layout_candidates)
                raise FileNotFoundError(
                    "Could not find paired Sen1Floods11 data in flat, nested, "
                    f"or STAC catalog layout below {root_dir!r}. Expected {expected}."
                )

        if requested_ids is not None:
            requested = set(requested_ids)
            self.samples = [pair for pair in self.samples if _chip_id(pair[0]) in requested]

        # augmentation pipeline (safe for 2-channel SAR, no color jitter)
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=0.5),
                A.Resize(img_size, img_size),
            ])
        else:
            self.transform = A.Compose([A.Resize(img_size, img_size)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sar_path, label_path = self.samples[idx]

        sar = load_tif(sar_path)              # (H, W, 2) -> VV, VH
        label = load_tif(label_path)[:, :, 0]  # (H, W)

        sar = normalize_sar(sar).astype(np.float32)

        # treat "no data" (-1) as background (0) for simplicity in a hackathon MVP
        label = np.where(label == -1, 0, label).astype(np.float32)

        transformed = self.transform(image=sar, mask=label)
        sar, label = transformed["image"], transformed["mask"]

        sar = torch.from_numpy(sar).permute(2, 0, 1).float()   # (2, H, W)
        label = torch.from_numpy(label).unsqueeze(0).float()   # (1, H, W)

        return sar, label
