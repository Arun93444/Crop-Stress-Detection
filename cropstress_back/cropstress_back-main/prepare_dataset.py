"""
================================================================================
  CROP MONITORING SYSTEM — DATASET PREPARATION
  prepare_dataset.py

  What this script does:
    1. Downloads VegAnn (3,775 images) from Zenodo automatically
    2. Merges with your existing plant_stress_dataset + soil datasets
    3. Converts all PNG segmentation masks → YOLOv8 polygon .txt format
    4. Splits everything into train / val / test (70 / 20 / 10)
    5. Writes data.yaml for YOLOv8-seg training
    6. Prints a full dataset summary

  Expected existing layout BEFORE running:
    plant_stress_dataset/
        train/  healthy/  stressed/
        test/   healthy/  stressed/
    soil/
        train/  dry/  good/  wet/
        test/   dry/  good/  wet/

  Output layout AFTER running:
    segmentation_dataset/
        train/  images/  labels/
        val/    images/  labels/
        test/   images/  labels/
        data.yaml

  Segmentation classes:
    0 = plant   (vegetation — from VegAnn mask pixel 255)
    1 = soil    (bare ground — inferred as non-vegetation in soil images)

  Run:
    pip install requests tqdm opencv-python numpy
    python prepare_dataset.py
================================================================================
"""

import os
import sys
import cv2
import json
import math
import shutil
import random
import zipfile
import hashlib
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

random.seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT             = Path(".")
VEGANN_ZIP       = ROOT / "VegAnn_dataset.zip"
VEGANN_DIR       = ROOT / "VegAnn_dataset"
CROP_ROOT        = ROOT / "plant_stress_dataset"
SOIL_ROOT        = ROOT / "soil"
SEG_OUT          = ROOT / "segmentation_dataset"
VEGANN_URL       = "https://zenodo.org/records/7636408/files/VegAnn_dataset.zip?download=1"

# ── Split ratios ───────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

# ── Segmentation classes ───────────────────────────────────────────────────────
CLASS_NAMES = {0: "plant", 1: "soil"}
MIN_CONTOUR_AREA = 150    # pixels — ignore tiny noise contours
POLY_EPSILON     = 0.003  # polygon approximation tightness

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — DOWNLOAD VegAnn
# ──────────────────────────────────────────────────────────────────────────────

def download_vegann():
    """Download VegAnn zip from Zenodo if not already present."""
    if VEGANN_DIR.exists() and any(VEGANN_DIR.rglob("*.png")):
        print(f"  VegAnn already extracted at {VEGANN_DIR} — skipping download.")
        return

    if not VEGANN_ZIP.exists():
        print(f"\n  Downloading VegAnn (1.9 GB) from Zenodo...")
        print(f"  URL: {VEGANN_URL}")
        print(f"  This will take a few minutes depending on your connection.\n")

        resp = requests.get(VEGANN_URL, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(VEGANN_ZIP, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc="  VegAnn_dataset.zip"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"\n  Download complete: {VEGANN_ZIP}")
    else:
        print(f"  VegAnn zip already downloaded: {VEGANN_ZIP}")

    print(f"  Extracting VegAnn...")
    with zipfile.ZipFile(VEGANN_ZIP, "r") as zf:
        zf.extractall(ROOT)
    print(f"  Extracted to: {VEGANN_DIR}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — MASK → YOLO POLYGON CONVERSION
# ──────────────────────────────────────────────────────────────────────────────

def mask_to_yolo_polygons(mask_path: Path, class_id: int,
                           pixel_val: int = 255) -> list[str]:
    """
    Convert a binary PNG mask to YOLO segmentation polygon lines.

    VegAnn masks:
        255 = vegetation (plant)  → class_id 0
        0   = background (soil)   → class_id 1

    Returns list of YOLO label strings:
        "<class_id> x1 y1 x2 y2 ... xn yn"
    All coordinates normalised to [0, 1].
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    h, w = mask.shape

    # Threshold: select the target pixel value
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask == pixel_val] = 255

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        eps    = POLY_EPSILON * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        coords = " ".join(
            f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in approx
        )
        lines.append(f"{class_id} {coords}")

    return lines


def soil_image_to_full_mask_label(img_path: Path) -> list[str]:
    """
    Soil images have no segmentation mask — the entire image IS soil.
    Generate a single full-frame polygon for class 1 (soil).
    """
    return ["1 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000 0.000000 1.000000"]


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — COLLECT ALL SAMPLES
# ──────────────────────────────────────────────────────────────────────────────

def collect_vegann_samples() -> list[dict]:
    """
    Walk VegAnn extracted directory.
    Expected structure inside zip:
        VegAnn_dataset/
            images/   *.jpg  or  *.png
            masks/    *.png   (binary, 0/255)
    or flat with _mask suffix — we handle both.
    """
    samples = []

    img_dir  = VEGANN_DIR / "images"
    mask_dir = VEGANN_DIR / "masks"

    if not img_dir.exists():
        # Try flat layout where masks have _mask suffix
        all_imgs = [p for p in VEGANN_DIR.rglob("*")
                    if p.suffix.lower() in IMG_EXTS and "_mask" not in p.stem]
        for img_path in all_imgs:
            mask_path = img_path.parent / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                mask_path = img_path.parent / (img_path.stem + ".png")
            if mask_path.exists() and mask_path != img_path:
                samples.append({
                    "image": img_path,
                    "mask":  mask_path,
                    "type":  "vegann",
                    "class": 0,       # plant class for vegetation pixels
                    "pixel_val": 255,
                })
        return samples

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        mask_path = mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = mask_dir / (img_path.stem + img_path.suffix)
        if mask_path.exists():
            samples.append({
                "image":     img_path,
                "mask":      mask_path,
                "type":      "vegann",
                "class":     0,
                "pixel_val": 255,
            })

    return samples


def collect_soil_samples() -> list[dict]:
    """
    Soil images → full-frame soil (class 1) polygon labels.
    Reads from soil/train and soil/test.
    """
    samples = []
    for split in ["train", "test"]:
        split_dir = SOIL_ROOT / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTS:
                    samples.append({
                        "image": img_path,
                        "mask":  None,
                        "type":  "soil",
                        "class": 1,
                    })
    return samples


def collect_crop_samples() -> list[dict]:
    """
    Crop health images → full-frame plant (class 0) polygon labels.
    The entire image is a plant patch (PlantVillage style).
    Reads from plant_stress_dataset/train and /test.
    """
    samples = []
    for split in ["train", "test"]:
        split_dir = CROP_ROOT / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTS:
                    samples.append({
                        "image": img_path,
                        "mask":  None,
                        "type":  "crop",
                        "class": 0,
                    })
    return samples


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — WRITE SAMPLES TO SPLIT DIRS
# ──────────────────────────────────────────────────────────────────────────────

def split_samples(samples: list[dict]) -> dict[str, list[dict]]:
    """Shuffle and split samples into train / val / test."""
    random.shuffle(samples)
    n     = len(samples)
    n_tr  = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return {
        "train": samples[:n_tr],
        "val":   samples[n_tr:n_tr + n_val],
        "test":  samples[n_tr + n_val:],
    }


def unique_stem(img_path: Path, counter: dict) -> str:
    """Generate a unique filename stem to avoid collisions across datasets."""
    h = hashlib.md5(str(img_path).encode()).hexdigest()[:8]
    stem = f"{img_path.parent.parent.name}_{img_path.stem}_{h}"
    return stem


def write_split(split_name: str, samples: list[dict],
                stats: dict) -> None:
    """Copy images and write YOLO label files for one split."""
    img_out = SEG_OUT / split_name / "images"
    lbl_out = SEG_OUT / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    skipped = 0

    for s in tqdm(samples, desc=f"  Writing {split_name:5s}", ncols=80):
        img_path = s["image"]
        stem     = unique_stem(img_path, {})
        ext      = img_path.suffix.lower()

        dst_img = img_out / (stem + ext)
        dst_lbl = lbl_out / (stem + ".txt")

        # ── Generate YOLO label ─────────────────────────────────
        if s["type"] == "vegann":
            lines = mask_to_yolo_polygons(s["mask"], class_id=0, pixel_val=255)
            # Also add soil class for background pixels in VegAnn
            soil_lines = mask_to_yolo_polygons(s["mask"], class_id=1, pixel_val=0)
            lines.extend(soil_lines[:3])  # cap soil polygons to avoid noise
        elif s["type"] == "soil":
            lines = soil_image_to_full_mask_label(img_path)
        elif s["type"] == "crop":
            lines = ["0 0.000000 0.000000 1.000000 0.000000 "
                     "1.000000 1.000000 0.000000 1.000000"]
        else:
            lines = []

        if not lines:
            skipped += 1
            continue

        dst_lbl.write_text("\n".join(lines))
        shutil.copy2(img_path, dst_img)
        stats[split_name][s["type"]] += 1
        stats[split_name]["total"]   += 1

    if skipped:
        print(f"  [{split_name}] Skipped {skipped} samples (empty masks).")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — WRITE data.yaml
# ──────────────────────────────────────────────────────────────────────────────

def write_data_yaml():
    yaml_path = SEG_OUT / "data.yaml"
    content = f"""# YOLOv8 Segmentation Dataset — Crop Monitoring System
# Auto-generated by prepare_dataset.py

path: {SEG_OUT.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: 2
names:
  0: plant    # all vegetation pixels (leaves, stems, flowers)
  1: soil     # bare soil / background

# Dataset sources:
#   VegAnn    : 3,775 ground-level multi-crop images, binary masks
#   PlantVillage / plant_stress_dataset : your existing 70,166 crop patches
#   Soil dataset : your existing 90 soil images (dry/good/wet)
"""
    yaml_path.write_text(content)
    print(f"\n  data.yaml written: {yaml_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("  CROP MONITORING — DATASET PREPARATION")
    print("=" * 72)

    # ── Step 1: Download VegAnn ──────────────────────────────────
    print("\n[1/5] VegAnn dataset")
    download_vegann()

    # ── Step 2: Collect all samples ──────────────────────────────
    print("\n[2/5] Collecting samples from all sources...")
    vegann_samples = collect_vegann_samples()
    soil_samples   = collect_soil_samples()
    crop_samples   = collect_crop_samples()

    print(f"  VegAnn samples   : {len(vegann_samples):>6,}")
    print(f"  Soil samples     : {len(soil_samples):>6,}")
    print(f"  Crop samples     : {len(crop_samples):>6,}")

    # Use a capped subset of crop samples for segmentation training
    # (full 70k is too many for seg training — 5k is representative)
    random.shuffle(crop_samples)
    crop_samples_seg = crop_samples[:5000]
    print(f"  Crop samples (seg subset) : {len(crop_samples_seg):>6,}  "
          f"(capped at 5,000 for seg model)")

    all_samples = vegann_samples + soil_samples + crop_samples_seg
    print(f"  TOTAL            : {len(all_samples):>6,}")

    # ── Step 3: Split ────────────────────────────────────────────
    print("\n[3/5] Splitting into train / val / test...")
    splits = split_samples(all_samples)
    for s, slist in splits.items():
        print(f"  {s:5s} : {len(slist):,} samples")

    # ── Step 4: Write files ──────────────────────────────────────
    print("\n[4/5] Writing segmentation dataset...")
    if SEG_OUT.exists():
        print(f"  Removing existing {SEG_OUT} ...")
        shutil.rmtree(SEG_OUT)

    stats = {
        "train": defaultdict(int),
        "val":   defaultdict(int),
        "test":  defaultdict(int),
    }

    for split_name, sample_list in splits.items():
        write_split(split_name, sample_list, stats)

    # ── Step 5: data.yaml ────────────────────────────────────────
    print("\n[5/5] Writing data.yaml...")
    write_data_yaml()

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DATASET SUMMARY")
    print("=" * 72)
    for split_name, s in stats.items():
        print(f"  {split_name:5s} | total={s['total']:>5,}  "
              f"vegann={s['vegann']:>4,}  "
              f"soil={s['soil']:>4,}  "
              f"crop={s['crop']:>4,}")
    print("=" * 72)
    print("\n  Done. Run train_seg.py next to train the segmentation model.\n")


if __name__ == "__main__":
    main()