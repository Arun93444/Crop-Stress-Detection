"""
================================================================================
  CROP MONITORING SYSTEM — SEGMENTATION MODEL TRAINING
  train_seg.py

  PATH BUG FIX:
    YOLOv8 sometimes nests output like runs\segment\runs\seg\crop_monitor_seg
    instead of runs\seg\crop_monitor_seg depending on where you run the script.
    This version auto-searches for best.pt and last.pt anywhere under the
    project folder so the script works regardless of nesting.

  RESUME FIX:
    Loads last.pt directly as model, NOT yolov8s-seg.pt.
    Calls model.train(resume=True) with NO other arguments.
    Falls back to most recent epochN.pt backup if last.pt is corrupted.

  Run AFTER prepare_dataset.py:
    pip install ultralytics
    python train_seg.py

  Force fresh start:
    python train_seg.py --fresh
================================================================================
"""

import os
import sys
import argparse
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_YAML    = Path("segmentation_dataset/data.yaml")
RUN_NAME     = "crop_monitor_seg"

# YOLOv8 output root — script writes here, but may nest further.
# We use find_weights_dir() below to locate the actual weights folder.
TRAIN_PROJECT = Path("runs/seg")


# ──────────────────────────────────────────────────────────────────────────────
#  AUTO-LOCATE WEIGHTS FOLDER
#  YOLOv8 can nest output unpredictably depending on working directory.
#  Search the entire project tree for the weights folder.
# ──────────────────────────────────────────────────────────────────────────────

def find_weights_dir() -> Path | None:
    """
    Search everywhere under the current directory for:
        */crop_monitor_seg/weights/
    Returns the first match found, or None.

    This handles the nesting bug where YOLOv8 creates:
        runs/segment/runs/seg/crop_monitor_seg/weights/
    instead of the expected:
        runs/seg/crop_monitor_seg/weights/
    """
    cwd = Path(".")

    # Walk all directories looking for the weights folder
    for candidate in cwd.rglob(f"{RUN_NAME}/weights"):
        if candidate.is_dir():
            return candidate

    return None


def find_resume_checkpoint() -> Path | None:
    """
    Returns best available checkpoint to resume from:
      1. last.pt   — most recent epoch, size-checked (>1MB guards corrupt writes)
      2. epochN.pt — most recent numbered backup if last.pt is corrupt
      3. None      — no checkpoint exists, do a fresh start
    """
    weights_dir = find_weights_dir()
    if weights_dir is None:
        return None

    last_pt = weights_dir / "last.pt"

    # Priority 1: last.pt with size check
    if last_pt.exists() and last_pt.stat().st_size > 1_000_000:
        return last_pt

    # Priority 2: most recent epochN.pt backup
    backups = sorted(
        weights_dir.glob("epoch*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    for backup in backups:
        if backup.stat().st_size > 1_000_000:
            print(f"\n  WARNING: last.pt missing or corrupted.")
            print(f"  Falling back to backup checkpoint: {backup}")
            return backup

    return None


def find_best_pt() -> Path | None:
    """Locate best.pt wherever YOLOv8 saved it."""
    weights_dir = find_weights_dir()
    if weights_dir is None:
        return None
    best = weights_dir / "best.pt"
    return best if best.exists() else None


def find_last_pt() -> Path | None:
    """Locate last.pt wherever YOLOv8 saved it."""
    weights_dir = find_weights_dir()
    if weights_dir is None:
        return None
    last = weights_dir / "last.pt"
    return last if last.exists() else None


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force a fresh training run, ignoring any existing checkpoints"
    )
    args = parser.parse_args()

    # ── Dependencies ──────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    try:
        import torch
    except ImportError:
        print("  ERROR: torch not installed. Run: pip install torch")
        sys.exit(1)

    # ── Dataset check ─────────────────────────────────────────────
    if not DATA_YAML.exists():
        print(f"\n  ERROR: {DATA_YAML} not found.")
        print("  Run prepare_dataset.py first.")
        sys.exit(1)

    # ── Hardware ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        device  = "0"
        batch   = 16
        workers = 8
        epochs  = 50
        imgsz   = 640
        print(f"\n  Hardware : GPU ({torch.cuda.get_device_name(0)})")
        print(f"  Settings : imgsz=640  batch=16  epochs=50")
    else:
        device  = "cpu"
        batch   = 8
        workers = 2
        epochs  = 25
        imgsz   = 416
        print(f"\n  Hardware : CPU only")
        print(f"  Settings : imgsz=416  batch=8  epochs=25")
        print(f"  Est time : 3-5 hrs  (your last run did 25 epochs in 3.3 hrs)")

    # ── Show where we will look for / save checkpoints ────────────
    existing_weights = find_weights_dir()
    if existing_weights:
        print(f"\n  Existing weights found at:")
        print(f"  {existing_weights}")
    else:
        print(f"\n  No existing weights found. Will save to:")
        print(f"  {TRAIN_PROJECT / RUN_NAME / 'weights'}")

    # ── Decide: resume or fresh ───────────────────────────────────
    checkpoint = None if args.fresh else find_resume_checkpoint()

    print("\n" + "=" * 72)

    if checkpoint is not None:
        # ════════════════════════════════════════════════════════════
        #  RESUME MODE
        #
        #  THE FIX — two rules that must both be followed:
        #
        #  1. Load last.pt as the model argument, NOT yolov8s-seg.pt.
        #     Loading the base model loses all training state and
        #     causes a silent restart from epoch 1.
        #
        #  2. Call model.train(resume=True) with NO other arguments.
        #     Every config value is stored inside last.pt and read
        #     automatically. Passing extra args overrides stored state.
        # ════════════════════════════════════════════════════════════
        print("  RESUMING TRAINING")
        print("=" * 72)
        print(f"  Checkpoint : {checkpoint}")
        print(f"  Epoch counter + optimizer state + LR schedule restored")
        print(f"  from checkpoint. No other arguments needed.")
        print("=" * 72 + "\n")

        # CORRECT: load YOUR checkpoint, not the base pretrained model
        model = YOLO(str(checkpoint))

        # CORRECT: resume=True only — no data, epochs, batch, device etc
        results = model.train(resume=True)

    else:
        # ════════════════════════════════════════════════════════════
        #  FRESH START
        # ════════════════════════════════════════════════════════════
        reason = "--fresh flag" if args.fresh else "no checkpoint found"
        print(f"  FRESH START  ({reason})")
        print("=" * 72)
        print(f"  Model        : yolo11n-seg  (nano — fastest, still accurate)")
        print(f"  Data         : {DATA_YAML}")
        print(f"  Image size   : {imgsz}px")
        print(f"  Epochs       : {epochs}")
        print(f"  Batch size   : {batch}")
        print(f"  Epoch backup : every 5 epochs")
        print(f"  Classes      : 0=plant   1=soil")
        print("=" * 72 + "\n")

        # CORRECT: base pretrained model for a fresh start only
        model = YOLO("yolo11n-seg.pt")

        results = model.train(
            data         = str(DATA_YAML.resolve()),  # absolute path avoids nesting
            epochs       = epochs,
            imgsz        = imgsz,
            batch        = batch,
            device       = device,
            workers      = workers,
            project      = str(TRAIN_PROJECT.resolve()),  # absolute path
            name         = RUN_NAME,
            exist_ok     = True,
            patience     = 10,
            save         = True,
            save_period  = 5,        # epoch5.pt, epoch10.pt crash backups
            optimizer    = "AdamW",
            lr0          = 1e-3,
            lrf          = 0.01,
            momentum     = 0.937,
            weight_decay = 5e-4,
            warmup_epochs= 2,
            mosaic       = 1.0,
            mixup        = 0.1,
            copy_paste   = 0.2,
            degrees      = 10.0,
            fliplr       = 0.5,
            flipud       = 0.1,
            hsv_h        = 0.015,
            hsv_s        = 0.7,
            hsv_v        = 0.4,
            val          = True,
            plots        = True,
            verbose      = True,
        )

    # ── Find weights wherever YOLOv8 actually saved them ─────────
    # Re-search after training because the folder now exists
    best_pt  = find_best_pt()
    last_pt  = find_last_pt()
    wdir     = find_weights_dir()

    print("\n" + "=" * 72)
    print("  CHECKPOINT STATUS")
    print("=" * 72)

    if best_pt and best_pt.exists():
        size_mb = best_pt.stat().st_size / 1_000_000
        print(f"  best.pt      : {best_pt.resolve()}  ({size_mb:.1f} MB)")
        print(f"  -> Copy this path into pipeline.py as SEG_MODEL_PATH")
        # Print the exact string the user can paste into pipeline.py
        print(f"\n  Paste this into pipeline.py line 52:")
        print(f'  SEG_MODEL_PATH = r"{best_pt.resolve()}"')
    else:
        print("  best.pt      : NOT FOUND")
        print("  Check training logs above for errors.")

    if last_pt and last_pt.exists():
        size_mb = last_pt.stat().st_size / 1_000_000
        print(f"\n  last.pt      : {last_pt.resolve()}  ({size_mb:.1f} MB)")
        print(f"  -> Run python train_seg.py again to resume if interrupted")

    if wdir:
        backups = sorted(wdir.glob("epoch*.pt"))
        if backups:
            print(f"\n  Epoch backups: {', '.join(p.name for p in backups)}")

    print("=" * 72)

    # ── Final test set validation ─────────────────────────────────
    if best_pt and best_pt.exists():
        print("\n  Running final validation on test set...")
        try:
            from ultralytics import YOLO as _YOLO
            test_model   = _YOLO(str(best_pt))
            test_results = test_model.val(
                data    = str(DATA_YAML.resolve()),
                split   = "test",
                imgsz   = imgsz,
                device  = device,
                verbose = False,
            )
            print(f"\n  Test mAP50     : {test_results.seg.map50:.3f}")
            print(f"  Test mAP50-95  : {test_results.seg.map:.3f}")
            print(f"\n  Training complete. Copy SEG_MODEL_PATH above into pipeline.py\n")
        except Exception as e:
            print(f"  Validation skipped ({e})")
            print(f"  best.pt is valid. Copy SEG_MODEL_PATH above into pipeline.py\n")


if __name__ == "__main__":
    main()