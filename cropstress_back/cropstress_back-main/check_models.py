"""
check_models.py
===============
Run this from your project root to verify both ML models
load correctly before starting the server.

    python check_models.py

What it checks:
  1. .env file present and model paths configured
  2. SEG_MODEL_PATH (best.pt) — file exists + YOLO loads it
  3. CLS_MODEL_PATH (.keras)  — file exists + Keras loads it
  4. Both models can actually run a dummy inference
  5. Prints the exact model_used string the API will report
"""

import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEP = "=" * 60

def ok(msg):   print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def err(msg):  print(f"  ❌  {msg}")


def check_env():
    print(f"\n{SEP}")
    print("  STEP 1 — Environment / .env")
    print(SEP)

    env_file = Path(".env")
    if not env_file.exists():
        err(".env not found — copy _env to .env and fill in values")
        return False
    ok(".env found")

    seg = os.getenv("SEG_MODEL_PATH", "")
    cls = os.getenv("CLS_MODEL_PATH", "")

    if not seg:
        warn("SEG_MODEL_PATH not set in .env — segmentation model will be skipped")
    else:
        print(f"  SEG_MODEL_PATH = {seg}")

    if not cls:
        warn("CLS_MODEL_PATH not set in .env — classifier model will be skipped")
    else:
        print(f"  CLS_MODEL_PATH = {cls}")

    return seg, cls


def check_seg_model(seg_path: str):
    print(f"\n{SEP}")
    print("  STEP 2 — Segmentation Model  (best.pt / YOLO)")
    print(SEP)

    if not seg_path:
        warn("SEG_MODEL_PATH not configured — skipping")
        return None

    p = Path(seg_path)
    if not p.exists():
        err(f"File not found: {p}")
        err("  → Run train_seg.py to train, then set SEG_MODEL_PATH in .env")
        return None
    ok(f"File exists ({p.stat().st_size / 1e6:.1f} MB): {p}")

    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        err("ultralytics not installed")
        err("  → pip install ultralytics>=8.2.0")
        return None
    ok("ultralytics imported")

    # Load model
    try:
        model = YOLO(str(p))
        ok(f"YOLO model loaded")
    except Exception as e:
        err(f"Failed to load YOLO model: {e}")
        return None

    # Dummy inference
    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy, conf=0.1, verbose=False)
        ok(f"Dummy inference OK — classes: {model.names}")
    except Exception as e:
        err(f"Inference failed: {e}")
        return None

    return model


def check_cls_model(cls_path: str):
    print(f"\n{SEP}")
    print("  STEP 3 — Classifier Model  (.keras / TensorFlow)")
    print(SEP)

    if not cls_path:
        warn("CLS_MODEL_PATH not configured — skipping")
        return None

    p = Path(cls_path)
    if not p.exists():
        err(f"File not found: {p}")
        err("  → Run app.py to train, then set CLS_MODEL_PATH in .env")
        return None
    ok(f"File exists ({p.stat().st_size / 1e6:.1f} MB): {p}")

    # Import tensorflow
    try:
        import tensorflow as tf
        ok(f"TensorFlow {tf.__version__} imported")
        # Suppress verbose GPU messages
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    except ImportError:
        err("tensorflow not installed")
        err("  → pip install tensorflow>=2.16.0")
        return None

    # Load model
    try:
        model = tf.keras.models.load_model(str(p), compile=False)
        ok("Keras model loaded")
    except Exception as e:
        err(f"Failed to load Keras model: {e}")
        return None

    # Print architecture summary (heads)
    try:
        output_names = [o.name for o in model.outputs]
        ok(f"Output heads: {output_names}")
        ok(f"Input shape: {model.input_shape}")
    except Exception:
        pass

    # Dummy inference
    try:
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        preds = model.predict(dummy, verbose=0)
        ok(f"Dummy inference OK — {len(preds)} output tensors")
        for i, p_arr in enumerate(preds):
            print(f"       head[{i}] shape={p_arr.shape}  sample={p_arr[0][:3]}")
    except Exception as e:
        err(f"Inference failed: {e}")
        return None

    return model


def check_combined(seg_model, cls_model):
    print(f"\n{SEP}")
    print("  STEP 4 — Combined Pipeline Status")
    print(SEP)

    if seg_model and cls_model:
        ok("Both models loaded — model_used will be: yolo+classifier")
    elif seg_model:
        warn("Only segmentation model loaded — model_used: yolo+hsv")
        warn("  Classifier adds health/stress accuracy. Train app.py to enable.")
    elif cls_model:
        warn("Only classifier loaded — model_used: vi+lab_hsv+classifier")
        warn("  Segmentation adds zone detection. Train train_seg.py to enable.")
    else:
        warn("Neither model loaded — running VI+Lab HSV only (vi+lab_hsv)")
        warn("  Results still work but models improve accuracy significantly.")

    print()
    print(f"  model_used logic:")
    print(f"    No models      →  vi+lab_hsv")
    print(f"    YOLO only      →  yolo+hsv")
    print(f"    Classifier only→  vi+lab_hsv+classifier")
    print(f"    Both           →  yolo+classifier  ← best accuracy")


def main():
    print(SEP)
    print("  CROP MONITOR — Model Verification")
    print(SEP)

    result = check_env()
    if not result:
        sys.exit(1)

    seg_path, cls_path = result

    seg_model = check_seg_model(seg_path)
    cls_model = check_cls_model(cls_path)
    check_combined(seg_model, cls_model)

    print(f"\n{SEP}")
    if seg_model or cls_model:
        print("  ✅  At least one model loaded. Server is ready.")
    else:
        print("  ⚠️   No ML models loaded. Server will use VI+Lab analysis only.")
        print("  Run train_seg.py and app.py to train your models.")
    print(SEP)


if __name__ == "__main__":
    main()