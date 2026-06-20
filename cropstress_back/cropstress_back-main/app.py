"""
====================================================================
  CROP MONITORING SYSTEM — PRODUCTION MULTI-TASK CLASSIFIER v3
  app.py  |  TensorFlow / Keras  |  MobileNetV3Large backbone

  WHAT THIS MODEL DOES:
    Classifies individual image patches (single leaf, single plant,
    soil close-up) extracted by pipeline.py after segmentation.
    It does NOT process wide-area frames — that is train_seg.py's job.

  WHY VegAnn IS NOT USED HERE:
    VegAnn contains wide-field ground-level images with binary
    vegetation/background masks. It is the right data for training
    the YOLO segmentation model (train_seg.py), but wrong for this
    classifier which expects close-up patches of leaves and soil.
    Using VegAnn here would add noise and hurt accuracy.

  DATASETS USED:
    plant_stress_dataset/   70,166 close-up leaf images (healthy/stressed)
    soil/                       90 close-up soil images (dry/good/wet)
    Total: ~70,256 images — all close-up patches, correct for this model.

  5 OUTPUT HEADS:
    1. disease_out      : healthy / stressed        (binary softmax)
    2. soil_out         : dry / good / wet          (3-class softmax)
    3. stress_out       : severity 0-100            (sigmoid regression)
    4. soil_present_out : soil visible yes/no       (binary sigmoid)
    5. stress_type_out  : 6 stress types            (softmax)
       healthy_green | yellow_leaves | brown_rot |
       rust_spots    | wilting       | necrosis

  SPEED vs previous version:
    EfficientNetV2-S (21M params) → MobileNetV3Large (3.7M params)
    Steps/epoch 200 → 100 on CPU
    Unfreeze last 20 layers in Phase 2 (was 50)
    Estimated time: 2-3 hrs CPU

  Phase 2 tf.function crash fix:
    Eager optimizer pre-build before first tf.function call.
    Per-phase tf.function recompile after backbone unfreeze.
====================================================================
"""

import os
import sys
import time
import cv2
import json
import glob
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
tf.get_logger().setLevel("ERROR")

AUTOTUNE = tf.data.AUTOTUNE


# ──────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────

CROP_HEALTHY_KEYWORDS  = {"healthy"}
CROP_STRESSED_KEYWORDS = {"stressed", "diseased", "sick", "infected", "stress"}

SOIL_MAP = {"dry": 0, "good": 1, "wet": 2}
SOIL_INV = {0: "dry", 1: "good_moisture", 2: "wet_waterlogged"}

STRESS_TYPE_MAP = {
    "healthy_green": 0,
    "yellow_leaves": 1,
    "brown_rot":     2,
    "rust_spots":    3,
    "wilting":       4,
    "necrosis":      5,
}
STRESS_TYPE_INV  = {v: k for k, v in STRESS_TYPE_MAP.items()}
NUM_STRESS_TYPES = 6

IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
IMG_SIZE     = 224
NUM_DISEASES = 2

# Loss weights
W_DISEASE     = 1.5
W_SOIL        = 1.0
W_STRESS      = 0.5
W_SOIL_PRES   = 0.8
W_STRESS_TYPE = 1.2


# ──────────────────────────────────────────────────────────────────
#  SECTION 1 — DATA PIPELINE
#  Only uses: plant_stress_dataset/ and soil/
#  VegAnn is NOT included here — it belongs in train_seg.py only.
# ──────────────────────────────────────────────────────────────────

def _is_image(f: str) -> bool:
    return Path(f).suffix.lower() in IMG_EXTS


def _folder_to_crop_label(folder_name: str) -> int:
    fn = folder_name.lower().replace("_", " ").replace("-", " ")
    for kw in CROP_HEALTHY_KEYWORDS:
        if kw in fn:
            return 0
    for kw in CROP_STRESSED_KEYWORDS:
        if kw in fn:
            return 1
    return -1  # intermediate folder — skip, descend into it


def _infer_stress_type(folder_name: str, crop_label: int) -> int:
    """
    Infer stress type from folder name keywords.
    PlantVillage folder names often contain disease names like
    'Apple_scab', 'Tomato_Early_blight', 'Grape_Black_rot' etc.
    We map these to our 6 stress type classes.
    Healthy folders always return healthy_green (0).
    """
    if crop_label == 0:
        return STRESS_TYPE_MAP["healthy_green"]

    fn = folder_name.lower()
    if any(k in fn for k in ["yellow", "chloros", "mosaic", "virus"]):
        return STRESS_TYPE_MAP["yellow_leaves"]
    if any(k in fn for k in ["blight", "rot", "brown", "scab", "spot", "mold"]):
        return STRESS_TYPE_MAP["brown_rot"]
    if any(k in fn for k in ["rust"]):
        return STRESS_TYPE_MAP["rust_spots"]
    if any(k in fn for k in ["wilt", "drought", "water", "curl"]):
        return STRESS_TYPE_MAP["wilting"]
    if any(k in fn for k in ["necrosis", "dead", "black", "burn", "fire"]):
        return STRESS_TYPE_MAP["necrosis"]
    # Generic stressed — default to wilting
    return STRESS_TYPE_MAP["wilting"]


def collect_files(crop_root: str, soil_root: str) -> tuple:
    """
    Collect all labeled close-up patches from:
      - plant_stress_dataset/  (crop health — 70k images)
      - soil/                  (soil condition — 90 images)

    VegAnn is deliberately excluded. It contains wide-field images
    that would confuse this patch classifier. VegAnn is only used
    by prepare_dataset.py to train the YOLO segmentation model.
    """
    paths, l_disease, l_soil, l_stress = [], [], [], []
    l_soil_present, l_stress_type      = [], []

    print("\n" + "=" * 62)
    print("  SCANNING DATASETS")
    print("  (VegAnn excluded — used only for segmentation model)")
    print("=" * 62)

    # ── Crop images ────────────────────────────────────────────────
    healthy_count = stressed_count = 0

    if os.path.exists(crop_root):
        for dirpath, _, filenames in os.walk(crop_root):
            folder     = os.path.basename(dirpath)
            crop_label = _folder_to_crop_label(folder)
            if crop_label == -1:
                continue   # intermediate folder like "train/" — descend

            img_files = [f for f in filenames if _is_image(f)]
            if not img_files:
                continue

            is_stressed = (crop_label == 1)
            stress_type = _infer_stress_type(folder, crop_label)

            for f in img_files:
                paths.append(os.path.join(dirpath, f))
                l_disease.append(crop_label)
                l_soil.append(-1)       # no soil label for crop patches
                l_stress.append(
                    float(np.random.uniform(0, 15))    # healthy: low stress
                    if not is_stressed
                    else float(np.random.uniform(55, 100))  # stressed: high
                )
                l_soil_present.append(0)  # leaf patches — no visible soil
                l_stress_type.append(stress_type)

            if is_stressed:
                stressed_count += len(img_files)
            else:
                healthy_count  += len(img_files)

        print(f"  Healthy images   : {healthy_count:,}")
        print(f"  Stressed images  : {stressed_count:,}")
        print(f"  Crop total       : {healthy_count + stressed_count:,}")
    else:
        print(f"  WARNING: Crop root not found: {crop_root}")

    # ── Soil images — with oversampling ───────────────────────────
    # Problem: 90 soil images in 70k total = 0.13% of data.
    # With batch=16 and shuffle, most batches have zero soil images,
    # so the soil head gets almost no gradient — hence 0.0% soil acc.
    # Fix: repeat soil images 15x = 1,350 soil samples = ~1.9% of total.
    # Augmentation makes each repeated image look different each epoch.
    SOIL_OVERSAMPLE = 15

    dry_count = good_count = wet_count = 0
    soil_paths_raw, soil_labels_raw = [], []

    if os.path.exists(soil_root):
        for dirpath, _, filenames in os.walk(soil_root):
            folder = os.path.basename(dirpath).lower().strip()
            s_idx  = SOIL_MAP.get(folder, -1)
            if s_idx == -1:
                continue

            img_files = [f for f in filenames if _is_image(f)]
            for f in img_files:
                soil_paths_raw.append(os.path.join(dirpath, f))
                soil_labels_raw.append(s_idx)

            if s_idx == 0: dry_count  += len(img_files)
            if s_idx == 1: good_count += len(img_files)
            if s_idx == 2: wet_count  += len(img_files)

        # Repeat soil samples SOIL_OVERSAMPLE times
        raw_soil_total = len(soil_paths_raw)
        for _ in range(SOIL_OVERSAMPLE):
            for sp, si in zip(soil_paths_raw, soil_labels_raw):
                paths.append(sp)
                l_disease.append(-1)
                l_soil.append(si)
                l_stress.append(-1.0)
                l_soil_present.append(1)
                l_stress_type.append(-1)

        oversampled = raw_soil_total * SOIL_OVERSAMPLE
        print(f"  Soil dry         : {dry_count:,}  (raw)")
        print(f"  Soil good        : {good_count:,}  (raw)")
        print(f"  Soil wet         : {wet_count:,}  (raw)")
        print(f"  Soil total       : {raw_soil_total:,} raw "
              f"-> {oversampled:,} after {SOIL_OVERSAMPLE}x oversample")
    else:
        print(f"  WARNING: Soil root not found: {soil_root}")

    print(f"  TOTAL LABELED    : {len(paths):,}")
    print("=" * 62)

    return (paths, l_disease, l_soil, l_stress,
            l_soil_present, l_stress_type)


def make_tf_dataset(paths, l_disease, l_soil, l_stress,
                    l_soil_present, l_stress_type,
                    batch_size=32, augment=True, shuffle=True,
                    img_size=IMG_SIZE):

    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(paths,          dtype=tf.string),
        tf.constant(l_disease,      dtype=tf.int32),
        tf.constant(l_soil,         dtype=tf.int32),
        tf.constant(l_stress,       dtype=tf.float32),
        tf.constant(l_soil_present, dtype=tf.int32),
        tf.constant(l_stress_type,  dtype=tf.int32),
    ))

    def load_and_preprocess(path, ld, ls, lst, lsp, lstype):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
        img.set_shape([img_size, img_size, 3])
        return img, ld, ls, lst, lsp, lstype

    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(10000, max(100, len(paths))),
            reshuffle_each_iteration=True,
            seed=42,
        )

    ds = ds.batch(batch_size, drop_remainder=False)

    if augment:
        def augment_batch(img, ld, ls, lst, lsp, lstype):
            p = 2.0 / 6.0
            img = tf.cond(tf.random.uniform([]) < p,
                          lambda: tf.image.random_flip_left_right(img),
                          lambda: img)
            img = tf.cond(tf.random.uniform([]) < p,
                          lambda: tf.image.random_flip_up_down(img),
                          lambda: img)
            img = tf.cond(tf.random.uniform([]) < p,
                          lambda: tf.image.random_brightness(img, 0.2),
                          lambda: img)
            img = tf.cond(tf.random.uniform([]) < p,
                          lambda: tf.image.random_contrast(img, 0.75, 1.25),
                          lambda: img)
            img = tf.cond(tf.random.uniform([]) < p,
                          lambda: tf.image.random_saturation(img, 0.75, 1.25),
                          lambda: img)
            img = tf.cond(tf.random.uniform([]) < p,
                          lambda: tf.image.random_hue(img, 0.05),
                          lambda: img)
            img = tf.clip_by_value(img, -1.0, 1.0)
            return img, ld, ls, lst, lsp, lstype

        ds = ds.map(augment_batch, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def prepare_batch(img, l_disease, l_soil, l_stress,
                  l_soil_present, l_stress_type):
    m_disease = tf.cast(tf.not_equal(l_disease,     -1),   tf.float32)
    y_disease = tf.one_hot(tf.maximum(l_disease, 0),        depth=NUM_DISEASES)
    m_soil    = tf.cast(tf.not_equal(l_soil,        -1),   tf.float32)
    y_soil    = tf.one_hot(tf.maximum(l_soil, 0),           depth=3)
    m_stress  = tf.cast(tf.not_equal(l_stress,     -1.0),  tf.float32)
    y_stress  = tf.maximum(l_stress, 0.0)
    y_sp      = tf.cast(l_soil_present, tf.float32)
    m_stype   = tf.cast(tf.not_equal(l_stress_type, -1),   tf.float32)
    y_stype   = tf.one_hot(tf.maximum(l_stress_type, 0),   depth=NUM_STRESS_TYPES)
    return (img, y_disease, y_soil, y_stress, y_sp, y_stype,
            m_disease, m_soil, m_stress, m_stype)


# ──────────────────────────────────────────────────────────────────
#  SECTION 2 — MODEL
#  MobileNetV3Large: 3.7M params, built-in squeeze-excitation,
#  99.5% on PlantVillage in published research, 6x faster than
#  EfficientNetV2-S on CPU.
# ──────────────────────────────────────────────────────────────────

def build_multi_task_model(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            num_diseases=NUM_DISEASES,
                            num_stress_types=NUM_STRESS_TYPES,
                            dropout_rate=0.35):
    print("\n" + "=" * 62)
    print("  BUILDING MODEL  (MobileNetV3Large + 5 heads)")
    print("=" * 62)

    backbone = tf.keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,  # we handle preprocessing in dataset pipeline
    )
    backbone.trainable = False
    print(f"  Backbone: MobileNetV3Large | params=3.7M | frozen ✓")

    inputs = keras.Input(shape=input_shape, name="image_input")
    x      = backbone(inputs, training=False)
    x      = layers.GlobalAveragePooling2D(name="gap")(x)
    x      = layers.Dropout(dropout_rate, name="drop1")(x)
    shared = layers.Dense(256, activation="relu", name="shared_dense")(x)
    shared = layers.BatchNormalization(name="shared_bn")(shared)
    shared = layers.Dropout(dropout_rate * 0.7, name="drop2")(shared)

    # Disease + stress severity branch
    dis_h   = layers.Dense(128, activation="relu", name="dis_dense")(shared)
    dis_h   = layers.Dropout(0.2, name="dis_drop")(dis_h)

    # Soil classification branch
    soil_h  = layers.Dense(64, activation="relu", name="soil_dense")(shared)

    # Stress type branch
    stype_h = layers.Dense(128, activation="relu", name="stype_dense")(shared)
    stype_h = layers.Dropout(0.2, name="stype_drop")(stype_h)

    # 5 output heads
    disease_out     = layers.Dense(
        num_diseases,    activation="softmax", name="disease_out")(dis_h)
    soil_out        = layers.Dense(
        3,               activation="softmax", name="soil_out")(soil_h)
    stress_out      = layers.Dense(
        1,               activation="sigmoid", name="stress_out")(shared)
    soil_pres_out   = layers.Dense(
        1,               activation="sigmoid", name="soil_present_out")(shared)
    stress_type_out = layers.Dense(
        num_stress_types, activation="softmax", name="stress_type_out")(stype_h)

    model = keras.Model(
        inputs=inputs,
        outputs=[disease_out, soil_out, stress_out,
                 soil_pres_out, stress_type_out],
        name="CropMonitor_v3",
    )

    total     = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w)
                    for w in model.trainable_variables)
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Heads            : disease | soil | stress | soil_pres | stress_type")
    print("=" * 62)
    return model, backbone


# ──────────────────────────────────────────────────────────────────
#  SECTION 3 — LOSSES
# ──────────────────────────────────────────────────────────────────

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss — down-weights easy examples, focuses on hard/rare ones."""
    y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    ce_loss = -y_true * tf.math.log(y_pred)
    p_t     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
    fl      = alpha * tf.pow(1.0 - p_t, gamma) * ce_loss
    return tf.reduce_sum(fl, axis=-1)


def masked_focal(y_true, y_pred, mask, gamma=2.0):
    fl = focal_loss(y_true, y_pred, gamma=gamma)
    return tf.reduce_sum(fl * mask) / (tf.reduce_sum(mask) + 1e-7)


def masked_cce(y_true, y_pred, mask):
    cce_fn = keras.losses.CategoricalCrossentropy(reduction="none")
    return tf.reduce_sum(
        cce_fn(y_true, y_pred) * mask
    ) / (tf.reduce_sum(mask) + 1e-7)


def masked_mse(y_true, y_pred_sigmoid, mask):
    y_pred = tf.squeeze(y_pred_sigmoid, axis=-1)
    sq_err = tf.square(y_true / 100.0 - y_pred)
    return tf.reduce_sum(sq_err * mask) / (tf.reduce_sum(mask) + 1e-7)


def masked_bce(y_true, y_pred, mask=None):
    bce_fn = keras.losses.BinaryCrossentropy(reduction="none")
    loss   = bce_fn(tf.reshape(y_true, [-1, 1]), y_pred)
    if mask is not None:
        return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-7)
    return tf.reduce_mean(loss)


# ──────────────────────────────────────────────────────────────────
#  SECTION 4 — TRAINING
# ──────────────────────────────────────────────────────────────────

def train_step(model, optimizer, img,
               y_disease, y_soil, y_stress, y_soil_pres, y_stype,
               m_disease, m_soil, m_stress, m_stype):
    with tf.GradientTape() as tape:
        p_dis, p_soil, p_stress, p_sp, p_stype = model(img, training=True)

        loss_d  = masked_focal(y_disease, p_dis,    m_disease)
        loss_s  = masked_cce(y_soil,      p_soil,   m_soil)
        loss_st = masked_mse(y_stress,    p_stress, m_stress)
        loss_sp = masked_bce(y_soil_pres, p_sp)
        loss_ty = masked_cce(y_stype,     p_stype,  m_stype)

        total = (W_DISEASE     * loss_d  +
                 W_SOIL        * loss_s  +
                 W_STRESS      * loss_st +
                 W_SOIL_PRES   * loss_sp +
                 W_STRESS_TYPE * loss_ty)

    grads = tape.gradient(total, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total, loss_d, loss_s, loss_st, loss_ty


def make_optimizer(lr, decay_steps):
    return keras.optimizers.AdamW(
        learning_rate=keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=max(decay_steps, 1),
            t_mul=1.5,
            alpha=lr * 0.01,
        ),
        weight_decay=1e-5,
        clipnorm=5.0,
    )


def _build_optimizer_eagerly(optimizer, model):
    """
    Pre-allocate all optimizer momentum/velocity tf.Variables eagerly,
    outside any tf.function graph. This prevents the Phase 2 crash where
    the optimizer tries to create new Variables inside a traced function.
    """
    dummy = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(dummy, model.trainable_variables))


def train_model(model, backbone, dataset, total_samples,
                batch_size=32, phase1_epochs=5, phase2_epochs=7,
                steps_per_epoch=None, early_stop_patience=4,
                save_path="crop_monitor_best.keras"):

    total_epochs = phase1_epochs + phase2_epochs
    if steps_per_epoch is None:
        steps_per_epoch = min(500, max(50, total_samples // batch_size))

    optimizer = make_optimizer(2e-3, phase1_epochs * steps_per_epoch)
    _build_optimizer_eagerly(optimizer, model)
    compiled_step = tf.function(train_step)

    print("\n" + "=" * 62)
    print("  TRAINING CONFIGURATION")
    print("=" * 62)
    print(f"  Backbone         : MobileNetV3Large  (3.7M params)")
    print(f"  Total images     : {total_samples:,}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Steps / epoch    : {steps_per_epoch}")
    print(f"  Phase 1          : {phase1_epochs} epochs  frozen    AdamW lr=2e-3")
    print(f"  Phase 2          : {phase2_epochs} epochs  finetune  AdamW lr=2e-4")
    print(f"  Early stop       : patience {early_stop_patience}")
    print(f"  Loss weights     : disease={W_DISEASE}  soil={W_SOIL}  "
          f"stress={W_STRESS}  soil_pres={W_SOIL_PRES}  stype={W_STRESS_TYPE}")
    print("=" * 62)

    best_loss, patience_counter = float("inf"), 0
    in_phase2   = False
    train_start = time.time()
    history     = defaultdict(list)
    ds_iter     = iter(dataset.repeat())

    for epoch in range(total_epochs):

        # ── Phase 2 switch ─────────────────────────────────────────
        if epoch == phase1_epochs and not in_phase2:
            in_phase2 = True
            print("\n" + "=" * 62)
            print("  PHASE 2 — Fine-tuning last 20 backbone layers")
            backbone.trainable = True
            # MobileNetV3Large has fewer layers than EfficientNetV2-S
            # Unfreeze only last 20 to keep training stable and fast
            for layer in backbone.layers[:-20]:
                layer.trainable = False
            trainable_now = sum(tf.keras.backend.count_params(w)
                                for w in model.trainable_variables)
            print(f"  Trainable params now : {trainable_now:,}")

            # Rebuild optimizer eagerly BEFORE recompiling tf.function
            # This pre-allocates momentum/velocity slots for new variables
            # and prevents the ValueError: tf.function only supports
            # singleton tf.Variables crash
            optimizer = make_optimizer(2e-4, phase2_epochs * steps_per_epoch)
            _build_optimizer_eagerly(optimizer, model)
            compiled_step = tf.function(train_step)
            print("  Optimizer rebuilt + tf.function recompiled ✓")
            print("=" * 62)

        epoch_start  = time.time()
        losses       = []
        d_hit, d_tot = 0, 0
        t_hit, t_tot = 0, 0
        s_hit, s_tot = 0, 0
        st_err, st_n = 0.0, 0

        phase_tag = "P1-Frozen  " if not in_phase2 else "P2-Finetune"

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"  Epoch {epoch+1:>2}/{total_epochs} [{phase_tag}]",
            ncols=115,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
            colour="cyan",
            file=sys.stdout,
        )

        for _ in pbar:
            img, ld, ls, lst, lsp, lstype = next(ds_iter)
            (img, y_dis, y_soil, y_stress, y_sp, y_stype,
             m_dis, m_soil, m_stress, m_stype) = prepare_batch(
                img, ld, ls, lst, lsp, lstype
            )

            total_loss, loss_d, loss_s, loss_st, loss_ty = compiled_step(
                model, optimizer, img,
                y_dis, y_soil, y_stress, y_sp, y_stype,
                m_dis, m_soil, m_stress, m_stype
            )

            # Compute metrics on the same batch
            outputs     = model(img, training=False)
            p_dis_np    = outputs[0].numpy()
            p_soil_np   = outputs[1].numpy()
            p_stress_np = outputs[2].numpy().flatten() * 100.0
            p_stype_np  = outputs[4].numpy()
            mc, ms, mst, mty = (m_dis.numpy(),  m_soil.numpy(),
                                 m_stress.numpy(), m_stype.numpy())
            yd, ys, yst, ystype = (y_dis.numpy(), y_soil.numpy(),
                                    y_stress.numpy(), y_stype.numpy())

            for i in range(len(p_dis_np)):
                if mc[i]:
                    d_tot += 1
                    d_hit += int(np.argmax(p_dis_np[i]) == np.argmax(yd[i]))
                if mty[i]:
                    t_tot += 1
                    t_hit += int(np.argmax(p_stype_np[i]) == np.argmax(ystype[i]))
            for i in range(len(p_soil_np)):
                if ms[i]:
                    s_tot += 1
                    s_hit += int(np.argmax(p_soil_np[i]) == np.argmax(ys[i]))
            for i in range(len(p_stress_np)):
                if mst[i]:
                    st_n   += 1
                    st_err += abs(p_stress_np[i] - yst[i])

            losses.append(float(total_loss.numpy()))
            avg_l    = float(np.mean(losses))
            dis_acc  = (d_hit / d_tot * 100) if d_tot else 0.0
            type_acc = (t_hit / t_tot * 100) if t_tot else 0.0
            soil_acc = (s_hit / s_tot * 100) if s_tot else 0.0
            s_mae    = (st_err / st_n)        if st_n  else 0.0

            pbar.set_postfix({
                "Loss":    f"{avg_l:.4f}",
                "CropAcc": f"{dis_acc:.1f}%",
                "TypeAcc": f"{type_acc:.1f}%",
                "SoilAcc": f"{soil_acc:.1f}%",
                "SMAE":    f"{s_mae:.1f}",
            }, refresh=True)

        pbar.close()

        avg_loss    = float(np.mean(losses))
        ep_time     = time.time() - epoch_start
        dis_acc_ep  = (d_hit / d_tot * 100) if d_tot else 0.0
        type_acc_ep = (t_hit / t_tot * 100) if t_tot else 0.0
        soil_acc_ep = (s_hit / s_tot * 100) if s_tot else 0.0
        smae_ep     = (st_err / st_n)        if st_n  else 0.0

        history["loss"].append(avg_loss)
        history["dis_acc"].append(dis_acc_ep)
        history["type_acc"].append(type_acc_ep)
        history["soil_acc"].append(soil_acc_ep)
        history["stress_mae"].append(smae_ep)

        print(f"  ┌─ Loss             : {avg_loss:.4f}")
        print(f"  ├─ Crop Health Acc  : {dis_acc_ep:.2f}%")
        print(f"  ├─ Stress Type Acc  : {type_acc_ep:.2f}%")
        print(f"  ├─ Soil Accuracy    : {soil_acc_ep:.2f}%")
        print(f"  ├─ Stress MAE       : {smae_ep:.2f}")
        print(f"  └─ Time             : {ep_time:.0f}s\n")

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            model.save(save_path)
            print(f"  ✅ Best model saved  (loss = {best_loss:.4f})\n")
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement  "
                  f"({patience_counter}/{early_stop_patience})\n")
            if patience_counter >= early_stop_patience:
                print(f"  🛑 EARLY STOPPING at epoch {epoch + 1}")
                break

    h, r = divmod(int(time.time() - train_start), 3600)
    m, s = divmod(r, 60)
    print("=" * 62)
    print(f"  🎉 TRAINING COMPLETE  |  {h}h {m}m {s}s  "
          f"|  Best loss: {best_loss:.4f}")
    print("=" * 62)
    return model, dict(history)


# ──────────────────────────────────────────────────────────────────
#  SECTION 5 — INFERENCE  (single image / single patch)
# ──────────────────────────────────────────────────────────────────

def run_inference(model, image_path: str) -> dict:
    """
    Run all 5 heads on a single close-up image.
    Use this for:
      - Testing the model on a single leaf photo
      - Debugging individual patches extracted by pipeline.py
    For wide-area camera frames, use pipeline.py instead.
    """
    print(f"\n  Inference: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    img_pre     = tf.keras.applications.mobilenet_v3.preprocess_input(
                      img_resized[np.newaxis]
                  )

    p_crop, p_soil, p_stress_raw, p_soil_pres, p_stype = model.predict(
        img_pre, verbose=0
    )

    healthy_prob  = float(p_crop[0][0]) * 100
    stressed_prob = float(p_crop[0][1]) * 100
    is_stressed   = stressed_prob > 50.0
    stress_score  = float(np.clip(p_stress_raw[0, 0] * 100, 0, 100))
    soil_pres     = float(p_soil_pres[0, 0])
    soil_cond     = SOIL_INV[int(np.argmax(p_soil[0]))]
    stype_idx     = int(np.argmax(p_stype[0]))
    stype_label   = STRESS_TYPE_INV.get(stype_idx, "unknown")
    stype_conf    = float(p_stype[0][stype_idx]) * 100

    result = {
        "image":                  image_path,
        "crop_health":            "stressed" if is_stressed else "healthy",
        "healthy_prob":           round(healthy_prob, 2),
        "stressed_prob":          round(stressed_prob, 2),
        "stress_score":           round(stress_score, 2),
        "stress_type":            stype_label,
        "stress_type_confidence": round(stype_conf, 2),
        "soil_detected":          soil_pres > 0.5,
        "soil_confidence":        round(soil_pres * 100, 2),
        "soil_condition":         soil_cond if soil_pres > 0.5 else "not_detected",
    }

    print("\n" + "=" * 62)
    print("  RESULT")
    print("=" * 62)
    print(f"  Crop Health     : {result['crop_health'].upper()}")
    print(f"  Stress Score    : {result['stress_score']}/100")
    print(f"  Stress Type     : {result['stress_type'].upper()} "
          f"({result['stress_type_confidence']:.1f}% confident)")
    print(f"  Healthy prob    : {result['healthy_prob']}%")
    print(f"  Stressed prob   : {result['stressed_prob']}%")
    soil_str = (f"{result['soil_condition'].upper()} "
                f"({result['soil_confidence']:.1f}% confident)")
    print(f"  Soil            : "
          f"{'DETECTED — ' + soil_str if result['soil_detected'] else 'NOT DETECTED'}")
    print("=" * 62)
    print(json.dumps(result, indent=2))
    return result


# ──────────────────────────────────────────────────────────────────
#  SECTION 6 — TRAINING PLOT
# ──────────────────────────────────────────────────────────────────

def plot_history(history: dict, save="training_history.png"):
    if not history.get("loss"):
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ep    = range(1, len(history["loss"]) + 1)
    specs = [
        (axes[0, 0], "loss",       "Combined Loss",            "steelblue"),
        (axes[0, 1], "dis_acc",    "Crop Health Accuracy (%)", "seagreen"),
        (axes[0, 2], "type_acc",   "Stress Type Accuracy (%)", "coral"),
        (axes[1, 0], "soil_acc",   "Soil Accuracy (%)",        "firebrick"),
        (axes[1, 1], "stress_mae", "Stress MAE",               "darkorchid"),
    ]
    for a, key, title, col in specs:
        if key in history and history[key]:
            a.plot(ep, history[key], color=col, lw=2, marker="o", markersize=4)
            a.set_title(title, fontsize=11)
            a.set_xlabel("Epoch")
            a.grid(alpha=0.3)
    for a in [axes[0, 1], axes[0, 2], axes[1, 0]]:
        a.set_ylim(0, 100)
    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save}")


# ──────────────────────────────────────────────────────────────────
#  SECTION 7 — CLASS WEIGHT COMPUTATION
# ──────────────────────────────────────────────────────────────────

def compute_class_weights(l_disease: list) -> dict:
    counts = defaultdict(int)
    for d in l_disease:
        if d >= 0:
            counts[d] += 1
    if not counts:
        return {}
    total   = sum(counts.values())
    n_cls   = len(counts)
    weights = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
    names   = {0: "healthy", 1: "stressed"}
    for k, w in weights.items():
        print(f"    Class {names.get(k, k):<10}: {counts[k]:>6,}  weight={w:.3f}")
    return weights


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "#" * 62)
    print("#  CROP MONITORING — PATCH CLASSIFIER v3                   #")
    print("#  MobileNetV3Large · 5 heads · ~2-3 hrs CPU               #")
    print("#  Datasets: plant_stress_dataset + soil  (no VegAnn)      #")
    print("#" * 62)

    gpus   = tf.config.list_physical_devices("GPU")
    ON_GPU = len(gpus) > 0

    if ON_GPU:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n  Hardware: GPU ({len(gpus)} device(s))")
        BATCH_SIZE     = 32
        PHASE1_EPOCHS  = 8
        PHASE2_EPOCHS  = 10
        STEPS_OVERRIDE = None   # auto-calculate from dataset size
    else:
        print(f"\n  Hardware: CPU only")
        BATCH_SIZE     = 16
        PHASE1_EPOCHS  = 5
        PHASE2_EPOCHS  = 7
        # FIX: 100 steps was too few — only 1,600 images/epoch out of 70k.
        # 400 steps = 6,400 images/epoch = ~9% of dataset per epoch.
        # At 12 epochs the model sees the full dataset roughly once.
        # Time: ~6 min/epoch × 12 epochs = ~1.2 hrs total.
        STEPS_OVERRIDE = 400
        total_ep       = PHASE1_EPOCHS + PHASE2_EPOCHS
        print(f"  Batch={BATCH_SIZE}  Steps={STEPS_OVERRIDE}  "
              f"Epochs={total_ep}")
        print(f"  Estimated time: ~1.2-1.5 hrs on CPU")

    CROP_ROOT = "plant_stress_dataset"
    SOIL_ROOT = "soil"
    SAVE_PATH = "crop_monitor_best.keras"

    # Collect only crop + soil patches — no VegAnn
    (paths, l_dis, l_soil, l_stress,
     l_sp, l_stype) = collect_files(CROP_ROOT, SOIL_ROOT)

    if not paths:
        print("  ERROR: No labeled images found.")
        print("  Check that plant_stress_dataset/ and soil/ exist.")
        sys.exit(1)

    class_weights = compute_class_weights(l_dis)
    print(f"\n  Class weights for {len(class_weights)} disease classes computed")

    train_ds = make_tf_dataset(
        paths, l_dis, l_soil, l_stress, l_sp, l_stype,
        batch_size=BATCH_SIZE, augment=True, shuffle=True,
    )

    model, backbone = build_multi_task_model()

    model, history = train_model(
        model, backbone, train_ds,
        total_samples       = len(paths),
        batch_size          = BATCH_SIZE,
        phase1_epochs       = PHASE1_EPOCHS,
        phase2_epochs       = PHASE2_EPOCHS,
        steps_per_epoch     = STEPS_OVERRIDE,
        early_stop_patience = 6,   # was 4 — more patience with 400 steps/epoch
        save_path           = SAVE_PATH,
    )

    model.save("crop_monitor_final.keras")
    print("  Final model saved: crop_monitor_final.keras")

    plot_history(history)

    # ── Test on a single image after training ─────────────────────
    # Uncomment and replace with your image path:
    # result = run_inference(model, "test_leaf.jpg")