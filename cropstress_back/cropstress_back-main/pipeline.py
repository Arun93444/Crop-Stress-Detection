"""
================================================================================
  CROP MONITORING SYSTEM — UNIFIED BACKEND
  pipeline.py  |  FastAPI  |  MongoDB  |  GPT-4o-mini  |  Open-Meteo

  Endpoints:
    POST /api/auth/register
    POST /api/auth/login
    POST /api/upload              - image inference + store result
    GET  /api/history             - all past records
    GET  /api/latest              - most recent record
    GET  /api/weather             - Open-Meteo live weather
    POST /api/llm                 - GPT-4o-mini analysis
    GET  /api/alerts              - rule-based + trend alerts
    GET  /api/fields              - list fields
    POST /api/fields              - create field
    GET  /api/esp32/capture       - trigger ESP32 capture
    POST /api/esp32/scheduled     - background scheduler endpoint

  Install:
    pip install fastapi uvicorn python-multipart pymongo motor
                python-jose passlib bcrypt httpx openai python-dotenv
                aiofiles apscheduler

  Run:
    uvicorn pipeline:app --host 0.0.0.0 --port 8000 --reload
================================================================================
"""

import os
import io
import cv2
import base64
import hashlib
import logging
import numpy as np
import httpx
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Depends,
    BackgroundTasks, status, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

import motor.motor_asyncio
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop-monitor")

# ── Config ─────────────────────────────────────────────────────────────────────
MONGO_URI       = os.getenv("MONGO_URI", "mongodb+srv://vice50778_db_user:FmdAksIqxcYrHhSh@cluster0.skcyfo3.mongodb.net/?appName=Cluster0")
SECRET_KEY      = os.getenv("SECRET_KEY", "cropmonitor-secret-key-2024-change-in-production")
ALGORITHM       = "HS256"
TOKEN_EXPIRE    = 60 * 24  # 24 hours in minutes
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER       = os.getenv("SMTP_USER", "")
SMTP_PASS       = os.getenv("SMTP_PASS", "")
ESP32_IP        = os.getenv("ESP32_IP", "")   # Set after ESP32 connects
ESP32_SECRET    = os.getenv("ESP32_SECRET", "cropmonitor-esp32-key")  # Shared key for ESP32 auth
WEATHER_URL     = "https://api.open-meteo.com/v1/forecast?latitude=13.08&longitude=80.27&current_weather=true&hourly=relative_humidity_2m,precipitation"
UPLOAD_DIR      = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALERT_STRESS_THRESHOLD = 60
SEG_MODEL_PATH  = os.getenv("SEG_MODEL_PATH", "")
CLS_MODEL_PATH  = os.getenv("CLS_MODEL_PATH", "crop_monitor_best.keras")

# ── DB setup ───────────────────────────────────────────────────────────────────
client  = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db      = client["crop_monitor"]
users_col     = db["users"]
crop_col      = db["crop_data"]
fields_col    = db["fields"]
sessions_col  = db["sessions"]

# ── Auth ───────────────────────────────────────────────────────────────────────
pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await users_col.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ── ML Models (lazy-loaded) ────────────────────────────────────────────────────
_seg_model  = None
_cls_model  = None

def load_seg_model():
    global _seg_model
    if _seg_model is not None:
        return _seg_model
    try:
        from ultralytics import YOLO
        if SEG_MODEL_PATH and Path(SEG_MODEL_PATH).exists():
            _seg_model = YOLO(SEG_MODEL_PATH)
            logger.info(f"Segmentation model loaded: {SEG_MODEL_PATH}")
        else:
            for p in Path(".").rglob("crop_monitor_seg/weights/best.pt"):
                if p.stat().st_size > 1_000_000:
                    _seg_model = YOLO(str(p))
                    logger.info(f"Seg model found: {p}")
                    break
    except Exception as e:
        logger.warning(f"Seg model not loaded: {e}")
    return _seg_model

def load_cls_model():
    global _cls_model
    if _cls_model is not None:
        return _cls_model
    try:
        import tensorflow as tf
        if Path(CLS_MODEL_PATH).exists():
            _cls_model = tf.keras.models.load_model(CLS_MODEL_PATH, compile=False)
            logger.info("Classifier model loaded")
    except Exception as e:
        logger.warning(f"Cls model not loaded: {e}")
    return _cls_model

# ── Inference ──────────────────────────────────────────────────────────────────
STRESS_TYPE_INV = {
    0: "healthy_green", 1: "yellow_leaves", 2: "brown_rot",
    3: "rust_spots",    4: "wilting",        5: "necrosis"
}
SOIL_INV = {0: "dry", 1: "good_moisture", 2: "wet_waterlogged"}
DISEASE_NAMES = {
    "healthy_green": "No disease detected",
    "yellow_leaves": "Panama disease / Fusarium wilt / Chlorosis",
    "brown_rot":     "Black Sigatoka / Leaf spot / Anthracnose",
    "rust_spots":    "Rust disease / Leaf rust",
    "wilting":       "Water stress / Root rot",
    "necrosis":      "Bacterial wilt / Fire blight / Late blight",
}

# ── Alert type logic ───────────────────────────────────────────────────────────
def determine_alert_type(crop_health: str, stress_type: str, soil_condition: str) -> dict:
    """
    Smart alert classification:
      - Soil dry/wet but plant looks healthy → WATER_ALERT (don't panic about plant)
      - Soil good but plant stressed         → PLANT_STRESS_ALERT
      - Both bad                             → COMBINED_ALERT
      - All good                             → NONE
    """
    soil_problem = soil_condition in ("dry", "wet_waterlogged")
    plant_problem = crop_health == "stressed" and stress_type != "healthy_green"
    plant_ok = crop_health == "healthy" or stress_type == "healthy_green"

    if soil_problem and plant_ok:
        if soil_condition == "dry":
            return {
                "alert_type": "WATER_ALERT",
                "alert_message": "🌊 Soil is dry but plant looks healthy — irrigate now to prevent stress",
                "alert_severity": "WARN",
            }
        else:
            return {
                "alert_type": "DRAINAGE_ALERT",
                "alert_message": "💧 Soil is waterlogged but plant looks healthy — improve drainage to prevent root rot",
                "alert_severity": "WARN",
            }
    elif plant_problem and not soil_problem:
        return {
            "alert_type": "PLANT_STRESS_ALERT",
            "alert_message": f"🌿 Plant is stressed ({stress_type.replace('_',' ')}) but soil moisture is good — check for disease or pests",
            "alert_severity": "ALERT",
        }
    elif soil_problem and plant_problem:
        return {
            "alert_type": "COMBINED_ALERT",
            "alert_message": f"⚠️ Both plant stressed ({stress_type.replace('_',' ')}) and soil {soil_condition.replace('_',' ')} — urgent attention needed",
            "alert_severity": "CRITICAL",
        }
    else:
        return {
            "alert_type": "NONE",
            "alert_message": "✅ Plant and soil are both in good condition",
            "alert_severity": "OK",
        }
SEVERITY = {
    "healthy_green": "OK", "wilting": "WARN", "yellow_leaves": "ALERT",
    "rust_spots": "ALERT", "brown_rot": "ALERT", "necrosis": "CRITICAL",
    "unknown": "INFO"
}

def analyse_stress_hsv(img_rgb: np.ndarray) -> dict:
    """
    HSV stress + soil analysis with plant-presence check.
    OpenCV HSV: H=0-179, S=0-255, V=0-255

    Key fix: before classifying stress, check if image actually
    contains a plant. Brown pots/backgrounds should NOT trigger
    brown_rot. Only classify stress if plant pixels > 5%.
    """
    hsv     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    total   = float(h.size)

    def pct(mask): return float(np.sum(mask)) / total * 100

    # ── Plant colour masks ────────────────────────────────────
    # Healthy green: H 35-85, decent saturation & brightness
    green_mask    = (h >= 35)  & (h <= 85)  & (s > 50)  & (v > 50)

    # Yellow/chlorosis: H 18-36, high saturation (plant yellow, not background)
    yellow_mask   = (h >= 18)  & (h <= 36)  & (s > 90)  & (v > 110)

    # Brown rot on LEAF: raw candidate pixels (colour range only)
    # s>=100 already excludes most soil/pot browns (they sit at s~80-90),
    # but terracotta and dark soil can still reach s=170+.
    # So we add a SPATIAL FILTER: only count brown pixels that are
    # adjacent to green pixels — i.e. on a leaf, not a standalone object.
    brown_candidate = (h >= 6)  & (h <= 22)  & (s >= 100) & (s <= 210) & (v >= 55) & (v <= 175)
    # Dilate green mask by 20px — any brown pixel within 20px of green counts
    _green_u8   = green_mask.astype(np.uint8)
    _kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    _green_near = cv2.dilate(_green_u8, _kernel)
    brown_leaf_mask = brown_candidate & (_green_near > 0)   # only brown near green

    # Rust/orange spots: H 0-14, very high saturation
    rust_mask     = (h <= 14)  & (s > 130)  & (v > 80)

    # Necrosis/dead tissue: very dark AND low saturation (dead edges, black spots)
    # v < 40 catches genuinely dark eaten edges without triggering on dark-green leaves.
    # SPATIAL FILTER: only count dark pixels adjacent to green — pure-black image
    # borders and shadows far from the plant won't fire.
    necro_candidate = (v < 40)   & (s < 55)
    necro_mask      = necro_candidate & (_green_near > 0)

    # ── WHITE/PALE LESION MASK ────────────────────────────────
    # Detects: powdery mildew, scale insects, white spots, fungal lesions.
    # These appear as bright white or pale grey pixels ON the leaf surface.
    # Key signature: very high V (bright), very low S (desaturated = white/grey).
    # SPATIAL FILTER: only count pale pixels adjacent to green — a white
    # background/wall has no green neighbours; leaf lesions always do.
    lesion_candidate = (s < 25) & (v > 180)   # bright white / pale grey
    lesion_mask      = lesion_candidate & (_green_near > 0)   # only near green

    # Wilting: desaturated + mid-brightness, but ONLY when green is also low
    # (avoids white walls, pale backgrounds counting as wilted)
    wilt_mask     = (s < 30)   & (v > 100)  & (v < 200)

    # ── Spatial filter for rust ───────────────────────────────
    # Rust pixels alone (no green neighbours) = brown/orange object, not a plant.
    # Reuse _green_near (already computed above for brown filter).
    rust_leaf_mask  = rust_mask & (_green_near > 0)   # rust only near green
    # Recompute plant_mask with spatially-filtered rust
    plant_mask      = green_mask | yellow_mask | brown_leaf_mask | rust_leaf_mask

    # ── Plant presence check ──────────────────────────────────
    # Use a STRICTER presence mask: rust/brown ALONE don't count as plant.
    # Only green and yellow pixels (undeniably plant) determine presence.
    # This prevents terracotta pots or orange backgrounds from being
    # misclassified as a rusty stressed plant.
    has_plant_mask  = green_mask | yellow_mask
    has_plant_pct   = pct(has_plant_mask)
    has_plant       = has_plant_pct > 8.0 or (pct(plant_mask) > 15.0 and has_plant_pct > 2.0)

    # ── Soil colour masks ─────────────────────────────────────
    # Dry sandy soil: tan/beige, low-medium sat
    soil_dry_mask  = (h >= 10) & (h <= 25)  & (s >= 15) & (s <= 90)  & (v >= 80)  & (v <= 200)
    # Wet dark soil: dark brown, very low sat
    soil_wet_mask  = (h >= 5)  & (h <= 22)  & (s >= 10) & (s <= 70)  & (v >= 15)  & (v < 75)
    # Good moisture soil: medium brown
    soil_good_mask = (h >= 10) & (h <= 28)  & (s >= 25) & (s <= 90)  & (v >= 60)  & (v <= 150)

    g_pct     = pct(green_mask)
    y_pct     = pct(yellow_mask)
    b_pct     = pct(brown_leaf_mask)       # spatially-filtered brown (near green only)
    r_pct     = pct(rust_leaf_mask)        # spatially-filtered rust (near green only)
    n_pct     = pct(necro_mask)
    l_pct     = pct(lesion_mask)           # white/pale lesions on leaf (near green only)
    w_pct     = pct(wilt_mask)
    plant_pct = pct(plant_mask)
    sd_pct    = pct(soil_dry_mask)
    sw_pct    = pct(soil_wet_mask)
    sg_pct    = pct(soil_good_mask)
    # has_plant already computed above using has_plant_mask (green+yellow only)

    # ── Stress classification ─────────────────────────────────
    # Compute lesion and necrosis RELATIVE TO LEAF AREA (green pixels).
    # A few white spots on a large green leaf = small absolute % but
    # significant relative to the leaf surface → must be stressed.
    # Using absolute % caused the "healthy" miss on spotted leaves.
    leaf_area   = max(g_pct, 1.0)   # use green as proxy for leaf size
    l_rel       = l_pct / leaf_area * 100   # lesion % of leaf area
    n_rel       = n_pct / leaf_area * 100   # necro  % of leaf area

    # Weighted stress contribution from lesions and dark edges on the leaf
    # l_rel > 2% of leaf = noticeable spots; > 5% = significant infection
    lesion_stress = min(40.0, l_rel * 4.0)     # 2.5% lesion of leaf → 10 pts; 5% → 20 pts
    necro_stress  = min(25.0, n_rel * 1.5)     # 10% dark edges of leaf → 15 pts

    if not has_plant:
        # No meaningful plant pixels — soil-only or non-plant image
        stress_type  = "healthy_green"
        stress_score = 5.0

    elif g_pct > 45:
        # ── STRONG GREEN DOMINANCE ────────────────────────────
        stress_pct   = y_pct + b_pct + r_pct
        stress_ratio = stress_pct / max(g_pct, 1.0)
        extra_stress = lesion_stress + necro_stress   # now relative to leaf

        if extra_stress > 7:
            # Lesions/dark edges cover enough leaf surface = genuinely stressed
            stress_type  = "rust_spots" if l_rel > 2 else "necrosis"
            # base 42 + extra_stress (>=7) = >=49; with l_rel~2.6 → 42+10.4=52.4 ✓
            stress_score = min(85.0, 42 + extra_stress + stress_pct * 0.5)
        elif stress_ratio < 0.12 and extra_stress < 6:
            stress_type  = "healthy_green"
            stress_score = max(5.0, stress_pct * 1.5 + extra_stress * 0.3)
        elif stress_ratio < 0.35:
            stress_type  = "yellow_leaves" if y_pct >= b_pct else "brown_rot"
            stress_score = min(55.0, 20 + stress_pct * 1.2 + extra_stress * 0.5)
        else:
            stress_type  = "yellow_leaves" if y_pct >= b_pct else "brown_rot"
            stress_score = min(78.0, 40 + stress_pct * 1.1 + extra_stress * 0.4)

    elif g_pct > 20:
        # ── MODERATE GREEN ────────────────────────────────────
        stress_pct   = y_pct + b_pct + r_pct
        extra_stress = lesion_stress + necro_stress
        total_stress = stress_pct + extra_stress

        if total_stress < 8:
            stress_type  = "healthy_green"
            stress_score = max(8.0, total_stress * 2.5)
        elif total_stress < 25:
            stress_type  = "rust_spots" if (l_pct > b_pct and l_pct > y_pct) else ("yellow_leaves" if y_pct >= b_pct else "brown_rot")
            stress_score = min(65.0, 28 + total_stress * 1.6)
        else:
            stress_type  = "rust_spots" if (l_pct > b_pct and l_pct > y_pct) else ("yellow_leaves" if y_pct >= b_pct else "brown_rot")
            stress_score = min(85.0, 45 + total_stress * 1.3)

    elif l_pct > 1 or (n_pct > 5 and l_pct > 0.5):
        # Low green but lesions/damage present
        stress_type  = "rust_spots" if l_pct >= n_pct else "necrosis"
        stress_score = min(80.0, 35 + lesion_stress + necro_stress)

    elif y_pct > 15:
        stress_type  = "yellow_leaves"
        stress_score = min(100.0, 45 + y_pct * 1.8)

    elif b_pct > 18:
        stress_type  = "brown_rot"
        stress_score = min(100.0, 48 + b_pct * 1.8)

    elif r_pct > 10:
        stress_type  = "rust_spots"
        stress_score = min(100.0, 42 + r_pct * 2.5)

    elif n_pct > 18 or (n_pct > 8 and l_pct > 1):
        stress_type  = "necrosis"
        stress_score = min(100.0, 55 + n_pct * 1.5 + l_pct * 1.0)

    elif w_pct > 35 and g_pct < 15:
        stress_type  = "wilting"
        stress_score = min(100.0, 32 + w_pct * 1.0)

    elif y_pct > 8 or b_pct > 12:
        stress_type  = "yellow_leaves" if y_pct > b_pct else "brown_rot"
        stress_score = min(55.0, 22 + (y_pct + b_pct) * 1.8)

    else:
        stress_type  = "healthy_green"
        stress_score = max(5.0, min(22.0, (y_pct + b_pct + l_pct) * 2.0))

    # ── Soil detection ────────────────────────────────────────
    # Soil pixels must be present AND image not dominated by green plant
    total_soil_pct = sd_pct + sw_pct + sg_pct
    # Only detect soil if enough soil pixels and not just green plant
    if total_soil_pct > 10 and g_pct < 65:
        if sw_pct > sg_pct and sw_pct > sd_pct:
            soil_condition = "wet_waterlogged"
        elif sg_pct >= sd_pct:
            soil_condition = "good_moisture"
        else:
            soil_condition = "dry"
    elif total_soil_pct > 5 and plant_pct < 20:
        # Mostly soil image
        soil_condition = "dry" if v.mean() > 100 else "good_moisture"
    else:
        soil_condition = "not_detected"

    return {
        "stress_type":    stress_type,
        "stress_score":   round(stress_score, 1),
        "soil_condition": soil_condition,
        "has_plant":      has_plant,
        "colour_analysis": {
            "green_pct":  round(g_pct, 1),
            "yellow_pct": round(y_pct, 1),
            "brown_pct":  round(b_pct, 1),
            "rust_pct":   round(r_pct, 1),
            "necro_pct":  round(n_pct, 1),
            "lesion_pct": round(l_pct, 1),
            "soil_pct":   round(total_soil_pct, 1),
        }
    }

def run_full_inference(img_bytes: bytes) -> dict:
    """Run segmentation + classification on image bytes."""
    nparr  = np.frombuffer(img_bytes, np.uint8)
    img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img.shape[:2]

    # HSV analysis always runs
    hsv_result = analyse_stress_hsv(img_rgb)

    # Seed result from HSV (always reliable, no model needed)
    result = {
        "crop_health":    "unknown",
        "stress_score":   hsv_result["stress_score"],
        "stress_type":    hsv_result["stress_type"],
        "soil_condition": hsv_result["soil_condition"],   # HSV soil detection
        "confidence":     0.0,
        "colour_analysis": hsv_result["colour_analysis"],
        "disease_name":   DISEASE_NAMES.get(hsv_result["stress_type"], "Unknown"),
        "severity":       SEVERITY.get(hsv_result["stress_type"], "INFO"),
        "plant_type":     "unknown",
        "zones":          [],
        "model_used":     "hsv_only",
    }

    # Try segmentation model — refine stress on detected plant patch
    seg = load_seg_model()
    if seg:
        try:
            seg_res = seg(img_rgb, conf=0.12, iou=0.45, verbose=False)
            zones   = []
            for r in seg_res:
                if r.boxes is None:
                    continue
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    zones.append({
                        "class":  "plant" if cls_id == 0 else "soil",
                        "conf":   round(conf, 3),
                        "bbox":   [x1, y1, x2, y2],
                    })
            result["zones"]      = zones
            result["model_used"] = "yolo+hsv"

            if zones:
                plant_zones = [z for z in zones if z["class"] == "plant"]
                soil_zones  = [z for z in zones if z["class"] == "soil"]

                if plant_zones:
                    # Run HSV on the best plant patch for more focused analysis
                    best = max(plant_zones, key=lambda z: z["conf"])
                    bx   = best["bbox"]
                    ph, pw = bx[3]-bx[1], bx[2]-bx[0]
                    if ph > 20 and pw > 20:
                        patch = img_rgb[bx[1]:bx[3], bx[0]:bx[2]]
                        if patch.size > 0:
                            ph_result = analyse_stress_hsv(patch)
                            # Blend patch score with full-image score (60/40)
                            blended_score = ph_result["stress_score"] * 0.6 + result["stress_score"] * 0.4
                            result["stress_type"]  = ph_result["stress_type"]
                            result["stress_score"] = round(blended_score, 1)
                    result["confidence"] = round(best["conf"] * 100, 1)

                # Soil from segmentation — if model detected soil zone, trust it
                if soil_zones:
                    best_soil = max(soil_zones, key=lambda z: z["conf"])
                    bx = best_soil["bbox"]
                    sh, sw = bx[3]-bx[1], bx[2]-bx[0]
                    if sh > 15 and sw > 15:
                        soil_patch = img_rgb[bx[1]:bx[3], bx[0]:bx[2]]
                        sp_result  = analyse_stress_hsv(soil_patch)
                        # If HSV says soil condition on this patch, use it
                        if sp_result["soil_condition"] != "not_detected":
                            result["soil_condition"] = sp_result["soil_condition"]
                        else:
                            # Fallback: classify by patch brightness
                            patch_v = cv2.cvtColor(soil_patch, cv2.COLOR_RGB2HSV)[:,:,2].mean()
                            if patch_v < 60:
                                result["soil_condition"] = "wet_waterlogged"
                            elif patch_v < 120:
                                result["soil_condition"] = "good_moisture"
                            else:
                                result["soil_condition"] = "dry"
        except Exception as e:
            logger.warning(f"Seg inference error: {e}")

    # Try classifier model — highest priority if loaded
    cls = load_cls_model()
    if cls:
        try:
            import tensorflow as tf
            cls_input = cv2.resize(img_rgb, (224, 224)).astype(np.float32)
            cls_input = tf.keras.applications.mobilenet_v3.preprocess_input(cls_input[np.newaxis])
            preds = cls.predict(cls_input, verbose=0)
            p_crop, p_soil, p_stress, p_soil_pres, p_stype = preds

            cls_stressed_prob = float(p_crop[0][1])   # 0-1
            cls_healthy_prob  = float(p_crop[0][0])   # 0-1
            cls_stress_raw    = float(np.clip(p_stress[0, 0] * 100, 0, 100))
            stype_idx         = int(np.argmax(p_stype[0]))
            stype_conf        = float(p_stype[0][stype_idx]) * 100
            soil_pres         = float(p_soil_pres[0, 0])
            soil_cond         = SOIL_INV[int(np.argmax(p_soil[0]))]

            # ── Stress score: weighted blend ──────────────────
            # Core problem with old logic:
            #   Gate was 0.65 — a 60% confident healthy prediction fell
            #   into "uncertain" and blended 50/50 with a bad HSV score.
            #   Fix: lower gate to 0.55 so moderately confident predictions
            #   are respected, and give healthy classifier more authority.
            hsv_score = result["stress_score"]

            if cls_healthy_prob > 0.55:
                # Classifier says healthy with reasonable confidence.
                # Weight classifier 70%, HSV 30%. Cap at 48 (below 50 threshold).
                blended_stress = cls_stress_raw * 0.25 + hsv_score * 0.75
                blended_stress = min(blended_stress, 48.0)
                is_stressed    = blended_stress > 50

            elif cls_stressed_prob > 0.55:
                # Classifier confident stressed.
                # Weight classifier 65%, HSV 35%. Floor at 50.
                blended_stress = cls_stress_raw * 0.65 + hsv_score * 0.35
                blended_stress = max(blended_stress, 50.0)
                is_stressed    = True

            else:
                # Both probabilities < 0.55 — genuinely uncertain.
                # Use 40/60 classifier/HSV but require score > 50 for stress.
                # This prevents HSV noise from flipping a borderline case.
                blended_stress = cls_stress_raw * 0.4 + hsv_score * 0.6
                is_stressed    = blended_stress > 50

            # ── Stress type: use classifier if confident ──────
            # Lower gate from 60% → 50% so classifier wins more often.
            # Also: if classifier says healthy, force healthy_green regardless.
            if not is_stressed:
                final_stress_type = "healthy_green"
            elif stype_conf > 50:
                final_stress_type = STRESS_TYPE_INV.get(stype_idx, result["stress_type"])
            else:
                final_stress_type = result["stress_type"]

            # ── Soil: use classifier only when it's confident ─
            # Old gate was 0.3 — too low, weak soil signals overrode
            # good HSV soil reads. Raise to 0.45.
            if soil_pres > 0.45:
                final_soil = soil_cond
            elif result["soil_condition"] != "not_detected":
                final_soil = result["soil_condition"]
            else:
                final_soil = "not_detected"

            result.update({
                "crop_health":    "stressed" if is_stressed else "healthy",
                "stress_score":   round(blended_stress, 1),
                "stress_type":    final_stress_type,
                "soil_condition": final_soil,
                "confidence":     round(max(cls_stressed_prob, cls_healthy_prob) * 100, 1),
                "model_used":     "yolo+classifier",
            })
            result["disease_name"] = DISEASE_NAMES.get(result["stress_type"], "Unknown")
            result["severity"]     = SEVERITY.get(result["stress_type"], "INFO")
        except Exception as e:
            logger.warning(f"Cls inference error: {e}")

    # Final health label from stress score if still unknown
    # Raised threshold from 42 → 50. A score of 42-49 is borderline/mild —
    # it should NOT be labelled "stressed". Stressed means genuinely sick.
    if result["crop_health"] == "unknown":
        result["crop_health"] = "stressed" if result["stress_score"] > 50 else "healthy"

    # Ensure severity is always set
    result["disease_name"] = DISEASE_NAMES.get(result["stress_type"], "Unknown")
    result["severity"]     = SEVERITY.get(result["stress_type"], "INFO")

    # ── Smart alert type: soil vs plant vs combined ───────────────
    alert_info = determine_alert_type(
        result["crop_health"], result["stress_type"], result["soil_condition"]
    )
    result.update(alert_info)

    logger.info(
        f"Inference done | health={result['crop_health']} score={result['stress_score']} "
        f"type={result['stress_type']} soil={result['soil_condition']} "
        f"alert={result['alert_type']} model={result['model_used']}"
    )
    return result

# ── Weather ────────────────────────────────────────────────────────────────────
_weather_cache = {"data": None, "ts": None}

async def fetch_weather() -> dict:
    now = datetime.utcnow()
    if _weather_cache["data"] and _weather_cache["ts"]:
        age = (now - _weather_cache["ts"]).total_seconds()
        if age < 600:  # 10 min cache
            return _weather_cache["data"]
    try:
        async with httpx.AsyncClient(timeout=10) as client_http:
            r = await client_http.get(WEATHER_URL)
            r.raise_for_status()
            raw = r.json()
            cw  = raw.get("current_weather", {})
            hourly = raw.get("hourly", {})
            humidity = hourly.get("relative_humidity_2m", [None])[0]
            data = {
                "temperature": cw.get("temperature"),
                "windspeed":   cw.get("windspeed"),
                "weathercode": cw.get("weathercode"),
                "is_day":      cw.get("is_day"),
                "humidity":    humidity,
                "location":    "Chennai, Tamil Nadu",
                "timestamp":   now.isoformat(),
            }
            _weather_cache["data"] = data
            _weather_cache["ts"]   = now
            return data
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return {"error": str(e), "temperature": None, "humidity": None}

# ── LLM ───────────────────────────────────────────────────────────────────────
async def call_llm(
    crop_health: str, stress_score: float, stress_type: str,
    soil_condition: str, temperature: float, humidity: float,
    language: str = "english", plant_type: str = "unknown",
    alert_type: str = "NONE"
) -> dict:
    # Normalise: frontend sends "ta", some callers send "tamil" — treat both as tamil
    language = "tamil" if language.lower() in ("ta", "tamil") else "english"

    # Build context-aware situation summary
    if alert_type == "WATER_ALERT":
        situation = (f"IMPORTANT: The soil is {soil_condition} but the plant itself looks healthy. "
                     f"This is a SOIL MOISTURE ALERT only — the plant is NOT diseased. "
                     f"Focus remedies on irrigation/water management, NOT on disease treatment.")
    elif alert_type == "PLANT_STRESS_ALERT":
        situation = (f"IMPORTANT: The soil moisture is good ({soil_condition}) but the plant is stressed "
                     f"with {stress_type.replace('_',' ')}. The problem is NOT water-related. "
                     f"Focus remedies on disease/pest control and plant-specific treatment.")
    elif alert_type == "DRAINAGE_ALERT":
        situation = (f"IMPORTANT: The soil is waterlogged but the plant looks healthy. "
                     f"This is a DRAINAGE ALERT — act before root rot develops.")
    elif alert_type == "COMBINED_ALERT":
        situation = (f"CRITICAL: Both soil ({soil_condition}) and plant ({stress_type.replace('_',' ')}) "
                     f"have problems simultaneously. Address both issues urgently.")
    else:
        situation = "The crop and soil are both in acceptable condition."

    plant_context = (
        f"Plant type identified: {plant_type}. Tailor all advice specifically for {plant_type}."
        if plant_type and plant_type not in ("unknown", "")
        else (
            "Plant type is unknown. Based on the stress type, soil condition, and disease pattern, "
            "make your best educated guess of the likely plant type (e.g. tomato, banana, rice, chilli, "
            "maize, cotton, mango — common Tamil Nadu crops). Put your guess in plant_identified."
        )
    )

    tamil_header = ""

    prompt = f"""You are an expert agricultural scientist specializing in crop disease and soil management.

Analyze this crop monitoring data and provide detailed, PLANT-SPECIFIC recommendations:

Plant Type: {plant_type}
Crop Health: {crop_health}
Stress Score: {stress_score}/100
Stress Type: {stress_type}
Soil Condition: {soil_condition}
Alert Type: {alert_type}
Temperature: {temperature}°C
Humidity: {humidity}%

Situation Context: {situation}
{plant_context}

Respond with a JSON object with these exact keys:
{{
  "plant_identified": "best guess of plant type from visual cues (e.g. tomato, banana, rice, unknown)",
  "explanation": "what is happening — clearly distinguish if the issue is soil-only, plant-only, or both (2-3 sentences)",
  "cause": "root cause specific to this plant type and condition",
  "remedy": "numbered steps — if WATER_ALERT mention only irrigation steps; if PLANT_STRESS_ALERT mention disease/pest steps specific to this plant type",
  "prevention": "long-term prevention for this specific plant type",
  "urgency": "LOW | MEDIUM | HIGH | CRITICAL",
  "fertilizer_tip": "specific fertilizer for this plant type — skip if water alert only",
  "irrigation_advice": "precise watering advice based on soil condition and plant type"
}}

Return ONLY valid JSON, no markdown."""

    system_message = (
        "You are an expert agricultural scientist. "
        "You MUST respond ENTIRELY in Tamil language (தமிழ்). "
        "Every word in your JSON values must be in Tamil script. "
        "Do NOT use any English words inside the JSON values. JSON keys stay in English."
        if language == "tamil"
        else "You are an expert agricultural scientist. Respond in clear English."
    )

    if OPENAI_API_KEY:
        try:
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.3,
                max_tokens=900,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            import json
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")
            raise HTTPException(status_code=503, detail=f"GPT analysis unavailable: {e}")
    else:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured. Set OPENAI_API_KEY in environment.")

# ── Email Alerts ───────────────────────────────────────────────────────────────
def send_email_alert(to_email: str, subject: str, body: str):
    if not SMTP_USER or not SMTP_PASS:
        logger.info(f"Email not configured. Would send to {to_email}: {subject}")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = to_email
        html = f"""
        <html><body style="font-family:sans-serif;padding:20px">
        <h2 style="color:#e53e3e">🌱 Crop Monitor Alert</h2>
        <p>{body}</p>
        <hr><small>Crop Monitoring System — Auto Alert</small>
        </body></html>"""
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        logger.info(f"Alert email sent to {to_email}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")

# ── Trend Analysis ─────────────────────────────────────────────────────────────
async def compute_trend(field_id: Optional[str] = None) -> dict:
    query = {}
    if field_id:
        query["field_id"] = field_id
    cursor = crop_col.find(query).sort("timestamp", -1).limit(10)
    records = await cursor.to_list(10)
    if len(records) < 2:
        return {"trend": "insufficient_data", "direction": None, "scores": []}

    scores = [r.get("stress_score", 0) for r in reversed(records)]
    diffs  = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0

    if avg_diff > 3:
        direction = "increasing"
    elif avg_diff < -3:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "trend":     direction,
        "avg_change": round(avg_diff, 2),
        "scores":    scores,
        "direction": direction,
    }

# ── Scheduler (ESP32 auto-capture every 4h) ────────────────────────────────────
scheduler = AsyncIOScheduler()

async def scheduled_esp32_capture():
    if not ESP32_IP:
        return
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"http://{ESP32_IP}/capture")
            if r.status_code == 200:
                img_bytes = r.content
                result    = run_full_inference(img_bytes)
                fname     = f"auto_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
                fpath     = UPLOAD_DIR / fname
                with open(fpath, "wb") as f:
                    f.write(img_bytes)
                doc = {
                    "timestamp":      datetime.utcnow().isoformat(),
                    "source":         "esp32_auto",
                    "image_path":     str(fpath),
                    "crop_health":    result["crop_health"],
                    "stress_score":   result["stress_score"],
                    "stress_type":    result["stress_type"],
                    "soil_condition": result["soil_condition"],
                    "confidence":     result["confidence"],
                    "zones":          result["zones"],
                    "severity":       result["severity"],
                    "disease_name":   result["disease_name"],
                }
                await crop_col.insert_one(doc)
                logger.info(f"Auto-capture saved: {fname}")
    except Exception as e:
        logger.warning(f"Scheduled capture failed: {e}")

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(scheduled_esp32_capture, "interval", hours=4, id="esp32_capture")
    scheduler.start()
    logger.info("Scheduler started (ESP32 every 4h)")
    yield
    scheduler.shutdown()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Crop Monitor API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ────────────────────────────────────────────────────────────
class RegisterReq(BaseModel):
    name:     str
    email:    str
    password: str

class LoginReq(BaseModel):
    email:    str
    password: str

class LLMReq(BaseModel):
    crop_health:    str
    stress_score:   float
    stress_type:    str
    soil_condition: str
    language:       str = "english"
    plant_type:     str = "unknown"
    alert_type:     str = "NONE"

class FieldReq(BaseModel):
    field_name: str
    latitude:   float
    longitude:  float

# ── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/api/auth/register")
async def register(req: RegisterReq):
    existing = await users_col.find_one({"email": req.email})
    if existing:
        raise HTTPException(400, "Email already registered")
    user_id = hashlib.md5(req.email.encode()).hexdigest()
    user = {
        "_id":      user_id,
        "name":     req.name,
        "email":    req.email,
        "password": hash_password(req.password),
        "created":  datetime.utcnow().isoformat(),
    }
    await users_col.insert_one(user)
    token = create_token({"sub": user_id, "email": req.email, "name": req.name})
    return {"token": token, "user": {"id": user_id, "name": req.name, "email": req.email}}

@app.post("/api/auth/login")
async def login(req: LoginReq):
    user = await users_col.find_one({"email": req.email})
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")
    token = create_token({"sub": user["_id"], "email": user["email"], "name": user["name"]})
    return {"token": token, "user": {"id": user["_id"], "name": user["name"], "email": user["email"]}}

# ── Upload / Inference ─────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file:     UploadFile = File(...),
    field_id: Optional[str] = None,
    source:   str = "manual",
    user = Depends(get_current_user),
):
    img_bytes = await file.read()
    if len(img_bytes) == 0:
        raise HTTPException(400, "Empty file")

    # Run inference
    try:
        result = run_full_inference(img_bytes)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    # Save image
    fname = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    fpath = UPLOAD_DIR / fname
    with open(fpath, "wb") as f:
        f.write(img_bytes)

    # Encode thumbnail
    nparr  = np.frombuffer(img_bytes, np.uint8)
    img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    small  = cv2.resize(img, (320, 240))
    _, buf  = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    thumb64 = base64.b64encode(buf).decode()

    doc = {
        "user_id":        user["_id"],
        "field_id":       field_id,
        "timestamp":      datetime.utcnow().isoformat(),
        "source":         source,
        "image_path":     str(fpath),
        "thumbnail":      thumb64,
        "crop_health":    result["crop_health"],
        "stress_score":   result["stress_score"],
        "stress_type":    result["stress_type"],
        "soil_condition": result["soil_condition"],
        "confidence":     result["confidence"],
        "zones":          result["zones"],
        "severity":       result["severity"],
        "disease_name":   result["disease_name"],
        "colour_analysis": result.get("colour_analysis", {}),
        "model_used":     result["model_used"],
        "alert_type":     result.get("alert_type", "NONE"),
        "alert_message":  result.get("alert_message", ""),
        "alert_severity": result.get("alert_severity", "OK"),
        "plant_type":     result.get("plant_type", "unknown"),
    }
    res = await crop_col.insert_one(doc)
    doc["_id"] = str(res.inserted_id)

    # Background: check alerts and email
    async def check_and_alert():
        alert_type = result.get("alert_type", "NONE")
        should_alert = (
            result["stress_score"] > ALERT_STRESS_THRESHOLD or
            alert_type in ("WATER_ALERT", "DRAINAGE_ALERT", "COMBINED_ALERT", "PLANT_STRESS_ALERT")
        )
        if should_alert:
            user_doc = await users_col.find_one({"_id": user["_id"]})
            if user_doc:
                body = result.get("alert_message") or (
                    f"Alert: {alert_type}. Score: {result['stress_score']}/100. "
                    f"Soil: {result['soil_condition']}. Type: {result['stress_type']}."
                )
                send_email_alert(user_doc["email"], f"🚨 Crop Monitor Alert: {alert_type}", body)
    background_tasks.add_task(check_and_alert)

    return {**doc, "thumbnail": thumb64}

# ── History ────────────────────────────────────────────────────────────────────
@app.get("/api/history")
async def get_history(
    limit:    int = 50,
    field_id: Optional[str] = None,
    user = Depends(get_current_user),
):
    query = {"user_id": user["_id"]}
    if field_id:
        query["field_id"] = field_id
    cursor  = crop_col.find(query).sort("timestamp", -1).limit(limit)
    records = await cursor.to_list(limit)
    for r in records:
        r["_id"] = str(r["_id"])
        r.pop("thumbnail", None)  # don't send in list
    return records

# ── Latest ─────────────────────────────────────────────────────────────────────
@app.get("/api/latest")
async def get_latest(user = Depends(get_current_user)):
    doc = await crop_col.find_one(
        {"user_id": user["_id"]},
        sort=[("timestamp", -1)]
    )
    if not doc:
        return {}
    doc["_id"] = str(doc["_id"])
    return doc

# ── Weather ────────────────────────────────────────────────────────────────────
@app.get("/api/weather")
async def get_weather():
    return await fetch_weather()

# ── LLM ───────────────────────────────────────────────────────────────────────
@app.post("/api/llm")
async def llm_analysis(req: LLMReq, user = Depends(get_current_user)):
    weather = await fetch_weather()
    result  = await call_llm(
        crop_health    = req.crop_health,
        stress_score   = req.stress_score,
        stress_type    = req.stress_type,
        soil_condition = req.soil_condition,
        temperature    = weather.get("temperature") or 30.0,
        humidity       = weather.get("humidity") or 65.0,
        language       = req.language,
        plant_type     = req.plant_type,
        alert_type     = req.alert_type,
    )
    return result

# ── Alerts ─────────────────────────────────────────────────────────────────────
@app.get("/api/alerts")
async def get_alerts(user = Depends(get_current_user)):
    trend   = await compute_trend()
    records = await crop_col.find({"user_id": user["_id"]}).sort("timestamp", -1).limit(20).to_list(20)
    alerts  = []
    for r in records:
        if r.get("stress_score", 0) > ALERT_STRESS_THRESHOLD:
            alerts.append({
                "id":        str(r["_id"]),
                "type":      "stress_threshold",
                "message":   f"High stress detected: {r['stress_score']}/100 ({r.get('stress_type','')})",
                "severity":  r.get("severity", "ALERT"),
                "timestamp": r["timestamp"],
                "field_id":  r.get("field_id"),
            })
        if r.get("severity") == "CRITICAL":
            alerts.append({
                "id":        str(r["_id"]) + "_crit",
                "type":      "critical_disease",
                "message":   f"CRITICAL: {r.get('disease_name', 'Unknown disease')} detected",
                "severity":  "CRITICAL",
                "timestamp": r["timestamp"],
                "field_id":  r.get("field_id"),
            })

    if trend["direction"] == "increasing" and len(trend["scores"]) >= 3:
        alerts.insert(0, {
            "id":       "trend_warn",
            "type":     "trend",
            "message":  f"Stress is increasing over last {len(trend['scores'])} readings. Avg change: +{trend['avg_change']} pts/reading",
            "severity": "WARN",
            "timestamp": datetime.utcnow().isoformat(),
        })

    return {"alerts": alerts[:20], "trend": trend}

# ── Fields ─────────────────────────────────────────────────────────────────────
@app.get("/api/fields")
async def list_fields(user = Depends(get_current_user)):
    cursor = fields_col.find({"user_id": user["_id"]})
    fields = await cursor.to_list(50)
    for f in fields:
        f["_id"] = str(f["_id"])
    return fields

@app.post("/api/fields")
async def create_field(req: FieldReq, user = Depends(get_current_user)):
    doc = {
        "user_id":    user["_id"],
        "field_name": req.field_name,
        "latitude":   req.latitude,
        "longitude":  req.longitude,
        "created":    datetime.utcnow().isoformat(),
    }
    res = await fields_col.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    return doc

# ── ESP32 trigger ──────────────────────────────────────────────────────────────
@app.get("/api/esp32/capture")
async def trigger_esp32_capture(user = Depends(get_current_user)):
    """Manually trigger ESP32 camera capture from UI."""
    esp_ip = ESP32_IP
    if not esp_ip:
        raise HTTPException(400, "ESP32 IP not configured. Check ESP32_IP env variable.")
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"http://{esp_ip}/capture")
        if r.status_code != 200:
            raise HTTPException(502, "ESP32 did not respond correctly")
        result = run_full_inference(r.content)
        fname  = f"esp32_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
        fpath  = UPLOAD_DIR / fname
        with open(fpath, "wb") as f:
            f.write(r.content)
        nparr   = np.frombuffer(r.content, np.uint8)
        img     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        small   = cv2.resize(img, (320, 240))
        _, buf  = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        thumb64 = base64.b64encode(buf).decode()
        doc = {
            "user_id": user["_id"], "field_id": None,
            "timestamp": datetime.utcnow().isoformat(), "source": "esp32_manual",
            "image_path": str(fpath), "thumbnail": thumb64,
            "crop_health":    result["crop_health"],
            "stress_score":   result["stress_score"],
            "stress_type":    result["stress_type"],
            "soil_condition": result["soil_condition"],
            "confidence":     result["confidence"],
            "severity":       result["severity"],
            "disease_name":   result["disease_name"],
            "zones":          result.get("zones", []),
            "colour_analysis": result.get("colour_analysis", {}),
            "model_used":     result.get("model_used", "hsv_only"),
            "alert_type":     result.get("alert_type", "NONE"),
            "alert_message":  result.get("alert_message", ""),
            "alert_severity": result.get("alert_severity", "OK"),
            "plant_type":     result.get("plant_type", "unknown"),
        }
        res = await crop_col.insert_one(doc)
        doc["_id"] = str(res.inserted_id)
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"ESP32 capture error: {e}")

@app.post("/api/esp32/auto-upload")
async def esp32_auto_upload(request: Request, key: str = ""):
    """
    Called by ESP32 device directly — no JWT needed.
    Authenticated via shared secret key query param: ?key=cropmonitor-esp32-key
    Accepts raw JPEG bytes in request body.
    """
    if key != ESP32_SECRET:
        raise HTTPException(403, "Invalid ESP32 key")
    body = await request.body()
    if not body or len(body) < 1000:
        raise HTTPException(400, "No valid image data received")
    try:
        fname  = f"esp32_auto_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
        fpath  = UPLOAD_DIR / fname
        fpath.write_bytes(body)
        img_array = np.frombuffer(body, dtype=np.uint8)
        img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")
        result = run_inference(img)
        _, buf  = cv2.imencode('.jpg', cv2.resize(img, (320, 240)))
        thumb64 = base64.b64encode(buf).decode()
        doc = {
            "user_id": None, "field_id": None,
            "timestamp": datetime.utcnow().isoformat(), "source": "esp32_auto",
            "image_path": str(fpath), "thumbnail": thumb64,
            "crop_health":    result["crop_health"],
            "stress_score":   result["stress_score"],
            "stress_type":    result["stress_type"],
            "soil_condition": result["soil_condition"],
            "confidence":     result["confidence"],
            "severity":       result["severity"],
            "disease_name":   result["disease_name"],
            "zones":          result.get("zones", []),
            "colour_analysis": result.get("colour_analysis", {}),
            "model_used":     result.get("model_used", "hsv_only"),
            "alert_type":     result.get("alert_type", "NONE"),
            "alert_message":  result.get("alert_message", ""),
            "alert_severity": result.get("alert_severity", "OK"),
            "plant_type":     result.get("plant_type", "unknown"),
        }
        await db.readings.insert_one(doc)
        logger.info(f"ESP32 auto-upload saved: {fname}")
        return {"status": "ok", "crop_health": result["crop_health"], "stress_score": result["stress_score"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Auto-upload error: {e}")

@app.get("/api/esp32/register-ip")
@app.post("/api/esp32/register-ip")
async def register_esp32_ip(ip: str):
    """Called by ESP32 on boot to register its IP. Also persists to .env."""
    global ESP32_IP
    ESP32_IP = ip
    logger.info(f"ESP32 registered at {ip}")
    # Persist to .env so IP survives server restarts
    try:
        env_path = Path(".env")
        if env_path.exists():
            text = env_path.read_text()
            import re
            if re.search(r"^ESP32_IP\s*=", text, re.MULTILINE):
                text = re.sub(r"^(ESP32_IP\s*=).*$", f"ESP32_IP={ip}", text, flags=re.MULTILINE)
            else:
                text += f"\nESP32_IP={ip}\n"
            env_path.write_text(text)
            logger.info(f"ESP32 IP saved to .env: {ip}")
    except Exception as e:
        logger.warning(f"Could not save ESP32 IP to .env: {e}")
    return {"status": "ok", "ip": ip}

@app.get("/api/esp32/status")
async def esp32_status(user = Depends(get_current_user)):
    """Return current known ESP32 IP and reachability."""
    reachable = False
    if ESP32_IP:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"http://{ESP32_IP}/status")
                reachable = r.status_code == 200
        except Exception:
            reachable = False
    return {"ip": ESP32_IP or "", "reachable": reachable}

@app.post("/api/esp32/set-ip")
async def set_esp32_ip(request: Request, user = Depends(get_current_user)):
    """Manually set ESP32 IP from the UI."""
    global ESP32_IP
    body = await request.json()
    ip = body.get("ip", "").strip()
    if not ip:
        raise HTTPException(400, "IP address required")
    ESP32_IP = ip
    logger.info(f"ESP32 IP manually set to {ip}")
    try:
        env_path = Path(".env")
        if env_path.exists():
            text = env_path.read_text()
            import re
            if re.search(r"^ESP32_IP\s*=", text, re.MULTILINE):
                text = re.sub(r"^(ESP32_IP\s*=).*$", f"ESP32_IP={ip}", text, flags=re.MULTILINE)
            else:
                text += f"\nESP32_IP={ip}\n"
            env_path.write_text(text)
    except Exception as e:
        logger.warning(f"Could not save IP to .env: {e}")
    return {"status": "ok", "ip": ip}

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pipeline:app", host="0.0.0.0", port=8000, reload=True)