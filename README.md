# 🌱 Crop Stress Detection — AI-Powered Crop Health Monitoring System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)
![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNetV3-orange?logo=tensorflow)

An end-to-end AI-powered crop health monitoring system that detects plant stress, soil conditions, and disease types from field images using computer vision and deep learning.

---

## 🚀 Live Demo

> Upload a crop image → Get instant AI diagnosis → View stress score, soil condition, disease type, and remedies

---

## ✨ Features

- 🔬 **AI Crop Health Detection** — Detects healthy vs stressed crops using HSV color analysis + MobileNetV3Large classifier
- 🌿 **Stress Type Classification** — Identifies 6 stress types: Healthy, Yellow Leaves, Brown Rot, Rust Spots, Wilting, Necrosis
- 🪨 **Soil Condition Analysis** — Classifies soil as Dry, Good Moisture, or Wet/Waterlogged
- 📊 **Stress Score** — Gives a 0-100 stress severity score
- 🌤 **Live Weather Integration** — Real-time weather data via Open-Meteo API (Chennai, Tamil Nadu)
- 🤖 **AI Analysis** — GPT-4o-mini powered agricultural recommendations with remedy, prevention, and fertilizer tips
- 📈 **History & Trend Charts** — Track crop stress over time with interactive charts
- 🔔 **Smart Alerts** — Water alerts, drainage alerts, plant stress alerts, combined alerts
- 🌐 **Bilingual UI** — Full English and Tamil (தமிழ்) language support
- 📡 **ESP32-CAM Support** — Auto-capture from field camera every 4 hours
- 🔐 **JWT Authentication** — Secure login/register with bcrypt password hashing
- 📧 **Email Alerts** — Automatic email notifications for critical stress levels

---

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| FastAPI | REST API framework |
| TensorFlow / Keras | MobileNetV3Large crop classifier (5 output heads) |
| OpenCV + HSV Analysis | Computer vision stress detection |
| MongoDB Atlas + Motor | Async database (users, crop data, fields) |
| Python-Jose + Passlib | JWT auth + bcrypt password hashing |
| APScheduler | Background ESP32 auto-capture every 4h |
| OpenAI GPT-4o-mini | Agricultural AI analysis & recommendations |
| Open-Meteo API | Live weather data |
| Uvicorn | ASGI server |

### Frontend
| Technology | Purpose |
|---|---|
| React 19 + Vite | Frontend framework |
| Recharts | Stress trend charts |
| Context API | Global state management |
| CSS Variables | Dark/light mode theming |

---

## 🧠 ML Model Architecture

```
MobileNetV3Large Backbone (3.7M params, ImageNet pretrained)
         │
    GlobalAveragePooling2D
         │
    Shared Dense (256) + BatchNorm
         │
   ┌─────┼──────┬──────────┬─────────────┐
   │     │      │          │             │
disease  soil  stress  soil_present  stress_type
 (2cls) (3cls) (0-100)   (binary)     (6cls)
```

**5 Output Heads:**
1. `disease_out` — Healthy / Stressed (binary softmax)
2. `soil_out` — Dry / Good / Wet (3-class softmax)
3. `stress_out` — Severity 0–100 (sigmoid regression)
4. `soil_present_out` — Soil visible yes/no (binary sigmoid)
5. `stress_type_out` — 6 stress types (softmax)

**Training:** 2-phase training — Phase 1 frozen backbone, Phase 2 fine-tune last 20 layers. Focal loss for class imbalance. ~70,000 PlantVillage images.

---

## 📁 Project Structure

```
crop-stress-detection/
├── cropstress_back/
│   └── cropstress_back-main/
│       ├── pipeline.py          # FastAPI app — all endpoints
│       ├── app.py               # MobileNetV3 model training script
│       ├── train_seg.py         # YOLO segmentation training
│       ├── prepare_dataset.py   # Dataset preparation
│       ├── check_models.py      # Model verification
│       ├── requirements.txt     # Python dependencies
│       └── .env                 # Environment variables (not committed)
│
└── cropstress_front/
    └── cropstress_front-main/
        ├── src/
        │   ├── App.jsx          # Main React app (all components)
        │   ├── main.jsx         # Entry point
        │   └── index.css        # Global styles + CSS variables
        ├── public/
        ├── package.json
        └── vite.config.js
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB Atlas account

### 1. Clone the repository
```bash
git clone https://github.com/Arun93444/Crop-Stress-Detection.git
cd Crop-Stress-Detection
```

### 2. Backend Setup
```bash
cd cropstress_back/cropstress_back-main
pip install -r requirements.txt
```

Create a `.env` file:
```env
MONGO_URI=your_mongodb_atlas_connection_string
SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_openai_key_optional
ESP32_IP=
SMTP_USER=
SMTP_PASS=
```

Run the backend:
```bash
uvicorn pipeline:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd cropstress_front/cropstress_front-main
npm install
npm run dev
```

### 4. Open the app
```
http://localhost:5173
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login user |
| POST | `/api/upload` | Upload image + run inference |
| GET | `/api/history` | Get past crop records |
| GET | `/api/latest` | Get most recent reading |
| GET | `/api/weather` | Live weather data |
| POST | `/api/llm` | GPT-4o-mini AI analysis |
| GET | `/api/alerts` | Get alerts + trend |
| GET/POST | `/api/fields` | Manage farm fields |
| GET | `/api/esp32/capture` | Trigger ESP32 camera |

---

## 📸 How It Works

1. **User uploads** a crop/leaf/soil image
2. **HSV Analysis** runs instantly — detects green %, yellow %, brown %, rust %, necrosis %
3. **YOLO Segmentation** (if model loaded) — detects plant vs soil zones
4. **MobileNetV3 Classifier** (if model loaded) — runs 5-head prediction
5. **Smart blending** — combines HSV + classifier scores for final result
6. **Alert generation** — Water alert / Plant stress alert / Combined alert
7. **Result saved** to MongoDB with thumbnail
8. **Optional GPT analysis** — sends crop data + weather to GPT-4o-mini for remedy advice

---



## 👨‍💻 Developer

**Arun Prasath**
- 🎓 B.Tech AI & Data Science — United Institute of Technology, Coimbatore (2026)
- 💼 Seeking entry-level AI/ML & GenAI roles
- 🐙 GitHub: [github.com/Arun93444](https://github.com/Arun93444)
- 💼 LinkedIn: [linkedin.com/in/arun-prasath-7366b2265](https://linkedin.com/in/arun-prasath-7366b2265)

---

## 📄 License

This project was developed as a final year college project. Feel free to use it for learning purposes.
