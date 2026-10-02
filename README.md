# 🏆 AI Fitness Assistant — Sport AI Project

> A real-time computer vision and machine learning system that analyzes your athletic movements and recommends the sport you're best suited for.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-orange?logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)

---

## 📖 Overview

The **AI Fitness Assistant** captures your movements via webcam, extracts real biomechanical metrics (joint angles, rep count, jump height, balance sway, cadence), and feeds them into a trained **Random Forest Classifier** that predicts your athletic profile and recommends an ideal sport — all in real time.

**Two frontends. One powerful backend.**

| Frontend | Tech | Best for |
|---|---|---|
| Desktop CV App | OpenCV + MediaPipe | Live demo with skeleton overlay |
| Streamlit Web App | Streamlit + Matplotlib | Browser-based demo with radar chart |

---

## 🏗️ Architecture

```
Webcam
  │
  ▼
┌──────────────────────┐     POST /analyze-workout      ┌───────────────────────┐
│  Desktop CV App      │ ──────────────────────────────► │   FastAPI Backend     │
│  (OpenCV + MediaPipe)│                                 │   (scikit-learn RF    │
│  • rep count         │     POST /coaching              │    talent_model.pkl)  │
│  • knee angle        │ ◄────────────────────────────── │                       │
│  • balance sway      │                                 │   /analyze-frames     │
│  • jump height       │                                 │   (server-side CV)    │
└──────────────────────┘                                 └───────────────────────┘
                                                                    ▲
┌──────────────────────┐     POST /analyze-workout                  │
│  Streamlit Web App   │ ──────────────────────────────────────────►│
│  (ui_app.py)         │                                            │
└──────────────────────┘                                            │
```

---

## 📁 Project Structure

```
sport_ai_project/
├── app/
│   ├── main_app.py          # Desktop CV app (OpenCV + MediaPipe, live overlay)
│   └── ui_app.py            # Streamlit web app with radar chart
├── backend_api/
│   ├── main.py              # FastAPI server — ML inference + coaching endpoint
│   ├── requirements.txt     # Backend Python dependencies
│   └── README.md            # Backend-specific notes
├── src/
│   ├── pose_detection.py    # MediaPipe pose landmark extraction
│   ├── feature_extraction.py# Biomechanical feature derivation
│   ├── train_model.py       # Model training pipeline
│   └── test_model.py        # Model evaluation script
├── models/
│   └── talent_model.pkl     # Pre-trained Random Forest (sklearn 1.6.1)
├── data/
│   └── required_player_stats.csv  # Training dataset
└── project_report.md        # Detailed project report
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A webcam
- *(Optional)* OpenAI API key for AI coaching blurbs

### 1. Clone the Repository

```bash
git clone https://github.com/codee-19imran/sport_ai_project.git
cd sport_ai_project
```

### 2. Start the Backend API

```bash
cd backend_api
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verify it's running:
- Health check: `GET http://localhost:8000/health`
- Interactive API docs: `http://localhost:8000/docs`

### 3. Run the Desktop CV App *(Recommended for live demo)*

```bash
cd app
pip install opencv-python mediapipe requests numpy
python main_app.py
```

- **10-second** movement capture (press `ESC` to stop early)
- Live overlay: **rep count**, **knee angle**, **balance sway**, **speed**, **jump height**
- Auto-POSTs 4 real computed metrics to the backend at capture end

### 4. Run the Streamlit Web App *(Browser-based alternative)*

```bash
cd app
pip install streamlit matplotlib mediapipe opencv-python-headless requests
streamlit run ui_app.py
```

- Enter height / weight / age via sliders
- 8-second webcam capture + reaction time test
- Returns a **radar/spider chart** of the five athletic traits + sport recommendation

---

## 🔬 API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server + model status |
| `POST` | `/analyze-workout` | Infer traits & recommend sport from raw metrics |
| `POST` | `/analyze-frames` | Server-side CV: send base64 frames, backend extracts metrics |
| `POST` | `/coaching` | AI coaching blurb (GPT-4o-mini or template fallback) |

---

### `POST /analyze-workout`

**Request body:**
```json
{
  "height": 175,
  "weight": 70,
  "age": 22,
  "cadence": 132.5,
  "speed": 2.8,
  "jump": 0.45,
  "stability": 0.85,
  "reaction_time": 0.4
}
```

**Response:**
```json
{
  "talent_level": "Advanced",
  "recommended_sport": "Sprinting 🏃",
  "traits": {
    "stamina": 78,
    "strength": 82,
    "balance": 91,
    "agility": 74,
    "reaction_score": 65
  }
}
```

---

### `POST /analyze-frames`

Send raw webcam frames for **server-side pose estimation** — no MediaPipe needed on the client.

**Request body:**
```json
{
  "height": 175,
  "weight": 70,
  "age": 22,
  "frames": ["<base64 image>", "<base64 image>", "..."]
}
```

The backend decodes each frame, runs MediaPipe Pose, derives all metrics, and returns the same shape as `/analyze-workout`.

---

### `POST /coaching`

**Request body:**
```json
{
  "stamina": 66,
  "strength": 74,
  "balance": 85,
  "agility": 70,
  "reaction_score": 60,
  "sport": "Sprinting 🏃",
  "talent_level": "Advanced"
}
```

> Set the `OPENAI_API_KEY` environment variable to use GPT-4o-mini. A high-quality fallback template is used automatically if no key is provided.

---

## 📊 Roadmap

### ✅ Completed
- [x] Live skeleton overlay — rep count, knee angle, balance sway, jump height
- [x] End-to-end loop — 4 real computed metrics POSTed to backend
- [x] Beautiful results screen — radar chart + sport recommendation card
- [x] Demo resilience — localhost backend with graceful "no backend" fallback
- [x] Real jump test — jump height from ankle landmarks, normalised
- [x] Two contrasting demo personas — distinct sport recommendations for different inputs
- [x] LLM coaching blurb — `/coaching` endpoint (OpenAI GPT-4o-mini or template)
- [x] Sensor-source-agnostic API — `/analyze-frames` lets any client send raw frames

### 🔲 In Progress / Future
- [ ] On-device MediaPipe inference for mobile (React Native / Expo)
- [ ] Expanded training dataset — more sports & body types
- [ ] Live skeleton on mobile (MediaPipe Tasks)

---

## ⚙️ Key Dependencies

| Package | Version | Reason |
|---|---|---|
| `scikit-learn` | `==1.6.1` | Must match version used to train `talent_model.pkl` |
| `mediapipe` | `==0.10.14` | Pose landmark API used by both the desktop app and backend |
| `fastapi` | latest | REST API framework |
| `uvicorn` | latest | ASGI server |
| `opencv-python` | latest | Webcam capture & frame processing |
| `streamlit` | latest | Browser-based UI |
| `joblib` | latest | Model serialization |
| `pandas` / `numpy` | latest | Data handling & feature arrays |

---

## 👤 Author

**Imran Pasha**
- GitHub: [@codee-19imran](https://github.com/codee-19imran)
- Email: 11cshaikimranpasha@gmail.com

