from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import os
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

app = FastAPI(title="AI Fitness Assistant API")

# Allow CORS for mobile app connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/talent_model.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")

class WorkoutData(BaseModel):
    height: float
    weight: float
    age: int
    cadence: float
    speed: float
    jump: float
    stability: float
    reaction_time: float

class TraitsPayload(BaseModel):
    stamina: int
    strength: int
    balance: int
    agility: int
    reaction_score: int
    sport: str
    talent_level: str

class WorkoutFramesPayload(BaseModel):
    height: float
    weight: float
    age: int
    frames: list[str]


def _decode_frame(frame_b64: str):
    if not frame_b64:
        return None

    if frame_b64.startswith("data:image"):
        frame_b64 = frame_b64.split(",", 1)[1]

    try:
        # Works for both plain base64 and data URLs
        frame_bytes = base64.b64decode(frame_b64)
        image = np.frombuffer(frame_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return decoded
    except Exception:
        return None


def _extract_metrics_from_frames(frames: list[str]):
    if not frames:
        return {
            "speed": 0.0,
            "cadence": 0.0,
            "jump": 0.0,
            "stability": 0.0,
            "reaction_time": 1.0,
        }

    positions = []
    detected_count = 0

    for image in frames:
        decoded = _decode_frame(image)
        if decoded is None:
            continue

        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            ankle = landmarks[27] if landmarks[27].visibility > landmarks[28].visibility else landmarks[28]
            if ankle.visibility > 0.5:
                positions.append((ankle.x, ankle.y))
                detected_count += 1

    if len(positions) < 3:
        return {
            "speed": 0.0,
            "cadence": 0.0,
            "jump": 0.0,
            "stability": 0.0,
            "reaction_time": 1.0,
        }

    ys = np.array([p[1] for p in positions])
    deltas = [np.linalg.norm(np.array(curr) - np.array(prev)) for prev, curr in zip(positions, positions[1:])]
    if len(ys) > 2:
        local_minima = sum(1 for i in range(1, len(ys) - 1) if ys[i - 1] > ys[i] < ys[i + 1])
    else:
        local_minima = 0

    speed = float(np.mean(deltas) * 7.5)
    cadence = float(local_minima * 9.0)
    jump = float((np.max(ys) - np.min(ys)) * 5.0)
    stability = float(max(0.0, min(1.0, 1.0 / (1.0 + np.std(deltas) * 20.0))))
    reaction_time = float(max(0.2, min(1.0, 1.0 - (speed / 8.0))))

    return {
        "speed": float(np.clip(speed, 0.0, 5.0)),
        "cadence": float(np.clip(cadence, 0.0, 180.0)),
        "jump": float(np.clip(jump, 0.0, 3.0)),
        "stability": float(np.clip(stability, 0.0, 1.0)),
        "reaction_time": reaction_time,
        "detected_frames": detected_count,
    }


def calculate_traits(data: WorkoutData):
    stamina        = int(min(100, data.cadence / 2))
    strength       = int(min(100, (data.jump * 200) + (data.speed * 20)))
    balance        = int(min(100, data.stability * 100))
    agility        = int(min(100, (data.speed * 10) + data.cadence / 3))
    reaction_score = int(max(0, 100 - (data.reaction_time * 100)))
    return stamina, strength, balance, agility, reaction_score

def recommend_sport(height, speed, cadence, jump, strength):
    if speed > 2 and cadence > 120:
        return "Sprinting 🏃"
    elif height > 180 and jump > 0.3:
        return "Basketball 🏀"
    elif cadence > 110:
        return "Football ⚽"
    elif strength > 70:
        return "Wrestling 🤼"
    else:
        return "General Fitness 🏋️"

def _template_coaching(t: TraitsPayload) -> str:
    """Rule-based coaching blurb — zero external dependencies."""
    scores = {
        "stamina": t.stamina, "strength": t.strength,
        "balance": t.balance, "agility": t.agility,
        "reactions": t.reaction_score
    }
    top_trait = max(scores, key=scores.get)
    tips = {
        "stamina"  : ("exceptional endurance", "push aerobic boundaries with interval runs"),
        "strength" : ("explosive power", "add plyometric drills to channel that strength into speed"),
        "balance"  : ("outstanding body control", "challenge proprioception with single-leg stability work"),
        "agility"  : ("lightning-quick movement", "cone drills and ladder work will sharpen that edge"),
        "reactions": ("razor-sharp reaction speed", "reflex training with a reaction ball will level you up"),
    }
    quality, tip = tips[top_trait]
    return (
        f"Your athletic profile shows {quality} — a key trait for {t.sport}. "
        f"At talent level '{t.talent_level}', you're already ahead of the curve. "
        f"To accelerate your progress, {tip}. "
        f"Keep showing up, keep pushing limits — champions are built in the details. 🔥"
    )

@app.post("/analyze-workout")
def analyze_workout(data: WorkoutData):
    if model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    stamina, strength, balance, agility, reaction_score = calculate_traits(data)

    features = pd.DataFrame([[
        data.height, data.weight, data.age,
        stamina, strength, balance,
        data.speed,
        agility,
        reaction_score, 75, 68
    ]], columns=[
        'height', 'weight', 'age', 'stamina', 'strength', 'balance',
        'sprint_speed', 'agility', 'acceleration', 'jumping', 'reactions'
    ])

    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    sport = recommend_sport(data.height, data.speed, data.cadence, data.jump, strength)

    return {
        "traits": {
            "stamina"       : stamina,
            "strength"      : strength,
            "balance"       : balance,
            "agility"       : agility,
            "reaction_score": reaction_score,
        },
        "recommendation": {
            "talent_level": str(prediction),
            "sport"       : sport,
        },
        "raw_metrics": {
            "speed"        : data.speed,
            "cadence"      : data.cadence,
            "jump_height"  : data.jump,
            "stability"    : data.stability,
            "reaction_time": data.reaction_time,
        },
    }


@app.post("/analyze-frames")
def analyze_frames(data: WorkoutFramesPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    metrics = _extract_metrics_from_frames(data.frames)
    if metrics.get("detected_frames", 0) == 0:
        raise HTTPException(status_code=400, detail="No pose landmarks were detected in the submitted frames.")

    speed = float(metrics["speed"])
    cadence = float(metrics["cadence"])
    jump = float(metrics["jump"])
    stability = float(metrics["stability"])
    reaction_time = float(metrics["reaction_time"])

    stamina = int(min(100, cadence / 2))
    strength = int(min(100, (jump * 200) + (speed * 20)))
    balance = int(min(100, stability * 100))
    agility = int(min(100, (speed * 10) + cadence / 3))
    reaction_score = int(max(0, 100 - (reaction_time * 100)))

    features = pd.DataFrame([[
        data.height, data.weight, data.age,
        stamina, strength, balance,
        speed,
        agility,
        reaction_score, 75, 68
    ]], columns=[
        'height', 'weight', 'age', 'stamina', 'strength', 'balance',
        'sprint_speed', 'agility', 'acceleration', 'jumping', 'reactions'
    ])

    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    sport = recommend_sport(data.height, speed, cadence, jump, strength)

    return {
        "traits": {
            "stamina": stamina,
            "strength": strength,
            "balance": balance,
            "agility": agility,
            "reaction_score": reaction_score,
        },
        "recommendation": {
            "talent_level": str(prediction),
            "sport": sport,
        },
        "raw_metrics": {
            "speed": speed,
            "cadence": cadence,
            "jump_height": jump,
            "stability": stability,
            "reaction_time": reaction_time,
        },
    }

@app.post("/coaching")
def get_coaching(traits: TraitsPayload):
    """
    Tier 3.9 — LLM coaching blurb.
    Uses OpenAI if OPENAI_API_KEY env var is set; falls back to a
    high-quality template so the demo always works without any API key.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            import openai
            openai.api_key = api_key
            prompt = (
                f"You are an elite sports performance coach. "
                f"A user completed an AI-analysed workout with these traits (0-100): "
                f"Stamina={traits.stamina}, Strength={traits.strength}, "
                f"Balance={traits.balance}, Agility={traits.agility}, "
                f"Reactions={traits.reaction_score}. "
                f"Recommended sport: {traits.sport}. Talent level: {traits.talent_level}. "
                f"Write a punchy, motivational 3-sentence coaching paragraph personalised to these results."
            )
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            blurb = resp.choices[0].message.content.strip()
        except Exception:
            blurb = _template_coaching(traits)   # graceful fallback on any error
    else:
        blurb = _template_coaching(traits)

    return {"coaching": blurb}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
