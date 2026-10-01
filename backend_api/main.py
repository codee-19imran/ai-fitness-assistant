from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="AI Fitness Assistant API")

# Allow CORS for mobile app connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

def calculate_traits(data: WorkoutData):
    stamina = int(min(100, data.cadence / 2))
    strength = int(min(100, (data.jump * 200) + (data.speed * 20)))
    balance = int(min(100, data.stability * 100))
    agility = int(min(100, (data.speed * 10) + data.cadence / 3))
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

@app.post("/analyze-workout")
def analyze_workout(data: WorkoutData):
    if model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    stamina, strength, balance, agility, reaction_score = calculate_traits(data)
    
    # ML INPUT matches the original UI app
    features = [[
        data.height, data.weight, data.age,
        stamina, strength, balance,
        data.speed,
        agility,
        reaction_score, 75, 68 # Static values from original code
    ]]

    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    sport = recommend_sport(data.height, data.speed, data.cadence, data.jump, strength)

    return {
        "traits": {
            "stamina": stamina,
            "strength": strength,
            "balance": balance,
            "agility": agility,
            "reaction_score": reaction_score
        },
        "recommendation": {
            "talent_level": str(prediction),
            "sport": sport
        },
        "raw_metrics": {
            "speed": data.speed,
            "cadence": data.cadence,
            "jump_height": data.jump,
            "stability": data.stability,
            "reaction_time": data.reaction_time
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
