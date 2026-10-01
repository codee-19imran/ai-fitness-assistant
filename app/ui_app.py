import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

# Load ML model
model = joblib.load("models/talent_model.pkl")

mp_pose = mp.solutions.pose

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def analyze_video(height_cm, duration=8):

    pose = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    prev = None
    speeds = []
    y_positions = []
    timestamps = []
    pixel_heights = []

    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            head = lm[0].y
            ankle = lm[27]

            if ankle.visibility > 0.6:
                pixel_height = abs(head - ankle.y)
                pixel_heights.append(pixel_height)

                x, y = ankle.x, ankle.y
                t = time.time()

                y_positions.append(y)
                timestamps.append(t)

                if prev:
                    dist = np.linalg.norm(np.array(prev) - np.array([x, y]))
                    if dist < 0.003:
                        dist = 0
                    speeds.append(dist)

                prev = [x, y]

        if time.time() - start > duration:
            break

    cap.release()

    # SCALE
    if pixel_heights:
        scale = (height_cm / 100) / np.mean(pixel_heights)
    else:
        scale = 1

    # SPEED
    speed = (np.mean(speeds) * scale * fps) if speeds else 0

    # CADENCE
    steps = 0
    for i in range(1, len(y_positions)-1):
        if y_positions[i-1] > y_positions[i] < y_positions[i+1]:
            steps += 1

    total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1
    cadence = (steps / total_time) * 60

    # JUMP HEIGHT
    jump = (max(y_positions) - min(y_positions)) * scale if y_positions else 0

    # STABILITY
    stability = 1 / (1 + np.std(speeds)) if speeds else 0

    # -----------------------------
    # REACTION TEST
    # -----------------------------
    st.info("Get ready... react to MOVE signal!")

    time.sleep(2)
    st.warning("MOVE!")

    start_react = time.time()
    reaction_time = 1.0

    for _ in range(40):
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            ankle = lm[27]

            if ankle.visibility > 0.6:
                movement = abs(ankle.y - prev[1]) if prev else 0
                if movement > 0.02:
                    reaction_time = time.time() - start_react
                    break

    return speed, cadence, jump, stability, reaction_time


# -----------------------------
# SPORT RECOMMENDATION
# -----------------------------
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


# -----------------------------
# RADAR CHART
# -----------------------------
def plot_radar(data, labels):

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    data += data[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={'polar': True})
    ax.plot(angles, data)
    ax.fill(angles, data, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    return fig


# -----------------------------
# UI
# -----------------------------
st.title("🏅 AI Sports Talent System (Final)")

height = st.slider("Height (cm)", 140, 200, 170)
weight = st.slider("Weight (kg)", 40, 100, 65)
age = st.slider("Age", 10, 40, 20)

if st.button("Start Full Analysis 🚀"):

    speed, cadence, jump, stability, reaction_time = analyze_video(height)

    # Validation
    if speed < 0.2 and cadence < 30:
        st.warning("No movement detected")
        st.stop()

    # TRAITS
    stamina = int(min(100, cadence / 2))
    strength = int(min(100, (jump * 200) + (speed * 20)))
    balance = int(min(100, stability * 100))
    agility = int(min(100, (speed * 10) + cadence / 3))
    reaction_score = int(max(0, 100 - (reaction_time * 100)))

    # ML INPUT
    features = [[
        height, weight, age,
        stamina, strength, balance,
        speed,
        agility,
        reaction_score, 75, 68
    ]]

    prediction = model.predict(features)[0]

    sport = recommend_sport(height, speed, cadence, jump, strength)

    # OUTPUT
    st.success("Analysis Complete")

    st.write(f"Speed: {speed:.2f} m/s")
    st.write(f"Cadence: {cadence:.1f} steps/min")
    st.write(f"Jump Height: {jump:.2f} m")
    st.write(f"Stability: {stability:.2f}")
    st.write(f"Reaction Time: {reaction_time:.2f} sec")

    st.write("---")

    st.write(f"Stamina: {stamina}")
    st.write(f"Strength: {strength}")
    st.write(f"Balance: {balance}")
    st.write(f"Agility: {agility}")

    st.write("---")

    st.write(f"Talent Level: {prediction}")
    st.write(f"Recommended Sport: {sport}")

    # Radar Chart
    chart = plot_radar(
        [stamina, strength, balance, agility, reaction_score],
        ["Stamina", "Strength", "Balance", "Agility", "Reaction"]
    )

    st.pyplot(chart)
