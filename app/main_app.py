# app/main_app.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load trained model
model = joblib.load("models/talent_model.pkl")

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

prev_left_ankle = None
speeds = []
y_positions = []
timestamps = []

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

print("System running... Move for 10 seconds")

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        left_ankle_lm = landmarks[27]

        if left_ankle_lm.visibility > 0.6:
            x, y = left_ankle_lm.x, left_ankle_lm.y
            current_time = time.time()

            y_positions.append(y)
            timestamps.append(current_time)

            left_ankle = [x, y]

            if prev_left_ankle is not None:
                speed = calculate_distance(prev_left_ankle, left_ankle)

                if speed < 0.01:
                    speed = 0

                speeds.append(speed)

            prev_left_ankle = left_ankle
        else:
            prev_left_ankle = None

    cv2.imshow("AI Talent Detection", frame)

    # Run for 10 seconds
    if time.time() - start_time > 10:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# FEATURE CALCULATION
# -----------------------------

# Speed
avg_speed = np.mean(speeds) if speeds else 0

# Frequency
steps = 0
for i in range(1, len(y_positions)-1):
    if y_positions[i-1] > y_positions[i] < y_positions[i+1]:
        steps += 1

if len(timestamps) > 1:
    total_time = timestamps[-1] - timestamps[0]
    frequency = steps / total_time
else:
    frequency = 0

# Combined score
motion_score = (0.4 * avg_speed) + (0.6 * frequency)

# -----------------------------
# PREPARE ML INPUT
# -----------------------------

# Dummy + real features (must match 11 features)
input_features = [[
    175, 70, 21,          # height, weight, age
    70, 65, 60,           # stamina, strength, balance
    avg_speed * 1000,     # speed scaled
    65,                   # agility
    70, 75, 68            # acceleration, reactions, jumping
]]

# Prediction
prediction = model.predict(input_features)[0]

# -----------------------------
# OUTPUT
# -----------------------------

print("\n--- FINAL RESULT ---")
print(f"Average Speed: {avg_speed:.4f}")
print(f"Step Frequency: {frequency:.2f}")
print(f"Motion Score: {motion_score:.4f}")
print(f"Predicted Talent Level: {prediction}")
