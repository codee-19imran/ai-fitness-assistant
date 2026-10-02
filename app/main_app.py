# app/main_app.py
import cv2
import mediapipe as mp
import numpy as np
import requests
import time

# ─────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ─────────────────────────────────────────
# State variables
# ─────────────────────────────────────────
prev_left_ankle  = None
speeds           = []
y_positions      = []
timestamps       = []
hip_x_positions  = []

# Rep counter — only count confirmed, deliberate squats
rep_count        = 0
rep_state        = "up"       # "up" | "down"
SQUAT_DOWN_THRESH = 115       # angle below this = "squatting"
SQUAT_UP_THRESH   = 155       # angle above this = "stood up"
MIN_ANGLE_FRAMES  = 4         # angle must hold for N frames before state changes
down_frame_count  = 0
up_frame_count    = 0

# Jump detection — track ankle baseline once stable
ankle_baseline   = None
baseline_samples = []
BASELINE_FRAMES  = 30         # collect 1 second of standing before enabling jump
max_jump_delta   = 0.0        # max upward deviation from baseline

# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def calculate_angle(a, b, c):
    """Angle at joint B given three (x, y) points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

def draw_metric(frame, label, value, row, color=(0, 255, 128)):
    """Render a semi-transparent metric pill."""
    y = 40 + row * 48
    cv2.rectangle(frame, (10, y - 26), (360, y + 12), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, y - 26), (360, y + 12), color, 2)
    cv2.putText(frame, f"{label}: {value}", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

print("System running... Perform squats / jumps for 10 seconds")
print("Press ESC to stop early.\n")

start_time = time.time()
frame_count = 0

# ─────────────────────────────────────────
# Main capture loop
# ─────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    frame_count += 1

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    knee_angle    = None
    balance_sway  = None
    current_speed = 0.0

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # ── Draw skeleton ──────────────────────────────────────
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # ── Speed from ankle movement (threshold noise) ────────
        la = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        if la.visibility > 0.65:
            ankle_pos = [la.x, la.y]
            if prev_left_ankle is not None:
                dist = np.linalg.norm(np.array(ankle_pos) - np.array(prev_left_ankle))
                # Only count as movement if delta > 0.005 (filter camera noise)
                if dist > 0.005:
                    speeds.append(dist)
                    current_speed = dist
                else:
                    speeds.append(0.0)
            prev_left_ankle = ankle_pos
            y_positions.append(la.y)
            timestamps.append(time.time())

            # ── Jump height — establish standing baseline first ──
            if len(baseline_samples) < BASELINE_FRAMES:
                baseline_samples.append(la.y)
            else:
                if ankle_baseline is None:
                    ankle_baseline = float(np.median(baseline_samples))
                # Ankle going UP = y decreasing in image coords
                delta = ankle_baseline - la.y
                if delta > max_jump_delta:
                    max_jump_delta = delta
        else:
            prev_left_ankle = None

        # ── Knee angle (squat depth) with noise guard ──────────
        l_hip   = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee  = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        if all(p.visibility > 0.55 for p in [l_hip, l_knee, l_ankle]):
            knee_angle = calculate_angle(
                [l_hip.x, l_hip.y],
                [l_knee.x, l_knee.y],
                [l_ankle.x, l_ankle.y]
            )

            # ── Debounced rep counter ──────────────────────────
            if knee_angle < SQUAT_DOWN_THRESH:
                down_frame_count += 1
                up_frame_count    = 0
                if down_frame_count >= MIN_ANGLE_FRAMES and rep_state == "up":
                    rep_state = "down"
            elif knee_angle > SQUAT_UP_THRESH:
                up_frame_count   += 1
                down_frame_count  = 0
                if up_frame_count >= MIN_ANGLE_FRAMES and rep_state == "down":
                    rep_state = "up"
                    rep_count += 1
            else:
                down_frame_count = 0
                up_frame_count   = 0

        # ── Balance sway (hip midpoint std-dev, last 2 s) ─────
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        if l_hip.visibility > 0.55 and r_hip.visibility > 0.55:
            hip_mid_x = (l_hip.x + r_hip.x) / 2.0
            hip_x_positions.append(hip_mid_x)
            window = hip_x_positions[-60:]   # ~2 s window
            if len(window) >= 10:
                balance_sway = float(np.std(window)) * 1000

    # ── Overlay metrics (use only confirmed/stable values) ────
    current_avg_speed = float(np.mean(speeds[-30:])) * 1000 if len(speeds) >= 5 else 0.0
    jump_px           = max_jump_delta * h

    time_left = max(0, 10 - int(time.time() - start_time))
    draw_metric(frame, "Time left",    f"{time_left}s",                              0, (0, 180, 255))
    draw_metric(frame, "Reps",         f"{rep_count}",                               1, (0, 255, 128))
    draw_metric(frame, "Knee angle",
                f"{knee_angle:.0f} deg" if knee_angle else "—",                      2, (0, 255, 128))
    draw_metric(frame, "Balance sway",
                f"{balance_sway:.1f}" if balance_sway else "calibrating...",        3, (0, 255, 128))
    draw_metric(frame, "Speed",        f"{current_avg_speed:.1f}",                   4, (0, 255, 128))
    draw_metric(frame, "Jump height",
                f"{jump_px:.0f} px" if ankle_baseline else "stand still...",        5, (0, 255, 128))

    cv2.imshow("AI Talent Detection", frame)

    if time.time() - start_time > 10:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────
# POST-SESSION FEATURE CALCULATION
# ─────────────────────────────────────────
avg_speed   = float(np.mean(speeds)) if speeds else 0.0
avg_sway    = float(np.std(hip_x_positions)) if len(hip_x_positions) > 1 else 0.15
stability   = max(0.0, min(1.0, 1.0 - avg_sway * 10))

# Step frequency from ankle y — deduplicated after session
steps = 0
for i in range(1, len(y_positions) - 1):
    if y_positions[i - 1] > y_positions[i] < y_positions[i + 1]:
        # Only count peaks with a meaningful amplitude
        amplitude = max(y_positions[i - 1], y_positions[i + 1]) - y_positions[i]
        if amplitude > 0.005:
            steps += 1

if len(timestamps) > 1:
    total_time = timestamps[-1] - timestamps[0]
    frequency  = steps / total_time if total_time > 0 else 0.0
else:
    frequency  = 0.0

jump_norm = max(0.0, min(1.0, max_jump_delta / 0.05))  # 5 % of frame height = good jump

print(f"\n--- SESSION SUMMARY ---")
print(f"  Reps counted  : {rep_count}")
print(f"  Stability     : {stability:.2f}")
print(f"  Jump (norm)   : {jump_norm:.2f}")
print(f"  Cadence       : {frequency * 60:.1f} steps/min")
print(f"  Avg speed     : {avg_speed * 1000:.2f}")

# ─────────────────────────────────────────
# SEND TO BACKEND API
# ─────────────────────────────────────────
workout_data = {
    "height"       : 175.0,
    "weight"       : 70.0,
    "age"          : 22,
    "cadence"      : frequency * 60,   # REAL
    "speed"        : avg_speed * 1000, # REAL
    "jump"         : jump_norm,        # REAL
    "stability"    : stability,        # REAL
    "reaction_time": 0.4               # simulated
}

print("\n--- SENDING DATA TO BACKEND ---")
try:
    response = requests.post("http://localhost:8000/analyze-workout", json=workout_data, timeout=5)
    if response.status_code == 200:
        result = response.json()
        print("\n--- FINAL RESULT FROM API ---")
        print(f"  Traits          : {result['traits']}")
        print(f"  Recommended Sport: {result['recommendation']['sport']}")
        print(f"  Talent Level    : {result['recommendation']['talent_level']}")
    else:
        print(f"Backend error: {response.status_code} — {response.text}")
except Exception as e:
    print(f"Could not reach backend: {e}")
