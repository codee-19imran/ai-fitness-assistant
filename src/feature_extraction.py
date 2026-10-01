import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose


def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def analyze_video(height_cm, duration=8):

    pose = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or fps is None:
        fps = 30

    prev = None
    speeds = []
    y_positions = []
    timestamps = []
    pixel_heights = []

    st_frame = None  # placeholder if needed later

    start = time.time()

    # -----------------------------
    # MAIN MOTION CAPTURE
    # -----------------------------
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            head_y = lm[0].y
            ankle = lm[27]

            if ankle.visibility > 0.6:

                # Scale estimation
                pixel_height = abs(head_y - ankle.y)
                pixel_heights.append(pixel_height)

                x, y = ankle.x, ankle.y
                t = time.time()

                y_positions.append(y)
                timestamps.append(t)

                if prev is not None:
                    dist = calculate_distance(prev, [x, y])

                    # Noise filtering
                    if dist < 0.003:
                        dist = 0

                    speeds.append(dist)

                prev = [x, y]
            else:
                prev = None

        if time.time() - start > duration:
            break

    # -----------------------------
    # SCALE (pixels → meters)
    # -----------------------------
    if pixel_heights:
        avg_pixel_height = np.mean(pixel_heights)
        real_height_m = height_cm / 100
        scale = real_height_m / avg_pixel_height
    else:
        scale = 1

    # -----------------------------
    # SPEED (m/s)
    # -----------------------------
    if speeds:
        avg_norm_speed = np.mean(speeds)
        speed = avg_norm_speed * scale * fps
    else:
        speed = 0

    # -----------------------------
    # CADENCE
    # -----------------------------
    steps = 0
    for i in range(1, len(y_positions) - 1):
        if y_positions[i - 1] > y_positions[i] < y_positions[i + 1]:
            steps += 1

    if len(timestamps) > 1:
        total_time = timestamps[-1] - timestamps[0]
        frequency = steps / total_time
    else:
        frequency = 0

    cadence = frequency * 60

    # -----------------------------
    # JUMP HEIGHT
    # -----------------------------
    if y_positions:
        jump_norm = max(y_positions) - min(y_positions)
        jump = jump_norm * scale
    else:
        jump = 0

    # -----------------------------
    # STABILITY
    # -----------------------------
    if speeds:
        stability = 1 / (1 + np.std(speeds))
    else:
        stability = 0

    # -----------------------------
    # REACTION TIME (FIXED VERSION)
    # -----------------------------
    reaction_time = 1.0  # default slow

    if prev is not None:

        print("Prepare for reaction test...")
        time.sleep(2)
        print("MOVE NOW!")

        start_react = time.time()

        prev_react = prev

        for _ in range(60):  # ~2 seconds window

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                ankle = lm[27]

                if ankle.visibility > 0.6:
                    x, y = ankle.x, ankle.y

                    movement = calculate_distance(prev_react, [x, y])

                    if movement > 0.02:
                        reaction_time = time.time() - start_react
                        break

                    prev_react = [x, y]

    cap.release()

    return speed, cadence, jump, stability, reaction_time
