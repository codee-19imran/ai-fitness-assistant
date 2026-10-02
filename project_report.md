# AI Fitness Assistant - Project Report

## 1. Executive Summary
The **AI Fitness Assistant** is a multi-platform computer vision and machine learning application designed to analyze human sports movements in real-time. By extracting physical metrics from a user's movements, the system leverages a pre-trained Random Forest Classifier to evaluate athletic traits (stamina, strength, agility) and recommend an optimal sport for the user.

## 2. System Architecture
The project is broken down into modular components catering to different platforms and responsibilities:

*   **Mobile Frontend (`mobile_app/`)**: A React Native (Expo) application serving as the primary user interface. It utilizes the device's camera to capture movement and sends metric data to the backend.
*   **Backend API (`backend_api/`)**: A FastAPI server acting as the bridge between the frontend applications and the machine learning model. It processes incoming data, formats it into a Pandas DataFrame, and returns model predictions.
*   **Standalone Python Application (`app/` & `src/`)**: A direct desktop implementation using OpenCV and Google MediaPipe to perform real-time pose detection and landmark extraction directly via a webcam.
*   **Machine Learning Pipeline (`models/`, `data/`, `src/`)**: Scripts and datasets used to train the Random Forest Classifier on player statistics.

---

## 3. Component Deep Dive

### 3.1 Backend API (FastAPI)
The backend is responsible for receiving raw physical metrics (e.g., speed, cadence, jump height) and calculating derived athletic traits.
*   **Endpoints**: 
    *   `POST /analyze-workout`: Receives JSON payload with metrics, computes derived traits (stamina, balance, agility, reaction score), constructs an 11-feature Pandas DataFrame, and queries the `talent_model.pkl` to predict a talent level and recommend a sport.
    *   `GET /health`: Returns the API status and whether the ML model is successfully loaded in memory.
*   **Dependencies**: `fastapi`, `uvicorn`, `scikit-learn==1.6.1`, `pandas`, `joblib`.

### 3.2 Mobile Application (React Native / Expo)
The mobile app provides an accessible interface for users to perform their workouts. 
*   **Key File**: `WorkoutScreen.js`
*   **Functionality**: Uses Expo's `<CameraView>` to provide a live camera feed. Upon initiating a workout, it currently simulates the extraction of pose metrics (to ensure hackathon stability) and POSTs the payload to the FastAPI backend. It then displays the returned traits and sport recommendation.

### 3.3 Core AI & Computer Vision (`src/`)
*   **`pose_detection.py`**: Utilizes `mediapipe.solutions.pose` to extract 33 3D body landmarks from an OpenCV video stream in real-time.
*   **`train_model.py`**: A pipeline script that ingests `data/required_player_stats.csv`, drops the label column, and trains a `RandomForestClassifier`. The trained model expects 11 specific features (e.g., height, weight, age, stamina, strength, balance, sprint_speed, agility, acceleration, jumping, reactions).

---

## 4. Troubleshooting & Resolutions Applied
During the recent development and setup phase, several critical fixes were implemented:
1.  **Mobile App Dependencies**: Resolved a peer dependency conflict between `@tensorflow/tfjs-react-native` and `expo-camera` by utilizing `--legacy-peer-deps`.
2.  **Expo SDK Upgrade**: Upgraded the mobile app to Expo SDK 57, replacing deprecated `<Camera>` APIs with the modern `<CameraView>` component and fixing related React Native absolute positioning warnings.
3.  **ML Model Versioning**: Pinned `scikit-learn` to `1.6.1` in the backend to match the exact version used during model training, preventing `InconsistentVersionWarning` unpickling errors.
4.  **Feature Mapping Bug**: Fixed an issue where the backend passed a raw Python list to the Scikit-learn model, causing a feature-name mismatch warning. The input is now correctly wrapped in a Pandas DataFrame with the exact 11 column names.

## 5. Future Roadmap
*   **On-Device Inference**: Transition the simulated metrics in the Expo app to use actual on-device `@tensorflow-models/pose-detection` operating on camera frames via `requestAnimationFrame`.
*   **Model Retraining**: Expand the `required_player_stats.csv` dataset to include a wider variety of sports and body types to improve the Random Forest Classifier's accuracy.
