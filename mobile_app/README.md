# AI Fitness Assistant - Mobile App

This folder contains the React Native (Expo) code for the frontend of the AI Fitness Assistant.

## How to Run the App

1. Make sure you have Node.js installed.
2. Install the dependencies:
   ```bash
   npm install
   ```
3. Update the `BACKEND_URL` in `screens/WorkoutScreen.js` to point to your computer's local IP address (e.g., `http://192.168.1.15:8000/analyze-workout`).
4. Start the Expo development server:
   ```bash
   npx expo start
   ```
5. Download the **Expo Go** app on your iOS or Android device.
6. Scan the QR code shown in your terminal using the Expo Go app.

## Project Structure

- `App.js`: Contains the bottom tab navigation structure.
- `screens/HomeScreen.js`: The welcome dashboard.
- `screens/WorkoutScreen.js`: The camera view that captures movement and sends data to the Python backend.
- `screens/ProfileScreen.js`: User statistics.

## Note on MediaPipe

To keep this hackathon project stable, the `WorkoutScreen.js` currently captures real video but simulates the extraction of the 3D joint coordinates before sending them to the FastAPI backend. To fully integrate on-device MediaPipe, you would use `@tensorflow-models/pose-detection` within a `requestAnimationFrame` loop on the camera stream.
