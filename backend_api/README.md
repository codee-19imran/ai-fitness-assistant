# AI Fitness Assistant - Backend API

This folder contains the FastAPI backend that runs the Machine Learning model.

## How to Run the Backend

1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. The API will be available at `http://localhost:8000`. You can view the interactive documentation at `http://localhost:8000/docs`.

Make sure your mobile device and your computer are on the same WiFi network, and update the IP address in the React Native app to match your computer's local IP address.
