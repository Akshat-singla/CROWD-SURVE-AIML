# Real-Time AI Surveillance System

## Project Overview
The Real-Time AI Surveillance System is a modular, Python-based application that detects multiple persons in video feeds, tracks them across frames, and analyzes their movement behavior. It classifies activities (e.g., Normal Movement, Loitering, Running, Restricted Area Intrusion) using a hybrid rule-based and Random Forest machine learning model. The system can generate alerts, log events to CSV, and provide visual evidence via JPEG snapshots. It offers both a standalone CLI application and a live Flask-based web dashboard.

## Main Technologies
*   **Language:** Python 3.9+
*   **Deep Learning & Vision:** YOLOv8 (Ultralytics) for detection, ByteTrack for tracking, OpenCV for video processing and frame annotation.
*   **Machine Learning:** Scikit-Learn (Random Forest pipeline) for activity classification.
*   **Web Dashboard:** Flask (serves MJPEG stream and status updates via API).
*   **Data & Utilities:** NumPy, Pandas for feature computation and event logging, Psutil for memory monitoring.

## Key Directories & Files
*   `config/settings.py`: **Central configuration file.** Contains all tuneable parameters (video source, thresholds, zones, flask host/port). Avoid hardcoding values in the modules.
*   `main.py`: The standalone CLI pipeline runner. Can process live webcam feeds or recorded video files.
*   `dashboard/app.py`: The Flask web application. Implements a multi-threaded architecture (Capture Thread, Processing Thread, Log Worker) to run the surveillance pipeline and serve the UI.
*   `train_classifier.py`: Script to generate synthetic data and train the Random Forest ML model.
*   `modules/`: Contains the core Object-Oriented processing modules: `video_input.py`, `detector.py`, `tracker.py`, `behaviour_buffer.py`, `feature_extractor.py`, `classifier.py`, `alert_logger.py`, and `annotator.py`.
*   `data/`: Directory for input videos, annotated output videos, event logs (`logs/`), and evidence snapshots (`snapshots/`).
*   `models/`: Stores downloaded YOLOv8 weights (`yolov8n.pt`) and the trained scikit-learn classifier (`activity_classifier.pkl`).

## Building and Running

### 1. Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
# Activate the environment (Windows)
venv\Scripts\activate
# Activate the environment (macOS/Linux)
source venv/bin/activate

pip install -r requirements.txt
```
*(Note: YOLOv8 weights are auto-downloaded on the first run).*

### 2. Running the System
**Option A: Standalone CLI (OpenCV Window)**
```bash
python main.py
# Or specify a source:
python main.py --source 0  # Webcam
python main.py --source data/input/cctv.mp4
# Run headless (no GUI)
python main.py --no-display
```

**Option B: Flask Dashboard Only**
```bash
python dashboard/app.py
```
*Access the dashboard at `http://localhost:5000`.*

**Option C: Combined CLI & Dashboard**
```bash
python main.py --dashboard
```

### 3. Training the Classifier
Train the machine learning model using synthetic data:
```bash
python train_classifier.py --generate
```
Or train with your own labeled CSV data:
```bash
python train_classifier.py --data data/my_labelled_data.csv
```

## Development Conventions
*   **Configuration:** Always refer to `config/settings.py` for thresholds, file paths, model paths, and UI settings.
*   **Threading:** Be aware of the multi-threaded nature of the dashboard app (`dashboard/app.py`). There are explicit threads for Capture, Processing, and Logging to maintain performance without blocking the web server. State is shared securely via queues and locks.
*   **Type Hinting:** Use standard Python type hinting (e.g., `Dict`, `Optional`, `np.ndarray`) extensively across module boundaries.
*   **Logging:** Utilize Python's built-in `logging` module rather than `print()` for pipeline progress, warnings, and errors.
*   **Memory Management:** The system explicitly monitors memory usage (via `psutil`) and cleans up stale tracking buffers and explicitly deletes frame references to ensure long-term stability. Maintain these practices when adding new features.
