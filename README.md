# Real-Time AI Surveillance System

A modular, production-oriented Python surveillance system that detects people in video feeds, tracks them with persistent IDs, analyses their movement behaviour, classifies violations, and streams a live annotated feed through a Flask web dashboard. The system also performs secondary object detection to catch weapons, abandoned bags, and possible theft events in real time.

---

## Features

### Person Tracking & Behaviour Analysis
- **Multi-person detection** using YOLOv8 (Ultralytics) at configurable confidence and resolution
- **Persistent tracking** with selectable backend — ByteTrack (default) or DeepSORT
- **10-second sliding observation window** per person for stable behaviour classification
- **Five activity categories**: Normal Movement · Loitering · Running / Sudden Motion · Suspicious Lingering · Restricted Area Intrusion
- **Hybrid classifier** — rule-based heuristics backed by an optional Random Forest ML model

### Object Detection & Security Events
- **Weapon detection** — knives and scissors (COCO) flagged immediately; custom firearm models supported
- **Unsafe activity alerts** — irregular high-energy movement is flagged as `Unsafe Activity`
- **Abandoned bag detection** — backpacks, handbags, and suitcases tracked with IoU matching; alert fires when a bag has been stationary for ≥ 30 s with no nearby person
- **Heuristic theft detection** — when a tracked bag disappears and a person was within configurable proximity of it

### Alerting & Logging
- **Per-person alert cooldown** to suppress duplicate alerts
- **JPEG evidence snapshots** rendered with full annotation at alert time (dedicated worker thread)
- **CSV event log** written by a background log-queue worker (Thread 3) with no blocking on the pipeline
- **Snapshot integrity check** — processing only signals completion after the log queue is fully drained

### Flask Web Dashboard
- **Concurrent workers**: Capture → Person tracking/processing, asynchronous object detection, and evidence/log worker
- **MJPEG live video stream** paced to source FPS; auto-reconnects on drop
- **Native OS file browser** for selecting input video files
- **Real-time statistics panel**: current/peak/total-unique person counts, FPS, memory usage, bag stats
- **Alert event log** with activity-type filter chips, refreshed every 3 s
- **Snapshot gallery tab** with lightbox full-size view
- **Offline preprocessing mode** — full inference pass saved as annotated MP4, then streamed at native FPS via `/processed_feed`
- **Two-feed stampede-control mode** — compare live entry/exit cameras or two uploaded, synchronized videos; track current people counts and center-line crossings, and raise dashboard/evidence alerts when counts differ beyond the configured threshold

### Operational & Performance
- **GPU acceleration** — auto-selects CUDA → Apple MPS → CPU
- **Configurable tracking cadence** — person tracking runs at `TRACKING_INTERVAL`; intermediate frames reuse the latest tracks
- **Duplicate-frame guard** — each captured frame is processed once instead of repeating YOLO on the same camera frame
- **Aspect-preserving YOLO resizing** — source frames are letterboxed by Ultralytics rather than distorted into a square
- **Memory management** — `psutil` RAM monitoring, bounded log queue, periodic stale-track cleanup via `threading.Timer`, explicit frame reference deletion
- **Logging** — Python `logging` module throughout; configurable via `LOG_LEVEL` env var

---

## Project Structure

```
Real-Time-AI-Surveillance-System/
│
├── config/
│   └── settings.py              # CLI pipeline configuration (all tuneable params)
│
├── dashboard/
│   ├── app.py                   # Flask app — concurrent worker pipeline + all routes
│   ├── config.py                # Dashboard-specific configuration
│   ├── templates/
│   │   └── index.html           # Dark-theme dashboard page
│   └── static/
│       ├── css/style.css        # Dashboard stylesheet
│       └── js/dashboard.js      # Live polling, snapshot gallery, UI logic
│
├── modules/
│   ├── video_input.py           # OpenCV video source abstraction (CLI pipeline)
│   ├── detector.py              # YOLOv8 person detection (CLI pipeline)
│   ├── tracker.py               # ByteTrack / DeepSORT factory — PersonTracker
│   ├── object_detector.py       # ObjectDetector (weapons & bags) + BagTracker
│   ├── behaviour_buffer.py      # Per-person sliding observation window
│   ├── feature_extractor.py     # 7-feature motion & zone-dwell computation
│   ├── classifier.py            # Hybrid rule-based + Random Forest classifier
│   ├── alert_logger.py          # Alert cooldown, JPEG snapshots, CSV event log
│   └── annotator.py             # Frame overlay drawing (boxes, trails, zones)
│
├── data/
│   ├── input/                   # Drop video files here
│   ├── output/                  # Annotated output videos
│   ├── logs/                    # CSV event log + confusion matrix PNG
│   ├── snapshots/               # Evidence JPEG snapshots
│   └── training_data.csv        # Feature vectors for classifier training
│
├── models/
│   ├── yolov8n.pt               # YOLOv8 nano weights (auto-downloaded)
│   ├── yolov8s.pt               # YOLOv8 small weights (dashboard default)
│   └── activity_classifier.pkl  # Trained scikit-learn classifier (generated)
│
├── main.py                      # Standalone CLI pipeline runner
├── train_classifier.py          # Trains & evaluates the Random Forest model
├── extract_ucf_features.py      # Offline feature extraction from labelled video folders
└── requirements.txt             # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- A CUDA-capable GPU (optional but strongly recommended)
- `pip` and `venv`

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** YOLOv8 weights (`yolov8n.pt` / `yolov8s.pt`) are automatically downloaded by Ultralytics on first run.

### 3. GPU acceleration (optional but recommended)

The system auto-detects CUDA. To install a CUDA-enabled PyTorch build explicitly:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Train a custom weapon detector

Prepare a YOLO-format dataset YAML with weapon classes, then train:

```bash
python train_weapon_detector.py --data data/weapon_dataset/data.yaml --epochs 50
```

The resulting `models/weapon_yolov8.pt` is used by the dashboard when
`WEAPON_MODEL_PATH=models/weapon_yolov8.pt` is set. Without a custom model,
the built-in COCO detector still alerts on knives and scissors.

**Important:** COCO does not include guns/firearms. Until a labeled firearm
dataset has been trained and the custom weights are present, the dashboard
will show that firearm detection is unavailable; it must not be treated as a
gun detector. The weapon training dataset YAML should name firearm classes
with labels such as `gun`, `pistol`, `handgun`, `rifle`, or `firearm`.

### Live inference performance

Live video uses independent capture, person-processing, evidence-logging, and
object-detection workers. The object worker keeps only one inference in flight
so it cannot build a stale frame backlog. Person detection defaults to the
YOLOv8 nano weights at 416 px, CUDA half precision when available, and a
configurable tracking cadence. Processing now waits for a new captured frame
instead of re-running inference on duplicate camera frames. The dashboard also
reports when firearm weights are absent.

```powershell
$env:INFERENCE_IMGSZ = "416"
$env:TRACKING_INTERVAL = "3"
$env:OBJECT_DETECTION_INTERVAL = "4"
```

Lower tracking intervals improve track refresh but use more GPU; higher
intervals generally improve throughput but make tracks/behaviour less responsive.
FPS depends on input resolution, camera, GPU load, model weights, and scene density.

### Two-camera crowd-flow mode

Choose **Crowd** in the dashboard, then select **Live cameras** or **Upload
videos**. For cameras, enter two different camera indexes. For videos, upload
one entry clip and one exit clip; both play at their recorded frame rate and
comparison ends when the first clip reaches its end. Use clips of the same
event and synchronized start time for a meaningful comparison. Set the
entry/exit crossing direction, rolling time window, and allowed count
mismatch. The current default is 60 seconds and 5 people.

The dashboard compares the people currently visible in both feeds and also
counts tracked centroid crossings of each center line (with a small
hysteresis band to reduce jitter). It logs an evidence snapshot when either
the visible people-count difference or the rolling entry/exit crossing-count
difference reaches the threshold. The snapshot gallery keeps a separate JPEG
per alert. Alerts currently stay in the dashboard and local event log;
external dispatch to authorities is not configured.

The two feeds must represent the same controlled entrance/exit boundary and,
for uploaded videos, the same time period. Camera placement, unsynchronized
clips, occlusion, missed detections, track-ID switches, or people crossing in
groups can affect the count. This is an operational warning aid, not a
guaranteed crowd-safety or emergency-dispatch system.

### 4. DeepSORT tracker (optional)

Only needed if you set `TRACKER_TYPE = "deepsort"`:

```bash
pip install deep-sort-realtime
```

---

## Configuration

The project has **two separate configuration files** — one for each entry point.

| File | Used by |
|------|---------|
| `config/settings.py` | `main.py` (CLI standalone pipeline) |
| `dashboard/config.py` | `dashboard/app.py` (Flask dashboard) |

Both files are fully commented. Key parameters are summarised below.

### Video Source

```python
# config/settings.py
VIDEO_SOURCE = 0                          # default webcam
VIDEO_SOURCE = "data/input/cctv.mp4"     # recorded video file

# dashboard/config.py
DEFAULT_SOURCE = 0                        # can be overridden from the UI
```

### Detection & Tracking

```python
YOLO_MODEL_PATH      = "models/yolov8s.pt"   # yolov8n = faster, yolov8s = more accurate
DETECTION_CONFIDENCE = 0.5                    # 0–1
TRACKER_TYPE         = "bytetrack"            # "bytetrack" | "deepsort"
INFERENCE_IMGSZ      = 640                    # inference resolution (px)
```

### Behaviour Thresholds

```python
OBSERVATION_WINDOW_SEC   = 10.0   # sliding window length (video seconds)
MIN_WINDOW_SAMPLES       = 10     # min data-points before classifying
MIN_SPEED_THRESHOLD      = 15.0   # px/s — below this = stationary
RUNNING_SPEED_THRESHOLD  = 180.0  # px/s — above this = Running
MOTION_VARIANCE_THRESHOLD= 50.0   # erratic motion → Running / Sudden Motion
LOITERING_STILLNESS      = 0.80   # fraction of window spent stationary
LOITERING_DISPLACEMENT   = 40.0   # max straight-line travel (px) for Loitering
PACE_RATIO_THRESHOLD     = 4.0    # back-and-forth ratio → Suspicious Lingering
VIOLATION_COOLDOWN       = 20.0   # min seconds between alerts per person
```

### Object Detection

```python
OBJECT_CONF_THRESHOLD      = 0.45  # confidence threshold for weapon/bag detector
OBJECT_TARGET_CLASSES      = [24, 26, 28, 43, 76]  # backpack, handbag, suitcase, knife, scissors
OBJECT_DETECTION_INTERVAL  = 4     # run every N frames
WEAPON_MODEL_PATH          = None  # set to a custom model path for gun detection
```

### Abandoned Bag & Theft Detection

```python
ABANDONED_BAG_SECONDS  = 30   # seconds stationary before "abandoned" alert
ABANDONED_BAG_COOLDOWN = 60   # seconds before re-alerting for same bag
BAG_DISAPPEAR_TIMEOUT  = 10   # seconds before removing a bag from tracking
THEFT_PROXIMITY_PX     = 80   # max px from person centroid to disappeared bag
THEFT_COOLDOWN         = 30   # seconds between theft alerts per track ID
```

### Configuring Restricted Zones (CLI pipeline)

Zones are defined as lists of `(x, y)` pixel coordinate polygons in `config/settings.py`:

```python
RESTRICTED_ZONES = [
    # Zone 1 — server room doorway
    [(100, 50), (400, 50), (400, 300), (100, 300)],
    # Zone 2 — emergency exit corridor
    [(800, 400), (1200, 400), (1200, 700), (800, 700)],
]
```

> **Tip:** Open a still frame from your camera in any image viewer, note the pixel coordinates of zone corners, and paste them here.

### Memory Management

```python
MAX_MEMORY_MB                 = 500   # RSS threshold (MB) before forced gc.collect()
BEHAVIOUR_BUFFER_MAX_PER_TRACK= 300   # max buffered entries per track
STALE_CLEANUP_INTERVAL        = 30    # seconds between cleanup cycles
STALE_TRACK_TIMEOUT           = 60    # seconds before a track is pruned
```

### Flask Dashboard

```python
FLASK_HOST  = "0.0.0.0"   # listen on all network interfaces
FLASK_PORT  = 5000
FLASK_DEBUG = False
```

---

## Running the System

### Option A — Standalone CLI (OpenCV window)

```bash
# Default source from config
python main.py

# Specific webcam
python main.py --source 0

# Video file
python main.py --source data/input/cctv.mp4

# Save annotated output video to data/output/
python main.py --source data/input/cctv.mp4 --save

# Headless — no display window (ideal for servers)
python main.py --no-display

# Combined: CLI window + Flask dashboard
python main.py --dashboard
```

Press **Q** in the OpenCV window or **Ctrl+C** in the terminal to stop.

### Option B — Flask Dashboard Only (recommended)

```bash
python dashboard/app.py
```

Open **http://localhost:5000** in your browser.

The dashboard allows you to:
1. Browse and select any video source via the native OS file browser
2. Start / stop the pipeline without restarting the server
3. Monitor live video, violation events, people-counter stats, and snapshot evidence
4. Preprocess a recorded video offline and stream the result

---

## Dashboard Architecture

```
Browser
  │  HTTP / MJPEG
  ▼
Flask App (main thread)
  ├── GET  /              → dashboard HTML
  ├── GET  /video_feed    → MJPEG stream (Thread 2 → latest_frame → here)
  ├── GET  /processed_feed→ MJPEG stream for offline-preprocessed video
  ├── POST /start         → launch pipeline (Threads 1, 2, 3)
  ├── POST /stop          → clean pipeline shutdown
  ├── GET  /events        → latest violation events as JSON
  ├── GET  /status        → pipeline status (running, FPS, track count)
  ├── GET  /stats         → people-counter + memory + bag stats
  ├── GET  /progress      → offline processing progress (0–100 %)
  ├── GET  /browse        → OS native file picker dialog
  ├── GET  /snapshots     → snapshot gallery metadata
  └── GET  /snapshot-image/<filename> → serve individual JPEG

Thread 1 — Capture
  Reads raw frames at source FPS → latest_raw_frame (lock-protected)
  Video timestamps derived from frame_index / source_fps (not wall-clock)
  Loops recorded video automatically at end of file

Thread 2 — Processing
  Reads each newly captured frame → YOLOv8 + ByteTrack/DeepSORT (configurable cadence)
  Ultralytics letterboxes source images and returns boxes in source coordinates
  Behaviour buffer → feature extraction → violation classification
  Reads asynchronous object worker results → weapon alerts, abandoned bag, theft
  Annotates frame → latest_frame (lock-protected)

Object Worker
  One in-flight YOLO object-detection job at a time, independent of person tracking

Log Worker (AlertLogger.run_worker)
  Drains log_queue (bounded, maxsize=50)
  Renders annotated snapshot JPEG → saves to data/snapshots/
  Appends row to data/logs/event_log.csv
  Updates in-memory event deque (read by /events)

Cleanup Timer (threading.Timer, every 30 s)
  Prunes stale tracks from behaviour buffer and alert cooldown dicts
  Calls BagTracker.cleanup_stale()
  Forces gc.collect() if RAM > MAX_MEMORY_MB
```

---

## System Pipeline (CLI)

```
Video Input
    │
    ▼
Person Detection (YOLOv8)
    │  List[Detection]
    ▼
Multi-Person Tracking (ByteTrack / DeepSORT)
    │  List[Track] + unique persistent IDs
    ▼
Behaviour Buffer  (10-s sliding window per person)
    │  List[BufferEntry]
    ▼
Feature Extraction  (7 motion features)
    │  avg_speed, max_speed, total_displacement, total_distance,
    │  stillness_ratio, pace_ratio, motion_variance
    ▼
Activity Classifier  (rules → optional Random Forest)
    │  "Normal" / "Loitering" / "Running" / "Suspicious Lingering" / "Intrusion"
    ├──▶ Alert Logger  → JPEG snapshot + CSV event log
    └──▶ Frame Annotator → annotated BGR frame
              │
              ├──▶ OpenCV Display Window  (CLI mode)
              └──▶ Flask MJPEG Stream → Browser Dashboard
```

---

## Training the Classifier

The system operates with **rule-based heuristics only** out of the box. Train the ML model to handle edge cases:

### Step 1 — Quick start with synthetic data

```bash
python train_classifier.py --generate
```

Generates `data/training_data.csv` (~2 000 synthetic samples) and trains a `StandardScaler + RandomForestClassifier` pipeline. Outputs:
- `models/activity_classifier.pkl`
- `data/logs/confusion_matrix.png`
- Console: accuracy, classification report, 5-fold CV scores

### Step 2 — Train on your own labelled data

Prepare a CSV with these columns:

| avg_speed | max_speed | total_displacement | total_distance | stillness_ratio | pace_ratio | motion_variance | label |
|-----------|-----------|--------------------|----------------|-----------------|------------|-----------------|-------|

Then run:

```bash
python train_classifier.py --data data/training_data.csv
```

---

## Extracting Features from Real Videos (UCF-Crime / Custom Dataset)

`extract_ucf_features.py` automates feature extraction from labelled video folders using the full YOLOv8 + ByteTrack pipeline.

```bash
# Extract from a folder of "Normal" clips
python extract_ucf_features.py --folder "D:/UCF-Crime/Normal" --label "Normal"

# Extract from "Loitering" clips
python extract_ucf_features.py --folder "D:/UCF-Crime/Loitering" --label "Loitering"

# Supported labels
python extract_ucf_features.py --label Running   # or "Suspicious Lingering"

# Force GPU or CPU
python extract_ucf_features.py --folder ... --label ... --device cuda
python extract_ucf_features.py --folder ... --label ... --device cpu
```

**Supported labels:** `Normal` · `Loitering` · `Running` · `Suspicious Lingering`

**What it does:**
- Recursively finds `.mp4`, `.avi`, `.mov` files in the given folder
- Runs YOLOv8 + ByteTrack on every 2nd frame (skips 1 for speed)
- Derives video-time timestamps from frame index / FPS (not wall-clock time)
- When a 10-second observation window fills for a track, extracts 7 features and appends a row to `data/training_data.csv`
- Clears the buffer after each extraction to avoid correlated duplicate rows
- Prints progress every 100 feature vectors extracted
- Prints a final summary of videos processed, skipped, and rows added

Once enough data is collected, train the model:

```bash
python train_classifier.py --data data/training_data.csv
```

---

## Dashboard Features

| Section | Description |
|---------|-------------|
| **Live Feed** | MJPEG stream paced to source FPS; auto-reconnects on drop |
| **Status Bar** | Live / Offline indicator, person count, FPS, alert count, uptime |
| **People Counter** | Current / peak / total-unique persons; session start time |
| **Memory Stats** | Current and peak RSS (MB); bags tracked and abandoned count |
| **Alert Events** | Real-time table refreshed every 3 s; filter chips by violation type; client-side search |
| **Snapshot Gallery** | Thumbnail strip with full-size lightbox on click |
| **File Browser** | Native OS dialog for selecting video files without typing paths |
| **Offline Processing** | Processes a full video offline, shows progress bar, then streams result |

---

## Violation Categories

| Label | Trigger condition |
|-------|-------------------|
| **Normal Movement** | No violation detected |
| **Loitering** | `stillness_ratio ≥ 0.80` AND `total_displacement ≤ 40 px` |
| **Running / Sudden Motion** | `avg_speed > 180 px/s` OR high `motion_variance` |
| **Suspicious Lingering** | `pace_ratio > 4.0` (back-and-forth without net displacement) |
| **Weapon Detected** | Knife or scissors detected by object detector (immediate, no cooldown) |
| **Abandoned Bag** | Bag stationary ≥ 30 s with no nearby person |
| **Possible Theft** | Tracked bag disappears and a person was within 80 px of its last position |

---

## Performance Tips

- Use `yolov8n.pt` for maximum CPU speed; use `yolov8s.pt` or `yolov8m.pt` for better accuracy on GPU.
- Lower `INFERENCE_IMGSZ` (e.g. `416`) to reduce GPU memory and increase throughput.
- Increase `OBJECT_DETECTION_INTERVAL` (e.g. `8`) if object detection is causing latency.
- Run the dashboard headless on a server — there is no need for a display.
- Set `LOG_LEVEL=DEBUG` (environment variable) to see per-frame diagnostic output.
- For recorded video, use the **offline preprocessing** mode — the full video is processed once, saved as an MP4, then streamed at its native FPS with no real-time processing overhead.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 detection + ByteTrack tracking |
| `opencv-python` | Frame capture, drawing, JPEG encoding, video I/O |
| `torch` / `torchvision` | Deep learning backend (CPU or CUDA) |
| `numpy` / `pandas` | Feature computation, event logging, CSV I/O |
| `scikit-learn` / `joblib` | Random Forest classifier, model serialisation |
| `flask` | Web dashboard server and MJPEG streaming |
| `psutil` | Process memory monitoring |
| `matplotlib` / `seaborn` | Confusion matrix and statistics visualisation |
| `deep-sort-realtime` | *(Optional)* DeepSORT tracker backend |

---

## License

MIT License — free to use, modify, and distribute.
