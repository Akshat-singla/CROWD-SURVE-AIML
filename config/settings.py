# =============================================================================
# config/settings.py
# Global configuration for the Real-Time AI Surveillance System.
# All tuneable parameters live here so every other module can import a single
# source of truth instead of hard-coding values.
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Project root — everything is relative to this so the project stays portable
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# 1.  VIDEO INPUT
# ===========================================================================

# Use an integer (e.g. 0) for the default webcam, or a file path string for
# a recorded video.  Examples:
#   VIDEO_SOURCE = 0                              # live webcam
#   VIDEO_SOURCE = "data/input/cctv_sample.mp4"  # recorded clip
VIDEO_SOURCE = 0

# Target frames-per-second for processing.  Frames captured faster than this
# are skipped so downstream modules are not overwhelmed.
# Set to None to process every frame the camera/video provides.
TARGET_FPS = 15


# ===========================================================================
# 2.  YOLOV8 DETECTION
# ===========================================================================

# Path to the YOLOv8 weights file.  Ultralytics will auto-download the
# specified model name on first run if the file is not found locally.
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")

# Minimum confidence score for a detection to be kept (0–1).
DETECTION_CONFIDENCE = 0.5

# Only detect the "person" class (COCO class index 0).
DETECTION_CLASSES = [0]

# Maximum number of tracked persons to process per frame (caps CPU/GPU load).
MAX_TRACKED_PERSONS = 20


# ===========================================================================
# 3.  TRACKING
# ===========================================================================

# Tracker to use: "bytetrack" or "botsort" (both shipped with Ultralytics).
TRACKER_TYPE = "bytetrack"

# Path to the tracker YAML config (Ultralytics looks for these in its package;
# set to None to use the built-in defaults).
TRACKER_CONFIG = None


# ===========================================================================
# 4.  BEHAVIOUR ANALYSIS
# ===========================================================================

# Duration (seconds) of the sliding observation window per tracked person.
OBSERVATION_WINDOW_SEC = 10

# Minimum number of data-points required in a window before classification
# is attempted (avoids noisy predictions at track start).
MIN_WINDOW_SAMPLES = 10

# Pixel-distance-per-second thresholds for speed-based classification.
SPEED_RUN_THRESHOLD = 80      # px/s — above this → "Running / Sudden Motion"
SPEED_WALK_THRESHOLD = 20     # px/s — below this AND loitering → "Loitering"

# Loitering: how many seconds a person must stay in the same area.
LOITERING_TIME_SEC = 8

# Loitering: maximum pixel radius from the centroid to still be "staying put".
LOITERING_RADIUS_PX = 60


# ===========================================================================
# 5.  RESTRICTED ZONES
# ===========================================================================

# List of polygon zones as lists of (x, y) pixel co-ordinates.
# Each polygon defines one restricted area on the frame.
# These are placeholder values — adjust them to match your camera layout.
RESTRICTED_ZONES = [
    # Example Zone 1 — top-left quadrant of a 1280×720 frame
    [(50, 50), (400, 50), (400, 350), (50, 350)],
    # Example Zone 2 — bottom-right corner
    [(880, 400), (1230, 400), (1230, 670), (880, 670)],
]

# Fill colour for zone overlays (BGR, semi-transparent via alpha blending).
ZONE_OVERLAY_COLOR = (0, 0, 255)   # red
ZONE_OVERLAY_ALPHA = 0.25          # transparency (0 = invisible, 1 = solid)


# ===========================================================================
# 6.  ALERT & EVENT LOGGING
# ===========================================================================

# Minimum seconds between two consecutive alerts for the SAME person.
# Prevents alert spam when a person stays in a suspicious state.
ALERT_COOLDOWN_SEC = 15

# Activity labels produced by the classifier.
ACTIVITY_LABELS = [
    "Normal Movement",
    "Loitering",
    "Running / Sudden Motion",
    "Restricted Area Intrusion",
]

# Activities that should trigger an alert (must match labels above exactly).
SUSPICIOUS_ACTIVITIES = [
    "Loitering",
    "Running / Sudden Motion",
    "Restricted Area Intrusion",
]

# Directory where JSON event logs are written.
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")

# Directory where evidence frame snapshots are saved when an alert fires.
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "snapshots")

# Ensure directories exist at import time.
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ===========================================================================
# 7.  OUTPUT VIDEO
# ===========================================================================

# Directory for annotated output video files.
OUTPUT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "output")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# FourCC codec for the output video writer.
OUTPUT_VIDEO_CODEC = "mp4v"

# Annotated frame JPEG quality (1–100) used when streaming via Flask.
STREAM_JPEG_QUALITY = 80


# ===========================================================================
# 8.  ACTIVITY CLASSIFIER MODEL
# ===========================================================================

# Path to the serialised Scikit-learn classifier (.pkl).
# Generated by running training/train_classifier.py.
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "models", "activity_classifier.pkl")


# ===========================================================================
# 9.  OBJECT DETECTION  (weapons, bags)
# ===========================================================================

# Confidence threshold for weapon / bag object detector.
OBJECT_CONF_THRESHOLD = 0.45

# Custom weapon-detection model (2 classes: person, weapon).
# Trained via: python train_weapon_detector.py --data data/weapon_dataset/data.yaml
# When None, falls back to YOLO_MODEL_PATH (COCO classes only: knife/scissors).
_DEFAULT_WEAPON_MODEL = os.path.join(BASE_DIR, "models", "weapon_yolov8.pt")
WEAPON_MODEL_PATH = (
    _DEFAULT_WEAPON_MODEL if os.path.isfile(_DEFAULT_WEAPON_MODEL) else None
)

# Run object detection every N frames.
OBJECT_DETECTION_INTERVAL = 4

# COCO object classes used as fallback when WEAPON_MODEL_PATH is None:
#   24=backpack, 26=handbag, 28=suitcase, 43=knife, 76=scissors
OBJECT_TARGET_CLASSES = [24, 26, 28, 43, 76]


# ===========================================================================
# 10.  FLASK DASHBOARD
# ===========================================================================

FLASK_HOST = "0.0.0.0"   # listen on all interfaces
FLASK_PORT = 5000
FLASK_DEBUG = False       # set True only during development

# How many recent alert entries to display in the dashboard alert panel.
DASHBOARD_MAX_ALERTS = 50


# ===========================================================================
# 11.  ANNOTATION / DISPLAY
# ===========================================================================

# Bounding-box colours per activity category (BGR).
ACTIVITY_COLORS = {
    "Normal Movement":           (0, 200, 0),    # green
    "Loitering":                 (0, 165, 255),  # orange
    "Running / Sudden Motion":   (0, 0, 255),    # red
    "Restricted Area Intrusion": (128, 0, 128),  # purple
    "Unknown":                   (200, 200, 200) # grey (before first prediction)
}

# Whether to draw the movement trail (recent centroid positions) per person.
DRAW_TRAILS = True
TRAIL_LENGTH = 30          # number of past centroids to draw
TRAIL_COLOR = (255, 255, 0)  # yellow (BGR)
