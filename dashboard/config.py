# =============================================================================
# dashboard/config.py
# Centralised configuration for the AI Surveillance Dashboard.
#
# All tunable thresholds, paths, and server settings live here so they can
# be adjusted without touching any pipeline logic.
# =============================================================================

import os
import torch

# ---------------------------------------------------------------------------
# Device selection — GPU if available, otherwise CPU.
# ---------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Project root — makes all relative paths work regardless of CWD
# ---------------------------------------------------------------------------
_DASH_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(_DASH_DIR)

# ===========================================================================
# 1.  VIDEO / CAPTURE
# ===========================================================================

# Default video source used if none is supplied via the UI.
# Use an integer for a webcam (e.g. 0) or a file path string.
DEFAULT_SOURCE = 0

# JPEG quality for the MJPEG stream (1-100).  60 is sufficient for
# surveillance display and significantly reduces bandwidth + stream latency.
# (FIX 5: was 75)
STREAM_JPEG_QUALITY = 60

# JPEG quality for evidence snapshots (higher = better forensic detail).
SNAPSHOT_JPEG_QUALITY = 95

# ===========================================================================
# 2.  YOLO DETECTION  (person tracking model)
# ===========================================================================

# Path to the YOLOv8 weights file.  Ultralytics auto-downloads on first run.
# Nano is substantially faster for live surveillance. Use YOLO_MODEL_PATH to
# opt into the larger model when accuracy is more important than FPS.
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "yolov8n.pt"),
)

# Minimum detection confidence (0–1).
DETECTION_CONFIDENCE = 0.5

# COCO class index for "person".  We only detect people for tracking.
DETECTION_CLASSES = [0]

# YOLOv8 inference image size.  Full 640px is used now that GPU is available.
INFERENCE_IMGSZ = int(os.environ.get("INFERENCE_IMGSZ", "416"))
TRACKING_INTERVAL = int(os.environ.get("TRACKING_INTERVAL", "3"))

# ===========================================================================
# 3.  TRACKER SELECTION
# ===========================================================================

# Use "bytetrack" for most scenarios — faster, no Re-ID model needed,
# performs well in low-to-moderate occlusion environments.
# Use "deepsort" when persons frequently overlap and IDs keep switching,
# as DeepSORT uses appearance features to recover correct identities.
# NOTE: "deepsort" requires:  pip install deep-sort-realtime
TRACKER_TYPE = "bytetrack"   # options: "bytetrack" | "deepsort"

# ===========================================================================
# 4.  BEHAVIOUR ANALYSIS WINDOW
# ===========================================================================

# Duration (seconds) of the sliding per-track observation window.
OBSERVATION_WINDOW_SEC = 10.0

# Minimum number of data-points in a window before classification runs.
# Prevents noisy predictions at the very start of a track.
MIN_WINDOW_SAMPLES = 10

# ===========================================================================
# 5.  VIOLATION THRESHOLDS
# ===========================================================================
#
# IMPORTANT — speed units (Fix 3 & Fix 6):
# All speed thresholds below are in PIXELS PER SECOND of VIDEO TIME.
# Thread 1 computes video timestamps from frame_index / source_fps, so the
# feature extractor always measures real video-time motion regardless of how
# fast or slow the CPU processes frames.
#
# If you switch to a very different video resolution (e.g. 240p vs 1080p),
# re-tune MIN_SPEED_THRESHOLD and RUNNING_SPEED_THRESHOLD accordingly, because
# the same physical motion produces different pixel displacements at each
# resolution.  The other thresholds (ratios, displacement, cooldown) are
# resolution-independent.
# ===========================================================================

# ── Behaviour Classification Thresholds ───────────────────────────────────
# All speed values are in PIXELS PER SECOND of video time, calibrated for
# 640px inference size on typical surveillance footage.

# Minimum speed (px/sec) to consider a person moving.
MIN_SPEED_THRESHOLD = 15.0

# Speed (px/sec) above which motion is classified as running.
RUNNING_SPEED_THRESHOLD = 180.0

# Motion variance above this confirms erratic / sudden movement.
MOTION_VARIANCE_THRESHOLD = 50.0

# Stillness ratio above which Loitering is declared.
# (0.80 = person was stationary for 80 % of the window)
LOITERING_STILLNESS = 0.80

# Maximum total displacement (straight-line px) for Loitering to fire.
LOITERING_DISPLACEMENT = 40.0

# pace_ratio = total_distance / (total_displacement + ε)
# Above this threshold the person is going back-and-forth → Suspicious Lingering.
PACE_RATIO_THRESHOLD = 4.0

# High-energy irregular motion that is not confidently classified as running,
# loitering, or pacing is reported as Unsafe Activity.
UNSAFE_MOTION_VARIANCE_THRESHOLD = 1200.0
UNSAFE_MIN_SPEED = 25.0

# Seconds of VIDEO TIME before the same track ID can trigger another alert.
VIOLATION_COOLDOWN = 20.0

# ===========================================================================
# 6.  OBJECT DETECTION  (weapons, bags — Enhancement 3 & 4)
# ===========================================================================

# Confidence threshold for the object detector (separate from person tracker).
# Lower = more detections but more false positives.
OBJECT_CONF_THRESHOLD = float(os.environ.get("OBJECT_CONF_THRESHOLD", "0.30"))

# COCO classes to detect (indices into the COCO 80-class list):
#   24=backpack, 26=handbag, 28=suitcase, 43=knife, 76=scissors
OBJECT_TARGET_CLASSES = [24, 26, 28, 43, 76]

# NOTE ON GUN DETECTION:
# The standard COCO-trained YOLOv8n does not include a "gun" class.
# To add gun detection, train or obtain a custom weapon-detection model
# (e.g. one trained on the Open Images "Handgun" / "Rifle" classes)
# and set WEAPON_MODEL_PATH below.  The ObjectDetector class will use
# WEAPON_MODEL_PATH when it is not None, falling back to YOLO_MODEL_PATH.
_DEFAULT_WEAPON_MODEL = os.path.join(BASE_DIR, "models", "weapon_yolov8.pt")
WEAPON_MODEL_PATH = os.environ.get("WEAPON_MODEL_PATH") or (
    _DEFAULT_WEAPON_MODEL if os.path.isfile(_DEFAULT_WEAPON_MODEL) else None
)

# Run object detection every N frames (4 = 25% of frames at 15 fps ≈ ~3-4 det/s).
OBJECT_DETECTION_INTERVAL = int(os.environ.get("OBJECT_DETECTION_INTERVAL", "4"))

# Prevent repeated weapon alerts while the same object remains visible.
WEAPON_ALERT_COOLDOWN = 10.0

# Crowd-flow alerts are based on crossings within this rolling comparison.
CROWD_DEFAULT_WINDOW_SECONDS = 60
CROWD_DEFAULT_MISMATCH_THRESHOLD = 5
CROWD_ALERT_COOLDOWN = 30.0

# ===========================================================================
# 7.  ABANDONED BAG DETECTION  (Enhancement 4)
# ===========================================================================

# Seconds a bag must remain stationary (without a nearby person) before
# it is classified as abandoned.
ABANDONED_BAG_SECONDS = 30

# Cooldown (seconds) before re-alerting for the same abandoned bag.
ABANDONED_BAG_COOLDOWN = 60

# Remove bags from tracker if not seen for this many seconds.
BAG_DISAPPEAR_TIMEOUT = 10

# ===========================================================================
# 8.  THEFT DETECTION  (Enhancement 5)
# ===========================================================================

# Maximum pixel distance from a person's centroid to a disappeared bag's
# last known centre for a theft flag to be raised.
THEFT_PROXIMITY_PX = 80

# Cooldown (seconds) before re-flagging theft for the same track ID.
THEFT_COOLDOWN = 30

# ===========================================================================
# 9.  MEMORY MANAGEMENT  (Enhancement 2)
# ===========================================================================

# If the process RSS exceeds this value (MB), force gc.collect() + log a warning.
MAX_MEMORY_MB = 500

# Maximum snapshots kept per track ID in behaviour_buffer history.
# Oldest entries are trimmed when this limit is exceeded.
BEHAVIOUR_BUFFER_MAX_PER_TRACK = 300

# Interval (seconds) between stale-track cleanup cycles.
STALE_CLEANUP_INTERVAL = 30

# A track not seen for this many seconds is considered stale and purged.
STALE_TRACK_TIMEOUT = 60

# ===========================================================================
# 10.  LOGGING & SNAPSHOTS
# ===========================================================================

# Directory for evidence JPEG snapshots.
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "snapshots")

# Directory for the CSV event log.
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")

# CSV filename.
LOG_FILENAME = "event_log.csv"

# Maximum events kept in-memory (server side).
EVENT_LOG_MAXLEN = 200

# Maximum events returned by the /events endpoint.
DASHBOARD_MAX_EVENTS = 50

# Maximum jobs in the log_queue before drops occur (prevents unbounded growth).
LOG_QUEUE_MAXSIZE = 50

# Ensure directories exist at import time.
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===========================================================================
# 11.  FLASK SERVER
# ===========================================================================

FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False

# ===========================================================================
# 12.  ANNOTATION COLOURS  (BGR tuples — OpenCV convention)
# ===========================================================================

# Normal / non-flagged persons.
COLOR_NORMAL    = (0, 200, 0)       # green

# Behaviour-based violations (person-tracking violations).
COLOR_LOITERING = (0, 165, 255)     # orange
COLOR_RUNNING   = (0, 0, 220)       # bright red
COLOR_LINGERING = (128, 0, 200)     # purple  (Suspicious Lingering)
COLOR_UNSAFE    = (0, 80, 255)      # orange-red (unsafe / strange activity)

# Object-detection violations.
COLOR_WEAPON        = (0, 0, 255)   # bright red   (weapon detected)
COLOR_ABANDONED_BAG = (0, 140, 255) # orange       (abandoned bag)
COLOR_THEFT         = (255, 0, 255) # magenta      (possible theft)

COLOR_UNKNOWN   = (160, 160, 160)   # grey

# Mapping from violation type string → BGR colour (used by alert_logger renderer).
VIOLATION_COLOR_MAP = {
    "Running / Sudden Motion": COLOR_RUNNING,
    "Loitering":               COLOR_LOITERING,
    "Suspicious Lingering":    COLOR_LINGERING,
    "Unsafe Activity":         COLOR_UNSAFE,
    "Weapon Detected":         COLOR_WEAPON,
    "Abandoned Bag":           COLOR_ABANDONED_BAG,
    "Possible Theft":          COLOR_THEFT,
}
