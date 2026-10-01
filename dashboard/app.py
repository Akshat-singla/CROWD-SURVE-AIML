import gc
# from concurrent.futures import Future, ThreadPoolExecutor  # no longer needed
import logging
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
# queue is still imported for log_queue used inside AlertLogger
from typing import Optional, Union

import cv2
import numpy as np
import psutil
import torch
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# ---------------------------------------------------------------------------
# sys.path setup — must happen before any local imports
# ---------------------------------------------------------------------------
_DASH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_DASH_DIR)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _DASH_DIR in sys.path:
    sys.path.remove(_DASH_DIR)
sys.path.insert(0, _DASH_DIR)

import config  # resolves to dashboard/config.py

from modules.tracker           import PersonTracker
from modules.behaviour_buffer  import BehaviourBuffer
from modules.feature_extractor import FeatureExtractor
from modules.classifier        import ViolationClassifier
from modules.alert_logger      import AlertLogger
from modules.object_detector   import ObjectDetector, BagTracker, compute_iou
from modules.crowd_flow        import CrowdFlowMonitor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
cv2.setNumThreads(1)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(_DASH_DIR, "templates"),
    static_folder=os.path.join(_DASH_DIR, "static"),
)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# ===========================================================================
# Shared frame holders  (Fix 1 & Fix 2)
# Queues are replaced with single shared variables protected by locks.
# The MJPEG generator and Thread 2 always see the *latest* frame with zero
# possibility of serving stale backlogged frames.
# ===========================================================================

# Thread 1 → Thread 2: most-recent raw BGR frame.
latest_raw_frame = None
latest_raw_frame_id = 0
raw_frame_lock   = threading.Lock()

# Thread 2 → MJPEG generator: most-recent annotated BGR frame.
# Fix 1: pre-initialize with a placeholder so the generator serves something
# immediately, before Thread 2 has produced its first annotated frame.
_placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(
    _placeholder_frame, "Waiting for video...",
    (155, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA,
)
latest_frame = _placeholder_frame.copy()
frame_lock   = threading.Lock()

# Source FPS — written by Thread 1 so the MJPEG generator can pace itself.
source_fps: float = 25.0
source_fps_lock   = threading.Lock()

# Flag set by _launch_pipeline: True when source is a file (enables looping).
_is_recorded_video: bool = False

# Fix 1: video timestamp (seconds from video start) delivered alongside the
# raw frame by Thread 1 using frame_index / source_fps — NOT wall-clock time.
latest_video_timestamp: float = 0.0
video_timestamp_lock          = threading.Lock()

# Fix 5: processing-speed stats — written by Thread 2, read by /stats route.
_speed_stats_lock = threading.Lock()
_speed_stats: dict = {
    "processing_fps": 0.0,   # how fast Thread 2 is processing frames
    "video_fps":      0.0,   # native FPS of the source video
    "speed_ratio":    0.0,   # processing_fps / video_fps
}

# Pre-processing mode — for recorded video, the full video is processed
# offline first and saved as an annotated MP4, then streamed to the dashboard.
is_processing:      bool          = False
processing_progress: int          = 0
processed_video_path: Optional[str] = None
_preproc_lock = threading.Lock()

# Pipeline control.
_stop_event:        threading.Event           = threading.Event()
_log_stop_event:    threading.Event           = threading.Event()
_capture_thread:    Optional[threading.Thread] = None
_process_thread:    Optional[threading.Thread] = None
_log_worker_thread: Optional[threading.Thread] = None
_crowd_threads: list[threading.Thread] = []
_crowd_mode = False
_crowd_monitor: Optional[CrowdFlowMonitor] = None
_crowd_frames: dict[str, np.ndarray] = {}
_crowd_frames_lock = threading.Lock()
_crowd_alert_lock = threading.Lock()
_crowd_last_alert = 0.0
_crowd_camera_status: dict[str, str] = {"entry": "offline", "exit": "offline"}
_crowd_people_counts: dict[str, int] = {"entry": 0, "exit": 0}
_crowd_source_type = "camera"
_crowd_directions = {"entry": "left_to_right", "exit": "left_to_right"}
_cleanup_timer:     Optional[threading.Timer]  = None   # Enhancement 2

# AlertLogger singleton — created fresh each time the live pipeline starts.
_alert_logger: Optional[AlertLogger] = None

# Pre-processing alert logger — used for recorded video mode.
# Separate from _alert_logger which is used for live camera mode.
_preproc_alert_logger: Optional[AlertLogger] = None

# Status shared between Thread 2 and Flask routes.
_status_lock = threading.Lock()
_status: dict = {
    "pipeline_running": False,
    "fps":              0.0,
    "active_tracks":    0,
    "source":           "",
    "mode":             "offline",
}

# ===========================================================================
# People-counter + memory stats  (Enhancements 2)
# Protected by _stats_lock — written by Thread 2, read by /stats route.
# ===========================================================================
_stats_lock = threading.Lock()
_stats: dict = {
    "current_count":   0,
    "peak_count":      0,
    "total_unique":    0,
    "session_start":   None,
    "memory_mb":       0.0,
    "peak_memory_mb":  0.0,
    "bags_tracked":    0,
    "abandoned_bags":  0,
}
_unique_ids: set = set()

# psutil process handle — created once at module load.
_process = psutil.Process()


# ===========================================================================
# Memory helpers  (Enhancement 2)
# ===========================================================================

def _get_memory_mb() -> float:
    """Return current process RSS in megabytes."""
    try:
        return _process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _check_memory() -> float:
    """
    Read current RAM usage; force gc if over threshold.
    Returns current MB.
    """
    current_mb = _get_memory_mb()
    if current_mb > config.MAX_MEMORY_MB:
        logger.warning(
            "Memory usage %.1f MB exceeds threshold %d MB — forcing gc.collect().",
            current_mb, config.MAX_MEMORY_MB,
        )
        gc.collect()
    return current_mb


# ===========================================================================
# Placeholder JPEG
# ===========================================================================

def _make_placeholder() -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "No active feed", (155, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (55, 55, 55), 2, cv2.LINE_AA)
    cv2.putText(img, "Select source and press Start", (100, 265),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


_PLACEHOLDER_JPG = _make_placeholder()


# ===========================================================================
# Annotation helper (Thread 2 — live stream frame only)
# NOTE: violation snapshots are rendered separately in Thread 3 (alert_logger).
# ===========================================================================

def _annotate_frame(
    frame:           np.ndarray,
    tracks:          list,
    activities:      dict,
    fps:             float,
    person_count:    int,
    weapon_detects:  list,
    abandoned_bags:  list,
    theft_track_ids: set,
) -> np.ndarray:
    """
    Draw bounding boxes, HUD, weapon/bag overlays onto a copy of the frame.
    This is the live stream annotation — NOT the snapshot renderer.

    Parameters
    ----------
    frame           : Raw BGR frame.
    tracks          : Active Track objects.
    activities      : track_id → violation label or None.
    fps             : Current processing FPS.
    person_count    : Active person count.
    weapon_detects  : List of weapon ObjectDetection objects (current frame).
    abandoned_bags  : List of abandoned _BagEntry objects (current frame).
    theft_track_ids : Set of track_ids flagged for theft this frame.
    """
    COLOR_MAP = {
        "Running / Sudden Motion": config.COLOR_RUNNING,
        "Loitering":               config.COLOR_LOITERING,
        "Suspicious Lingering":    config.COLOR_LINGERING,
        None:                      config.COLOR_NORMAL,
    }

    out = frame.copy()

    # ── Person tracks ─────────────────────────────────────────────────
    for track in tracks:
        if track.track_id in theft_track_ids:
            color = config.COLOR_THEFT
            label_text = "⚠ POSSIBLE THEFT"
        else:
            label = activities.get(track.track_id)
            color = COLOR_MAP.get(label, config.COLOR_NORMAL)
            label_text = label if label else "Normal"

        box_label = f"ID:{track.track_id} | {label_text}"

        cv2.rectangle(out, (track.x1, track.y1), (track.x2, track.y2), color, 2)

        # Corner tick marks
        tick = 10
        for px, py, dx, dy in [
            (track.x1, track.y1,  1,  1),
            (track.x2, track.y1, -1,  1),
            (track.x1, track.y2,  1, -1),
            (track.x2, track.y2, -1, -1),
        ]:
            cv2.line(out, (px, py), (px + dx * tick, py), color, 2)
            cv2.line(out, (px, py), (px, py + dy * tick), color, 2)

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        (tw, th), _ = cv2.getTextSize(box_label, font, scale, thick)
        lx = track.x1
        ly = max(track.y1 - 6, th + 4)
        cv2.rectangle(out, (lx - 2, ly - th - 3), (lx + tw + 2, ly + 2),
                      color, cv2.FILLED)
        cv2.putText(out, box_label, (lx, ly),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    # ── Weapon detections ─────────────────────────────────────────────
    for wd in weapon_detects:
        x1, y1, x2, y2 = wd.bbox
        wlabel = f"\u26a0 WEAPON: {wd.label}"
        cv2.rectangle(out, (x1, y1), (x2, y2), config.COLOR_WEAPON, 3)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        (tw, th), _ = cv2.getTextSize(wlabel, font, scale, thick)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1 - 2, ly - th - 3), (x1 + tw + 2, ly + 2),
                      config.COLOR_WEAPON, cv2.FILLED)
        cv2.putText(out, wlabel, (x1, ly),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    # ── Abandoned bags ────────────────────────────────────────────────
    for bag in abandoned_bags:
        x1, y1, x2, y2 = bag.bbox
        blabel = f"\u26a0 ABANDONED BAG #{bag.bag_id}"
        cv2.rectangle(out, (x1, y1), (x2, y2), config.COLOR_ABANDONED_BAG, 3)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
        (tw, th), _ = cv2.getTextSize(blabel, font, scale, thick)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1 - 2, ly - th - 3), (x1 + tw + 2, ly + 2),
                      config.COLOR_ABANDONED_BAG, cv2.FILLED)
        cv2.putText(out, blabel, (x1, ly),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    # ── HUD: FPS (top-left) ───────────────────────────────────────────
    hud_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 26),
                hud_font, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    # ── HUD: person count (top-right) ─────────────────────────────────
    count_text = f"Persons: {person_count}"
    (cw, _), _ = cv2.getTextSize(count_text, hud_font, 0.65, 2)
    h_, w_ = out.shape[:2]
    cv2.putText(out, count_text, (w_ - cw - 10, 26),
                hud_font, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    return out


# ===========================================================================
# Thread 1 — Capture Thread
# ===========================================================================

def _capture_thread_fn(source, stop_event: threading.Event) -> None:
    """
    Read frames from the video source and push them into the shared frame
    holder alongside a video-time timestamp computed from the frame index.

    Fix 1 & Fix 2:
    - For recorded video: reads as fast as possible (no sleep) so Thread 2
      is never throttled by Thread 1.  Video timestamps are derived from
      frame_index / source_fps — not wall-clock time — so the behaviour
      buffer always measures accurate video-time windows.
    - For live camera: sleeps to match the hardware capture rate.
    """
    # Declare all globals first — Python requires global declaration before any
    # read or write of a module-level variable inside a function.
    global source_fps, latest_raw_frame, latest_raw_frame_id
    global latest_video_timestamp, _is_recorded_video

    logger.info("[Thread 1] Capture thread starting — source: %s", source)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logger.error("[Thread 1] Cannot open source '%s'.", source)
        with _status_lock:
            _status["pipeline_running"] = False
        return

    # Clamp source FPS to a sensible range.
    source_fps_val = cap.get(cv2.CAP_PROP_FPS)
    if source_fps_val <= 0 or source_fps_val > 120:
        source_fps_val = 25.0
    frame_interval = 1.0 / source_fps_val

    logger.info(
        "[Thread 1] Source opened — FPS=%.1f, recorded=%s",
        source_fps_val, _is_recorded_video,
    )

    # Publish source FPS for the MJPEG generator pacing and /stats endpoint.
    with source_fps_lock:
        source_fps = source_fps_val
    with _speed_stats_lock:
        _speed_stats["video_fps"] = source_fps_val

    # Fix 1: track frame index to compute video-time timestamps.
    frame_index = 0
    _first_ts_printed = False

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                # Fix 5 (carried forward): loop recorded video instead of stopping.
                if _is_recorded_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = 0
                    logger.info("[Thread 1] Video ended — restarting from beginning.")
                    print("[INFO] Video ended — restarting from beginning")
                    continue
                else:
                    logger.info("[Thread 1] Live source exhausted or disconnected.")
                    break

            # Fix 1: compute video-time timestamp from frame position.
            video_ts = frame_index / source_fps_val
            frame_index += 1

            # Confirm the first few timestamps in the terminal.
            if not _first_ts_printed and frame_index <= 5:
                print(f"[VTS] frame_index={frame_index-1}  video_ts={video_ts:.4f}s  "
                      f"(source_fps={source_fps_val:.1f})")
                if frame_index == 5:
                    _first_ts_printed = True

            # Update shared frame and its video timestamp atomically.
            with raw_frame_lock:
                latest_raw_frame = frame.copy()
                latest_raw_frame_id += 1
                latest_video_timestamp = video_ts

            # Fix 2: no sleep for recorded video — read as fast as possible.
            # For live camera, pace to avoid spinning the CPU.
            if not _is_recorded_video:
                elapsed    = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        cap.release()
        logger.info("[Thread 1] Capture thread stopped.")
        with _status_lock:
            _status["pipeline_running"] = False


# ===========================================================================
# Stale track cleanup  (Enhancement 2)
# ===========================================================================

def _schedule_stale_cleanup(
    stop_event:       threading.Event,
    buf:              BehaviourBuffer,
    last_alert_time:  dict,
    track_last_seen:  dict,
    theft_cooldown:   dict,
    bag_tracker:      BagTracker,
) -> None:
    """
    Schedule the next stale-track cleanup cycle via threading.Timer.
    Reschedules itself until stop_event is set.
    """
    if stop_event.is_set():
        return

    now       = time.time()
    cutoff    = now - config.STALE_TRACK_TIMEOUT
    stale_ids = [tid for tid, ts in track_last_seen.items() if ts < cutoff]

    if stale_ids:
        logger.info(
            "[Cleanup] Removing %d stale track(s): %s", len(stale_ids), stale_ids
        )
        for tid in stale_ids:
            buf.remove(tid)                   # remove this track's centroid buffer
            last_alert_time.pop(tid, None)
            track_last_seen.pop(tid, None)
            theft_cooldown.pop(tid, None)

        gc.collect()
        mem_mb = _get_memory_mb()
        logger.info(
            "[Cleanup] Stale cleanup done — %d track(s) cleared, RAM=%.1f MB.",
            len(stale_ids), mem_mb,
        )

    # Clean up abandoned bags in the same cycle
    bag_tracker.cleanup_stale(now)

    # Reschedule
    global _cleanup_timer
    _cleanup_timer = threading.Timer(
        config.STALE_CLEANUP_INTERVAL,
        _schedule_stale_cleanup,
        args=(stop_event, buf, last_alert_time, track_last_seen, theft_cooldown, bag_tracker),
    )
    _cleanup_timer.daemon = True
    _cleanup_timer.start()


# ===========================================================================
# Thread 2 — Processing Thread
# ===========================================================================

def _process_thread_fn(stop_event: threading.Event, alert_logger: AlertLogger) -> None:
    """
    Pull frames from raw_queue, run detection/tracking, classify violations,
    run object detection, check abandoned bags and theft, then annotate and
    push JPEG bytes into display_queue.
    """
    logger.info("[Thread 2] Processing thread starting.")

    # ── Initialise person tracking pipeline ──────────────────────────
    tracker   = PersonTracker(
        model_path=config.YOLO_MODEL_PATH,
        confidence=config.DETECTION_CONFIDENCE,
        imgsz=config.INFERENCE_IMGSZ,
        tracker_type=config.TRACKER_TYPE,
    )
    buf       = BehaviourBuffer(
        window_sec=config.OBSERVATION_WINDOW_SEC,
        min_samples=config.MIN_WINDOW_SAMPLES,
    )
    extractor = FeatureExtractor(
        min_samples=config.MIN_WINDOW_SAMPLES,
        min_speed_threshold=config.MIN_SPEED_THRESHOLD,  # px/sec of video time
    )
    classifier = ViolationClassifier(
        running_speed_threshold=config.RUNNING_SPEED_THRESHOLD,
        motion_variance_threshold=config.MOTION_VARIANCE_THRESHOLD,
        loitering_stillness=config.LOITERING_STILLNESS,
        loitering_displacement=config.LOITERING_DISPLACEMENT,
        pace_ratio_threshold=config.PACE_RATIO_THRESHOLD,
        min_speed_threshold=config.MIN_SPEED_THRESHOLD,
        unsafe_motion_variance_threshold=config.UNSAFE_MOTION_VARIANCE_THRESHOLD,
        unsafe_min_speed=config.UNSAFE_MIN_SPEED,
    )

    # ── Initialise object detector and bag tracker ────────────────────
    # Enhancement 3 & 4: separate model instance from the person tracker.
    obj_model_path = (config.WEAPON_MODEL_PATH
                      if config.WEAPON_MODEL_PATH else config.YOLO_MODEL_PATH)
    obj_detector = ObjectDetector(
        model_path=obj_model_path,
        conf_threshold=config.OBJECT_CONF_THRESHOLD,
        # Custom weapon models generally use different class IDs; allowing
        # all classes lets ObjectDetector classify them by model label.
        target_classes=(
            None if config.WEAPON_MODEL_PATH else config.OBJECT_TARGET_CLASSES
        ),
    )
    model_names = getattr(obj_detector._model, "names", {})
    logger.info(
        "[Thread 2] Object model=%s classes=%s device=%s",
        obj_model_path, model_names, obj_detector.device,
    )
    if not config.WEAPON_MODEL_PATH:
        logger.warning(
            "Firearm detection unavailable: default COCO weights have no gun class. "
            "Train/provide models/weapon_yolov8.pt or set WEAPON_MODEL_PATH."
        )
    elif not any(
        token in str(label).lower()
        for label in (model_names.values() if isinstance(model_names, dict) else model_names)
        for token in ("gun", "firearm", "pistol", "rifle", "handgun", "weapon")
    ):
        logger.error("Configured weapon model has no recognizable firearm class names: %s", model_names)
    bag_tracker = BagTracker(
        abandoned_seconds=config.ABANDONED_BAG_SECONDS,
        disappear_timeout=config.BAG_DISAPPEAR_TIMEOUT,
        alert_cooldown=config.ABANDONED_BAG_COOLDOWN,
    )

    # ── Per-track state dicts ─────────────────────────────────────────
    last_alert_time: dict = {}    # track_id → last violation Unix time
    track_last_seen: dict = {}    # track_id → last seen Unix time  (Enhancement 2)
    theft_cooldown:  dict = {}    # track_id → last theft alert Unix time
    weapon_alert_time: dict = {}  # weapon label → last alert video timestamp

    # Object detection state (Enhancement 3 & 5)
    prev_bag_detections: list = []   # bag ObjectDetections from previous det frame

    # Currently displayed abandoned-bag entries (for annotation)
    current_abandoned: list = []

    # Frame counters
    frame_number = 0
    last_tracks  = []

    # FPS measurement
    fps_counter = 0
    fps_ts      = time.perf_counter()
    current_fps = 0.0

    # Session start (for snapshot footer)
    session_start: Optional[float] = None
    with _stats_lock:
        session_start = _stats.get("session_start")

    # ── Start stale cleanup timer ─────────────────────────────────────
    _schedule_stale_cleanup(
        stop_event, buf, last_alert_time, track_last_seen, theft_cooldown, bag_tracker
    )

    logger.info("[Thread 2] All pipeline modules initialised.")

    # FIX 6: detect and log the inference device at startup
    _infer_device = tracker.detector.device if hasattr(tracker, 'detector') else "cpu"
    print(f"[INFO] Running person-tracker inference on: {_infer_device}")
    logger.info("[Thread 2] Inference device: %s", _infer_device)

    # FIX 2/4: inference resolution
    _INFER_SIZE = config.INFERENCE_IMGSZ

    # Fix 2 (CRITICAL): declare globals so assignments update module-level vars.
    global latest_frame, latest_raw_frame, latest_raw_frame_id

    # Fix 4: wait until Thread 1 has delivered the first raw frame.
    logger.info("[Thread 2] Waiting for first frame from Thread 1...")
    while latest_raw_frame is None and not stop_event.is_set():
        time.sleep(0.05)
    if stop_event.is_set():
        return
    print("[INFO] Thread 2 received first frame, starting processing")
    logger.info("[Thread 2] First frame received — processing loop starting.")

    # Fix 5: wall-clock FPS counter for the processing-speed indicator.
    _proc_fps_counter = 0
    _proc_fps_ts      = time.perf_counter()
    last_processed_frame_id = 0

    # Track whether we've confirmed the first successful frame write.
    _first_frame_written = False
    latest_weapon_detects: list = []
    # Object detection now runs synchronously on the current frame for zero delay
    # (OBJECT_DETECTION_INTERVAL=1 means every frame; increase for performance)

    try:
        while not stop_event.is_set():
            # Read the most-recent raw frame and its video-time timestamp.
            with raw_frame_lock:
                frame = latest_raw_frame
                frame_id = latest_raw_frame_id
                video_ts = latest_video_timestamp

            if frame is None:
                time.sleep(0.005)
                continue
            if frame_id == last_processed_frame_id:
                time.sleep(0.001)
                continue
            last_processed_frame_id = frame_id

            # Fix 1: use video_ts (video time) instead of time.time() everywhere
            # so all behaviour windows are measured in actual video time.
            timestamp    = video_ts
            frame_number += 1

            # FIX 8: always work on a copy so the raw frame is never mutated
            original_frame = frame.copy()
            # FIX 2: Detection + Tracking (every 3rd frame instead of 2nd) ─
            if frame_number % config.TRACKING_INTERVAL == 0:
                # Ultralytics resizes with aspect-preserving letterboxing and
                # returns boxes in source-frame coordinates.
                raw_tracks = tracker.update(original_frame, timestamp)
                tracks      = raw_tracks
                last_tracks = tracks
            else:
                tracks = last_tracks
                for t in tracks:
                    t = t.__class__(
                        track_id=t.track_id, x1=t.x1, y1=t.y1, x2=t.x2, y2=t.y2,
                        cx=t.cx, cy=t.cy, confidence=t.confidence, timestamp=timestamp,
                    )

            active_ids = {t.track_id for t in tracks}

            # ── Behaviour buffer update ───────────────────────────────
            for track in tracks:
                buf.update(
                    track_id=track.track_id,
                    cx=track.cx,
                    cy=track.cy,
                    timestamp=timestamp,
                )

                # Enhancement 2: enforce max buffer size per track
                window = buf.get(track.track_id)
                if window and len(window) > config.BEHAVIOUR_BUFFER_MAX_PER_TRACK:
                    # trim oldest: replace with a deque/list slice
                    trimmed = list(window)[-config.BEHAVIOUR_BUFFER_MAX_PER_TRACK:]
                    buf._buffers[track.track_id].clear()
                    for entry in trimmed:
                        buf._buffers[track.track_id].append(entry)

            buf.remove_stale(active_ids)

            # ── Update track_last_seen with video timestamp ────────────
            for track in tracks:
                track_last_seen[track.track_id] = timestamp  # video time

            # ── Per-track classification ──────────────────────────────
            activities: dict = {}

            for track in tracks:
                if not buf.is_ready(track.track_id):
                    activities[track.track_id] = None
                    continue

                features  = extractor.extract(buf.get(track.track_id))
                violation = classifier.classify(features) if features else None
                activities[track.track_id] = violation

                if violation is not None:
                    last_ts = last_alert_time.get(track.track_id, 0.0)
                    if (timestamp - last_ts) >= config.VIOLATION_COOLDOWN:
                        last_alert_time[track.track_id] = timestamp
                        alert_logger.submit(
                            track_id=track.track_id,
                            violation=violation,
                            timestamp=time.time(),
                            raw_frame=frame,          # raw un-annotated frame
                            all_tracks=tracks,
                            session_start=session_start,
                        )

            # Run object detection synchronously on this frame for zero-delay weapon detection
            # (runs every OBJECT_DETECTION_INTERVAL frames, default=1 for every frame)
            theft_track_ids: set = set()
            all_obj_detections = []
            bag_detections = []
            object_timestamp = timestamp
            object_frame = original_frame

            if frame_number % config.OBJECT_DETECTION_INTERVAL == 0:
                all_obj_detections = obj_detector.detect(original_frame)

                latest_weapon_detects = [
                    d for d in all_obj_detections if d.category == "weapon"
                ]
                bag_detections = [
                    d for d in all_obj_detections if d.category == "bag"
                ]
                for wd in latest_weapon_detects:
                    last_weapon_ts = weapon_alert_time.get(wd.label, -float("inf"))
                    if (object_timestamp - last_weapon_ts) >= config.WEAPON_ALERT_COOLDOWN:
                        weapon_alert_time[wd.label] = object_timestamp
                        alert_logger.submit(
                            track_id=-1,
                            violation=f"Weapon Detected: {wd.label}",
                            timestamp=time.time(),
                            raw_frame=object_frame,
                            all_tracks=tracks,
                            session_start=session_start,
                        )
                    logger.warning(
                        "WEAPON DETECTED: %s (conf=%.2f)", wd.label, wd.confidence
                    )

                person_bboxes = [t.bbox for t in tracks]
                for prev_bag in prev_bag_detections:
                    if any(compute_iou(prev_bag.bbox, item.bbox) >= 0.35
                           for item in bag_detections):
                        continue
                    pbcx = (prev_bag.bbox[0] + prev_bag.bbox[2]) // 2
                    pbcy = (prev_bag.bbox[1] + prev_bag.bbox[3]) // 2
                    nearest_track = min(
                        tracks,
                        key=lambda t: ((t.cx - pbcx) ** 2 + (t.cy - pbcy) ** 2) ** 0.5,
                        default=None,
                    )
                    if nearest_track is None:
                        continue
                    nearest_dist = ((nearest_track.cx - pbcx) ** 2
                                    + (nearest_track.cy - pbcy) ** 2) ** 0.5
                    tid = nearest_track.track_id
                    last_theft_ts = theft_cooldown.get(tid, 0.0)
                    if (nearest_dist <= config.THEFT_PROXIMITY_PX
                            and object_timestamp - last_theft_ts >= config.THEFT_COOLDOWN):
                        theft_cooldown[tid] = object_timestamp
                        theft_track_ids.add(tid)
                        alert_logger.submit(
                            track_id=tid,
                            violation="Possible Theft",
                            timestamp=time.time(),
                            raw_frame=object_frame,
                            all_tracks=tracks,
                            session_start=session_start,
                        )
                        logger.warning(
                            "POSSIBLE THEFT — track_id=%d (dist=%.1fpx).",
                            tid, nearest_dist,
                        )

                newly_abandoned = bag_tracker.update(
                    bag_detections=bag_detections,
                    person_bboxes=person_bboxes,
                    timestamp=object_timestamp,
                )
                current_abandoned = [
                    entry for entry in bag_tracker._bag_tracker.values()
                    if entry.is_abandoned
                ]
                for abandoned in newly_abandoned:
                    alert_logger.submit(
                        track_id=abandoned.bag_id,
                        violation="Abandoned Bag",
                        timestamp=time.time(),
                        raw_frame=object_frame,
                        all_tracks=tracks,
                        session_start=session_start,
                    )
                prev_bag_detections = bag_detections

            weapon_detects = latest_weapon_detects

            # FIX 8: annotate on a copy of the original full-resolution frame
            annotated_frame = original_frame.copy()
            annotated_frame = _annotate_frame(
                frame=annotated_frame,
                tracks=tracks,
                activities=activities,
                fps=current_fps,
                person_count=len(tracks),
                weapon_detects=weapon_detects,
                abandoned_bags=current_abandoned,
                theft_track_ids=theft_track_ids,
            )

            # Fix 2: write the annotated frame into the module-level shared holder.
            # The global declaration above ensures this updates the right variable.
            with frame_lock:
                latest_frame = annotated_frame.copy()

            # Fix 2: print one-time confirmation when first frame is committed
            if not _first_frame_written:
                _first_frame_written = True
                print("[INFO] latest_frame updated for the first time — stream is live")
                logger.info("[Thread 2] First annotated frame written to latest_frame.")

            # Enhancement 2 / FIX 8: explicit frame reference cleanup
            annotated_frame = None
            del annotated_frame
            original_frame  = None
            del original_frame
            frame           = None
            del frame

            # ── Wall-clock FPS for the display-stream counter ─────────
            fps_counter += 1
            now_perf     = time.perf_counter()
            elapsed_fps  = now_perf - fps_ts
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_ts      = now_perf

            # Fix 5: processing-speed stats (wall-clock FPS vs video FPS).
            _proc_fps_counter += 1
            _proc_elapsed = time.perf_counter() - _proc_fps_ts
            if _proc_elapsed >= 1.0:
                proc_fps = _proc_fps_counter / _proc_elapsed
                _proc_fps_counter = 0
                _proc_fps_ts      = time.perf_counter()
                with _speed_stats_lock:
                    vid_fps = _speed_stats["video_fps"]
                    _speed_stats["processing_fps"] = round(proc_fps, 1)
                    _speed_stats["speed_ratio"]    = round(
                        proc_fps / vid_fps if vid_fps > 0 else 0.0, 2
                    )

            if frame_number % 30 == 0:
                with _speed_stats_lock:
                    ratio = _speed_stats["speed_ratio"]
                    vfps  = _speed_stats["video_fps"]
                print(
                    f"[PERF] frame={frame_number:6d} | "
                    f"proc_FPS={current_fps:.1f} | video_FPS={vfps:.1f} | "
                    f"ratio={ratio:.2f}x | "
                    f"device={_infer_device} | tracks={len(tracks)}"
                )

            # ── Update shared status ──────────────────────────────────
            with _status_lock:
                _status["active_tracks"] = len(tracks)
                _status["fps"]           = round(current_fps, 1)

            # ── Update people + memory stats ──────────────────────────
            current_mb = _check_memory()   # also triggers gc if over threshold
            with _stats_lock:
                _unique_ids.update(active_ids)
                _stats["current_count"] = len(tracks)
                _stats["total_unique"]  = len(_unique_ids)
                if len(tracks) > _stats["peak_count"]:
                    _stats["peak_count"] = len(tracks)
                _stats["memory_mb"]       = round(current_mb, 1)
                if current_mb > _stats["peak_memory_mb"]:
                    _stats["peak_memory_mb"] = round(current_mb, 1)
                _stats["bags_tracked"]    = bag_tracker.bags_tracked
                _stats["abandoned_bags"]  = bag_tracker.abandoned_count

    except Exception as exc:
        logger.error("[Thread 2] Unexpected error: %s", exc, exc_info=True)
    finally:
        logger.info("[Thread 2] Processing thread stopped.")
        with _status_lock:
            _status["pipeline_running"] = False
            _status["fps"]              = 0.0
            _status["active_tracks"]    = 0


# ===========================================================================
# MJPEG stream generator  (Fix 1, Fix 2, Fix 3, Fix 4)
# ===========================================================================

def _mjpeg_generator():
    """
    Yields MJPEG multipart frames from the latest_frame holder.

    Fix 2: reads the module-level latest_frame via explicit global declaration.
    Fix 3: robust exception handling so a single bad frame never kills the stream.
    Fix 4: paces output to source FPS so the browser receives smooth video
           at the same rate frames are captured, not as fast as possible.
    """
    global latest_frame
    while True:
        try:
            loop_start = time.time()

            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            if frame is None:
                time.sleep(0.02)
                continue

            ret, buffer = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60]
            )
            if not ret:
                time.sleep(0.02)
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                buffer.tobytes() +
                b'\r\n'
            )

            # Fix 4: pace the stream to match source FPS
            with source_fps_lock:
                fps = source_fps
            frame_interval = 1.0 / max(fps, 1.0)
            elapsed    = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as exc:
            print(f"[STREAM ERROR] {exc}")
            time.sleep(0.05)
            continue


def _publish_crowd_frames() -> None:
    """Publish a labeled side-by-side view of the latest entry/exit feeds."""
    with _crowd_frames_lock:
        entry_frame = _crowd_frames.get("entry")
        exit_frame = _crowd_frames.get("exit")
        entry_copy = entry_frame.copy() if entry_frame is not None else None
        exit_copy = exit_frame.copy() if exit_frame is not None else None

    with _status_lock:
        people_counts = dict(_crowd_people_counts)
        feeds_ready = all(
            _crowd_camera_status[role] == "online"
            for role in ("entry", "exit")
        )

    panel_height = 480
    panels = []
    labels = {
        "entry": ("ENTRY VIDEO" if _crowd_source_type == "videos" else "ENTRY CAMERA"),
        "exit": ("EXIT VIDEO" if _crowd_source_type == "videos" else "EXIT CAMERA"),
    }
    for role, frame in (("entry", entry_copy), ("exit", exit_copy)):
        label = labels[role]
        if frame is None:
            panel = np.zeros((panel_height, 640, 3), dtype=np.uint8)
            cv2.putText(
                panel, f"{label} OFFLINE", (24, panel_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 130, 160), 2, cv2.LINE_AA,
            )
        else:
            scale = panel_height / frame.shape[0]
            panel = cv2.resize(
                frame,
                (max(1, int(frame.shape[1] * scale)), panel_height),
                interpolation=cv2.INTER_AREA,
            )
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (12, 24, 40), cv2.FILLED)
            cv2.putText(
                panel, f"{label}  PEOPLE {people_counts[role]}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 220, 255), 2, cv2.LINE_AA,
            )
        panels.append(panel)

    combined = np.concatenate(panels, axis=1)
    count_difference = people_counts["entry"] - people_counts["exit"]
    count_alert = feeds_ready and abs(count_difference) >= (
        _crowd_monitor.alert_threshold if _crowd_monitor is not None else 1
    )
    if _crowd_monitor is not None:
        state = _crowd_monitor.snapshot()
        flow_alert = state["alert_active"]
        alert_text = (
            f"CROWD COUNT MISMATCH  ENTRY {people_counts['entry']}  "
            f"EXIT {people_counts['exit']}  DIFF {count_difference:+d}"
            if count_alert else
            "WAITING FOR BOTH FEEDS"
            if not feeds_ready else
            f"FLOW IMBALANCE  IN {state['entry_count']}  OUT {state['exit_count']}  "
            f"DIFF {state['difference']:+d}"
            if flow_alert else
            f"FLOW BALANCED  IN {state['entry_count']}  OUT {state['exit_count']}"
        )
        alert_active = count_alert or (feeds_ready and flow_alert)
        color = (0, 0, 255) if alert_active else (0, 190, 0)
        cv2.rectangle(combined, (0, panel_height - 42), (combined.shape[1], panel_height), (12, 24, 40), cv2.FILLED)
        cv2.putText(
            combined, alert_text, (14, panel_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )


    with frame_lock:
        global latest_frame
        latest_frame = combined


def _crowd_camera_worker(
    role: str,
    source: Union[int, str],
    direction: str,
    stop_event: threading.Event,
    alert_logger: AlertLogger,
) -> None:
    """Read one camera/video, count people and crossings, and capture alerts."""
    global _crowd_last_alert
    source_label = (
        f"{role} video '{os.path.basename(source)}'"
        if isinstance(source, str)
        else f"{role} camera index {source}"
    )
    cap = cv2.VideoCapture(source)
    if not isinstance(source, str):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logger.error("[Crowd] Cannot open %s.", source_label)
        with _status_lock:
            _crowd_camera_status[role] = "error"
        stop_event.set()
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) if isinstance(source, str) else 0.0
    if source_fps <= 0 or source_fps > 120:
        source_fps = 25.0
    frame_interval = 1.0 / source_fps
    try:
        tracker = PersonTracker(
            model_path=config.YOLO_MODEL_PATH,
            confidence=config.DETECTION_CONFIDENCE,
            imgsz=config.INFERENCE_IMGSZ,
            tracker_type=config.TRACKER_TYPE,
        )
        logger.info(
            "[Crowd] %s device=%s direction=%s",
            source_label, tracker.device, direction,
        )
        last_tracks = []
        frame_number = 0
        fps_started = time.perf_counter()
        fps_count = 0
        camera_started = time.monotonic()
        while not stop_event.is_set():
            loop_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, str):
                    logger.info("[Crowd] Reached end of %s.", source_label)
                    with _status_lock:
                        _crowd_camera_status[role] = "ended"
                else:
                    logger.error("[Crowd] Lost feed from %s.", source_label)
                    with _status_lock:
                        _crowd_camera_status[role] = "error"
                stop_event.set()
                break

            frame_number += 1
            now = time.monotonic()
            if frame_number % config.TRACKING_INTERVAL == 0:
                last_tracks = tracker.update(frame, now - camera_started)
                _crowd_monitor.update_tracks(
                    role, last_tracks, frame.shape[1], direction, timestamp=now
                )
                with _status_lock:
                    _crowd_camera_status[role] = "online"
            current_count = len(last_tracks)
            with _status_lock:
                _crowd_people_counts[role] = current_count

            fps_count += 1
            elapsed = time.perf_counter() - fps_started
            fps = fps_count / elapsed if elapsed >= 1.0 else 0.0
            if elapsed >= 1.0:
                fps_started = time.perf_counter()
                fps_count = 0

            annotated = _annotate_frame(
                frame=frame,
                tracks=last_tracks,
                activities={},
                fps=fps,
                person_count=current_count,
                weapon_detects=[],
                abandoned_bags=[],
                theft_track_ids=set(),
            )
            line_x = annotated.shape[1] // 2
            cv2.line(annotated, (line_x, 0), (line_x, annotated.shape[0]), (255, 190, 0), 2)
            with _crowd_frames_lock:
                _crowd_frames[role] = annotated
            _publish_crowd_frames()

            state = _crowd_monitor.snapshot(timestamp=now)
            with _status_lock:
                people_counts = dict(_crowd_people_counts)
                feeds_ready = all(
                    _crowd_camera_status[camera_role] == "online"
                    for camera_role in ("entry", "exit")
                )
            count_difference = people_counts["entry"] - people_counts["exit"]
            count_alert = (
                feeds_ready
                and abs(count_difference) >= state["alert_threshold"]
            )
            if feeds_ready and (count_alert or state["alert_active"]):
                with _crowd_alert_lock:
                    if now - _crowd_last_alert >= config.CROWD_ALERT_COOLDOWN:
                        _crowd_last_alert = now
                        with frame_lock:
                            evidence = latest_frame.copy() if latest_frame is not None else annotated.copy()
                        if count_alert:
                            violation = (
                                "Crowd Count Mismatch "
                                f"(entry={people_counts['entry']}, exit={people_counts['exit']}, "
                                f"difference={count_difference:+d})"
                            )
                        else:
                            violation = (
                                "Crowd Flow Imbalance "
                                f"(in={state['entry_count']}, out={state['exit_count']}, "
                                f"difference={state['difference']:+d})"
                            )
                        alert_logger.submit(
                            track_id=-1,
                            violation=violation,
                            timestamp=time.time(),
                            raw_frame=evidence,
                            all_tracks=[],
                            session_start=_stats.get("session_start"),
                        )
                        logger.critical("[Crowd] %s", violation)

            if isinstance(source, str):
                sleep_time = frame_interval - (time.perf_counter() - loop_started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except Exception:
        logger.exception("[Crowd] %s worker failed.", source_label)
        with _status_lock:
            _crowd_camera_status[role] = "error"
        stop_event.set()
    finally:
        cap.release()
        with _status_lock:
            if _crowd_camera_status[role] == "online":
                _crowd_camera_status[role] = "offline"


# ===========================================================================
# Pipeline lifecycle helpers
# ===========================================================================

def _launch_crowd_pipeline(
    entry_source: Union[int, str],
    exit_source: Union[int, str],
    source_type: str,
    window_seconds: int,
    alert_threshold: int,
    entry_direction: str,
    exit_direction: str,
) -> None:
    """Start independent entry/exit camera workers and shared alert logging."""
    global _alert_logger, _stop_event, _log_stop_event, _log_worker_thread
    global _crowd_threads, _crowd_monitor
    global _crowd_mode, _crowd_frames, _crowd_last_alert, _crowd_directions
    global _crowd_source_type, _preproc_alert_logger
    global latest_frame

    _crowd_mode = True
    _stop_event = threading.Event()
    _log_stop_event = threading.Event()
    _preproc_alert_logger = None
    _crowd_source_type = source_type
    _crowd_monitor = CrowdFlowMonitor(
        window_seconds=window_seconds,
        alert_threshold=alert_threshold,
    )
    _crowd_last_alert = 0.0
    _crowd_directions = {"entry": entry_direction, "exit": exit_direction}
    with _crowd_frames_lock:
        _crowd_frames = {}
    with frame_lock:
        latest_frame = None
    with _status_lock:
        _crowd_camera_status.update({"entry": "starting", "exit": "starting"})
        _crowd_people_counts.update({"entry": 0, "exit": 0})
        _status.update({
            "pipeline_running": True,
            "source": f"crowd {source_type}: entry, exit",
            "fps": 0.0,
            "active_tracks": 0,
            "mode": "crowd",
            "crowd_source_type": source_type,
        })
    with _stats_lock:
        _stats["session_start"] = time.time()
        _stats["current_count"] = 0
        _stats["peak_count"] = 0
        _stats["total_unique"] = 0

    _alert_logger = AlertLogger(
        snapshot_dir=config.SNAPSHOT_DIR,
        log_dir=config.LOG_DIR,
        log_filename=config.LOG_FILENAME,
        event_log_maxlen=config.EVENT_LOG_MAXLEN,
        log_queue_maxsize=config.LOG_QUEUE_MAXSIZE,
        snapshot_quality=config.SNAPSHOT_JPEG_QUALITY,
    )
    _log_worker_thread = threading.Thread(
        target=_alert_logger.run_worker,
        args=(_log_stop_event,),
        name="CrowdLogWorker",
        daemon=True,
    )
    _log_worker_thread.start()
    _crowd_threads = [
        threading.Thread(
            target=_crowd_camera_worker,
            args=("entry", entry_source, entry_direction, _stop_event, _alert_logger),
            name="CrowdEntrySource",
            daemon=True,
        ),
        threading.Thread(
            target=_crowd_camera_worker,
            args=("exit", exit_source, exit_direction, _stop_event, _alert_logger),
            name="CrowdExitSource",
            daemon=True,
        ),
    ]
    for thread in _crowd_threads:
        thread.start()


def _launch_pipeline(source) -> None:
    """Start all three background threads for a new pipeline session."""
    global _capture_thread, _process_thread, _log_worker_thread
    global _stop_event, _log_stop_event, _alert_logger, _cleanup_timer
    global _is_recorded_video
    global _crowd_mode
    global _preproc_alert_logger
    _crowd_mode = False
    _preproc_alert_logger = None

    # Fix 5: detect whether source is a file path (recorded) or webcam index
    _is_recorded_video = isinstance(source, str) and len(source) > 0

    _stop_event = threading.Event()
    _log_stop_event = threading.Event()

    _alert_logger = AlertLogger(
        snapshot_dir=config.SNAPSHOT_DIR,
        log_dir=config.LOG_DIR,
        log_filename=config.LOG_FILENAME,
        event_log_maxlen=config.EVENT_LOG_MAXLEN,
        log_queue_maxsize=config.LOG_QUEUE_MAXSIZE,
        snapshot_quality=config.SNAPSHOT_JPEG_QUALITY,
    )

    # Fix 1 & 2: reset the shared frame holders on each new session
    global latest_raw_frame, latest_frame
    with raw_frame_lock:
        latest_raw_frame = None
    with frame_lock:
        latest_frame = None

    with _status_lock:
        _status["pipeline_running"] = True
        _status["source"]           = str(source)
        _status["mode"]             = "surveillance"
        _status["fps"]              = 0.0
        _status["active_tracks"]    = 0

    sess_start = time.time()
    with _stats_lock:
        _unique_ids.clear()
        _stats["current_count"]   = 0
        _stats["peak_count"]      = 0
        _stats["total_unique"]    = 0
        _stats["session_start"]   = sess_start
        _stats["memory_mb"]       = 0.0
        _stats["peak_memory_mb"]  = 0.0
        _stats["bags_tracked"]    = 0
        _stats["abandoned_bags"]  = 0

    # Thread 3 — log/snapshot worker (starts first)
    _log_worker_thread = threading.Thread(
        target=_alert_logger.run_worker,
        args=(_log_stop_event,),
        name="LogWorker",
        daemon=True,
    )
    _log_worker_thread.start()

    # Thread 1 — capture
    _capture_thread = threading.Thread(
        target=_capture_thread_fn,
        args=(source, _stop_event),
        name="Capture",
        daemon=True,
    )
    _capture_thread.start()

    # Thread 2 — processing
    _process_thread = threading.Thread(
        target=_process_thread_fn,
        args=(_stop_event, _alert_logger),
        name="Processing",
        daemon=True,
    )
    _process_thread.start()

    logger.info("Pipeline launched — source=%s, tracker=%s", source, config.TRACKER_TYPE)


def _shutdown_pipeline() -> None:
    """Signal all threads to stop and wait briefly for clean exit."""
    global _capture_thread, _process_thread, _log_worker_thread, _cleanup_timer
    global _crowd_threads, _crowd_mode, _crowd_monitor, _crowd_frames
    global _log_stop_event

    _stop_event.set()

    # Cancel the stale cleanup timer if it is still pending
    if _cleanup_timer is not None and _cleanup_timer.is_alive():
        _cleanup_timer.cancel()

    producer_threads = [
        (_capture_thread,    "Capture"),
        (_process_thread,    "Processing"),
    ] + [(thread, thread.name) for thread in _crowd_threads]
    for thread, name in producer_threads:
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("%s thread did not stop within 5 s.", name)

    if _alert_logger is not None and not any(
        thread is not None and thread.is_alive()
        for thread, _ in producer_threads
    ):
        _alert_logger.log_queue.join()
    elif _alert_logger is not None:
        logger.error("Evidence queue may still have pending jobs because a producer did not stop.")
    _log_stop_event.set()
    if _log_worker_thread is not None and _log_worker_thread.is_alive():
        _log_worker_thread.join(timeout=5.0)
        if _log_worker_thread.is_alive():
            logger.warning("LogWorker thread did not stop within 5 s.")

    # Fix 1 & 2: clear shared frame holders on shutdown
    global latest_raw_frame, latest_frame
    with raw_frame_lock:
        latest_raw_frame = None
    with frame_lock:
        latest_frame = None
    with _crowd_frames_lock:
        _crowd_frames = {}
    _crowd_threads = []
    _crowd_mode = False
    if _crowd_monitor is not None:
        _crowd_monitor.reset()
    _crowd_monitor = None

    with _status_lock:
        _status["pipeline_running"] = False
        _status["fps"]              = 0.0
        _status["active_tracks"]    = 0
        _status["mode"]             = "offline"
        _crowd_camera_status.update({"entry": "offline", "exit": "offline"})
        _crowd_people_counts.update({"entry": 0, "exit": 0})

    with _stats_lock:
        _unique_ids.clear()
        _stats["current_count"]   = 0
        _stats["peak_count"]      = 0
        _stats["total_unique"]    = 0
        _stats["session_start"]   = None
        _stats["memory_mb"]       = 0.0
        _stats["peak_memory_mb"]  = 0.0
        _stats["bags_tracked"]    = 0
        _stats["abandoned_bags"]  = 0

    gc.collect()
    logger.info("Pipeline shut down.")


def _drain_queue(q: queue.Queue) -> None:
    """Remove all items from a queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


# ===========================================================================
# Pre-processing worker  (recorded video mode)
# ===========================================================================

def process_and_save(video_path: str) -> None:
    """
    Offline pre-processing pass for a recorded video file.

    Reads every frame, runs the full detection / tracking / behaviour /
    classification pipeline (fresh instances so as not to conflict with a
    live pipeline session), annotates each frame, and writes the result to
    an MP4 in the 'outputs/' directory.

    Progress is reported via the module-level is_processing /
    processing_progress / processed_video_path globals.
    """
    global is_processing, processing_progress, processed_video_path

    # Fix 3 — capture session start time for AlertLogger snapshot footer.
    session_start_time = time.time()

    logger.info("[Preproc] Starting offline pass for: %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("[Preproc] Cannot open '%s'.", video_path)
        with _preproc_lock:
            is_processing = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    vid_fps      = cap.get(cv2.CAP_PROP_FPS)
    if vid_fps <= 0 or vid_fps > 120:
        vid_fps = 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output file
    out_dir  = os.path.join(_ROOT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_processed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, vid_fps, (width, height))

    # Fresh pipeline instances — isolated from any live Thread 2 session.
    _INFER_SIZE = config.INFERENCE_IMGSZ
    obj_model_path = (
        config.WEAPON_MODEL_PATH if config.WEAPON_MODEL_PATH
        else config.YOLO_MODEL_PATH
    )
    pp_tracker    = PersonTracker(
        model_path=config.YOLO_MODEL_PATH,
        confidence=config.DETECTION_CONFIDENCE,
        imgsz=_INFER_SIZE,
        tracker_type=config.TRACKER_TYPE,
    )
    pp_buf        = BehaviourBuffer(
        window_sec=config.OBSERVATION_WINDOW_SEC,
        min_samples=config.MIN_WINDOW_SAMPLES,
    )
    pp_extractor  = FeatureExtractor(
        min_samples=config.MIN_WINDOW_SAMPLES,
        min_speed_threshold=config.MIN_SPEED_THRESHOLD,
    )
    pp_classifier = ViolationClassifier(
        running_speed_threshold=config.RUNNING_SPEED_THRESHOLD,
        motion_variance_threshold=config.MOTION_VARIANCE_THRESHOLD,
        loitering_stillness=config.LOITERING_STILLNESS,
        loitering_displacement=config.LOITERING_DISPLACEMENT,
        pace_ratio_threshold=config.PACE_RATIO_THRESHOLD,
        min_speed_threshold=config.MIN_SPEED_THRESHOLD,
        unsafe_motion_variance_threshold=config.UNSAFE_MOTION_VARIANCE_THRESHOLD,
        unsafe_min_speed=config.UNSAFE_MIN_SPEED,
    )
    pp_obj_detector = ObjectDetector(
        model_path=obj_model_path,
        conf_threshold=config.OBJECT_CONF_THRESHOLD,
        target_classes=(
            None if config.WEAPON_MODEL_PATH else config.OBJECT_TARGET_CLASSES
        ),
    )
    pp_bag_tracker = BagTracker(
        abandoned_seconds=config.ABANDONED_BAG_SECONDS,
        disappear_timeout=config.BAG_DISAPPEAR_TIMEOUT,
        alert_cooldown=config.ABANDONED_BAG_COOLDOWN,
    )

    # Fresh AlertLogger for offline processing — isolated from any live session.
    pp_alert_logger = AlertLogger(
        snapshot_dir=config.SNAPSHOT_DIR,
        log_dir=config.LOG_DIR,
        log_filename=config.LOG_FILENAME,
        event_log_maxlen=config.EVENT_LOG_MAXLEN,
        log_queue_maxsize=config.LOG_QUEUE_MAXSIZE,
        snapshot_quality=config.SNAPSHOT_JPEG_QUALITY,
    )
    # Fix 2 — expose this instance at module level so /events and /snapshots
    # can read from it immediately once the queue has been drained.
    global _preproc_alert_logger
    _preproc_alert_logger = pp_alert_logger
    pp_log_stop   = threading.Event()
    pp_log_thread = threading.Thread(
        target=pp_alert_logger.run_worker,
        args=(pp_log_stop,),
        name="PreprocLogWorker",
        daemon=True,
    )
    pp_log_thread.start()
    pp_last_alert_time: dict = {}   # track_id -> last violation video_ts
    pp_weapon_alert_time: dict = {}
    pp_weapon_detects: list = []
    pp_abandoned_bags: list = []

    COLOR_MAP = {
        "Running / Sudden Motion": config.COLOR_RUNNING,
        "Loitering":               config.COLOR_LOITERING,
        "Suspicious Lingering":    config.COLOR_LINGERING,
        None:                      config.COLOR_NORMAL,
    }

    frame_index  = 0
    frame_number = 0
    last_tracks  = []

    # FPS tracking for GPU impact visibility
    fps_timer       = time.time()
    fps_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            video_ts  = frame_index / vid_fps
            frame_index  += 1
            frame_number += 1

            # Run tracker at the same configurable cadence as live mode.
            if frame_number % config.TRACKING_INTERVAL == 0:
                last_tracks = pp_tracker.update(frame, video_ts)
            tracks = last_tracks

            active_ids = {t.track_id for t in tracks}

            for track in tracks:
                pp_buf.update(
                    track_id=track.track_id,
                    cx=track.cx,
                    cy=track.cy,
                    timestamp=video_ts,
                )
            pp_buf.remove_stale(active_ids)

            activities: dict = {}
            for track in tracks:
                if not pp_buf.is_ready(track.track_id):
                    activities[track.track_id] = None
                    continue
                features  = pp_extractor.extract(pp_buf.get(track.track_id))
                violation = pp_classifier.classify(features) if features else None
                activities[track.track_id] = violation

            # Fix 2 — submit violations to AlertLogger BEFORE drawing annotations
            # so AlertLogger receives a clean un-annotated frame to render its
            # own evidence snapshot internally (Thread 3).
            for track in tracks:
                violation = activities.get(track.track_id)
                if violation and violation != 'Normal':
                    last_ts = pp_last_alert_time.get(track.track_id, -config.VIOLATION_COOLDOWN)
                    if (video_ts - last_ts) >= config.VIOLATION_COOLDOWN:
                        pp_last_alert_time[track.track_id] = video_ts
                        print(f"[VIOLATION] Track {track.track_id} | "
                              f"{violation} | Timestamp: {video_ts:.2f}s")
                        # Fix 2 — pass raw un-annotated frame copy and Unix timestamp.
                        # Fix 6 — tracks already have x1/y1/x2/y2 set by the
                        # scaling loop above, so no wrapper is needed.
                        pp_alert_logger.submit(
                            track_id=track.track_id,
                            violation=violation,
                            timestamp=time.time(),          # Unix time — not video_ts
                            raw_frame=frame.copy(),         # clean frame — no boxes drawn yet
                            all_tracks=tracks,
                            session_start=session_start_time,  # Fix 3
                        )

            # Recorded-video mode uses the same object detector and event
            # logger as live mode; no firearm model means COCO knives/scissors only.
            if frame_number % config.OBJECT_DETECTION_INTERVAL == 0:
                pp_detections = pp_obj_detector.detect(frame)
                pp_weapon_detects = [
                    item for item in pp_detections if item.category == "weapon"
                ]
                pp_bags = [item for item in pp_detections if item.category == "bag"]
                for detected in pp_weapon_detects:
                    last_ts = pp_weapon_alert_time.get(detected.label, -float("inf"))
                    if video_ts - last_ts >= config.WEAPON_ALERT_COOLDOWN:
                        pp_weapon_alert_time[detected.label] = video_ts
                        pp_alert_logger.submit(
                            track_id=-1,
                            violation=f"Weapon Detected: {detected.label}",
                            timestamp=time.time(),
                            raw_frame=frame.copy(),
                            all_tracks=tracks,
                            session_start=session_start_time,
                        )
                        logger.warning(
                            "[Preproc] WEAPON DETECTED: %s (conf=%.2f)",
                            detected.label, detected.confidence,
                        )
                newly_abandoned = pp_bag_tracker.update(
                    bag_detections=pp_bags,
                    person_bboxes=[track.bbox for track in tracks],
                    timestamp=video_ts,
                )
                pp_abandoned_bags = [
                    entry for entry in pp_bag_tracker._bag_tracker.values()
                    if entry.is_abandoned
                ]
                for abandoned in newly_abandoned:
                    pp_alert_logger.submit(
                        track_id=abandoned.bag_id,
                        violation="Abandoned Bag",
                        timestamp=time.time(),
                        raw_frame=frame.copy(),
                        all_tracks=tracks,
                        session_start=session_start_time,
                    )

            # Annotate frame AFTER submit so AlertLogger gets the raw frame.
            annotated = _annotate_frame(
                frame=frame.copy(),
                tracks=tracks,
                activities=activities,
                fps=vid_fps,
                person_count=len(tracks),
                weapon_detects=pp_weapon_detects,
                abandoned_bags=pp_abandoned_bags,
                theft_track_ids=set(),
            )
            writer.write(annotated)

            # FPS tracking — print every 50 frames so GPU impact is visible.
            fps_frame_count += 1
            if fps_frame_count % 50 == 0:
                elapsed_fps = time.time() - fps_timer
                current_fps = 50 / elapsed_fps if elapsed_fps > 0 else 0.0
                with _preproc_lock:
                    _prog = processing_progress
                print(
                    f"[INFO] Processing speed: {current_fps:.1f} FPS | "
                    f"Progress: {_prog}%"
                )
                fps_timer = time.time()

            # Update progress (clamped to 99 until the file is finalised).
            with _preproc_lock:
                processing_progress = min(
                    99, int((frame_index / total_frames) * 100)
                )

        # Finalise video file first.
        cap.release()
        writer.release()

        # Fix 1 + Fix 4 — block until every queued job has been fully processed
        # (snapshot written to disk, CSV row appended) before telling the
        # frontend that processing is complete.  log_queue.join() returns only
        # when every put_nowait() call has been matched by a task_done() call
        # inside run_worker's finally block.
        remaining = pp_alert_logger.log_queue.qsize()
        print(f"[INFO] Waiting for log worker to finish "
              f"({remaining} jobs remaining)...")
        logger.info("[Preproc] Draining log worker queue (%d jobs)...", remaining)

        pp_alert_logger.log_queue.join()   # Fix 1: blocks until all task_done() calls

        print(f"[INFO] Log worker finished — "
              f"total events: {pp_alert_logger.total_events}")
        logger.info("[Preproc] Log worker drained — total events=%d.",
                    pp_alert_logger.total_events)

        # Now stop the per-session worker thread (it is isolated to this offline
        # pass and is not the live-pipeline Thread 3).
        pp_log_stop.set()
        pp_log_thread.join(timeout=10)
        logger.info("[Preproc] Log worker thread stopped.")

        # Fix 3 — verify snapshots actually landed on disk.
        import glob as _glob
        saved_snaps = _glob.glob(
            os.path.join(config.SNAPSHOT_DIR, '*.jpg'))
        print(f"[INFO] Snapshots saved to disk: {len(saved_snaps)}")
        if len(saved_snaps) == 0 and pp_alert_logger.total_events > 0:
            print("[WARNING] Events logged but no snapshots found "
                  "— check SNAPSHOT_DIR path in config.py")
            print(f"[WARNING] SNAPSHOT_DIR = {config.SNAPSHOT_DIR}")
            print(f"[WARNING] Absolute path = "
                  f"{os.path.abspath(config.SNAPSHOT_DIR)}")

        # Fix 4 — only mark done AFTER all snapshots and CSV entries are saved.
        with _preproc_lock:
            processed_video_path = out_path
            processing_progress  = 100
            is_processing        = False

        logger.info("[Preproc] Saved annotated video to: %s", out_path)
        print(f"[INFO] Processing fully complete including all snapshots: {out_path}")

    except Exception as exc:
        logger.error("[Preproc] Error during offline processing: %s", exc, exc_info=True)
        cap.release()
        try:
            writer.release()
        except Exception:
            pass
        # Drain whatever is left in the queue then stop the worker cleanly.
        try:
            pp_alert_logger.log_queue.join()
        except Exception:
            pass
        pp_log_stop.set()
        pp_log_thread.join(timeout=5)
        with _preproc_lock:
            is_processing = False


# ===========================================================================
# Flask Routes
# ===========================================================================

@app.route("/")
def index():
    """Serve the single-page surveillance dashboard."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint."""
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/start", methods=["POST"])
def start():
    """
    Start the surveillance pipeline.

    Expected JSON body:
        {"source": "camera", "index": 0}
        {"source": "video",  "path": "/path/to/file.mp4"}

    For recorded video the system pre-processes the full file first, saves an
    annotated MP4, then streams it.  The response is {"status": "processing"}
    and the client should poll /progress until done == true, then point the
    stream at /processed_feed.

    For live camera the worker pipeline is launched immediately
    and the response is {"status": "started"}.
    """
    global is_processing, processing_progress, processed_video_path

    body = request.get_json(silent=True) or {}
    mode = body.get("source", "camera")

    if mode == "crowd":
        source_type = body.get("crowd_source_type", "cameras")
        try:
            window_seconds = int(body.get(
                "window_seconds", config.CROWD_DEFAULT_WINDOW_SECONDS
            ))
            alert_threshold = int(body.get(
                "alert_threshold", config.CROWD_DEFAULT_MISMATCH_THRESHOLD
            ))
        except (TypeError, ValueError):
            return jsonify({"error": "The time window and mismatch threshold must be integers."}), 400
        entry_direction = body.get("entry_direction", "left_to_right")
        exit_direction = body.get("exit_direction", "left_to_right")
        if source_type == "cameras":
            try:
                entry_source = int(body.get("entry_index", 0))
                exit_source = int(body.get("exit_index", 1))
            except (TypeError, ValueError):
                return jsonify({"error": "Camera indexes must be integers."}), 400
            if entry_source < 0 or exit_source < 0 or entry_source == exit_source:
                return jsonify({"error": "Choose two different, non-negative camera indexes."}), 400
        elif source_type == "videos":
            entry_source = body.get("entry_path", "")
            exit_source = body.get("exit_path", "")
            if not isinstance(entry_source, str) or not isinstance(exit_source, str):
                return jsonify({"error": "Upload both entry and exit videos first."}), 400
            entry_source = os.path.realpath(entry_source)
            exit_source = os.path.realpath(exit_source)
            upload_dir = os.path.realpath(os.path.join(config.BASE_DIR, "data", "input"))
            if entry_source == exit_source:
                return jsonify({"error": "Choose two different video files."}), 400
            for role, video_path in (("entry", entry_source), ("exit", exit_source)):
                try:
                    inside_upload_dir = (
                        os.path.commonpath((upload_dir, video_path)) == upload_dir
                    )
                except ValueError:
                    inside_upload_dir = False
                if not inside_upload_dir or not os.path.isfile(video_path):
                    return jsonify({"error": f"Upload a valid {role} video before starting."}), 400
                video_cap = cv2.VideoCapture(video_path)
                opened = video_cap.isOpened()
                video_cap.release()
                if not opened:
                    return jsonify({"error": f"OpenCV cannot open the {role} video."}), 400
        else:
            return jsonify({"error": "Crowd source must be cameras or videos."}), 400
        if not 10 <= window_seconds <= 3600:
            return jsonify({"error": "Comparison window must be between 10 and 3600 seconds."}), 400
        if not 1 <= alert_threshold <= 1000:
            return jsonify({"error": "Mismatch threshold must be between 1 and 1000 people."}), 400
        if (entry_direction not in {"left_to_right", "right_to_left"}
                or exit_direction not in {"left_to_right", "right_to_left"}):
            return jsonify({"error": "Crossing direction must be left_to_right or right_to_left."}), 400
        with _status_lock:
            if _status["pipeline_running"]:
                return jsonify({"error": "Pipeline already running."}), 400
        _launch_crowd_pipeline(
            entry_source, exit_source, source_type, window_seconds,
            alert_threshold, entry_direction, exit_direction,
        )
        return jsonify({
            "status": "started",
            "mode": "crowd",
            "crowd_source_type": source_type,
            "window_seconds": window_seconds,
            "alert_threshold": alert_threshold,
        }), 200

    elif mode == "video":
        # ── Recorded-video path: offline pre-processing ───────────────────
        source = body.get("path", "").strip()
        if not source:
            return jsonify({"error": "No video path provided."}), 400
        if not os.path.exists(source):
            return jsonify({"error": f"Video file not found: {source}"}), 400
        test_cap = cv2.VideoCapture(source)
        if not test_cap.isOpened():
            return jsonify({"error": f"OpenCV cannot open this file: {source}"}), 400
        test_cap.release()

        # Guard against double-start.
        with _preproc_lock:
            if is_processing:
                return jsonify({"error": "Pre-processing already running."}), 400
            is_processing        = True
            processing_progress  = 0
            processed_video_path = None

        # Fix 5 — clear the previous session's logger reference so the
        # frontend never sees stale events while the new pass is running.
        global _preproc_alert_logger
        _preproc_alert_logger = None

        threading.Thread(
            target=process_and_save,
            args=(source,),
            daemon=True,
        ).start()

        return jsonify({"status": "processing"}), 200

    elif mode == "camera":
        # ── Live-camera path: capture/processing/object/log worker pipeline ─
        with _status_lock:
            if _status["pipeline_running"]:
                return jsonify({"error": "Pipeline already running."}), 400
        try:
            source = int(body.get("index", 0))
        except (TypeError, ValueError):
            source = 0
        _launch_pipeline(source)
        return jsonify({"status": "started", "source": str(source)}), 200

    else:
        return jsonify({"error": f"Unknown source mode: {mode!r}"}), 400


@app.route("/progress")
def progress():
    """Return pre-processing progress for recorded-video mode."""
    with _preproc_lock:
        return jsonify({
            "is_processing": is_processing,
            "progress":      processing_progress,
            "done":          (processed_video_path is not None) and (not is_processing),
        })


def _stream_processed():
    """
    MJPEG generator that streams the pre-processed annotated MP4 at its
    native FPS, looping continuously.
    """
    with _preproc_lock:
        path = processed_video_path
    if not path or not os.path.isfile(path):
        return

    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = 1.0 / fps

    try:
        while True:
            t   = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            ret2, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ret2:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )

            elapsed = time.time() - t
            sleep_t = delay - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        cap.release()


@app.route("/processed_feed")
def processed_feed():
    """MJPEG stream of the pre-processed annotated video file."""
    return Response(
        _stream_processed(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stop", methods=["POST"])
def stop():
    """Stop the pipeline and release all resources."""
    with _status_lock:
        if not _status["pipeline_running"]:
            return jsonify({"status": "not_running"}), 200

    _shutdown_pipeline()
    return jsonify({"status": "stopped"}), 200


@app.route("/events")
def events():
    """Return the latest violation events as JSON (newest first).

    In recorded-video mode serves from _preproc_alert_logger.
    In live-camera mode serves from _alert_logger.
    """
    # Fix 3 — prefer the preproc logger when set (recorded-video mode).
    logger_to_use = (
        _preproc_alert_logger
        if _preproc_alert_logger is not None
        else _alert_logger
    )
    if logger_to_use is None:
        return jsonify([])
    return jsonify(logger_to_use.get_recent_events(n=config.DASHBOARD_MAX_EVENTS))


@app.route("/status")
def status():
    """Return current pipeline status as JSON."""
    with _status_lock:
        snap = dict(_status)
        snap["crowd_cameras"] = dict(_crowd_camera_status)
    custom_weapon_model = bool(
        config.WEAPON_MODEL_PATH and os.path.isfile(config.WEAPON_MODEL_PATH)
    )
    snap["firearm_detection_enabled"] = custom_weapon_model
    snap["firearm_detection_status"] = (
        "Custom weapon model configured"
        if custom_weapon_model
        else "No firearm model; default COCO model cannot detect guns"
    )
    snap["inference_device"] = config.DEVICE
    if torch.cuda.is_available():
        snap["inference_device_name"] = torch.cuda.get_device_name(0)
    snap["server_time"] = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
    return jsonify(snap)


@app.route("/stats")
def stats():
    """
    People-counter + memory + bag + processing-speed statistics.

    Returns:
        current_count, peak_count, total_unique, session_duration,
        memory_mb, peak_memory_mb, bags_tracked, abandoned_bags_count,
        processing_fps, video_fps, speed_ratio.
    """
    with _stats_lock:
        snap       = dict(_stats)
        sess_start = snap.get("session_start")

    if sess_start:
        elapsed = int(time.time() - sess_start)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        snap["session_duration"] = f"{h:02d}:{m:02d}:{s:02d}"
    else:
        snap["session_duration"] = "00:00:00"

    snap["abandoned_bags_count"] = snap.pop("abandoned_bags", 0)
    snap.pop("session_start", None)

    # Fix 5: attach processing-speed indicators.
    with _speed_stats_lock:
        snap["processing_fps"] = _speed_stats["processing_fps"]
        snap["video_fps"]      = _speed_stats["video_fps"]
        snap["speed_ratio"]    = _speed_stats["speed_ratio"]
    with _status_lock:
        snap["mode"] = _status.get("mode", "offline")

    crowd_monitor = _crowd_monitor
    snap["crowd_flow"] = (
        crowd_monitor.snapshot() if crowd_monitor is not None else {
            "entry_count": 0,
            "exit_count": 0,
            "difference": 0,
            "absolute_difference": 0,
            "window_seconds": config.CROWD_DEFAULT_WINDOW_SECONDS,
            "alert_threshold": config.CROWD_DEFAULT_MISMATCH_THRESHOLD,
            "alert_active": False,
        }
    )
    with _status_lock:
        snap["crowd_cameras"] = dict(_crowd_camera_status)
        snap["crowd_people"] = dict(_crowd_people_counts)
        snap["crowd_feeds_ready"] = all(
            snap["crowd_cameras"][role] == "online"
            for role in ("entry", "exit")
        )
    crowd_threshold = crowd_monitor.alert_threshold if crowd_monitor is not None else (
        config.CROWD_DEFAULT_MISMATCH_THRESHOLD
    )
    people_difference = (
        snap["crowd_people"]["entry"] - snap["crowd_people"]["exit"]
    )
    snap["crowd_people_difference"] = people_difference
    snap["crowd_people_alert_active"] = (
        snap["crowd_feeds_ready"]
        and abs(people_difference) >= crowd_threshold
    )

    return jsonify(snap)


@app.route("/upload", methods=["POST"])
def upload():
    """Receive an uploaded video file and save it to data/input/."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    from werkzeug.utils import secure_filename
    filename = secure_filename(f.filename)
    extension = os.path.splitext(filename)[1].lower()
    if not filename or extension not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        return jsonify({"error": "Choose an MP4, AVI, MOV, MKV, WEBM, or M4V video."}), 400
    upload_dir = os.path.join(config.BASE_DIR, "data", "input")
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:10]}_{filename}"
    save_path = os.path.join(upload_dir, filename)
    f.save(save_path)
    logger.info("[Upload] Saved uploaded file to: %s", save_path)
    return jsonify({"path": save_path, "filename": filename})


@app.route("/snapshots")
def snapshots():
    """Return metadata for all captured snapshots (gallery endpoint).

    In recorded-video mode serves from _preproc_alert_logger.
    In live-camera mode serves from _alert_logger.
    """
    # Fix 4 — prefer the preproc logger when set (recorded-video mode).
    logger_to_use = (
        _preproc_alert_logger
        if _preproc_alert_logger is not None
        else _alert_logger
    )
    if logger_to_use is None:
        return jsonify([])

    all_events = logger_to_use.get_recent_events(n=config.EVENT_LOG_MAXLEN)

    gallery = []
    for evt in all_events:
        snap_path = evt.get("snapshot_path", "")
        if not snap_path or not os.path.isfile(snap_path):
            continue

        filename = os.path.basename(snap_path)
        gallery.append({
            "event_id":       evt.get("event_id"),
            "track_id":       evt.get("track_id"),
            "violation_type": evt.get("violation_type"),
            "timestamp":      evt.get("timestamp"),
            "url":            f"/snapshot-image/{filename}",
            "filename":       filename,
        })

    return jsonify(gallery)


@app.route("/snapshot-image/<path:filename>")
def snapshot_image(filename: str):
    """Serve a snapshot JPEG from the configured snapshot directory."""
    return send_from_directory(config.SNAPSHOT_DIR, filename)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    logger.info("Starting Flask dashboard on http://localhost:%d", config.FLASK_PORT)
    # FIX 6: debug=False and use_reloader=False — both cause performance issues
    # when enabled (reloader spawns extra processes; debug disables optimisations).
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
