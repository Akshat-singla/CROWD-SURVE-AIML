# =============================================================================
# extract_ucf_features.py
# Offline feature extractor for labelled video folders (e.g. UCF-Crime).
#
# Usage:
#   python extract_ucf_features.py --folder "D:/UCF-Crime/Normal"   --label "Normal"
#   python extract_ucf_features.py --folder "D:/UCF-Crime/Loitering" --label "Loitering"
#   python extract_ucf_features.py --folder "D:/UCF-Crime/Running"   --label "Running"
#   python extract_ucf_features.py --folder "D:/UCF-Crime/Lingering" --label "Suspicious Lingering"
#   python extract_ucf_features.py --folder "D:/UCF-Crime/Unsafe" --label "Unsafe Activity"
#
# GPU acceleration:
#   The script auto-selects CUDA if available, or pass --device cuda to force it.
#   python extract_ucf_features.py --folder ... --label ... --device cuda
#   python extract_ucf_features.py --folder ... --label ... --device cpu
#
# Pipeline (reuses existing modules — no logic is duplicated):
#   PersonDetector (modules/detector.py)       — YOLO person detection
#   PersonTracker  (modules/tracker.py)        — ByteTrack multi-person tracking
#   BehaviourBuffer (modules/behaviour_buffer.py) — per-track sliding window
#   FeatureExtractor (modules/feature_extractor.py) — compute motion features
#
# For every tracked person whose 10-second observation window fills up,
# the script extracts 7 motion features + the given label and appends one
# row to data/training_data.csv.  The buffer for that track is then cleared
# so the next extraction starts a fresh 10-second window.
#
# CSV columns (matching train_classifier.py FEATURE_COLS + LABEL_COL):
#   avg_speed, max_speed, total_displacement, total_distance,
#   stillness_ratio, pace_ratio, motion_variance, label
# =============================================================================

import argparse
import csv
import logging
import os
import sys
import torch

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so module imports work from any CWD.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Import existing modules — no logic is duplicated here.
# ---------------------------------------------------------------------------
from modules.detector import PersonDetector
from modules.tracker import PersonTracker
from modules.behaviour_buffer import BehaviourBuffer
from modules.feature_extractor import FeatureExtractor

# ---------------------------------------------------------------------------
# Config constants (pulled from the same dashboard/config.py the live system uses)
# ---------------------------------------------------------------------------
from dashboard.config import (
    YOLO_MODEL_PATH,
    DETECTION_CONFIDENCE,
    INFERENCE_IMGSZ,
    TRACKER_TYPE,
    OBSERVATION_WINDOW_SEC,
    MIN_WINDOW_SAMPLES,
    MIN_SPEED_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_LABELS = {
    "Normal", "Loitering", "Running", "Suspicious Lingering", "Unsafe Activity"
}
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}

CSV_PATH = os.path.join(_SCRIPT_DIR, "data", "training_data.csv")

# Column order must match train_classifier.py FEATURE_COLS + LABEL_COL.
CSV_COLUMNS = [
    "avg_speed",
    "max_speed",
    "total_displacement",
    "total_distance",
    "stillness_ratio",
    "pace_ratio",
    "motion_variance",
    "label",
]

# Print a progress message every this many feature vectors extracted.
PROGRESS_INTERVAL = 100


# ===========================================================================
# Helper — motion variance
# ===========================================================================

def _compute_motion_variance(entries) -> float:
    """
    Variance of frame-to-frame centroid step distances (pixels).

    High variance  → erratic / sudden speed changes (running, fighting).
    Low variance   → uniform motion or sustained stillness.
    """
    if len(entries) < 2:
        return 0.0
    xs = np.array([e.cx for e in entries], dtype=np.float64)
    ys = np.array([e.cy for e in entries], dtype=np.float64)
    step_distances = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    return float(np.var(step_distances))


# ===========================================================================
# Helper — check 10-second window is fully spanned
# ===========================================================================

def _window_is_full(entries, window_sec: float) -> bool:
    """
    Return True when the oldest and newest buffer entries span ≥ window_sec.
    This confirms we have a complete observation window, not just the first
    few seconds of a new track.
    """
    if len(entries) < 2:
        return False
    return (entries[-1].timestamp - entries[0].timestamp) >= window_sec


# ===========================================================================
# Helper — discover video files recursively
# ===========================================================================

def _find_videos(folder: str) -> list[str]:
    """
    Recursively walk *folder* and return sorted absolute paths for every
    .mp4, .avi, and .mov file found.
    """
    video_paths = []
    for dirpath, _dirnames, filenames in os.walk(folder):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                video_paths.append(os.path.join(dirpath, fname))
    return sorted(video_paths)


# ===========================================================================
# Per-video processing
# ===========================================================================

def process_video(
    video_path: str,
    label: str,
    detector: PersonDetector,
    tracker: PersonTracker,
    extractor: FeatureExtractor,
    csv_writer,
    run_total: list,         # mutable single-element list so we can update it
) -> int:
    """
    Run the YOLO + ByteTrack pipeline on *video_path*.

    Every 2nd frame is skipped for speed.  When a track's BehaviourBuffer
    spans a full 10-second window the feature vector is extracted, written
    to the CSV, and the buffer is cleared so that track starts a fresh window.

    Returns the number of feature rows extracted from this video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open '%s' — skipping.", os.path.basename(video_path))
            return 0
    except Exception as exc:
        logger.warning("Error opening '%s': %s — skipping.", os.path.basename(video_path), exc)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0   # fall back to 25 fps if unreadable
    frame_idx = 0
    rows_extracted = 0

    # Fresh BehaviourBuffer per video — tracks from different clips never bleed.
    buffer = BehaviourBuffer(
        window_sec=OBSERVATION_WINDOW_SEC,
        min_samples=MIN_WINDOW_SAMPLES,
    )

    logger.info("  Processing '%s' @ %.1f fps…", os.path.basename(video_path), fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video or read error

        # ── Skip every 2nd frame for speed ──────────────────────────────
        if frame_idx % 2 != 0:
            frame_idx += 1
            continue

        # Derive video-time from the frame index so the 10-second window
        # reflects actual footage duration, not wall-clock processing time.
        video_time = frame_idx / fps

        # ── YOLOv8 detection ────────────────────────────────────────────
        # The detector returns List[Detection]; we convert to the format
        # that the tracker expects via tracker.update().
        try:
            tracks = tracker.update(frame, timestamp=video_time)
        except Exception as exc:
            logger.debug("Tracker error on frame %d: %s", frame_idx, exc)
            frame_idx += 1
            continue

        active_ids = {t.track_id for t in tracks}

        # ── Update BehaviourBuffer ───────────────────────────────────────
        for track in tracks:
            buffer.update(track.track_id, track.cx, track.cy, video_time)

        # ── Feature extraction ───────────────────────────────────────────
        for track in tracks:
            tid = track.track_id
            entries = buffer.get(tid)

            # Only extract when the buffer holds a complete 10-second window
            # AND has enough samples for reliable statistics.
            if not _window_is_full(entries, OBSERVATION_WINDOW_SEC):
                continue
            if not buffer.is_ready(tid):
                continue

            features = extractor.extract(entries)
            if features is None:
                continue

            motion_variance = _compute_motion_variance(entries)

            csv_writer.writerow({
                "avg_speed":          round(features["avg_speed"],          4),
                "max_speed":          round(features["max_speed"],          4),
                "total_displacement": round(features["total_displacement"], 4),
                "total_distance":     round(features["total_distance"],     4),
                "stillness_ratio":    round(features["stillness_ratio"],    4),
                "pace_ratio":         round(features["pace_ratio"],         4),
                "motion_variance":    round(motion_variance,                4),
                "label":              label,
            })

            rows_extracted += 1
            run_total[0]   += 1

            # ── Progress reporting every 100 feature vectors ─────────────
            if run_total[0] % PROGRESS_INTERVAL == 0:
                print(
                    f"  [Progress] {run_total[0]} feature vectors extracted so far "
                    f"(current video: {os.path.basename(video_path)})"
                )

            # ── Clear this track's buffer — start a fresh 10-second window ──
            # Removes all entries so the next window is completely new data,
            # avoiding highly correlated duplicate rows from the same clip.
            buffer.remove(tid)

        # ── Prune buffers for tracks no longer visible ───────────────────
        buffer.remove_stale(active_ids)

        frame_idx += 1

    cap.release()
    return rows_extracted


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract motion feature vectors from a labelled folder of videos "
            "and append them to data/training_data.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported labels:
  {', '.join(sorted(SUPPORTED_LABELS))}

Examples:
  python extract_ucf_features.py --folder "D:/UCF-Crime/Normal"    --label "Normal"
  python extract_ucf_features.py --folder "D:/UCF-Crime/Loitering" --label "Loitering"
  python extract_ucf_features.py --folder "D:/UCF-Crime/Running"   --label "Running"
  python extract_ucf_features.py --folder "D:/UCF-Crime/Lingering" --label "Suspicious Lingering"
  python extract_ucf_features.py --folder "D:/UCF-Crime/Unsafe"    --label "Unsafe Activity"
        """,
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the folder of videos (.mp4, .avi, .mov) to process (searched recursively).",
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=sorted(SUPPORTED_LABELS),
        help=(
            "Behaviour label to assign to every extracted feature vector. "
            f"Supported values: {', '.join(sorted(SUPPORTED_LABELS))}"
        ),
    )
    parser.add_argument(
        "--output",
        default=CSV_PATH,
        help=f"Path to the output CSV file (default: {CSV_PATH}).",
    )
    parser.add_argument(
        "--model",
        default=YOLO_MODEL_PATH,
        help=f"Path to YOLOv8 weights (default: {YOLO_MODEL_PATH}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DETECTION_CONFIDENCE,
        help=f"YOLO detection confidence threshold (default: {DETECTION_CONFIDENCE}).",
    )
    parser.add_argument(
        "--tracker",
        default=TRACKER_TYPE,
        choices=["bytetrack", "deepsort"],
        help=f"Tracker backend (default: {TRACKER_TYPE}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu", "mps"],
        help=(
            "Compute device for YOLO inference. "
            "Defaults to auto-detect (CUDA → MPS → CPU). "
            "Pass 'cuda' to force GPU, 'cpu' to force CPU."
        ),
    )
    args = parser.parse_args()

    # ── Validate folder ───────────────────────────────────────────────────────
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        logger.error("Folder not found: '%s'", folder)
        sys.exit(1)

    # ── Discover video files recursively ─────────────────────────────────────
    video_files = _find_videos(folder)
    if not video_files:
        logger.warning(
            "No .mp4 / .avi / .mov files found in '%s' (searched recursively).", folder
        )
        sys.exit(0)

    logger.info(
        "Found %d video file(s) in '%s'  |  label='%s'",
        len(video_files), folder, args.label,
    )

    # ── Ensure output directory exists ────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # ── Resolve compute device ────────────────────────────────────────────────
    if args.device is not None:
        # User explicitly requested a device — validate it.
        requested = args.device
        if requested == "cuda":
            if not torch.cuda.is_available():
                logger.error(
                    "--device cuda was requested but CUDA is not available. "
                    "Make sure you have a CUDA-enabled GPU and the correct "
                    "PyTorch build (pip install torch --index-url "
                    "https://download.pytorch.org/whl/cu121)."
                )
                sys.exit(1)
            device_label = f"cuda ({torch.cuda.get_device_name(0)})"
        elif requested == "mps":
            if not torch.backends.mps.is_available():
                logger.error(
                    "--device mps was requested but Apple MPS is not available "
                    "on this system."
                )
                sys.exit(1)
            device_label = "mps (Apple)"
        else:
            device_label = "cpu"
        device = requested
    else:
        # Auto-detect: CUDA → MPS → CPU.
        if torch.cuda.is_available():
            device = "cuda"
            device_label = f"cuda ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            device = "mps"
            device_label = "mps (Apple)"
        else:
            device = "cpu"
            device_label = "cpu"

    print(f"\n  🖥  Compute device : {device_label}")
    if device == "cpu":
        print(
            "  ⚠  Running on CPU — inference will be slow.\n"
            "     For GPU acceleration install CUDA PyTorch:\n"
            "     pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        )

    # ── Initialise pipeline objects (created once, reused across all videos) ──
    logger.info("Loading YOLOv8 model for detection…")
    detector = PersonDetector(
        model_path=args.model,
        confidence=args.confidence,
        imgsz=INFERENCE_IMGSZ,
        device=device,
    )

    logger.info("Loading PersonTracker (%s) on %s…", args.tracker, device_label)
    tracker = PersonTracker(
        model_path=args.model,
        confidence=args.confidence,
        imgsz=INFERENCE_IMGSZ,
        tracker_type=args.tracker,
        device=device,
    )

    extractor = FeatureExtractor(
        min_samples=MIN_WINDOW_SAMPLES,
        min_speed_threshold=MIN_SPEED_THRESHOLD,
    )

    # ── Determine initial row count in CSV (for the final summary) ────────────
    csv_exists = os.path.isfile(args.output) and os.path.getsize(args.output) > 0
    initial_row_count = 0
    if csv_exists:
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                # Subtract 1 for the header row.
                initial_row_count = max(0, sum(1 for _ in f) - 1)
        except Exception:
            initial_row_count = 0

    # ── Open (or create) CSV in append mode ───────────────────────────────────
    videos_done    = 0
    videos_skipped = 0
    run_total      = [0]   # mutable container so process_video can update it

    with open(args.output, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)

        # Write the header only when creating a new / empty file.
        if not csv_exists:
            writer.writeheader()
            logger.info("Created new CSV at '%s'.", args.output)
        else:
            logger.info(
                "Appending to existing CSV at '%s' (%d existing rows).",
                args.output, initial_row_count,
            )

        # ── Process each video ────────────────────────────────────────────
        for idx, video_path in enumerate(video_files, start=1):
            print(
                f"\n[{idx}/{len(video_files)}] {os.path.relpath(video_path, folder)}"
                f"  (run total so far: {run_total[0]})"
            )

            try:
                # Reset ByteTrack / DeepSORT state between videos so IDs
                # from one clip cannot carry over into the next.
                tracker.reset()

                rows = process_video(
                    video_path=video_path,
                    label=args.label,
                    detector=detector,
                    tracker=tracker,
                    extractor=extractor,
                    csv_writer=writer,
                    run_total=run_total,
                )
                videos_done += 1
                logger.info(
                    "  ✓ %s — %d feature vector(s) extracted.",
                    os.path.basename(video_path), rows,
                )

            except Exception as exc:
                # Corrupted or unreadable files are skipped gracefully.
                logger.warning(
                    "  ✗ Skipping '%s' — %s",
                    os.path.basename(video_path), exc,
                )
                videos_skipped += 1
                continue

    # ── Final summary ─────────────────────────────────────────────────────────
    total_rows_now = initial_row_count + run_total[0]
    print("\n" + "=" * 62)
    print(f"  Label               : {args.label}")
    print(f"  Videos processed    : {videos_done}")
    print(f"  Videos skipped      : {videos_skipped}")
    print(f"  Feature vectors added this run : {run_total[0]}")
    print(f"  Total rows now in CSV          : {total_rows_now}")
    print(f"  Output CSV          : {os.path.abspath(args.output)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
