# =============================================================================
# main.py
# Real-Time AI Surveillance System — standalone CLI pipeline runner.
#
# Usage:
#   python main.py                        # uses VIDEO_SOURCE from config
#   python main.py --source 0             # webcam index
#   python main.py --source path/to.mp4  # video file
#   python main.py --no-display           # headless (no cv2 window)
#   python main.py --save                 # write annotated output video
#   python main.py --dashboard            # also start Flask dashboard
#
# Press  Q  (in the OpenCV window) or  Ctrl+C  to stop.
# =============================================================================

import argparse
import logging
import os
import sys
import threading
import time
from typing import Dict, Optional, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path regardless of the working directory.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.settings import (
    VIDEO_SOURCE,
    TARGET_FPS,
    OUTPUT_VIDEO_DIR,
    OUTPUT_VIDEO_CODEC,
    STREAM_JPEG_QUALITY,
)
from modules.video_input      import VideoInput
from modules.detector         import PersonDetector
from modules.tracker          import PersonTracker
from modules.behaviour_buffer import BehaviourBuffer
from modules.feature_extractor import FeatureExtractor
from modules.classifier       import ViolationClassifier
from modules.alert_logger     import AlertLogger
from modules.annotator        import FrameAnnotator

# ---------------------------------------------------------------------------
# Logging — INFO by default; set LOG_LEVEL env var for DEBUG.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Pipeline class — wraps all modules and runs the main loop.
# ===========================================================================

class SurveillancePipeline:
    """
    Orchestrates the complete frame-by-frame surveillance pipeline.

    The pipeline runs in a single loop:
      VideoInput → Detector → Tracker → BehaviourBuffer → FeatureExtractor
        → ActivityClassifier → AlertLogger → FrameAnnotator → display/stream

    Parameters
    ----------
    source : Optional[Union[str, int]]
        Video source — camera index or file path.  None → uses config default.
    show_window : bool
        Whether to display annotated frames in a named OpenCV window.
    save_video : bool
        Whether to write annotated frames to an output .mp4 file.
    """

    def __init__(
        self,
        source=None,
        show_window: bool = True,
        save_video: bool = False,
    ):
        self._source      = source
        self._show_window = show_window
        self._save_video  = save_video
        self._running     = False

        # Shared annotated-frame buffer for the optional Flask stream.
        # Protected by a lock so the Flask thread can read safely.
        self._frame_lock = threading.Lock()
        self._latest_jpg: Optional[bytes] = None
        self._status: dict = {
            "active_tracks":    0,
            "fps":              0.0,
            "total_alerts":    0,
            "uptime_seconds":  0,
            "pipeline_running": False,
        }

        # Initialise all pipeline modules.
        logger.info("Initialising surveillance pipeline modules…")
        self.video     = VideoInput(source=self._source, target_fps=TARGET_FPS)
        self.tracker   = PersonTracker(
            model_path=YOLO_MODEL_PATH,
            confidence=DETECTION_CONFIDENCE,
            imgsz=640,
            tracker_type=TRACKER_TYPE,
        )
        self.buf       = BehaviourBuffer()
        self.extractor = FeatureExtractor()
        self.clf       = ViolationClassifier()
        self.alerter   = AlertLogger()
        self.annotator = FrameAnnotator()

        # OpenCV video writer (initialised lazily on first frame).
        self._writer: Optional[cv2.VideoWriter] = None

        logger.info("All modules initialised.")

    # =========================================================================
    # Public interface
    # =========================================================================

    def run(self) -> None:
        """
        Start the main pipeline loop.

        Blocks until the video source is exhausted or the user presses Q /
        sends KeyboardInterrupt.  Calls :meth:`stop` automatically on exit.
        """
        if not self.video.is_open():
            logger.error("Video source could not be opened.  Aborting.")
            return

        props = self.video.get_properties()
        logger.info(
            "Source: %s  |  %dx%d @ %.1f FPS  |  %d frames total",
            self._source or VIDEO_SOURCE,
            props.get("width", 0), props.get("height", 0),
            props.get("fps", 0.0),  props.get("frame_count", -1),
        )

        self._running = True
        self._status["pipeline_running"] = True
        start_time  = time.time()
        fps_counter = 0
        fps_ts      = time.time()
        current_fps = 0.0

        logger.info("Pipeline running — press Q (window) or Ctrl+C to stop.")

        try:
            for frame, timestamp in self.video.read_frames():
                if not self._running:
                    break

                # ── 1. Detection + Tracking (combined in PersonTracker) ──────
                tracks = self.tracker.update(frame=frame, timestamp=timestamp)
                active_ids = {t.track_id for t in tracks}

                # ── 3. Behaviour buffer update + stale cleanup ─────────────
                for track in tracks:
                    self.buf.update(
                        track_id=track.track_id,
                        cx=track.cx,
                        cy=track.cy,
                        timestamp=timestamp,
                    )
                self.buf.remove_stale(active_ids, current_time=timestamp)

                # ── 4. Feature extraction + classification + alerting ──────
                track_activities: Dict[int, str] = {}
                for track in tracks:
                    entries  = self.buf.get(track.track_id)
                    features = self.extractor.extract(entries)

                    if features is None:
                        track_activities[track.track_id] = "Unknown"
                        continue

                    activity = self.clf.classify(features)
                    track_activities[track.track_id] = activity

                    # Alert (no-op if cooldown, or if activity is Normal).
                    self.alerter.process(
                        track_id=track.track_id,
                        activity=activity,
                        frame=frame.copy(),   # snapshot copy before annotation
                        timestamp=timestamp,
                    )

                # ── 5. Frame annotation ────────────────────────────────────
                annotated = self.annotator.annotate(
                    frame=frame,
                    tracks=tracks,
                    track_activities=track_activities,
                    fps=current_fps,
                    alert_count=self.alerter.total_alerts,
                )

                # ── 6. FPS calculation ─────────────────────────────────────
                fps_counter += 1
                now     = time.time()
                elapsed = now - fps_ts
                if elapsed >= 1.0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_ts      = now

                # ── 7. Display window ──────────────────────────────────────
                if self._show_window:
                    cv2.imshow("AI Surveillance System — Press Q to quit", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q") or key == 27:
                        logger.info("Q pressed — shutting down.")
                        break

                # ── 8. Video writer ────────────────────────────────────────
                if self._save_video:
                    self._write_frame(annotated, frame.shape)

                # ── 9. Shared JPEG buffer for Flask stream ─────────────────
                ok, jpg = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY],
                )
                if ok:
                    with self._frame_lock:
                        self._latest_jpg = jpg.tobytes()

                # ── 10. Update shared status dict ──────────────────────────
                self._status.update({
                    "active_tracks":   len(tracks),
                    "fps":             round(current_fps, 1),
                    "total_alerts":   self.alerter.total_alerts,
                    "uptime_seconds": int(time.time() - start_time),
                })

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Release all resources cleanly."""
        self._running = False
        self._status["pipeline_running"] = False
        self.video.release()
        if self._writer is not None:
            self._writer.release()
            logger.info("Output video file closed.")
        if self._show_window:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")

    def get_latest_frame_jpg(self) -> Optional[bytes]:
        """Return the most recent JPEG-encoded annotated frame (thread-safe)."""
        with self._frame_lock:
            return self._latest_jpg

    def get_status(self) -> dict:
        """Return a copy of the current status dict (thread-safe)."""
        return dict(self._status)

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _write_frame(self, frame: np.ndarray, shape: tuple) -> None:
        """Lazily initialise the VideoWriter and write one frame."""
        h, w = shape[:2]
        if self._writer is None:
            os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
            filename  = f"output_{int(time.time())}.mp4"
            out_path  = os.path.join(OUTPUT_VIDEO_DIR, filename)
            fourcc    = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
            fps_out   = TARGET_FPS if TARGET_FPS and TARGET_FPS > 0 else 15
            self._writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w, h))
            logger.info("Recording output video → '%s'.", out_path)

        self._writer.write(frame)


# ===========================================================================
# Flask dashboard integration (optional)
# ===========================================================================

def _start_flask(pipeline: SurveillancePipeline) -> None:
    """
    Monkey-patch the dashboard app to pull frames / status from this pipeline
    instance instead of running its own pipeline thread, then start Flask.
    """
    try:
        import dashboard.app as dash_app
        import numpy as _np

        # Override the shared-state references so Flask routes use this pipeline.
        def _patched_gen():
            _BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while True:
                jpg = pipeline.get_latest_frame_jpg()
                if jpg:
                    yield _BOUNDARY + jpg + b"\r\n"
                time.sleep(1 / (TARGET_FPS or 15))

        # Replace the generator in the Flask route.
        dash_app._mjpeg_generator = _patched_gen

        # Override status endpoint source.
        @dash_app.app.route("/status", endpoint="status_patched")
        def _patched_status():
            from flask import jsonify
            from datetime import datetime, timezone
            stat = pipeline.get_status()
            stat["server_time"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return jsonify(stat)

        from config.settings import FLASK_HOST, FLASK_PORT
        logger.info(
            "Starting Flask dashboard on http://localhost:%d", FLASK_PORT
        )
        dash_app.app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except Exception as exc:
        logger.error("Failed to start Flask dashboard: %s", exc, exc_info=True)


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-Time AI Surveillance System\n"
            "Detects, tracks, and classifies multi-person behaviour in video."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", default=None,
        help=(
            "Video source: integer camera index (e.g. 0) or path to a video file. "
            "Defaults to VIDEO_SOURCE in config/settings.py."
        ),
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless — do not open an OpenCV display window.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Write the annotated output to data/output/ as an MP4 file.",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Also start the Flask web dashboard in a background thread.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse source — convert digit strings to integers for camera indices.
    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    pipeline = SurveillancePipeline(
        source=source,
        show_window=not args.no_display,
        save_video=args.save,
    )

    # Optionally start the Flask dashboard in a separate daemon thread.
    if args.dashboard:
        flask_thread = threading.Thread(
            target=_start_flask,
            args=(pipeline,),
            name="FlaskDashboard",
            daemon=True,
        )
        flask_thread.start()

    # Run the pipeline (blocks until exit).
    pipeline.run()


if __name__ == "__main__":
    main()
