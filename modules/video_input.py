# =============================================================================
# modules/video_input.py
# VideoInput — OpenCV-based video source abstraction.
#
# Responsibilities:
#   • Accept a webcam index OR a video file path as input source.
#   • Open and validate the source via cv2.VideoCapture.
#   • Yield frames one at a time together with a wall-clock timestamp.
#   • Honour an optional TARGET_FPS cap to control processing rate.
#   • Handle all error conditions gracefully (source missing, dropped frames,
#     end-of-file) without crashing the pipeline.
#   • Expose a clean release() method to free hardware/file resources.
# =============================================================================

import time
import logging
from typing import Optional

import cv2

from config.settings import TARGET_FPS

# Module-level logger — the root logging config is set up in main.py.
logger = logging.getLogger(__name__)


class VideoInput:
    """
    A lightweight wrapper around cv2.VideoCapture that provides a simple
    frame-by-frame iteration interface with timestamps and optional FPS capping.

    Usage example
    -------------
    >>> video = VideoInput(source="data/input/cctv.mp4")
    >>> for frame, timestamp in video.read_frames():
    ...     process(frame, timestamp)
    >>> video.release()

    Or as a context manager:
    >>> with VideoInput(source=0) as video:
    ...     for frame, ts in video.read_frames():
    ...         process(frame, ts)
    """

    def __init__(self, source=None, target_fps: float = None):
        """
        Initialise the video input.

        Parameters
        ----------
        source : int | str | None
            • int  → webcam index (0 = default camera, 1 = second camera, …)
            • str  → absolute or relative path to a video file.
            • None → falls back to the value of TARGET_FPS in config/settings.py.
        target_fps : float | None
            Maximum frames per second to yield.  Frames arriving faster than
            this rate are skipped.  Pass None (default) to use the value from
            config/settings.py; pass 0 or a negative number to disable capping.
        """
        # ── Resolve source ────────────────────────────────────────────────────
        if source is None:
            from config.settings import VIDEO_SOURCE
            source = VIDEO_SOURCE
        self.source = source

        # ── Resolve target FPS ────────────────────────────────────────────────
        self.target_fps: float = target_fps if target_fps is not None else (TARGET_FPS or 0)
        # Minimum interval (seconds) between two yielded frames; 0 → no cap.
        self._min_frame_interval: float = (1.0 / self.target_fps) if self.target_fps > 0 else 0.0

        # ── Internal state ────────────────────────────────────────────────────
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open: bool = False
        self._last_yield_time: float = 0.0   # wall-clock time of last yielded frame
        self._dropped_frames: int = 0        # counter for monitoring

        # Open the source immediately so callers know early if it is invalid.
        self._open()

    # =========================================================================
    # Public interface
    # =========================================================================

    def read_frames(self):
        """
        Generator that yields (frame, timestamp) tuples indefinitely until the
        source is exhausted or an unrecoverable error occurs.

        Yields
        ------
        frame : np.ndarray
            BGR image array of shape (H, W, 3).
        timestamp : float
            Unix wall-clock time (seconds) at the moment the frame was grabbed.
        """
        if not self._is_open:
            logger.error("Cannot read frames — video source is not open.")
            return

        logger.info(
            "Starting frame capture from '%s' (target FPS: %s).",
            self.source,
            self.target_fps if self.target_fps > 0 else "unlimited",
        )

        while True:
            # ── Guard: capture object must be valid ───────────────────────────
            if self._cap is None:
                logger.error("VideoCapture is None — stopping frame capture.")
                break

            # ── Grab the next frame ───────────────────────────────────────────
            ret, frame = self._cap.read()

            if not ret:
                # ret == False has two causes:
                #   1. End of a video file → normal termination.
                #   2. Failed hardware read / dropped frame → transient fault.
                if self._is_file_source():
                    logger.info("End of video file reached: '%s'.", self.source)
                else:
                    self._dropped_frames += 1
                    logger.warning(
                        "Failed to grab frame from source '%s' "
                        "(total dropped: %d). Retrying…",
                        self.source,
                        self._dropped_frames,
                    )
                    # Brief pause prevents a hot busy-loop on persistent failure.
                    time.sleep(0.05)
                    continue
                break   # Exit on end-of-file

            # ── Record wall-clock timestamp ───────────────────────────────────
            current_time = time.time()

            # ── FPS throttling ────────────────────────────────────────────────
            if self._min_frame_interval > 0:
                elapsed = current_time - self._last_yield_time
                if elapsed < self._min_frame_interval:
                    # Skip this frame — we're ahead of the target FPS.
                    continue

            self._last_yield_time = current_time
            yield frame, current_time

    def get_properties(self) -> dict:
        """
        Return a dictionary of key video source properties.

        Returns
        -------
        dict with keys: width, height, fps, frame_count, backend
        """
        if not self._is_open or self._cap is None:
            return {}

        return {
            "width":       int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":      int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":         self._cap.get(cv2.CAP_PROP_FPS),
            # frame_count is -1 / 0 for live cameras.
            "frame_count": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "backend":     self._cap.getBackendName(),
        }

    def is_open(self) -> bool:
        """Return True if the underlying VideoCapture is currently open."""
        return self._is_open and self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """
        Release the underlying cv2.VideoCapture resource.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("Video source '%s' released.", self.source)
        self._is_open = False
        self._cap = None

    # =========================================================================
    # Context-manager support  (with VideoInput(...) as v:)
    # =========================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        # Do not suppress exceptions — let them propagate normally.
        return False

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _open(self) -> None:
        """Open the video source and validate it.  Called once in __init__."""
        logger.info("Opening video source: '%s'.", self.source)

        try:
            self._cap = cv2.VideoCapture(self.source)
        except Exception as exc:  # pragma: no cover
            # VideoCapture rarely raises, but guard just in case.
            logger.error("Exception while opening source '%s': %s", self.source, exc)
            self._is_open = False
            return

        if not self._cap.isOpened():
            self._is_open = False
            if self._is_file_source():
                logger.error(
                    "Could not open video file '%s'. "
                    "Check that the path exists and the codec is installed.",
                    self.source,
                )
            else:
                logger.error(
                    "Could not open camera at index %s. "
                    "Check that the device is connected and not in use.",
                    self.source,
                )
            return

        self._is_open = True
        props = self.get_properties()
        logger.info(
            "Video source opened successfully — %dx%d @ %.1f FPS, "
            "%d total frames, backend: %s.",
            props.get("width", 0),
            props.get("height", 0),
            props.get("fps", 0.0),
            props.get("frame_count", -1),
            props.get("backend", "unknown"),
        )

    def _is_file_source(self) -> bool:
        """Return True if the source is a file path (str), False for cameras (int)."""
        return isinstance(self.source, str)
