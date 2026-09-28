# =============================================================================
# modules/tracker.py  (Enhanced — ByteTrack / DeepSORT factory)
# PersonTracker — multi-person tracking with selectable backend.
#
# Enhancement 6: TRACKER_TYPE factory pattern.
#   "bytetrack"  →  Ultralytics model.track(persist=True) — fast, no Re-ID.
#   "deepsort"   →  deep_sort_realtime.DeepSort — slower but uses appearance
#                   features to recover IDs after occlusion.
#
# Both backends return the same List[Track] output type so the rest of the
# pipeline is unaffected by the choice.
#
# Usage:
#   Set TRACKER_TYPE in dashboard/config.py before starting the pipeline.
#   Switching requires a pipeline restart.
# =============================================================================

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track dataclass — one entry per active person per frame
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """
    A single active tracked person for one video frame.

    Attributes
    ----------
    track_id : int      Persistent unique ID across frames.
    x1, y1, x2, y2 : int  Bounding box corners (pixels).
    cx, cy : int        Centroid of the bounding box.
    confidence : float  Detection confidence for this frame.
    timestamp : float   Unix wall-clock time this frame was captured.
    """
    track_id:   int
    x1:         int
    y1:         int
    x2:         int
    y2:         int
    cx:         int
    cy:         int
    confidence: float
    timestamp:  float

    @property
    def bbox(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


# ---------------------------------------------------------------------------
# PersonTracker — factory + unified interface
# ---------------------------------------------------------------------------

class PersonTracker:
    """
    Multi-person tracker with selectable detection backend.

    Parameters
    ----------
    model_path   : Path to YOLOv8 weights for ByteTrack detection.
    confidence   : Detection confidence threshold.
    imgsz        : YOLOv8 inference image size.
    device       : Torch device string; auto-selected when None.
    tracker_type : "bytetrack" (default) or "deepsort".
    """

    def __init__(
        self,
        model_path:   str,
        confidence:   float = 0.5,
        imgsz:        int   = 640,
        device:       Optional[str] = None,
        tracker_type: str   = "bytetrack",
    ):
        self.confidence   = confidence
        self.imgsz        = imgsz
        self.device       = device or self._auto_device()
        self.tracker_type = tracker_type.lower()

        if self.tracker_type == "bytetrack":
            self._init_bytetrack(model_path)
        elif self.tracker_type == "deepsort":
            self._init_deepsort(model_path)
        else:
            raise ValueError(
                f"Unknown TRACKER_TYPE '{tracker_type}'. "
                "Choose 'bytetrack' or 'deepsort'."
            )

    # ──────────────────────────────────────────────────────────────────
    # Backend initialisation
    # ──────────────────────────────────────────────────────────────────

    def _init_bytetrack(self, model_path: str) -> None:
        """
        ByteTrack via Ultralytics model.track(persist=True).

        This is the recommended default:
          • No Re-ID model required.
          • Kalman-filter motion prediction keeps IDs stable in
            low-to-moderate occlusion environments.
          • Lower CPU/GPU overhead than DeepSORT.
        """
        logger.info(
            "Loading ByteTrack model '%s' on device '%s'…", model_path, self.device
        )
        try:
            self._model = YOLO(model_path)
            self._model.to(self.device)
            self._deepsort = None
            logger.info("PersonTracker (ByteTrack) ready.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load ByteTrack model: {exc}") from exc

    def _init_deepsort(self, model_path: str) -> None:
        """
        DeepSORT via deep_sort_realtime library.

        Use when persons frequently overlap and ByteTrack keeps switching IDs.
        DeepSORT uses appearance (Re-ID) features extracted from each detection
        crop to match identities across occlusions.

        Requires:  pip install deep-sort-realtime

        The YOLOv8 model is still used for detection; DeepSORT handles tracking.
        """
        # Import is intentionally deferred so that the package is only required
        # when TRACKER_TYPE = "deepsort" is explicitly selected.
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise ImportError(
                "DeepSORT backend selected but 'deep-sort-realtime' is not installed. "
                "Run:  pip install deep-sort-realtime"
            ) from exc

        logger.info(
            "Loading DeepSORT detector model '%s' on device '%s'…", model_path, self.device
        )
        try:
            self._model    = YOLO(model_path)
            self._model.to(self.device)
            self._deepsort = DeepSort(max_age=30)
            logger.info("PersonTracker (DeepSORT) ready.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load DeepSORT tracker: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, timestamp: float) -> List[Track]:
        """
        Run detection + tracking on a single BGR frame.

        Parameters
        ----------
        frame     : np.ndarray  Shape (H, W, 3) BGR.
        timestamp : float       Wall-clock capture time.

        Returns
        -------
        List[Track]  Active persons in this frame, sorted by track_id.
        """
        if frame is None or frame.ndim != 3:
            return []

        if self.tracker_type == "bytetrack":
            return self._update_bytetrack(frame, timestamp)
        else:
            return self._update_deepsort(frame, timestamp)

    def reset(self) -> None:
        """
        Reset tracker state.  Call when the video source changes to prevent
        stale IDs being matched against the new stream.
        """
        if self.tracker_type == "bytetrack":
            try:
                self._model = YOLO(self._model.model.yaml["yaml_file"])
                self._model.to(self.device)
                logger.info("ByteTrack state reset.")
            except Exception:
                logger.warning("ByteTrack reset failed — re-loading model.")
        else:
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort
                self._deepsort = DeepSort(max_age=30)
                logger.info("DeepSORT state reset.")
            except Exception:
                logger.warning("DeepSORT reset failed.")

    # ──────────────────────────────────────────────────────────────────
    # Backend-specific update methods
    # ──────────────────────────────────────────────────────────────────

    def _update_bytetrack(self, frame: np.ndarray, timestamp: float) -> List[Track]:
        """
        ByteTrack path: model.track() with persist=True maintains internal
        ByteTrack state across consecutive calls so IDs stay stable even when
        called every other frame.
        """
        try:
            precision_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.device == "cuda" else nullcontext()
            )
            with precision_context:
                results = self._model.track(
                    source=frame,
                    conf=self.confidence,
                    classes=[0],          # person only
                    imgsz=self.imgsz,
                    persist=True,         # maintain ByteTrack state across calls
                    verbose=False,
                    device=self.device,
                )
        except Exception as exc:
            logger.error("ByteTrack inference error: %s", exc)
            return []

        return self._parse_ultralytics(results, timestamp)

    def _update_deepsort(self, frame: np.ndarray, timestamp: float) -> List[Track]:
        """
        DeepSORT path:
          1. Run YOLOv8 in plain predict() mode (no tracker) to get raw detections.
          2. Convert detections to the [[x, y, w, h], conf, class_id] format
             required by deep_sort_realtime.
          3. Call deepsort.update_tracks(detections, frame=frame).
          4. Parse DeepSORT track objects → our Track dataclass.
        """
        h, w = frame.shape[:2]

        try:
            precision_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.device == "cuda" else nullcontext()
            )
            with precision_context:
                results = self._model.predict(
                    source=frame,
                    conf=self.confidence,
                    classes=[0],
                    imgsz=self.imgsz,
                    verbose=False,
                    device=self.device,
                )
        except Exception as exc:
            logger.error("DeepSORT detect error: %s", exc)
            return []

        # Build detection list in [[x,y,w,h], conf, class_id] format
        raw_detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None or boxes.conf is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].item())
                    bw   = int(x2 - x1)
                    bh   = int(y2 - y1)
                    raw_detections.append(([int(x1), int(y1), bw, bh], conf, 0))

        try:
            ds_tracks = self._deepsort.update_tracks(raw_detections, frame=frame)
        except Exception as exc:
            logger.error("DeepSORT update_tracks error: %s", exc)
            return []

        tracks: List[Track] = []
        for dt in ds_tracks:
            if not dt.is_confirmed():
                continue
            tid = int(dt.track_id)
            ltrb = dt.to_ltrb()  # [x1, y1, x2, y2] as floats
            x1 = max(0, min(int(ltrb[0]), w - 1))
            y1 = max(0, min(int(ltrb[1]), h - 1))
            x2 = max(0, min(int(ltrb[2]), w - 1))
            y2 = max(0, min(int(ltrb[3]), h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # DeepSORT doesn't expose per-track confidence; use 1.0 as placeholder.
            tracks.append(Track(
                track_id=tid,
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=cx, cy=cy,
                confidence=1.0,
                timestamp=timestamp,
            ))

        tracks.sort(key=lambda t: t.track_id)
        logger.debug("DeepSORT: %d confirmed track(s).", len(tracks))
        return tracks

    # ──────────────────────────────────────────────────────────────────
    # Shared parse helpers
    # ──────────────────────────────────────────────────────────────────

    def _parse_ultralytics(self, results, timestamp: float) -> List[Track]:
        """Convert Ultralytics track results → List[Track]."""
        tracks: List[Track] = []

        if not results or results[0].boxes is None:
            return tracks

        result = results[0]
        boxes  = result.boxes

        if boxes.id is None:
            return tracks

        ids   = boxes.id.cpu().numpy().astype(int)
        xyxys = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        h, w  = result.orig_shape[:2]

        for track_id, xyxy, conf in zip(ids, xyxys, confs):
            x1, y1, x2, y2 = xyxy
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            tracks.append(Track(
                track_id=int(track_id),
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=cx, cy=cy,
                confidence=float(conf),
                timestamp=timestamp,
            ))

        tracks.sort(key=lambda t: t.track_id)
        logger.debug("ByteTrack: %d active track(s).", len(tracks))
        return tracks

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
