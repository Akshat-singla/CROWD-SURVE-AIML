# =============================================================================
# modules/detector.py  (redesigned — no zone logic, cleaner interface)
# PersonDetector — wraps YOLOv8 to return person detections per frame.
#
# Responsibilities:
#   • Load a YOLOv8 model once (GPU → CPU fallback).
#   • Run inference on a BGR frame with verbose=False and imgsz=416 (Fix 3).
#   • Filter to person class only (COCO class 0).
#   • Return a typed List[Detection] suitable for the tracker.
#
# FIX 3: Default inference resolution (imgsz) changed from 640 → 416.
#         Speeds up CPU inference with minimal accuracy loss for typical
#         surveillance distances.
# FIX 6: GPU is auto-detected via _auto_device().  Device is printed to
#         stdout at model-load time so it is immediately visible in the
#         terminal without needing to inspect logs.
# =============================================================================

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection dataclass — one detection per person per frame
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    A single person detection in one video frame.

    Attributes
    ----------
    x1, y1 : int   Top-left bounding box corner (pixels).
    x2, y2 : int   Bottom-right bounding box corner (pixels).
    confidence : float  Detection score in [0, 1].
    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    label: str = "person"

    @property
    def bbox(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def cx(self) -> int:
        """Horizontal centroid of the bounding box."""
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        """Vertical centroid of the bounding box."""
        return (self.y1 + self.y2) // 2


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class PersonDetector:
    """
    Wraps a YOLOv8 model for single-call, person-only detection.

    Parameters
    ----------
    model_path : str
        Path to the .pt weights file (auto-downloaded by Ultralytics if absent).
    confidence : float
        Minimum detection confidence threshold.
    imgsz : int
        Inference image size (shorter edge).  640 is the YOLOv8 default.
    device : Optional[str]
        Torch device ("cuda", "cpu", "mps").  Auto-selected when None.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        imgsz: int = 640,          # 640 — full resolution now that GPU is available
        device: Optional[str] = None,
    ):
        self.confidence = confidence
        self.imgsz      = imgsz
        self.device     = device or self._auto_device()

        # FIX 6: print device to stdout so it is visible in the terminal
        print(f"[INFO] PersonDetector — loading '{model_path}' on device: {self.device}")
        logger.info(
            "Loading YOLOv8 model '%s' on device '%s'…", model_path, self.device
        )
        try:
            self._model = YOLO(model_path)
            self._model.to(self.device)
            logger.info("YOLOv8 model loaded.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLOv8 model: {exc}") from exc

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray   Shape (H, W, 3) BGR.

        Returns
        -------
        List[Detection]   Sorted by confidence (highest first).
        """
        if frame is None or frame.ndim != 3:
            return []

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence,
                classes=[0],          # person only
                imgsz=self.imgsz,
                verbose=False,        # suppress per-frame console spam
                device=self.device,
            )
        except Exception as exc:
            logger.error("YOLOv8 inference error: %s", exc)
            return []

        return self._parse(results, frame.shape)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _parse(self, results, frame_shape: tuple) -> List[Detection]:
        """Convert Ultralytics Results → List[Detection]."""
        h, w = frame_shape[:2]
        detections: List[Detection] = []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls  = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = xyxy
            # Clamp to frame boundaries.
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box

            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=cls,
                label="person",
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _auto_device() -> str:
        """Select best available torch device (FIX 6: GPU auto-detection)."""
        if torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.current_device()}"
            print(f"[INFO] CUDA available — using GPU: {torch.cuda.get_device_name(0)}")
            logger.info("CUDA available — using GPU.")
            return dev
        if torch.backends.mps.is_available():
            print("[INFO] Apple MPS available — using MPS.")
            logger.info("Apple MPS available — using MPS.")
            return "mps"
        print("[INFO] No GPU detected — running inference on CPU.")
        logger.info("No GPU detected — using CPU.")
        return "cpu"
