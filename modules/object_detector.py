# =============================================================================
# modules/object_detector.py
# Object Detection — Weapons, Bags, and Abandoned Bag Logic
#
# Enhancement 3: ObjectDetector
#   Runs a second YOLOv8 instance on COCO classes for handbags, backpacks,
#   suitcases, knives, and scissors.  Weapon-like objects are flagged
#   immediately; bag-type objects are passed to BagTracker for abandonment
#   analysis.
#
# Enhancement 4: BagTracker
#   Maintains a per-session dictionary of observed bags.  A bag is classified
#   as abandoned when it has been stationary for ABANDONED_BAG_SECONDS and no
#   person bounding box overlaps with it.
#
# NOTE ON GUN DETECTION:
#   The standard COCO dataset does not contain a "gun" class.  To detect
#   firearms, replace WEAPON_MODEL_PATH in config.py with a path to a
#   custom-trained model (e.g. trained on Open Images "Handgun" / "Rifle").
#   The ObjectDetector will load that model instead of the default yolov8n.pt.
# =============================================================================

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class metadata for objects we care about
# ---------------------------------------------------------------------------

# Maps COCO class-id → (human label, category)
# category is "weapon" or "bag"
_COCO_OBJECT_CLASSES: Dict[int, Tuple[str, str]] = {
    24: ("backpack",  "bag"),
    26: ("handbag",   "bag"),
    28: ("suitcase",  "bag"),
    43: ("knife",     "weapon"),
    76: ("scissors",  "weapon"),
}

_WEAPON_LABELS = {
    "gun", "firearm", "handgun", "pistol", "rifle", "weapon", "shotgun",
    "revolver", "airgun", "machine gun", "knife", "scissors", "firearm",
    "weapon detection", "gun detection",
}
_BAG_LABELS = {"backpack", "handbag", "suitcase", "bag", "purse"}


# ---------------------------------------------------------------------------
# Detection result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ObjectDetection:
    """
    A single detected object in one video frame.

    Attributes
    ----------
    label      : Human-readable COCO class name.
    bbox       : (x1, y1, x2, y2) pixel bounding box.
    confidence : Detection confidence score (0–1).
    class_id   : Raw COCO class index.
    category   : "weapon" or "bag".
    """
    label:      str
    bbox:       Tuple[int, int, int, int]
    confidence: float
    class_id:   int
    category:   str   # "weapon" | "bag"

    @property
    def cx(self) -> int:
        return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def cy(self) -> int:
        return (self.bbox[1] + self.bbox[3]) // 2


# ---------------------------------------------------------------------------
# IoU helper (shared by BagTracker and theft detection in app.py)
# ---------------------------------------------------------------------------

def compute_iou(box1: Tuple[int, int, int, int],
                box2: Tuple[int, int, int, int]) -> float:
    """
    Intersection-over-Union for two (x1, y1, x2, y2) bounding boxes.

    Returns a float in [0, 1].  Returns 0.0 for degenerate inputs.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0

    area1 = max(0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ===========================================================================
# ObjectDetector
# ===========================================================================

class ObjectDetector:
    """
    Runs YOLOv8 (COCO-trained) on selected object classes every N frames.

    Only the OBJECT_TARGET_CLASSES defined in config are returned.
    Results are split into weapon detections (immediate alert) and bag
    detections (forwarded to BagTracker).

    Parameters
    ----------
    model_path        : Path to YOLOv8 weights (yolov8n.pt or custom).
    conf_threshold    : Minimum confidence to keep a detection.
    target_classes    : List of COCO class indices to detect.
    device            : Torch device string; auto-selected when None.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        self.conf_threshold  = conf_threshold
        # ``None`` means all classes for a custom model. An explicit list is
        # used for the COCO fallback so bags and weapon-like classes are fast.
        self.target_classes  = target_classes
        self.device          = device or self._auto_device()

        logger.info(
            "Loading object detector model '%s' on '%s'…",
            model_path, self.device,
        )
        try:
            self._model = YOLO(model_path)
            self._model.to(self.device)
            names = getattr(self._model, "names", {})
            labels = names.values() if isinstance(names, dict) else names
            firearm_labels = [
                str(label) for label in labels
                if any(token in str(label).lower()
                       for token in ("gun", "firearm", "pistol", "rifle", "handgun"))
            ]
            if not firearm_labels:
                logger.warning(
                    "Model '%s' has no named firearm classes. Gun detection will not work; "
                    "COCO only provides knife/scissors among weapon-like objects.",
                    model_path,
                )
            else:
                logger.info("Firearm classes enabled: %s", firearm_labels)
            logger.info("ObjectDetector model loaded.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load object detector: {exc}") from exc

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[ObjectDetection]:
        """
        Run inference on a single BGR frame and return detected objects.

        Parameters
        ----------
        frame : np.ndarray  Shape (H, W, 3) BGR.

        Returns
        -------
        List[ObjectDetection]  Filtered to target classes above conf threshold.
        """
        if frame is None or frame.ndim != 3:
            return []

        try:
            precision_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.device == "cuda" else nullcontext()
            )
            with precision_context:
                results = self._model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    classes=self.target_classes,
                    verbose=False,
                    device=self.device,
                )
        except Exception as exc:
            logger.error("ObjectDetector inference error: %s", exc)
            return []

        return self._parse(results)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _parse(self, results) -> List[ObjectDetection]:
        """Convert raw Ultralytics results → List[ObjectDetection]."""
        detections: List[ObjectDetection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        h, w  = results[0].orig_shape[:2]

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())

            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Custom weapon models use their own class IDs, so classify from
            # the model's label while retaining COCO compatibility.
            model_names = getattr(self._model, "names", {})
            raw_label = model_names.get(cls_id, "") if isinstance(model_names, dict) else ""
            label = str(raw_label).strip().lower().replace("_", " ").replace("-", " ")
            if any(name in label for name in _WEAPON_LABELS):
                category = "weapon"
            elif any(name in label for name in _BAG_LABELS):
                category = "bag"
            elif cls_id in _COCO_OBJECT_CLASSES:
                label, category = _COCO_OBJECT_CLASSES[cls_id]
            else:
                continue
            detections.append(ObjectDetection(
                label=label,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_id=cls_id,
                category=category,
            ))

        logger.debug("ObjectDetector: %d object(s) found.", len(detections))
        return detections

    @staticmethod
    def _auto_device() -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ===========================================================================
# BagTracker  (Enhancement 4 — Abandoned Bag Detection)
# ===========================================================================

@dataclass
class _BagEntry:
    """Internal state for a single tracked bag."""
    bag_id:       int
    bbox:         Tuple[int, int, int, int]
    first_seen:   float          # Unix time when first detected
    last_seen:    float          # Unix time when last detected
    last_alert:   float = 0.0   # Unix time of last abandoned-bag alert
    is_abandoned: bool  = False


class BagTracker:
    """
    Tracks bag-type objects across frames using IoU-based matching.

    A bag is classified as abandoned when:
      1. It has been continuously visible for >= ABANDONED_BAG_SECONDS seconds.
      2. No person bounding box significantly overlaps it (IoU < 0.1 with all
         current person bboxes).

    Parameters
    ----------
    abandoned_seconds : Seconds of stationary detection before flagging.
    disappear_timeout : Seconds without detection before removing a bag entry.
    alert_cooldown    : Minimum seconds between re-alerts for the same bag.
    iou_match_thresh  : IoU threshold for matching new detections to known bags.
    person_iou_thresh : IoU above this means a person is "near" the bag (not abandoned).
    """

    def __init__(
        self,
        abandoned_seconds:  float = 30.0,
        disappear_timeout:  float = 10.0,
        alert_cooldown:     float = 60.0,
        iou_match_thresh:   float = 0.40,
        person_iou_thresh:  float = 0.10,
    ):
        self.abandoned_seconds  = abandoned_seconds
        self.disappear_timeout  = disappear_timeout
        self.alert_cooldown     = alert_cooldown
        self.iou_match_thresh   = iou_match_thresh
        self.person_iou_thresh  = person_iou_thresh

        self._bag_tracker: Dict[int, _BagEntry] = {}
        self._next_bag_id: int = 1

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        bag_detections: List[ObjectDetection],
        person_bboxes:  List[Tuple[int, int, int, int]],
        timestamp:      float,
    ) -> List[_BagEntry]:
        """
        Update bag state and return newly (or re-)abandoned bags ready to alert.

        Parameters
        ----------
        bag_detections : Bag ObjectDetection objects from this frame.
        person_bboxes  : All current person bboxes as (x1,y1,x2,y2) tuples.
        timestamp      : Unix wall-clock time.

        Returns
        -------
        List[_BagEntry]  Bags that should trigger an alert right now.
        """
        # ── 1. Match incoming detections to known bags ─────────────────
        matched_ids: set = set()

        for det in bag_detections:
            best_iou  = 0.0
            best_id   = None

            for bag_id, entry in self._bag_tracker.items():
                iou = compute_iou(det.bbox, entry.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id  = bag_id

            if best_id is not None and best_iou >= self.iou_match_thresh:
                # Update existing bag — keep first_seen, update bbox + last_seen
                self._bag_tracker[best_id].bbox      = det.bbox
                self._bag_tracker[best_id].last_seen = timestamp
                matched_ids.add(best_id)
            else:
                # New bag
                new_id = self._next_bag_id
                self._next_bag_id += 1
                self._bag_tracker[new_id] = _BagEntry(
                    bag_id=new_id,
                    bbox=det.bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                matched_ids.add(new_id)

        # ── 2. Remove bags that have disappeared (not seen recently) ────
        to_remove = [
            bid for bid, entry in self._bag_tracker.items()
            if (timestamp - entry.last_seen) > self.disappear_timeout
        ]
        for bid in to_remove:
            del self._bag_tracker[bid]
            logger.debug("BagTracker: removed stale bag_id=%d.", bid)

        # ── 3. Check abandonment conditions ────────────────────────────
        newly_abandoned: List[_BagEntry] = []

        for bag_id, entry in self._bag_tracker.items():
            # Must have been in the scene long enough
            age = timestamp - entry.first_seen
            if age < self.abandoned_seconds:
                continue

            # No person must overlap with it
            person_nearby = any(
                compute_iou(entry.bbox, pbbox) >= self.person_iou_thresh
                for pbbox in person_bboxes
            )
            if person_nearby:
                entry.is_abandoned = False
                continue

            # Cooldown check
            entry.is_abandoned = True
            if (timestamp - entry.last_alert) >= self.alert_cooldown:
                entry.last_alert = timestamp
                newly_abandoned.append(entry)
                logger.info(
                    "BagTracker: bag_id=%d flagged as ABANDONED (age=%.0fs).",
                    bag_id, age,
                )

        return newly_abandoned

    @property
    def bags_tracked(self) -> int:
        """Total number of currently tracked bags."""
        return len(self._bag_tracker)

    @property
    def abandoned_count(self) -> int:
        """Number of bags currently flagged as abandoned."""
        return sum(1 for e in self._bag_tracker.values() if e.is_abandoned)

    def cleanup_stale(self, timestamp: float) -> None:
        """
        Explicit cleanup pass — call from the stale-track cleanup timer
        (Enhancement 2) to purge bags not seen recently.
        """
        to_remove = [
            bid for bid, entry in self._bag_tracker.items()
            if (timestamp - entry.last_seen) > self.disappear_timeout
        ]
        for bid in to_remove:
            del self._bag_tracker[bid]
        if to_remove:
            logger.info(
                "BagTracker cleanup: removed %d stale bag(s).", len(to_remove)
            )
