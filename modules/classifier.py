# =============================================================================
# modules/classifier.py  (pure rule-based — ML removed)
# ViolationClassifier — classifies behaviour for one tracked person.
#
# Classification uses only deterministic rules applied to the feature dict
# produced by FeatureExtractor.extract().  No model files are required.
#
# Priority order:
#   1. Running / Sudden Motion  — high max_speed AND high motion_variance
#   2. Loitering                — high stillness_ratio + low total_displacement
#   3. Suspicious Lingering     — high pace_ratio (going back and forth)
#   4. Normal                   — none of the above
#
# Object-detection labels (Weapon Detected, Abandoned Bag, Possible Theft)
# are passed through unchanged — they never enter the rule logic.
# =============================================================================

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Violation label constants — used throughout the system.
# ---------------------------------------------------------------------------
VIOLATION_RUNNING   = "Running / Sudden Motion"
VIOLATION_LOITERING = "Loitering"
VIOLATION_LINGERING = "Suspicious Lingering"
VIOLATION_UNSAFE    = "Unsafe Activity"

# Object-detection labels — always pass through, never go through rule logic.
_OBJECT_DETECTION_LABELS = frozenset({
    "Weapon Detected",
    "Abandoned Bag",
    "Possible Theft",
    VIOLATION_UNSAFE,
})


# ===========================================================================
# ViolationClassifier
# ===========================================================================

class ViolationClassifier:
    """
    Pure rule-based behaviour classifier.

    Thresholds are read from dashboard/config if available; the constructor
    parameters act as explicit overrides (or sensible defaults when the
    class is used outside the dashboard context).

    Parameters
    ----------
    running_speed_threshold : float
        max_speed (px/sec) above which Running may be declared.
    motion_variance_threshold : float
        motion_variance above this confirms erratic movement (used jointly
        with running_speed_threshold).
    loitering_stillness : float
        stillness_ratio above which Loitering may be declared.
    loitering_displacement : float
        Maximum total_displacement (px) for Loitering to trigger.
    pace_ratio_threshold : float
        pace_ratio above which Suspicious Lingering is declared.
    min_speed_threshold : float
        avg_speed must exceed this for Suspicious Lingering to trigger.
    """

    def __init__(
        self,
        running_speed_threshold:    float = 180.0,
        motion_variance_threshold:  float = 50.0,
        loitering_stillness:        float = 0.80,
        loitering_displacement:     float = 40.0,
        pace_ratio_threshold:       float = 4.0,
        min_speed_threshold:        float = 15.0,
        unsafe_motion_variance_threshold: float = 1200.0,
        unsafe_min_speed: float = 25.0,
    ):
        self.running_speed_threshold   = running_speed_threshold
        self.motion_variance_threshold = motion_variance_threshold
        self.loitering_stillness       = loitering_stillness
        self.loitering_displacement    = loitering_displacement
        self.pace_ratio_threshold      = pace_ratio_threshold
        self.min_speed_threshold       = min_speed_threshold
        self.unsafe_motion_variance_threshold = unsafe_motion_variance_threshold
        self.unsafe_min_speed = unsafe_min_speed

        logger.info(
            "ViolationClassifier initialised (pure rule-based) — "
            "run_spd=%.1f  mv_var=%.1f  loit_still=%.2f  loit_disp=%.1f  "
            "pace_ratio=%.1f  min_spd=%.1f",
            running_speed_threshold, motion_variance_threshold,
            loitering_stillness, loitering_displacement,
            pace_ratio_threshold, min_speed_threshold,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, features: Dict[str, float]) -> Optional[str]:
        """
        Classify behaviour for one tracked person.

        Object-detection labels passed in *features* under an ``"object_label"``
        key are returned immediately without entering the rule engine.

        Parameters
        ----------
        features : dict[str, float]
            Output of FeatureExtractor.extract(), optionally augmented with
            an ``"object_label"`` key for object-detection pass-throughs.

        Returns
        -------
        Optional[str]
            A violation label string, or None for normal behaviour.
        """
        if not features:
            return None

        # ── Object-detection pass-through ────────────────────────────────
        object_label = features.get("object_label")
        if object_label in _OBJECT_DETECTION_LABELS:
            logger.debug("Object-detection label '%s' passed through.", object_label)
            return object_label

        # ── Pure rule-based classification ────────────────────────────────
        return self._classify_rules(features)

    # ------------------------------------------------------------------
    # Rule engine
    # ------------------------------------------------------------------

    def _classify_rules(self, features: Dict[str, float]) -> Optional[str]:
        """
        Apply deterministic threshold rules.

        Priority order:
          1. Running / Sudden Motion  — max_speed spike + motion variance
          2. Loitering                — high stillness + low displacement
          3. Suspicious Lingering     — high pace_ratio (pacing back and forth)
          4. None (Normal)            — no threshold exceeded
        """
        avg_speed          = features.get("avg_speed",          0.0)
        max_speed          = features.get("max_speed",          0.0)
        total_displacement = features.get("total_displacement", 0.0)
        total_distance     = features.get("total_distance",     0.0)
        stillness_ratio    = features.get("stillness_ratio",    0.0)
        motion_variance    = features.get("motion_variance",    0.0)

        # Pace ratio — how much the person is going back and forth.
        pace_ratio = total_distance / (total_displacement + 1e-5)

        # ── Rule 1: Running / Sudden Motion ───────────────────────────────
        # High max speed AND high motion variance.
        if (max_speed > self.running_speed_threshold
                and motion_variance > self.motion_variance_threshold):
            logger.debug(
                "RULE: RUNNING — max_speed=%.1f > %.1f  motion_var=%.1f > %.1f",
                max_speed, self.running_speed_threshold,
                motion_variance, self.motion_variance_threshold,
            )
            return VIOLATION_RUNNING

        # ── Rule 2: Loitering ─────────────────────────────────────────────
        # Person is mostly still and has not moved far from start.
        if (stillness_ratio    > self.loitering_stillness
                and total_displacement < self.loitering_displacement):
            logger.debug(
                "RULE: LOITERING — stillness=%.2f > %.2f  disp=%.1f < %.1f",
                stillness_ratio, self.loitering_stillness,
                total_displacement, self.loitering_displacement,
            )
            return VIOLATION_LOITERING

        # ── Rule 3: Suspicious Lingering ──────────────────────────────────
        # Person is moving but going back and forth in a small area.
        if (pace_ratio > self.pace_ratio_threshold
                and avg_speed > self.min_speed_threshold):
            logger.debug(
                "RULE: LINGERING — pace_ratio=%.2f > %.2f  avg_speed=%.1f > %.1f",
                pace_ratio, self.pace_ratio_threshold,
                avg_speed, self.min_speed_threshold,
            )
            return VIOLATION_LINGERING

        # ── Rule 4: Unsafe / Strange Activity ───────────────────────────
        # Catch high-energy, irregular movement that does not fit the more
        # specific running or pacing labels.
        if (motion_variance > self.unsafe_motion_variance_threshold
                and avg_speed > self.unsafe_min_speed):
            logger.debug(
                "RULE: UNSAFE ACTIVITY — motion_var=%.1f speed=%.1f",
                motion_variance, avg_speed,
            )
            return VIOLATION_UNSAFE

        # ── Default: Normal Movement ──────────────────────────────────────
        logger.debug("RULE: NORMAL — no threshold exceeded.")
        return None
