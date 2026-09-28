# =============================================================================
# modules/feature_extractor.py  (redesigned — pure movement features)
# FeatureExtractor — derives motion features from a per-track buffer.
#
# Fix 3 & Fix 4: All speed metrics are now computed in pixels per second of
# VIDEO time, using the timestamp deltas stored in each BufferEntry.
# Previously speeds were pixels-per-step which conflated frame rate with speed.
# Now they correctly reflect actual motion speed regardless of CPU processing
# rate, because Thread 1 supplies video-time timestamps (frame_index / source_fps).
#
# Computes the six movement metrics required by the rule-based classifier:
#   avg_speed, max_speed, total_displacement, total_distance,
#   stillness_ratio, pace_ratio.
#
# Stateless — one instance can safely serve all track IDs concurrently.
# No zone logic, no config imports — receives all thresholds via constructor.
# =============================================================================

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from modules.behaviour_buffer import BufferEntry

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Computes motion features from one track's sliding centroid buffer.

    Parameters
    ----------
    min_samples : int
        Minimum buffer length before extraction is attempted.
    min_speed_threshold : float
        Fix 3 & 4: Speed threshold in pixels per second of VIDEO time.
        All speed calculations now use the timestamp deltas stored in each
        BufferEntry (which are video-time values from Thread 1) so behaviour
        windows are accurate regardless of CPU processing speed.
    """

    def __init__(
        self,
        min_samples: int = 10,
        min_speed_threshold: float = 20.0,
    ):
        self.min_samples         = min_samples
        self.min_speed_threshold = min_speed_threshold

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, entries: List[BufferEntry]) -> Optional[Dict[str, float]]:
        """
        Compute all six motion features from a track's buffer entries.

        Parameters
        ----------
        entries : List[BufferEntry]
            Chronological buffer for a single track from BehaviourBuffer.get().

        Returns
        -------
        dict[str, float] | None
            Feature dict, or None if the buffer is too small or time span zero.

        Feature definitions  (Fix 3 & 4 — all speeds in px/sec of video time)
        -------------------
        avg_speed          : mean per-step speed (px / video-second)
        max_speed          : peak per-step speed (px / video-second)
        total_displacement : straight-line distance first → last centroid (px)
        total_distance     : cumulative path length (px)
        stillness_ratio    : fraction of steps where speed < min_speed_threshold
        pace_ratio         : total_distance / (total_displacement + 1e-5)
                             High ratio = person pacing back and forth.
        """
        if len(entries) < self.min_samples:
            return None

        # ── Unpack centroids and video-time timestamps ────────────────────
        xs  = np.array([e.cx        for e in entries], dtype=np.float64)
        ys  = np.array([e.cy        for e in entries], dtype=np.float64)
        tss = np.array([e.timestamp for e in entries], dtype=np.float64)

        # ── Per-step pixel distances ──────────────────────────────────────
        dx = np.diff(xs)
        dy = np.diff(ys)
        step_distances = np.sqrt(dx ** 2 + dy ** 2)   # shape (N-1,)

        if len(step_distances) == 0:
            return None

        # ── Per-step time deltas (video seconds) ─────────────────────────
        dt = np.diff(tss)                              # shape (N-1,)
        # Guard: replace zero/negative dt with tiny positive value to avoid
        # division-by-zero when two consecutive entries share a timestamp
        # (e.g. detection was skipped so the same frame was processed twice).
        dt = np.where(dt > 0.0, dt, 1e-6)

        # Fix 3 & 4: speeds in pixels per second of video time.
        step_speeds = step_distances / dt              # px / video-second

        # ── Total video time spanned by this window ───────────────────────
        total_time = float(tss[-1] - tss[0])
        if total_time <= 0.0:
            return None

        # ── avg_speed (px/sec of video time) ─────────────────────────────
        avg_speed = float(np.mean(step_speeds))

        # ── max_speed (px/sec of video time) ─────────────────────────────
        max_speed = float(np.max(step_speeds))

        # ── total_displacement: straight-line first→last ──────────────────
        total_displacement = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))

        # ── total_distance: cumulative path ───────────────────────────────
        total_distance = float(np.sum(step_distances))

        # ── stillness_ratio (based on px/sec speed) ───────────────────────
        still_steps     = np.sum(step_speeds < self.min_speed_threshold)
        stillness_ratio = float(still_steps / len(step_speeds))

        # ── pace_ratio ────────────────────────────────────────────────────
        # Dividing by (displacement + ε) avoids ZeroDivisionError when the
        # person hasn't moved at all (displacement = 0 → ratio → large).
        pace_ratio = total_distance / (total_displacement + 1e-5)

        features = {
            "avg_speed":          avg_speed,
            "max_speed":          max_speed,
            "total_displacement": total_displacement,
            "total_distance":     total_distance,
            "stillness_ratio":    stillness_ratio,
            "pace_ratio":         pace_ratio,
        }

        logger.debug(
            "Features — avg_spd=%.1f px/s  max_spd=%.1f px/s  disp=%.1f  "
            "dist=%.1f  still=%.2f  pace=%.2f  span=%.2fs",
            avg_speed, max_speed, total_displacement,
            total_distance, stillness_ratio, pace_ratio, total_time,
        )
        return features
