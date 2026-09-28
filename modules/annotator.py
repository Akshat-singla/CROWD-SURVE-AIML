# =============================================================================
# modules/annotator.py
# FrameAnnotator — draws all visual overlays onto a video frame.
#
# Responsibilities:
#   • Zone polygons (semi-transparent coloured fill + edge).
#   • Bounding boxes colour-coded by activity.
#   • Track IDs, activity labels, and confidence badges.
#   • Per-person movement trail (recent centroid history).
#   • FPS counter and global stats banner.
# =============================================================================

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import (
    ACTIVITY_COLORS,
    ZONE_OVERLAY_COLOR,
    ZONE_OVERLAY_ALPHA,
    DRAW_TRAILS,
    TRAIL_LENGTH,
    TRAIL_COLOR,
    RESTRICTED_ZONES,
)
from modules.tracker import Track
from modules.behaviour_buffer import BehaviourBuffer

logger = logging.getLogger(__name__)


class FrameAnnotator:
    """
    Applies all visual overlays to a raw BGR frame in-place and returns the
    annotated copy.  Designed to be called once per processed frame.

    Parameters
    ----------
    restricted_zones : list | None
        Zone polygon definitions.  Defaults to ``RESTRICTED_ZONES`` in config.
    draw_trails : bool | None
        Whether to draw centroid movement trails.  Defaults to ``DRAW_TRAILS``.
    trail_length : int | None
        Maximum number of historic centroids to draw.  Defaults to ``TRAIL_LENGTH``.
    """

    def __init__(
        self,
        restricted_zones=None,
        draw_trails: bool = None,
        trail_length: int = None,
    ):
        self._zones = restricted_zones if restricted_zones is not None else RESTRICTED_ZONES
        self._draw_trails  = draw_trails  if draw_trails  is not None else DRAW_TRAILS
        self._trail_length = trail_length if trail_length is not None else TRAIL_LENGTH

        # Per-track recent centroid history for trail rendering.
        # deque auto-discards entries beyond maxlen.
        self._trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self._trail_length)
        )

        # Pre-compile zone polygons to numpy once.
        self._zone_polys = [
            np.array(z, dtype=np.int32).reshape(-1, 1, 2)
            for z in self._zones if len(z) >= 3
        ]
        logger.debug(
            "FrameAnnotator initialised — %d zones, trails=%s.",
            len(self._zone_polys), self._draw_trails,
        )

    # =========================================================================
    # Public interface
    # =========================================================================

    def annotate(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        track_activities: Dict[int, str],
        buf: Optional[BehaviourBuffer] = None,
        fps: float = 0.0,
        alert_count: int = 0,
    ) -> np.ndarray:
        """
        Draw all overlays on a copy of ``frame`` and return the annotated image.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from the video source.
        tracks : List[Track]
            Active tracks for this frame (from MultiPersonTracker).
        track_activities : Dict[int, str]
            Maps track_id → activity label string.
        buf : BehaviourBuffer | None
            Used to update internal trail histories from the buffer data.  Pass
            None to skip trail updates (trails will still be drawn from cache).
        fps : float
            Current pipeline FPS — rendered in the stats banner.
        alert_count : int
            Total alert count — rendered in the stats banner.

        Returns
        -------
        np.ndarray
            Annotated BGR image (same shape as ``frame``).
        """
        out = frame.copy()

        # ── 1. Zone overlays (drawn first so boxes appear on top) ─────────────
        self._draw_zones(out)

        # ── 2. Update trail history from current tracks ───────────────────────
        if self._draw_trails:
            for t in tracks:
                self._trails[t.track_id].append((t.cx, t.cy))

        # ── 3. Remove trails for expired tracks ───────────────────────────────
        active_ids = {t.track_id for t in tracks}
        stale_ids  = set(self._trails.keys()) - active_ids
        for sid in stale_ids:
            del self._trails[sid]

        # ── 4. Per-track overlays ─────────────────────────────────────────────
        for track in tracks:
            activity = track_activities.get(track.track_id, "Unknown")
            color    = ACTIVITY_COLORS.get(activity, ACTIVITY_COLORS.get("Unknown", (200, 200, 200)))

            if self._draw_trails:
                self._draw_trail(out, track.track_id, color)

            self._draw_box(out, track, color)
            self._draw_label(out, track, activity, color)

        # ── 5. Stats banner (top-left HUD) ────────────────────────────────────
        self._draw_hud(out, fps=fps, track_count=len(tracks), alert_count=alert_count)

        return out

    def reset_trails(self) -> None:
        """Clear all cached trail history (call when switching video sources)."""
        self._trails.clear()

    # =========================================================================
    # Private draw helpers
    # =========================================================================

    def _draw_zones(self, frame: np.ndarray) -> None:
        """Draw semi-transparent filled polygons + solid border for each zone."""
        if not self._zone_polys:
            return

        overlay = frame.copy()
        for poly in self._zone_polys:
            cv2.fillPoly(overlay, [poly], color=ZONE_OVERLAY_COLOR)

        # Blend overlay with original at configured alpha.
        cv2.addWeighted(overlay, ZONE_OVERLAY_ALPHA, frame, 1 - ZONE_OVERLAY_ALPHA, 0, frame)

        # Solid border on top of the fill.
        for poly in self._zone_polys:
            cv2.polylines(frame, [poly], isClosed=True, color=ZONE_OVERLAY_COLOR, thickness=2)

        # "RESTRICTED" text label at the top-left corner of each zone.
        for poly in self._zone_polys:
            pts = poly.reshape(-1, 2)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 6
            y = max(y, 14)
            cv2.putText(
                frame, "RESTRICTED", (x + 4, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ZONE_OVERLAY_COLOR, 1, cv2.LINE_AA,
            )

    def _draw_trail(self, frame: np.ndarray, track_id: int, color: tuple) -> None:
        """Draw a fading polyline through the recent centroids of this track."""
        pts = list(self._trails.get(track_id, []))
        if len(pts) < 2:
            return

        n = len(pts)
        for i in range(1, n):
            # Fade the trail from near-transparent (old) to full colour (recent).
            alpha = i / n
            # Interpolate colour toward dim base.
            c = tuple(int(ch * alpha) for ch in TRAIL_COLOR)
            thickness = max(1, int(2 * alpha))
            cv2.line(frame, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)

    def _draw_box(self, frame: np.ndarray, track: Track, color: tuple) -> None:
        """Draw a bounding box with a filled top banner for the label."""
        x1, y1, x2, y2 = track.x1, track.y1, track.x2, track.y2

        # Main rectangle.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner tick marks for a surveillance aesthetic.
        tick = 10
        for cx_, cy_, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx_, cy_), (cx_ + dx * tick, cy_), color, 3, cv2.LINE_AA)
            cv2.line(frame, (cx_, cy_), (cx_, cy_ + dy * tick), color, 3, cv2.LINE_AA)

        # Centroid dot.
        cv2.circle(frame, (track.cx, track.cy), 3, color, -1, cv2.LINE_AA)

    def _draw_label(
        self, frame: np.ndarray, track: Track, activity: str, color: tuple
    ) -> None:
        """Draw a filled pill label above the bounding box."""
        label   = f" ID:{track.track_id}  {activity} "
        font    = cv2.FONT_HERSHEY_SIMPLEX
        scale   = 0.48
        thick   = 1

        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

        # Position label above the box; clamp to frame top edge.
        lx = track.x1
        ly = max(track.y1 - 6, th + 4)

        # Filled background rectangle.
        pad = 3
        cv2.rectangle(
            frame,
            (lx - pad, ly - th - pad),
            (lx + tw + pad, ly + baseline + pad),
            color, cv2.FILLED,
        )
        # White text on the coloured background.
        cv2.putText(
            frame, label, (lx, ly),
            font, scale, (255, 255, 255), thick, cv2.LINE_AA,
        )

        # Confidence score — small grey text below bounding box.
        conf_label = f"{track.confidence:.0%}"
        cv2.putText(
            frame, conf_label,
            (track.x1, track.y2 + 13),
            font, 0.38, (160, 160, 160), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_hud(
        frame: np.ndarray,
        fps: float,
        track_count: int,
        alert_count: int,
    ) -> None:
        """
        Draw a heads-up display strip in the top-left corner with pipeline stats.
        """
        lines = [
            f"FPS: {fps:5.1f}",
            f"Persons: {track_count}",
            f"Alerts:  {alert_count}",
        ]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thick = 1
        pad   = 6
        line_h = 20

        # Semi-transparent dark background panel.
        panel_w = 130
        panel_h = len(lines) * line_h + pad * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            y = pad + (i + 1) * line_h - 4
            cv2.putText(frame, line, (pad, y), font, scale, (0, 220, 255), thick, cv2.LINE_AA)
