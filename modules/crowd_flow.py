"""Rolling two-camera crowd-flow counting and imbalance evaluation."""

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Optional, Dict, Deque, Any


@dataclass
class CrowdFlowMonitor:
    """Count track crossings and compare entry/exit totals over a time window."""

    window_seconds: int = 60
    alert_threshold: int = 5
    _crossings: Dict[str, Deque[float]] = field(
        default_factory=lambda: {"entry": deque(), "exit": deque()}
    )
    _last_side: Dict[str, Dict[int, int]] = field(
        default_factory=lambda: {"entry": {}, "exit": {}}
    )
    _lock: Lock = field(default_factory=Lock)

    def update_tracks(
        self,
        camera_role: str,
        tracks: list,
        frame_width: int,
        direction: str,
        timestamp: Optional[float] = None,
    ) -> int:
        """Record one event per person as their centroid crosses the center line."""
        if camera_role not in self._crossings:
            raise ValueError("camera_role must be 'entry' or 'exit'")
        if direction not in {"left_to_right", "right_to_left"}:
            raise ValueError("direction must be left_to_right or right_to_left")
        if frame_width < 2:
            raise ValueError("frame_width must be at least 2")

        now = monotonic() if timestamp is None else timestamp
        center = frame_width / 2
        hysteresis = frame_width * 0.025
        desired_sign = 1 if direction == "left_to_right" else -1
        added = 0
        visible_ids = {track.track_id for track in tracks}

        with self._lock:
            sides = self._last_side[camera_role]
            for track in tracks:
                x = track.cx
                if x < center - hysteresis:
                    side = -1
                elif x > center + hysteresis:
                    side = 1
                else:
                    continue

                previous = sides.get(track.track_id)
                if previous is not None and side != previous:
                    crossing_sign = 1 if side > previous else -1
                    if crossing_sign == desired_sign:
                        self._crossings[camera_role].append(now)
                        added += 1
                sides[track.track_id] = side

            stale = set(sides) - visible_ids
            for track_id in stale:
                sides.pop(track_id, None)

            self._prune_locked(now)
        return added

    def snapshot(self, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Return rolling counts and whether the configured mismatch is exceeded."""
        now = monotonic() if timestamp is None else timestamp
        with self._lock:
            self._prune_locked(now)
            entry = len(self._crossings["entry"])
            exit_count = len(self._crossings["exit"])
        difference = entry - exit_count
        return {
            "entry_count": entry,
            "exit_count": exit_count,
            "difference": difference,
            "absolute_difference": abs(difference),
            "window_seconds": self.window_seconds,
            "alert_threshold": self.alert_threshold,
            "alert_active": abs(difference) >= self.alert_threshold,
        }

    def reset(self) -> None:
        with self._lock:
            for events in self._crossings.values():
                events.clear()
            for tracks in self._last_side.values():
                tracks.clear()

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for events in self._crossings.values():
            while events and events[0] < cutoff:
                events.popleft()
