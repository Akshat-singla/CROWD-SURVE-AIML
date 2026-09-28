# =============================================================================
# modules/behaviour_buffer.py  (redesigned — no zone logic)
# BehaviourBuffer — per-track sliding centroid window.
#
# Stores a time-ordered list of (cx, cy, timestamp) observations for every
# active track ID.  Automatically trims entries outside the observation window.
# No zone detection — this module is purely a rolling data store.
# =============================================================================

import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single buffer entry — lightweight, no zone fields
# ---------------------------------------------------------------------------

class BufferEntry:
    """
    One timestamped centroid observation for a tracked person.

    Attributes
    ----------
    cx, cy : int        Centroid pixel coordinates (bottom-centre of bbox).
    timestamp : float   Unix wall-clock time when this frame was captured.
    """

    __slots__ = ("cx", "cy", "timestamp")

    def __init__(self, cx: int, cy: int, timestamp: float):
        self.cx        = cx
        self.cy        = cy
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"BufferEntry(cx={self.cx}, cy={self.cy}, ts={self.timestamp:.3f})"


# ---------------------------------------------------------------------------
# Buffer manager
# ---------------------------------------------------------------------------

class BehaviourBuffer:
    """
    Manages per-person sliding observation windows for behaviour analysis.

    Each active track ID owns a list of :class:`BufferEntry` objects ordered
    chronologically.  On every call to :meth:`update`, the entry is appended
    and old entries (outside ``window_sec``) are pruned from the front.

    Parameters
    ----------
    window_sec : float
        Duration (seconds) of each track's observation window.
    min_samples : int
        Minimum entries required before :meth:`is_ready` returns True.
    """

    def __init__(self, window_sec: float = 10.0, min_samples: int = 10):
        self.window_sec  = window_sec
        self.min_samples = min_samples

        # track_id → list of BufferEntry (chronological order)
        self._buffers: Dict[int, List[BufferEntry]] = defaultdict(list)

        logger.info(
            "BehaviourBuffer ready — window=%.1fs, min_samples=%d.",
            self.window_sec, self.min_samples,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, track_id: int, cx: int, cy: int, timestamp: float) -> None:
        """
        Append a new centroid observation and trim expired entries.

        Parameters
        ----------
        track_id : int   Unique tracker ID.
        cx, cy : int     Centroid pixel position.
        timestamp : float  Unix wall-clock time of this frame.
        """
        self._buffers[track_id].append(BufferEntry(cx, cy, timestamp))
        self._trim(track_id, timestamp)

    def get(self, track_id: int) -> List[BufferEntry]:
        """
        Return the current observation list for a track (read-only).

        Returns an empty list if the track has no data.
        """
        return self._buffers.get(track_id, [])

    def is_ready(self, track_id: int) -> bool:
        """
        True when the buffer contains at least ``min_samples`` entries —
        the minimum needed for reliable feature extraction.
        """
        return len(self._buffers.get(track_id, [])) >= self.min_samples

    def remove(self, track_id: int) -> None:
        """Delete the entire buffer for a lost (expired) track ID."""
        self._buffers.pop(track_id, None)

    def remove_stale(self, active_ids: set) -> None:
        """
        Remove buffers for all track IDs *not* in ``active_ids``.

        Call once per frame after the tracker update, passing the set of
        IDs still visible in the current frame.
        """
        stale = set(self._buffers.keys()) - active_ids
        for tid in stale:
            self.remove(tid)
            logger.debug("Buffer removed for lost track ID %d.", tid)

    def active_ids(self) -> List[int]:
        """Return a list of all track IDs currently holding buffer data."""
        return list(self._buffers.keys())

    def size(self, track_id: int) -> int:
        """Number of entries currently buffered for a track."""
        return len(self._buffers.get(track_id, []))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trim(self, track_id: int, current_time: float) -> None:
        """
        Remove entries from the front of the buffer that fall outside
        the ``window_sec`` observation window.

        Always keeps at least one entry so an active track's buffer is
        never completely empty after trimming.
        """
        cutoff = current_time - self.window_sec
        buf    = self._buffers[track_id]

        # Find the index of the first entry that is still within the window.
        first_valid = 0
        for i, entry in enumerate(buf):
            if entry.timestamp >= cutoff:
                first_valid = i
                break
        else:
            # Every entry is older than the window — keep just the last one.
            first_valid = max(0, len(buf) - 1)

        if first_valid > 0:
            del buf[:first_valid]
