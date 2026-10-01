# =============================================================================
# modules/alert_logger.py  (Enhanced — annotated violation snapshots)
# AlertLogger — snapshot saving, CSV logging, and in-memory event store.
#
# Enhancement 1: Violation-Highlighted Snapshots
#   All snapshot rendering now happens exclusively in Thread 3 (the worker).
#   The raw (un-annotated) frame and the full track list are passed via the
#   job queue and the final JPEG is composed here with:
#     • Thin green boxes for all non-violating persons.
#     • Thick coloured box (4 px) + filled label for the violating person.
#     • Semi-transparent coloured overlay (30 % opacity) inside violating box.
#     • Bottom-left timestamp + session info footer text.
#   Snapshots are saved at JPEG quality 95 (vs 75 for the stream).
#
# Enhancement 2: Memory cleanup
#   After saving a snapshot the local frame reference is explicitly set to None
#   and deleted so CPython can reclaim the numpy buffer immediately.
#
# Design (unchanged):
#   • AlertLogger is a passive data store + disk writer.
#   • Thread 2 submits jobs to log_queue (maxsize=50 — drops if full).
#   • run_worker() runs as Thread 3 — drains the queue, does all I/O there.
#   • In-memory deque(maxlen=EVENT_LOG_MAXLEN) is protected by threading.Lock.
# =============================================================================

import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CSV column schema.
_COLUMNS = ["event_id", "track_id", "violation_type", "timestamp", "snapshot_path"]

# Violation-type → (BGR colour, box thickness) for snapshot rendering.
# These match the colours defined in config.VIOLATION_COLOR_MAP.
_VIOLATION_STYLE: Dict[str, tuple] = {
    "Running / Sudden Motion": ((0, 0, 220),   4),   # bright red
    "Loitering":               ((0, 140, 255), 4),   # orange
    "Suspicious Lingering":    ((128, 0, 200), 4),   # purple
    "Weapon Detected":         ((0, 0, 255),   4),   # bright red
    "Abandoned Bag":           ((0, 140, 255), 4),   # orange
    "Possible Theft":          ((255, 0, 255), 4),   # magenta
}
_DEFAULT_VIOLATION_STYLE = ((0, 0, 220), 4)          # fallback bright red


class AlertLogger:
    """
    Manages violation event logging and snapshot saving — all disk writes
    happen exclusively in Thread 3 (the worker thread).

    Parameters
    ----------
    snapshot_dir      : str    Directory for JPEG snapshots.
    log_dir           : str    Directory for the CSV event log.
    log_filename      : str    CSV filename.
    event_log_maxlen  : int    Maximum events kept in-memory.
    log_queue_maxsize : int    Hard cap on the job queue (drops if full).
    snapshot_quality  : int    JPEG quality for snapshots (default 95).
    """

    def __init__(
        self,
        snapshot_dir:      str,
        log_dir:           str,
        log_filename:      str = "event_log.csv",
        event_log_maxlen:  int = 200,
        log_queue_maxsize: int = 50,
        snapshot_quality:  int = 95,
    ):
        self.snapshot_dir     = snapshot_dir
        self.log_dir          = log_dir
        self.csv_path         = os.path.join(log_dir, log_filename)
        self.event_log_maxlen = event_log_maxlen
        self.snapshot_quality = snapshot_quality

        os.makedirs(snapshot_dir, exist_ok=True)
        os.makedirs(log_dir,      exist_ok=True)

        # ── In-memory event log (thread-safe bounded deque) ───────────
        self._events: deque = deque(maxlen=event_log_maxlen)
        self._events_lock   = threading.Lock()

        # ── Auto-increment event ID ────────────────────────────────────
        self._next_id = self._load_next_id()

        # ── Job queue — maxsize prevents unbounded memory growth ───────
        # If full, submit() will drop the job with a warning (non-blocking).
        self.log_queue: queue.Queue = queue.Queue(maxsize=log_queue_maxsize)

        logger.info(
            "AlertLogger ready — snapshots='%s', csv='%s', next_id=%d, "
            "queue_maxsize=%d.",
            snapshot_dir, self.csv_path, self._next_id, log_queue_maxsize,
        )

    # ------------------------------------------------------------------
    # Thread 3 — worker entry point
    # ------------------------------------------------------------------

    def run_worker(self, stop_event: threading.Event) -> None:
        """
        Long-running worker that drains the log_queue and performs disk I/O.

        Runs in Thread 3.  Blocks on log_queue.get() with a short timeout
        so it wakes up promptly when stop_event is set.

        Each job dict contains:
            {
                "track_id":       int,
                "violation":      str,
                "timestamp":      float,       (Unix time)
                "raw_frame":      np.ndarray,  (un-annotated BGR — for snapshot)
                "all_tracks":     list,         (Track objects for all persons)
                "session_start":  Optional[float], (session start Unix time)
            }
        """
        logger.info("Log worker thread started.")
        while not stop_event.is_set() or not self.log_queue.empty():
            try:
                job = self.log_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_job(job)
            except Exception as exc:
                logger.error("Log worker error: %s", exc, exc_info=True)
            finally:
                self.log_queue.task_done()

        logger.info("Log worker thread stopped.")

    # ------------------------------------------------------------------
    # Public — called by Thread 2 (non-blocking)
    # ------------------------------------------------------------------

    def submit(
        self,
        track_id:      int,
        violation:     str,
        timestamp:     float,
        raw_frame:     np.ndarray,   # un-annotated BGR frame
        all_tracks:    list,         # List[Track] of ALL active persons
        session_start: Optional[float] = None,
    ) -> None:
        """
        Enqueue a logging job for Thread 3 to process asynchronously.

        Returns immediately — never blocks Thread 2.

        Parameters
        ----------
        track_id      : Unique tracker ID of the violating person.
        violation     : Violation label string.
        timestamp     : Unix time of the frame.
        raw_frame     : Un-annotated BGR frame (copied internally).
        all_tracks    : All currently active Track objects.
        session_start : Unix time when the current session started (for footer).
        """
        job = {
            "track_id":      track_id,
            "violation":     violation,
            "timestamp":     timestamp,
            "raw_frame":     raw_frame.copy(),  # copy so Thread 2 can reuse immediately
            "all_tracks":    list(all_tracks),  # shallow copy of the list
            "session_start": session_start,
        }
        try:
            self.log_queue.put_nowait(job)
        except queue.Full:
            logger.warning(
                "log_queue full — dropping violation job for track_id=%d (%s).",
                track_id, violation,
            )

    def get_recent_events(self, n: int = 50) -> List[dict]:
        """
        Return the most recent n events (newest first).  Thread-safe.
        """
        with self._events_lock:
            events = list(self._events)
        return list(reversed(events))[:n]

    def reset(self) -> None:
        """
        Clear all in-memory state and wipe the snapshot directory.
        Safe to call between sessions without recreating the instance.
        """
        import shutil
        with self._events_lock:
            self._events.clear()
        if os.path.exists(self.snapshot_dir):
            shutil.rmtree(self.snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        print("[INFO] AlertLogger reset — events cleared, snapshots wiped.")
        logger.info("AlertLogger reset.")

    @property
    def total_events(self) -> int:
        """Total events logged since startup."""
        with self._events_lock:
            return self._next_id - 1

    # ------------------------------------------------------------------
    # Private — run only inside Thread 3
    # ------------------------------------------------------------------

    def _process_job(self, job: dict) -> None:
        """
        Build the annotated violation snapshot, save to disk, write CSV,
        and update the in-memory event log.

        All rendering happens here in Thread 3 — never in Thread 2.
        """
        track_id      = job["track_id"]
        violation     = job["violation"]
        ts            = job["timestamp"]
        raw_frame     = job["raw_frame"]
        all_tracks    = job["all_tracks"]
        session_start = job.get("session_start")

        event_id = self._next_id
        self._next_id += 1

        dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        # ── Render violation-highlighted snapshot ─────────────────────
        snapshot_frame = self._render_violation_snapshot(
            raw_frame=raw_frame,
            violating_track_id=track_id,
            violation_type=violation,
            all_tracks=all_tracks,
            timestamp_str=dt_str,
            session_start=session_start,
        )

        # ── Save snapshot JPEG at quality 95 (evidence grade) ─────────
        safe_vtype  = violation.lower().replace(" ", "_").replace("/", "_")
        snap_name   = f"snapshot_{event_id}_{track_id}_{safe_vtype}_{int(ts)}.jpg"
        snap_path   = os.path.join(self.snapshot_dir, snap_name)
        snapshot_ok = False

        try:
            snapshot_ok = cv2.imwrite(
                snap_path,
                snapshot_frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.snapshot_quality],
            )
            if not snapshot_ok:
                logger.warning("cv2.imwrite failed for '%s'.", snap_path)
                snap_path = ""
        except Exception as exc:
            logger.error("Snapshot save error: %s", exc)
            snap_path = ""

        # Free the rendered snapshot immediately after saving
        snapshot_frame = None
        del snapshot_frame

        # Free the raw frame reference held in this job
        raw_frame = None
        del raw_frame

        # ── Build event record ─────────────────────────────────────────
        event = {
            "event_id":       event_id,
            "track_id":       track_id,
            "violation_type": violation,
            "timestamp":      dt_str,
            "snapshot_path":  snap_path,
        }

        # ── Append to CSV ──────────────────────────────────────────────
        try:
            row_df = pd.DataFrame([event], columns=_COLUMNS)
            write_header = not os.path.isfile(self.csv_path)
            row_df.to_csv(self.csv_path, mode="a", header=write_header, index=False)
        except Exception as exc:
            logger.error("CSV write error: %s", exc)

        # ── Update in-memory deque ─────────────────────────────────────
        with self._events_lock:
            self._events.append(event)

        logger.warning(
            "VIOLATION | Track %-4d | %-28s | %s",
            track_id, violation, dt_str,
        )

    # ------------------------------------------------------------------
    # Snapshot rendering helpers (Thread 3 only)
    # ------------------------------------------------------------------

    def _render_violation_snapshot(
        self,
        raw_frame:           np.ndarray,
        violating_track_id:  int,
        violation_type:      str,
        all_tracks:          list,
        timestamp_str:       str,
        session_start:       Optional[float],
    ) -> np.ndarray:
        """
        Compose a violation-highlighted frame from the raw (un-annotated) BGR image.

        Rendering rules
        ---------------
        Non-violating persons:
            • Thin (2 px) green bounding box.
            • Small ID label above the box.

        Violating person:
            • Thick (4 px) coloured box (colour depends on violation type).
            • Filled coloured rectangle label above the box showing
              "ID:{id} | {violation_type}".
            • 30 % opacity coloured overlay filling the bounding box interior.

        Footer:
            • Timestamp and session info in the bottom-left corner.
        """
        out = raw_frame.copy()
        h, w = out.shape[:2]

        # Determine violation style
        color, box_thickness = _VIOLATION_STYLE.get(
            violation_type, _DEFAULT_VIOLATION_STYLE
        )

        normal_color = (0, 200, 0)   # green  (BGR)

        # Identify the violating person's data from the track list
        violating_track = None
        for t in all_tracks:
            if t.track_id == violating_track_id:
                violating_track = t
                break

        # ── Draw non-violating persons ─────────────────────────────────
        for t in all_tracks:
            if t.track_id == violating_track_id:
                continue  # drawn separately below

            # Thin green box
            cv2.rectangle(out, (t.x1, t.y1), (t.x2, t.y2), normal_color, 2)

            # Small ID label above the box
            label      = f"ID:{t.track_id}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.40
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            ly = max(t.y1 - 4, th + 2)
            cv2.rectangle(out, (t.x1, ly - th - 2), (t.x1 + tw + 2, ly + 2),
                          normal_color, cv2.FILLED)
            cv2.putText(out, label, (t.x1, ly),
                        font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

        # ── Draw violating person ──────────────────────────────────────
        if violating_track is not None:
            vx1, vy1 = violating_track.x1, violating_track.y1
            vx2, vy2 = violating_track.x2, violating_track.y2

            # 30 % coloured overlay inside the bounding box
            overlay = out.copy()
            cv2.rectangle(overlay, (vx1, vy1), (vx2, vy2), color, cv2.FILLED)
            cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)

            # Thick coloured bounding box (4 px)
            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), color, box_thickness)

            # Filled label rectangle above the box
            label_txt  = f"ID:{violating_track.track_id} | {violation_type}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thick = 2
            (tw, th), _ = cv2.getTextSize(label_txt, font, font_scale, font_thick)
            lx = vx1
            ly = max(vy1 - 6, th + 6)
            pad = 4
            cv2.rectangle(out,
                          (lx - pad, ly - th - pad),
                          (lx + tw + pad, ly + pad),
                          color, cv2.FILLED)
            cv2.putText(out, label_txt, (lx, ly),
                        font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        # ── Bottom-left footer ─────────────────────────────────────────
        footer_font  = cv2.FONT_HERSHEY_SIMPLEX
        footer_scale = 0.44
        footer_thick = 1
        footer_color = (200, 200, 200)  # light grey

        if session_start is not None:
            elapsed = int(time.time() - session_start)
            hh, rem = divmod(elapsed, 3600)
            mm, ss  = divmod(rem, 60)
            session_str = f"Session: {hh:02d}:{mm:02d}:{ss:02d}"
        else:
            session_str = "Session: --:--:--"

        lines = [timestamp_str, session_str]
        margin = 8
        line_h = 18

        for i, line in enumerate(reversed(lines)):
            y_pos = h - margin - i * line_h
            # Subtle shadow for readability
            cv2.putText(out, line, (margin + 1, y_pos + 1),
                        footer_font, footer_scale, (0, 0, 0), footer_thick + 1, cv2.LINE_AA)
            cv2.putText(out, line, (margin, y_pos),
                        footer_font, footer_scale, footer_color, footer_thick, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    # Private — CSV bootstrap
    # ------------------------------------------------------------------

    def _load_next_id(self) -> int:
        """Read the existing CSV to determine the starting event_id."""
        if os.path.isfile(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path, usecols=["event_id"])
                if not df.empty:
                    return int(df["event_id"].max()) + 1
            except Exception:
                pass
        return 1
