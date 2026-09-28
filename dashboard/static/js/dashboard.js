/* =============================================================================
   dashboard/static/js/dashboard.js
   Client-side logic for the AI Surveillance Dashboard.

   Responsibilities:
     • Poll /status every 1 s  → update header status badges.
     • Poll /events every 3 s  → refresh the alert event table.
     • Populate evidence snapshot thumbnails from event data.
     • Filter chips for activity-type filtering.
     • Lightbox for full-size snapshot preview.
     • Hide video overlay once the MJPEG stream loads.
   ============================================================================= */

'use strict';

// ── Configuration ────────────────────────────────────────────────────────────
const STATUS_INTERVAL_MS  = 1000;   // how often to fetch /status (ms)
const EVENTS_INTERVAL_MS  = 3000;   // how often to fetch /events  (ms)
const MAX_SNAP_THUMBS     = 8;      // max thumbnails in snapshot strip

// ── State ─────────────────────────────────────────────────────────────────────
let activeFilter  = 'all';          // currently selected activity filter
let allEvents     = [];             // full event list from server
let knownSnapshots = new Set();     // snapshot paths already rendered

// ── DOM refs (resolved once on DOMContentLoaded) ─────────────────────────────
let refs = {};

document.addEventListener('DOMContentLoaded', () => {
  refs = {
    statusDot:      document.getElementById('status-dot'),
    statusLabel:    document.getElementById('status-label'),
    statTracks:     document.getElementById('stat-tracks'),
    statFps:        document.getElementById('stat-fps'),
    statAlerts:     document.getElementById('stat-alerts'),
    statUptime:     document.getElementById('stat-uptime'),
    eventsBody:     document.getElementById('events-body'),
    noEventsMsg:    document.getElementById('no-events-msg'),
    eventCountLbl:  document.getElementById('event-count-label'),
    lastRefreshLbl: document.getElementById('last-refresh-label'),
    snapshotStrip:  document.getElementById('snapshot-strip'),
    noSnapMsg:      document.getElementById('no-snapshot-msg'),
    videoOverlay:   document.getElementById('video-overlay'),
    videoStream:    document.getElementById('video-stream'),
    lightbox:       document.getElementById('lightbox'),
    lightboxImg:    document.getElementById('lightbox-img'),
    lightboxCap:    document.getElementById('lightbox-caption'),
  };

  // Hide video overlay once the MJPEG img fires its first load event.
  refs.videoStream.addEventListener('load', () => {
    refs.videoOverlay.classList.add('hidden');
  });

  // Start polling loops.
  fetchStatus();
  fetchEvents();
  setInterval(fetchStatus,  STATUS_INTERVAL_MS);
  setInterval(fetchEvents,  EVENTS_INTERVAL_MS);
});


// ══════════════════════════════════════════════════════════ STATUS POLLING ══

/**
 * Fetch /status and update the header badge values.
 */
async function fetchStatus() {
  try {
    const res  = await fetch('/status');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Running indicator
    const isRunning = data.pipeline_running;
    refs.statusDot.classList.toggle('offline', !isRunning);
    refs.statusLabel.textContent = isRunning ? 'Live' : 'Offline';

    // Counters
    refs.statTracks.textContent = data.active_tracks ?? 0;
    refs.statFps.textContent    = (data.fps ?? 0).toFixed(1);
    refs.statAlerts.textContent = data.total_alerts ?? 0;
    refs.statUptime.textContent = formatUptime(data.uptime_seconds ?? 0);

  } catch (err) {
    // Network failure: mark as offline without throwing.
    refs.statusDot.classList.add('offline');
    refs.statusLabel.textContent = 'Offline';
    console.warn('[status] fetch failed:', err.message);
  }
}


// ══════════════════════════════════════════════════════════ EVENTS POLLING ══

/**
 * Fetch /events, store the result, and re-render the table + snapshot strip.
 */
async function fetchEvents() {
  try {
    const res  = await fetch('/events');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allEvents  = await res.json();

    renderEventTable(allEvents);
    renderSnapshotStrip(allEvents);

    // Footer timestamps.
    refs.eventCountLbl.textContent  = `${allEvents.length} alert${allEvents.length !== 1 ? 's' : ''} logged`;
    refs.lastRefreshLbl.textContent = `Updated ${new Date().toLocaleTimeString()}`;

  } catch (err) {
    console.warn('[events] fetch failed:', err.message);
  }
}


// ══════════════════════════════════════════════════════════ EVENT TABLE ══════

/**
 * Render the event log table, respecting the current activity filter.
 *
 * @param {Array<Object>} events - full event array from /events
 */
function renderEventTable(events) {
  const filtered = activeFilter === 'all'
    ? events
    : events.filter(e => e.activity_type === activeFilter);

  if (filtered.length === 0) {
    refs.eventsBody.innerHTML = '';
    refs.noEventsMsg.style.display = 'block';
    return;
  }
  refs.noEventsMsg.style.display = 'none';

  const rows = filtered.map(ev => `
    <tr>
      <td title="${ev.datetime_utc || ''}">${formatTime(ev.datetime_utc || ev.timestamp)}</td>
      <td><strong>#${ev.track_id ?? '—'}</strong></td>
      <td>${activityBadge(ev.activity_type)}</td>
      <td>${ev.zone_name && ev.zone_name !== 'N/A' ? ev.zone_name : '<span style="color:var(--text-muted)">—</span>'}</td>
      <td>${ev.snapshot_path
            ? `<a class="snap-link" href="#" onclick="openLightbox('${encodeURIComponent(ev.snapshot_path)}', '${escapeHtml(ev.activity_type)} — ID ${ev.track_id}'); return false;" title="View snapshot">🖼</a>`
            : '<span style="color:var(--text-muted)">—</span>'
          }</td>
    </tr>
  `).join('');

  refs.eventsBody.innerHTML = rows;
}

/**
 * Return the HTML for a colour-coded activity badge.
 * @param {string} activity
 */
function activityBadge(activity) {
  const map = {
    'Loitering':                  'badge-loiter',
    'Running / Sudden Motion':    'badge-running',
    'Restricted Area Intrusion':  'badge-intrusion',
    'Normal Movement':            'badge-normal',
  };
  const cls = map[activity] || 'badge-normal';
  return `<span class="activity-badge ${cls}">${escapeHtml(activity ?? 'Unknown')}</span>`;
}


// ══════════════════════════════════════════════════════ SNAPSHOT STRIP ══════

/**
 * Populate the snapshot thumbnail strip with evidence images.
 * New thumbnails are prepended; already-rendered ones are skipped.
 *
 * @param {Array<Object>} events
 */
function renderSnapshotStrip(events) {
  // Collect events with a snapshot path, newest first.
  const withSnaps = events
    .filter(e => e.snapshot_path)
    .slice(0, MAX_SNAP_THUMBS);

  if (withSnaps.length === 0) {
    refs.noSnapMsg.style.display = 'block';
    return;
  }
  refs.noSnapMsg.style.display = 'none';

  // Remove the "no snapshots" placeholder text node.
  refs.snapshotStrip.innerHTML = '';

  withSnaps.forEach(ev => {
    const filename = ev.snapshot_path.split(/[\\/]/).pop();
    const thumb = document.createElement('div');
    thumb.className = 'snap-thumb';
    thumb.title     = `${ev.activity_type} — ID ${ev.track_id}`;
    thumb.innerHTML = `
      <img
        src="/snapshots/${encodeURIComponent(filename)}"
        alt="${escapeHtml(ev.activity_type)}"
        loading="lazy"
        onerror="this.parentElement.style.display='none'"
      />
      <div class="snap-label">#${ev.track_id} ${escapeHtml(ev.activity_type)}</div>
    `;
    thumb.addEventListener('click', () => {
      openLightbox(
        encodeURIComponent(filename),
        `${ev.activity_type} — ID ${ev.track_id} @ ${formatTime(ev.datetime_utc || ev.timestamp)}`
      );
    });
    refs.snapshotStrip.appendChild(thumb);
  });
}


// ══════════════════════════════════════════════════════════════ FILTER ══════

/**
 * Called by the filter chip buttons.
 * @param {HTMLElement} chipEl - the clicked chip element
 * @param {string}      value  - activity value or 'all'
 */
function setFilter(chipEl, value) {
  activeFilter = value;

  // Update chip active state.
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  chipEl.classList.add('active');

  renderEventTable(allEvents);
}

/** Reset filter to show all events. */
function clearFilter() {
  setFilter(document.querySelector('.chip[data-activity="all"]'), 'all');
}


// ══════════════════════════════════════════════════════════ LIGHTBOX ════════

/**
 * Open the snapshot lightbox.
 * @param {string} encodedFilename - URL-encoded filename
 * @param {string} caption
 */
function openLightbox(encodedFilename, caption) {
  refs.lightboxImg.src      = `/snapshots/${encodedFilename}`;
  refs.lightboxCap.textContent = caption;
  refs.lightbox.classList.add('open');
  document.body.style.overflow = 'hidden';
}

/** Close the lightbox. */
function closeLightbox() {
  refs.lightbox.classList.remove('open');
  refs.lightboxImg.src         = '';
  document.body.style.overflow = '';
}

// Close lightbox on Escape key.
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});


// ══════════════════════════════════════════════════════════ UTILITIES ════════

/**
 * Format a seconds integer as mm:ss or hh:mm:ss.
 * @param {number} totalSeconds
 */
function formatUptime(totalSeconds) {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${pad(h)}:${pad(m)}:${pad(s)}`;
  return `${pad(m)}:${pad(s)}`;
}

/**
 * Format a datetime_utc string or Unix timestamp to a short local time.
 * @param {string|number} value
 */
function formatTime(value) {
  try {
    const d = typeof value === 'number'
      ? new Date(value * 1000)
      : new Date(value + (value.endsWith('Z') ? '' : 'Z'));
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return String(value).slice(0, 8);
  }
}

/** Zero-pad a number to 2 digits. */
const pad = n => String(n).padStart(2, '0');

/** Escape HTML to prevent XSS in dynamically built table rows. */
function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
