"""SQLite cache of captured macOS notification banners.

The Accessibility watcher (notification_watcher.py) only sees banners as they are
drawn — there is no API for notification *history*. So every banner the announcer
captures is recorded here, whether or not it was read aloud, giving the agent a
durable log it can summarise ("5 arrived while you were quiet, 2 from Slack") and
read back on request.

A row is marked ``read`` once it has actually been surfaced to the user — either
read aloud by the announcer, or reported on request via ``recent()``. Anything
captured but never surfaced is "missed".

Banners carry no explicit sender; the banner *title* is the sender/source within
the app (contact for Messages, sender for Mail, channel/sender for Slack), so it
is stored as the sender proxy for the per-app / per-sender digest.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timedelta

try:
    from loguru import logger
except ModuleNotFoundError:  # bare Python (e.g. standalone tests)
    import logging
    logger = logging.getLogger("notification_store")


def _parse_ts(value, end_of_day: bool = False):
    """ISO date or date-time string (local time) → unix epoch, or None. A bare
    date used as an upper bound is bumped to end-of-day so the whole day counts."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if end_of_day and len(s) == 10:   # date-only upper bound → include the whole day
        dt = dt + timedelta(days=1)
    return dt.timestamp()


class NotificationStore:
    def __init__(self, db_path: str, retention_days: int = 30):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS notifications (
                       id    INTEGER PRIMARY KEY AUTOINCREMENT,
                       ts    REAL    NOT NULL,
                       app   TEXT    NOT NULL DEFAULT '',
                       title TEXT    NOT NULL DEFAULT '',   -- banner title ≈ sender
                       text  TEXT    NOT NULL DEFAULT '',   -- full line, for reading out
                       read  INTEGER NOT NULL DEFAULT 0,    -- 1 = read aloud / reported
                       uuid  TEXT    NOT NULL DEFAULT ''    -- stable per-notification id
                   )"""
            )
            # Migrate older DBs that predate the uuid column.
            try:
                self._db.execute(
                    "ALTER TABLE notifications ADD COLUMN uuid TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_notif_ts ON notifications(ts)")
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_notif_uuid ON notifications(uuid)")
        # Trim the cache to the retention window on startup so it can't grow forever.
        self._prune(retention_days)
        # Per-session counters, reset by begin_session() on each client connection.
        self._session_start = time.time()
        self._last_turn_id = self._max_id()

    def _prune(self, max_age_days: int) -> None:
        """Delete notifications older than max_age_days. Called once at startup."""
        if not max_age_days or max_age_days <= 0:
            return
        cutoff = time.time() - max_age_days * 86400
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM notifications WHERE ts < ?", (cutoff,))
        if cur.rowcount:
            logger.info(
                f"Notification cache: pruned {cur.rowcount} row(s) older than {max_age_days}d")

    def _max_id(self) -> int:
        with self._lock:
            return int(self._db.execute(
                "SELECT COALESCE(MAX(id), 0) FROM notifications").fetchone()[0])

    def begin_session(self) -> None:
        """Reset the per-session counters at the start of a client connection, so
        'missed this session' and 'new since last turn' start clean on each load."""
        self._session_start = time.time()
        self._last_turn_id = self._max_id()

    def record(self, app: str, title: str, text: str, uuid: str = "",
               ts: float | None = None) -> int:
        ts = time.time() if ts is None else ts
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO notifications (ts, app, title, text, uuid) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, (app or "").strip(), (title or "").strip(),
                 (text or "").strip(), (uuid or "").strip()),
            )
        return int(cur.lastrowid)

    def has_uuid(self, uuid: str) -> bool:
        """True if a notification with this stable UUID is already recorded — used
        to skip re-announcing the same notification (e.g. when it's re-listed as the
        user opens Notification Center)."""
        uuid = (uuid or "").strip()
        if not uuid:
            return False
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM notifications WHERE uuid = ? LIMIT 1", (uuid,)).fetchone()
        return row is not None

    def mark_read(self, ids) -> None:
        ids = [ids] if isinstance(ids, int) else [int(i) for i in ids if i is not None]
        if not ids:
            return
        with self._lock, self._db:
            self._db.executemany(
                "UPDATE notifications SET read = 1 WHERE id = ?", [(i,) for i in ids])

    def unread_count(self) -> int:
        """Unread ('missed') notifications captured this session — drives the
        client's faint notify-button dot."""
        with self._lock:
            return int(self._db.execute(
                "SELECT COUNT(*) FROM notifications WHERE read = 0 AND ts >= ?",
                (self._session_start,)).fetchone()[0])

    def turn_digest(self) -> dict:
        """The DELTA to surface this turn: banners that arrived since the previous
        turn AND are still unread, aggregated by app and sender. Advancing the
        per-turn marker each call means each banner is reported exactly once (the
        turn after it arrives) — never re-passed turn after turn. ``missed`` is the
        running unread total this session, a bare count for the 'how many overall'
        cue. Banners the announcer already spoke (read=1) are not resurfaced.

        Returns {new, missed, by_app: [{app, count, senders: [{name, count}]}]}
        where new == len(by_app rows) (the delta), missed == total unread.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT app, title FROM notifications "
                "WHERE id > ? AND read = 0 ORDER BY id",
                (self._last_turn_id,)).fetchall()
            missed = int(self._db.execute(
                "SELECT COUNT(*) FROM notifications WHERE read = 0 AND ts >= ?",
                (self._session_start,)).fetchone()[0])
            self._last_turn_id = int(self._db.execute(
                "SELECT COALESCE(MAX(id), 0) FROM notifications").fetchone()[0])
        # Aggregate the delta by app, then by sender (title proxy), most first.
        apps: dict = {}
        for r in rows:
            app = ((r["app"] or "").strip() or "Unknown")
            sender = (r["title"] or "").strip()
            a = apps.setdefault(app, {"app": app, "count": 0, "senders": {}})
            a["count"] += 1
            if sender and sender.lower() != app.lower():
                a["senders"][sender] = a["senders"].get(sender, 0) + 1
        by_app = []
        for a in sorted(apps.values(), key=lambda x: -x["count"]):
            senders = [{"name": n, "count": c}
                       for n, c in sorted(a["senders"].items(), key=lambda kv: -kv[1])]
            by_app.append({"app": a["app"], "count": a["count"], "senders": senders})
        return {"new": len(rows), "missed": missed, "by_app": by_app}

    def search(self, keywords: str = "", date_from=None, date_to=None,
               mark_reported: bool = True, limit: int = 40) -> list[dict]:
        """Most-recent-first captured notifications where ANY keyword matches ANY
        attribute (source app, sender, or text), within a time window. With neither
        date_from nor date_to given the window defaults to the LAST 6 HOURS; else it
        is [date_from or the beginning, date_to or now]. Empty keywords → no keyword
        filter. Marks the returned ones as reported so they stop counting as missed.
        (Keyword-only by design — callers can't target individual fields.)"""
        terms = str(keywords or "").lower().split()
        limit = max(1, min(int(limit or 40), 100))
        lo, hi = _parse_ts(date_from), _parse_ts(date_to, end_of_day=True)
        if lo is None and hi is None:
            hi = time.time()
            lo = hi - 6 * 3600            # default: last 6 hours
        else:
            hi = time.time() if hi is None else hi
            lo = 0.0 if lo is None else lo
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, app, title, text, read FROM notifications "
                "WHERE ts >= ? AND ts <= ? ORDER BY id DESC LIMIT 300",
                (lo, hi)).fetchall()
        items = []
        for r in rows:
            if terms:
                hay = f"{r['app']} {r['title']} {r['text']}".lower()
                if not any(t in hay for t in terms):
                    continue
            items.append(dict(r))
            if len(items) >= limit:
                break
        if mark_reported and items:
            self.mark_read([r["id"] for r in items])
        return items
