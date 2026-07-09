"""macOS notification watcher via the Accessibility API (Option B).

Observes the banner windows created by the Notification Center UI process and
extracts their text, so the voice agent can read new notifications aloud. It
uses the AXObserver C API, whose callbacks are delivered on a CFRunLoop — so
the watcher owns a dedicated thread running that run loop and hands each
captured notification back to the asyncio world through a thread-safe callback.

Requires Accessibility permission (System Settings -> Privacy & Security ->
Accessibility) for whatever process launches Python. Focus / Do Not Disturb
already suppress banners, so nothing surfaces here while the user is in a Focus
— which is exactly the desired "stay quiet" behaviour.

Probe standalone (no bot):
    python notification_watcher.py          # prints extracted banner text
    python notification_watcher.py --dump   # prints the full AX subtree too
Then fire a test banner from another terminal:
    osascript -e 'display notification "hello there" with title "Test" subtitle "Sub"'
"""
from __future__ import annotations

import sys
import threading

try:
    from loguru import logger
except ModuleNotFoundError:  # standalone probe under a bare Python (no venv)
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("notification_watcher")

try:
    import objc
    import ApplicationServices as _AX
    import CoreFoundation as _CF
    from AppKit import NSWorkspace
    _AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 — PyObjC frameworks may be absent
    _AVAILABLE = False
    _IMPORT_ERROR = _exc

# The process that renders notification banners.
_NC_BUNDLE = "com.apple.notificationcenterui"

# AX attribute / role / notification names are stable string constants.
_ATTR_ROLE = "AXRole"
_ATTR_SUBROLE = "AXSubrole"
_ATTR_CHILDREN = "AXChildren"
_ATTR_VALUE = "AXValue"
_ATTR_TITLE = "AXTitle"
_ATTR_DESC = "AXDescription"
_ATTR_IDENTIFIER = "AXIdentifier"
_TEXT_ROLES = {"AXStaticText"}
# Each banner is an AXGroup with this subrole; its AXDescription reads
# "App Name, title, subtitle, body" and its static texts are tagged with
# AXIdentifier "title" / "subtitle" / "body".
_BANNER_SUBROLE = "AXNotificationCenterBanner"
_FIELD_IDS = ("title", "subtitle", "body")
_BANNER_NOTIFS = ("AXWindowCreated",)  # each banner is a new NotificationCenter window
_MAX_DEPTH = 16


def accessibility_trusted(prompt: bool = False) -> bool:
    """Whether this process may read other apps' AX trees. prompt=True asks the
    user to grant it (opens the System Settings pane)."""
    if not _AVAILABLE:
        return False
    key = getattr(_AX, "kAXTrustedCheckOptionPrompt", "AXTrustedCheckOptionPrompt")
    return bool(_AX.AXIsProcessTrustedWithOptions({key: bool(prompt)}))


def _copy(el, attr):
    try:
        err, val = _AX.AXUIElementCopyAttributeValue(el, attr, None)
        return val if err == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _collect_texts(el, out: list[str], depth: int = 0) -> None:
    """Depth-first gather of every static-text string under an element, in
    order, de-duplicated. (Fallback for banners without tagged fields.)"""
    if el is None or depth > _MAX_DEPTH:
        return
    if _copy(el, _ATTR_ROLE) in _TEXT_ROLES:
        for attr in (_ATTR_VALUE, _ATTR_TITLE, _ATTR_DESC):
            v = _copy(el, attr)
            if isinstance(v, str) and v.strip():
                s = " ".join(v.split())
                if s not in out:
                    out.append(s)
                break
    for child in (_copy(el, _ATTR_CHILDREN) or []):
        _collect_texts(child, out, depth + 1)


def _collect_fields(el, fields: dict, depth: int = 0) -> None:
    """Gather static texts tagged title/subtitle/body by their AXIdentifier."""
    if el is None or depth > _MAX_DEPTH:
        return
    if _copy(el, _ATTR_ROLE) in _TEXT_ROLES:
        ident = _copy(el, _ATTR_IDENTIFIER)
        val = _copy(el, _ATTR_VALUE)
        if ident in _FIELD_IDS and isinstance(val, str) and val.strip():
            fields.setdefault(ident, " ".join(val.split()))
    for child in (_copy(el, _ATTR_CHILDREN) or []):
        _collect_fields(child, fields, depth + 1)


def _extract_banners(el, out: list[dict], depth: int = 0) -> None:
    """Find each notification banner under an element and pull structured
    fields: {app, title, subtitle, body}. The app name is the first field of
    the banner group's AXDescription ("App, title, subtitle, body")."""
    if el is None or depth > _MAX_DEPTH:
        return
    if _copy(el, _ATTR_SUBROLE) == _BANNER_SUBROLE:
        fields: dict = {}
        _collect_fields(el, fields)
        desc = _copy(el, _ATTR_DESC) or ""
        app = desc.split(", ", 1)[0].strip() if desc else ""
        # If the app name coincides with the title, there was no app prefix.
        if app and app == fields.get("title"):
            app = ""
        banner = {"app": app, "title": fields.get("title", ""),
                  "subtitle": fields.get("subtitle", ""), "body": fields.get("body", "")}
        if not (banner["title"] or banner["body"]):
            # Untagged banner: fall back to positional static texts.
            texts: list[str] = []
            _collect_texts(el, texts)
            banner["title"] = texts[0] if texts else ""
            banner["body"] = " ".join(texts[1:]) if len(texts) > 1 else ""
        if banner["title"] or banner["body"] or banner["app"]:
            out.append(banner)
        return  # don't descend into a banner we've already captured
    for child in (_copy(el, _ATTR_CHILDREN) or []):
        _extract_banners(child, out, depth + 1)


def _dump_tree(el, depth: int = 0, lines: list[str] | None = None) -> list[str]:
    if lines is None:
        lines = []
    if el is None or depth > _MAX_DEPTH:
        return lines
    lines.append("  " * depth + f"{_copy(el, _ATTR_ROLE)} "
                 f"title={_copy(el, _ATTR_TITLE)!r} value={_copy(el, _ATTR_VALUE)!r}")
    for child in (_copy(el, _ATTR_CHILDREN) or []):
        _dump_tree(child, depth + 1, lines)
    return lines


class NotificationWatcher:
    """Watches Notification Center banners and calls ``on_notification(texts)``
    on the watcher thread for each one. The caller marshals to its own loop."""

    def __init__(self, on_notification, dump: bool = False):
        self._cb = on_notification
        self._dump = dump
        self._thread: threading.Thread | None = None
        self._runloop = None
        self._observer = None   # kept alive so it isn't GC'd
        self._app_el = None
        self._pid = None
        self._stop = False

    def start(self) -> bool:
        if not _AVAILABLE:
            logger.warning(f"Notification watcher unavailable (PyObjC import failed): {_IMPORT_ERROR}")
            return False
        if not accessibility_trusted(prompt=True):
            logger.warning(
                "Notification watcher: Accessibility permission NOT granted. Enable it in "
                "System Settings -> Privacy & Security -> Accessibility for the app running "
                "this process (Terminal / your IDE), then restart."
            )
            return False
        self._pid = self._find_nc_pid()
        if self._pid is None:
            logger.warning(f"Notification watcher: {_NC_BUNDLE} not running; cannot observe banners")
            return False
        self._thread = threading.Thread(target=self._run, name="notif-ax-watcher", daemon=True)
        self._thread.start()
        return True

    @staticmethod
    def _find_nc_pid():
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if app.bundleIdentifier() == _NC_BUNDLE:
                return app.processIdentifier()
        return None

    def _handle(self, observer, element, notification, refcon):
        # Runs on the watcher thread's run loop. Keep it quick.
        try:
            if self._dump:
                logger.info("AX banner subtree:\n" + "\n".join(_dump_tree(element)))
            banners: list[dict] = []
            _extract_banners(element, banners)
            for banner in banners:
                self._cb(banner)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Notification extract failed: {exc}")

    def _run(self):
        self._app_el = _AX.AXUIElementCreateApplication(self._pid)

        # PyObjC needs the C-callback tagged with its signature (callbackFor),
        # and the thunk kept alive as long as the observer uses it.
        @objc.callbackFor(_AX.AXObserverCreate)
        def _observer_cb(observer, element, notification, refcon):
            self._handle(observer, element, notification, refcon)

        self._cb_thunk = _observer_cb
        err, observer = _AX.AXObserverCreate(self._pid, _observer_cb, None)
        if err != 0 or observer is None:
            logger.warning(f"Notification watcher: AXObserverCreate failed ({err})")
            return
        self._observer = observer
        for notif in _BANNER_NOTIFS:
            e = _AX.AXObserverAddNotification(observer, self._app_el, notif, None)
            if e != 0:
                logger.debug(f"AXObserverAddNotification({notif}) -> {e}")
        self._runloop = _CF.CFRunLoopGetCurrent()
        _CF.CFRunLoopAddSource(
            self._runloop, _AX.AXObserverGetRunLoopSource(observer), _CF.kCFRunLoopDefaultMode
        )
        logger.info("Notification watcher: observing Notification Center banners")
        # RunInMode (vs CFRunLoopRun) lets us poll the stop flag for clean shutdown.
        while not self._stop:
            _CF.CFRunLoopRunInMode(_CF.kCFRunLoopDefaultMode, 1.0, False)
        logger.info("Notification watcher stopped")

    def stop(self):
        self._stop = True
        if self._runloop is not None:
            try:
                _CF.CFRunLoopStop(self._runloop)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":  # standalone probe
    import time

    dump = "--dump" in sys.argv
    if not accessibility_trusted(prompt=True):
        print("Grant Accessibility permission to this terminal, then re-run.")
        sys.exit(1)

    def _print(banner):
        print(f"NOTIFICATION  app={banner['app']!r} title={banner['title']!r} "
              f"subtitle={banner['subtitle']!r} body={banner['body']!r}")

    w = NotificationWatcher(_print, dump=dump)
    if not w.start():
        sys.exit(1)
    print("Watching. Fire a test banner:\n"
          "  osascript -e 'display notification \"hello\" with title \"Test\" subtitle \"Sub\"'\n"
          "Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        w.stop()
