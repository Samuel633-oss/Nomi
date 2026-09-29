"""Nomi push notifications (via ntfy.sh).

How it works
------------
* You pick a private topic name and subscribe to it in the free ntfy app
  (Android / iOS) or at https://ntfy.sh/<your-topic> in a browser.
* Nomi sends a push to that topic through a plain HTTP request. No extra
  Python packages are needed.
* A small background thread checks once a minute. At your chosen time each day
  it sends one digest of tasks that are due today and overdue.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

import database as db

DEFAULT_SERVER = "https://ntfy.sh"
DEFAULT_TIME = "08:00"

_lock = threading.Lock()
_thread: Optional[threading.Thread] = None


class NotificationError(Exception):
    """A push problem that can be shown to the user as-is."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def get_config() -> dict:
    return {
        "enabled": db.get_setting("push_enabled", "0") == "1",
        "server": db.get_setting("push_server", DEFAULT_SERVER) or DEFAULT_SERVER,
        "topic": db.get_setting("push_topic", ""),
        "time": db.get_setting("push_time", DEFAULT_TIME) or DEFAULT_TIME,
    }


def save_config(enabled: bool, server: str, topic: str, send_time: str) -> None:
    topic = (topic or "").strip()
    server = (server or DEFAULT_SERVER).strip().rstrip("/")
    if not server.startswith(("http://", "https://")):
        raise NotificationError("The server must start with https:// (for example https://ntfy.sh).")
    if enabled and not topic:
        raise NotificationError("Enter a topic name before turning notifications on.")
    if topic and not all(c.isalnum() or c in "-_" for c in topic):
        raise NotificationError("The topic can only contain letters, numbers, - and _.")
    db.set_setting("push_enabled", "1" if enabled else "0")
    db.set_setting("push_server", server)
    db.set_setting("push_topic", topic)
    db.set_setting("push_time", send_time)


# --------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------
def send_push(title: str, message: str, priority: int = 3, tags: Optional[list] = None) -> None:
    """Publish one notification. Raises NotificationError on failure."""
    cfg = get_config()
    if not cfg["topic"]:
        raise NotificationError("No topic set yet.")
    payload = {
        "topic": cfg["topic"],
        "title": title,
        "message": message,
        "priority": priority,
        "tags": tags or [],
    }
    request = urllib.request.Request(
        cfg["server"],
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        raise NotificationError(f"The push server said {exc.code} {exc.reason}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NotificationError(f"Could not reach the push server: {exc}") from exc


def build_digest() -> Optional[Tuple[str, str, int]]:
    """Return (title, message, priority) for today, or None if nothing is due."""
    due_today = db.list_tasks(view="today")
    overdue = db.list_tasks(view="overdue")
    if not due_today and not overdue:
        return None

    lines = []
    if overdue:
        lines.append(f"Overdue ({len(overdue)}):")
        lines += [f"- {t['title']} (was due {t['due_date']})" for t in overdue[:8]]
        if len(overdue) > 8:
            lines.append(f"...and {len(overdue) - 8} more")
    if due_today:
        if lines:
            lines.append("")
        lines.append(f"Due today ({len(due_today)}):")
        lines += [f"- {t['title']} [{t['priority']}]" for t in due_today[:8]]
        if len(due_today) > 8:
            lines.append(f"...and {len(due_today) - 8} more")

    title = f"Nomi: {len(due_today)} due today" + (f", {len(overdue)} overdue" if overdue else "")
    has_high = any(t["priority"] == "High" for t in due_today + overdue)
    priority = 4 if (overdue or has_high) else 3
    return title, "\n".join(lines), priority


def send_digest_now() -> bool:
    """Send today's digest immediately. Returns False if there was nothing to send."""
    digest = build_digest()
    if not digest:
        return False
    title, message, priority = digest
    send_push(title, message, priority, tags=["calendar"])
    return True


# --------------------------------------------------------------------------
# Background scheduler
# --------------------------------------------------------------------------
def check_and_send() -> None:
    """Send the daily digest once per day, after the chosen time."""
    cfg = get_config()
    if not cfg["enabled"] or not cfg["topic"]:
        return
    today = db.today().isoformat()
    if db.get_setting("push_last_sent", "") == today:
        return
    if db.now().strftime("%H:%M") < cfg["time"]:
        return
    # Mark first so a slow or failing send can never spam you every minute.
    db.set_setting("push_last_sent", today)
    try:
        send_digest_now()
    except NotificationError as exc:
        db.set_setting("push_last_error", f"{db.now().strftime('%Y-%m-%d %H:%M')} - {exc}")
        db.set_setting("push_last_sent", "")  # try again on the next check


def _loop() -> None:
    while True:
        try:
            check_and_send()
        except Exception:  # noqa: BLE001 - the thread must never die
            pass
        time.sleep(60)


def start_scheduler() -> None:
    """Start the background checker once per process. Safe to call many times."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_loop, name="nomi-push", daemon=True)
        _thread.start()
