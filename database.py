"""Nomi database layer (SQLite).

Every function here talks to SQLite directly using parameterised queries.
Both the Streamlit pages and the AI tools call these same functions, so the UI
and the assistant always see exactly the same data.
"""

import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

try:  # .env support for local development (optional)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

STATUSES = ["Todo", "In Progress", "Completed"]
PRIORITIES = ["Low", "Medium", "High"]
PROJECT_STATUSES = ["Active", "On Hold", "Completed"]

_STATUS_ALIASES = {
    "todo": "Todo",
    "to do": "Todo",
    "to-do": "Todo",
    "open": "Todo",
    "pending": "Todo",
    "in progress": "In Progress",
    "in-progress": "In Progress",
    "inprogress": "In Progress",
    "doing": "In Progress",
    "started": "In Progress",
    "completed": "Completed",
    "complete": "Completed",
    "done": "Completed",
    "finished": "Completed",
}
_PROJECT_STATUS_ALIASES = {
    "active": "Active",
    "in progress": "Active",
    "on hold": "On Hold",
    "on-hold": "On Hold",
    "paused": "On Hold",
    "completed": "Completed",
    "complete": "Completed",
    "done": "Completed",
}

WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


class NomiError(Exception):
    """A problem with a request that can be shown to the user as-is."""


# --------------------------------------------------------------------------
# Connection, time helpers, schema
# --------------------------------------------------------------------------
def _db_path() -> str:
    return os.environ.get("NOMI_DB_PATH") or str(Path(__file__).parent / "nomi.db")


def _tz() -> ZoneInfo:
    name = os.environ.get("NOMI_TIMEZONE") or "Africa/Lagos"
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


def now() -> datetime:
    return datetime.now(_tz())


def today() -> date:
    return now().date()


def _stamp() -> str:
    return now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_conn():
    """Open a connection with foreign keys on; commit on success, roll back on error."""
    try:
        conn = sqlite3.connect(_db_path(), timeout=10)
    except sqlite3.Error as exc:
        raise NomiError(f"Could not open the database: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise NomiError(f"Database error: {exc}") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'Active'
                CHECK (status IN ('Active', 'On Hold', 'Completed')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'Todo'
                CHECK (status IN ('Todo', 'In Progress', 'Completed')),
    priority    TEXT NOT NULL DEFAULT 'Medium'
                CHECK (priority IN ('Low', 'Medium', 'High')),
    due_date    TEXT,
    project_id  INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
"""


def init_db() -> None:
    """Create the tables if they do not exist yet. Safe to call many times."""
    path = Path(_db_path())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NomiError(f"Could not create the database folder: {exc}") from exc
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------
def parse_due_date(value: Optional[str]) -> Optional[str]:
    """Turn user/AI date text into YYYY-MM-DD.

    Accepts YYYY-MM-DD, 'today', 'tomorrow', 'yesterday', 'in 3 days',
    and weekday names such as 'friday'. Empty values mean "no date".
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip().lower()
    if text in ("", "none", "null", "no date", "clear"):
        return None
    for prefix in ("on ", "by ", "this ", "next "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    base = today()
    if text == "today":
        return base.isoformat()
    if text == "tomorrow":
        return (base + timedelta(days=1)).isoformat()
    if text == "yesterday":
        return (base - timedelta(days=1)).isoformat()
    match = re.fullmatch(r"in (\d+) days?", text)
    if match:
        return (base + timedelta(days=int(match.group(1)))).isoformat()
    if text in WEEKDAYS:
        ahead = (WEEKDAYS.index(text) - base.weekday()) % 7
        return (base + timedelta(days=ahead)).isoformat()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise NomiError(
            f"Invalid date '{value}'. Use YYYY-MM-DD (for example {base.isoformat()}), "
            "or words like 'today' / 'tomorrow'."
        )


def _normalize(value: Optional[str], allowed: List[str], aliases: Dict[str, str], label: str) -> str:
    text = str(value or "").strip()
    for item in allowed:
        if text.lower() == item.lower():
            return item
    if text.lower() in aliases:
        return aliases[text.lower()]
    raise NomiError(f"Invalid {label} '{value}'. Allowed values: {', '.join(allowed)}.")


def normalize_status(value: Optional[str]) -> str:
    return _normalize(value, STATUSES, _STATUS_ALIASES, "status")


def normalize_priority(value: Optional[str]) -> str:
    return _normalize(value, PRIORITIES, {}, "priority")


def normalize_project_status(value: Optional[str]) -> str:
    return _normalize(value, PROJECT_STATUSES, _PROJECT_STATUS_ALIASES, "project status")


def _require_text(value: Optional[str], label: str) -> str:
    text = (value or "").strip()
    if not text:
        raise NomiError(f"{label} cannot be empty.")
    return text


def _rows(cursor) -> List[Dict[str, Any]]:
    return [dict(r) for r in cursor.fetchall()]


# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------
_PROJECT_SELECT = """
SELECT p.*,
       (SELECT COUNT(*) FROM tasks t WHERE t.project_id = p.id) AS task_count,
       (SELECT COUNT(*) FROM tasks t
         WHERE t.project_id = p.id AND t.status = 'Completed') AS completed_count
FROM projects p
"""


def create_project(name: str, description: str = "", status: str = "Active") -> Dict[str, Any]:
    name = _require_text(name, "Project name")
    status = normalize_project_status(status or "Active")
    stamp = _stamp()
    with get_conn() as conn:
        if conn.execute("SELECT 1 FROM projects WHERE name = ?", (name,)).fetchone():
            raise NomiError(f"A project named '{name}' already exists.")
        cur = conn.execute(
            "INSERT INTO projects (name, description, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, (description or "").strip(), status, stamp, stamp),
        )
        project_id = cur.lastrowid
    return get_project(project_id)


def get_project(project_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(_PROJECT_SELECT + " WHERE p.id = ?", (project_id,)).fetchone()
    return dict(row) if row else None


def find_project(name: str) -> Optional[Dict[str, Any]]:
    """Find a project by exact name (any case), or by a single partial match."""
    name = (name or "").strip()
    if not name:
        return None
    with get_conn() as conn:
        row = conn.execute(_PROJECT_SELECT + " WHERE p.name = ?", (name,)).fetchone()
        if row:
            return dict(row)
        rows = conn.execute(
            _PROJECT_SELECT + " WHERE p.name LIKE ?", (f"%{name}%",)
        ).fetchall()
    return dict(rows[0]) if len(rows) == 1 else None


def list_projects(status: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = _PROJECT_SELECT
    params: List[Any] = []
    if status:
        sql += " WHERE p.status = ?"
        params.append(normalize_project_status(status))
    sql += " ORDER BY p.status = 'Completed', p.name COLLATE NOCASE"
    with get_conn() as conn:
        return _rows(conn.execute(sql, params))


def update_project(project_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    project = get_project(project_id)
    if not project:
        raise NomiError(f"No project with id {project_id}.")
    fields: Dict[str, Any] = {}
    if updates.get("name") is not None:
        fields["name"] = _require_text(updates["name"], "Project name")
    if updates.get("description") is not None:
        fields["description"] = str(updates["description"]).strip()
    if updates.get("status") is not None:
        fields["status"] = normalize_project_status(updates["status"])
    if not fields:
        raise NomiError("Nothing to update.")
    with get_conn() as conn:
        if "name" in fields:
            clash = conn.execute(
                "SELECT 1 FROM projects WHERE name = ? AND id != ?", (fields["name"], project_id)
            ).fetchone()
            if clash:
                raise NomiError(f"A project named '{fields['name']}' already exists.")
        fields["updated_at"] = _stamp()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE projects SET {assignments} WHERE id = ?", [*fields.values(), project_id]
        )
    return get_project(project_id)


def delete_project(project_id: int) -> Dict[str, Any]:
    """Delete a project. Its tasks are kept but become unassigned (project_id NULL)."""
    project = get_project(project_id)
    if not project:
        raise NomiError(f"No project with id {project_id}.")
    with get_conn() as conn:
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return project


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
_TASK_SELECT = """
SELECT t.*, p.name AS project_name
FROM tasks t
LEFT JOIN projects p ON p.id = t.project_id
"""


def _check_project_exists(conn, project_id: Optional[int]) -> None:
    if project_id is not None:
        if not conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone():
            raise NomiError(f"No project with id {project_id}.")


def create_task(
    title: str,
    description: str = "",
    status: str = "Todo",
    priority: str = "Medium",
    due_date: Optional[str] = None,
    project_id: Optional[int] = None,
) -> Dict[str, Any]:
    title = _require_text(title, "Task title")
    status = normalize_status(status or "Todo")
    priority = normalize_priority(priority or "Medium")
    due = parse_due_date(due_date)
    stamp = _stamp()
    with get_conn() as conn:
        _check_project_exists(conn, project_id)
        cur = conn.execute(
            "INSERT INTO tasks (title, description, status, priority, due_date, project_id, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (title, (description or "").strip(), status, priority, due, project_id, stamp, stamp),
        )
        task_id = cur.lastrowid
    return get_task(task_id)


def get_task(task_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(_TASK_SELECT + " WHERE t.id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


def list_tasks(
    view: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    project_id: Optional[int] = None,
    due_on: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """List tasks.

    view: 'today' | 'overdue' | 'incomplete' | 'completed' | 'all'/None
    """
    where: List[str] = []
    params: List[Any] = []
    today_str = today().isoformat()

    view = (view or "all").strip().lower()
    if view == "today":
        where.append("t.due_date = ? AND t.status != 'Completed'")
        params.append(today_str)
    elif view == "overdue":
        where.append("t.due_date < ? AND t.status != 'Completed'")
        params.append(today_str)
    elif view == "incomplete":
        where.append("t.status != 'Completed'")
    elif view == "completed":
        where.append("t.status = 'Completed'")
    elif view != "all":
        raise NomiError(f"Unknown task view '{view}'.")

    if status:
        where.append("t.status = ?")
        params.append(normalize_status(status))
    if priority:
        where.append("t.priority = ?")
        params.append(normalize_priority(priority))
    if project_id is not None:
        where.append("t.project_id = ?")
        params.append(project_id)
    if due_on:
        where.append("t.due_date = ?")
        params.append(parse_due_date(due_on))
    if search:
        for word in search.split():
            where.append("(t.title LIKE ? OR t.description LIKE ?)")
            params.extend([f"%{word}%", f"%{word}%"])

    sql = _TASK_SELECT
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += (
        " ORDER BY t.status = 'Completed', t.due_date IS NULL, t.due_date,"
        " CASE t.priority WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, t.id"
        " LIMIT ?"
    )
    params.append(int(limit))
    with get_conn() as conn:
        return _rows(conn.execute(sql, params))


def update_task(task_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update only the keys present in `updates` (None values are ignored).

    due_date: pass '' / 'none' to clear it. project_id: pass 0 to unassign.
    """
    if not get_task(task_id):
        raise NomiError(f"No task with id {task_id}.")
    fields: Dict[str, Any] = {}
    if updates.get("title") is not None:
        fields["title"] = _require_text(updates["title"], "Task title")
    if updates.get("description") is not None:
        fields["description"] = str(updates["description"]).strip()
    if updates.get("status") is not None:
        fields["status"] = normalize_status(updates["status"])
    if updates.get("priority") is not None:
        fields["priority"] = normalize_priority(updates["priority"])
    if updates.get("due_date") is not None:
        fields["due_date"] = parse_due_date(updates["due_date"])
    if "project_id" in updates and updates["project_id"] is not None:
        pid = int(updates["project_id"])
        fields["project_id"] = pid if pid > 0 else None
    if not fields:
        raise NomiError("Nothing to update.")
    with get_conn() as conn:
        _check_project_exists(conn, fields.get("project_id"))
        fields["updated_at"] = _stamp()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE tasks SET {assignments} WHERE id = ?", [*fields.values(), task_id]
        )
    return get_task(task_id)


def delete_task(task_id: int) -> Dict[str, Any]:
    task = get_task(task_id)
    if not task:
        raise NomiError(f"No task with id {task_id}.")
    with get_conn() as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    return task


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
def create_note(title: Optional[str], content: Optional[str]) -> Dict[str, Any]:
    title = (title or "").strip()
    content = (content or "").strip()
    if not title and not content:
        raise NomiError("A note needs a title or some content.")
    if not title:
        first_line = content.splitlines()[0]
        title = first_line[:60] + ("…" if len(first_line) > 60 else "")
    stamp = _stamp()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO notes (title, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (title, content, stamp, stamp),
        )
        note_id = cur.lastrowid
    return get_note(note_id)


def get_note(note_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    return dict(row) if row else None


def list_notes(search: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM notes"
    params: List[Any] = []
    if search:
        clauses = []
        for word in search.split():
            clauses.append("(title LIKE ? OR content LIKE ?)")
            params.extend([f"%{word}%", f"%{word}%"])
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY updated_at DESC, id DESC LIMIT ?"
    params.append(int(limit))
    with get_conn() as conn:
        return _rows(conn.execute(sql, params))


def update_note(note_id: int, title: Optional[str] = None, content: Optional[str] = None) -> Dict[str, Any]:
    if not get_note(note_id):
        raise NomiError(f"No note with id {note_id}.")
    fields: Dict[str, Any] = {}
    if title is not None:
        fields["title"] = _require_text(title, "Note title")
    if content is not None:
        fields["content"] = content.strip()
    if not fields:
        raise NomiError("Nothing to update.")
    fields["updated_at"] = _stamp()
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE notes SET {assignments} WHERE id = ?", [*fields.values(), note_id])
    return get_note(note_id)


def delete_note(note_id: int) -> Dict[str, Any]:
    note = get_note(note_id)
    if not note:
        raise NomiError(f"No note with id {note_id}.")
    with get_conn() as conn:
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return note


# --------------------------------------------------------------------------
# Memories
# --------------------------------------------------------------------------
_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "can", "could", "did", "does", "for",
    "from", "have", "how", "into", "just", "like", "many", "much", "need", "not", "our",
    "please", "should", "that", "the", "them", "then", "there", "these", "they", "this",
    "want", "what", "when", "where", "which", "who", "will", "with", "would", "you",
    "your", "mine", "me", "my", "i", "is", "it", "to", "of", "in", "on", "do", "a", "an",
}
_BROAD_WORDS = ("plan", "organize", "organise", "schedule", "recommend", "suggest", "advice", "routine")


def save_memory(content: str) -> Dict[str, Any]:
    content = _require_text(content, "Memory")
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM memories WHERE lower(content) = lower(?)", (content,)
        ).fetchone()
        if existing:
            result = dict(existing)
            result["duplicate"] = True
            return result
        cur = conn.execute(
            "INSERT INTO memories (content, created_at) VALUES (?, ?)", (content, _stamp())
        )
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (cur.lastrowid,)).fetchone()
    result = dict(row)
    result["duplicate"] = False
    return result


def get_memory(memory_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return dict(row) if row else None


def list_memories(search: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM memories"
    params: List[Any] = []
    if search:
        clauses = []
        for word in search.split():
            clauses.append("content LIKE ?")
            params.append(f"%{word}%")
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with get_conn() as conn:
        return _rows(conn.execute(sql, params))


def delete_memory(memory_id: int) -> Dict[str, Any]:
    memory = get_memory(memory_id)
    if not memory:
        raise NomiError(f"No memory with id {memory_id}.")
    with get_conn() as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    return memory


def _keywords(text: str) -> set:
    """Lower-case words with a crude stem, so 'study' matches 'studying'."""
    result = set()
    for word in re.findall(r"[a-zA-Z']{3,}", (text or "").lower()):
        if word in _STOPWORDS:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
        result.add(word)
    return result


def relevant_memories(text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Cheap keyword-overlap retrieval (no vector database).

    Returns memories sharing meaningful words with `text`. For broad requests
    ("plan my day", "suggest something") a few recent memories are added too.
    """
    words = _keywords(text)
    all_memories = list_memories(limit=500)
    scored = []
    for memory in all_memories:
        mem_words = _keywords(memory["content"])
        score = len(words & mem_words)
        if score:
            scored.append((score, memory))
    scored.sort(key=lambda pair: (-pair[0], -pair[1]["id"]))
    picked = [m for _, m in scored[:limit]]
    if len(picked) < 3 and any(w in (text or "").lower() for w in _BROAD_WORDS):
        for memory in all_memories:  # newest first
            if memory not in picked:
                picked.append(memory)
            if len(picked) >= 3:
                break
    return picked[:limit]


# --------------------------------------------------------------------------
# Weekly activity
# --------------------------------------------------------------------------
def week_range(period: str = "this_week") -> tuple:
    """Return (start_date, end_date) as ISO strings. end is always today."""
    end = today()
    if period == "last_7_days":
        start = end - timedelta(days=6)
    else:  # this_week: Monday up to today
        start = end - timedelta(days=end.weekday())
    return start.isoformat(), end.isoformat()


def week_summary(start: str, end: str) -> Dict[str, Any]:
    """Collect real activity between two dates (inclusive) for the weekly review.

    Note: "completed this week" means tasks currently marked Completed whose
    updated_at falls in the range (Nomi does not store a separate completed_at).
    """
    in_range = "substr({col}, 1, 10) BETWEEN ? AND ?"
    today_str = today().isoformat()
    with get_conn() as conn:
        completed = _rows(
            conn.execute(
                _TASK_SELECT
                + " WHERE t.status = 'Completed' AND "
                + in_range.format(col="t.updated_at")
                + " ORDER BY t.updated_at",
                (start, end),
            )
        )
        created = _rows(
            conn.execute(
                _TASK_SELECT
                + " WHERE "
                + in_range.format(col="t.created_at")
                + " ORDER BY t.created_at",
                (start, end),
            )
        )
        incomplete = _rows(
            conn.execute(
                _TASK_SELECT
                + " WHERE t.status != 'Completed' ORDER BY t.due_date IS NULL, t.due_date, t.id"
            )
        )
        projects = _rows(
            conn.execute(
                """
                SELECT p.id, p.name, p.status,
                       (SELECT COUNT(*) FROM tasks t WHERE t.project_id = p.id
                         AND substr(t.updated_at, 1, 10) BETWEEN ? AND ?) AS tasks_touched,
                       (SELECT COUNT(*) FROM tasks t WHERE t.project_id = p.id
                         AND t.status = 'Completed'
                         AND substr(t.updated_at, 1, 10) BETWEEN ? AND ?) AS tasks_completed
                FROM projects p
                WHERE substr(p.updated_at, 1, 10) BETWEEN ? AND ?
                   OR EXISTS (SELECT 1 FROM tasks t WHERE t.project_id = p.id
                              AND substr(t.updated_at, 1, 10) BETWEEN ? AND ?)
                ORDER BY p.name COLLATE NOCASE
                """,
                (start, end, start, end, start, end, start, end),
            )
        )
        notes_created = _rows(
            conn.execute(
                "SELECT id, title, created_at FROM notes WHERE "
                + in_range.format(col="created_at")
                + " ORDER BY created_at",
                (start, end),
            )
        )
        notes_updated = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE "
            + in_range.format(col="updated_at")
            + " AND substr(created_at, 1, 10) < ?",
            (start, end, start),
        ).fetchone()[0]
    overdue = [t for t in incomplete if t["due_date"] and t["due_date"] < today_str]
    return {
        "start": start,
        "end": end,
        "today": today_str,
        "completed": completed,
        "created": created,
        "incomplete": incomplete,
        "overdue": overdue,
        "projects": projects,
        "notes_created": notes_created,
        "notes_updated": notes_updated,
    }
