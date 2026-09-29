"""LangChain tools for Nomi.

Each tool is a small wrapper around a function in database.py, so a tool call
really reads or writes SQLite. Tools return plain-text results that the model
reads and reports back to you. If a tool fails it returns "ERROR: ..." so the
assistant can tell you honestly that it did not work.
"""

import functools
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

import database as db
from database import NomiError


# --------------------------------------------------------------------------
# Formatting helpers (also used by the daily plan / weekly review prompts)
# --------------------------------------------------------------------------
def format_task(t: Dict[str, Any]) -> str:
    parts = [f"#{t['id']} [{t['priority']}] {t['title']} — {t['status']}"]
    if t.get("due_date"):
        parts.append(f"due {t['due_date']}")
        if t["status"] != "Completed" and t["due_date"] < db.today().isoformat():
            parts.append("OVERDUE")
    if t.get("project_name"):
        parts.append(f"project: {t['project_name']}")
    line = ", ".join(parts)
    if t.get("description"):
        line += f" (notes: {t['description'][:120]})"
    return line


def format_tasks(tasks: List[Dict[str, Any]], empty: str = "No matching tasks.") -> str:
    return "\n".join(format_task(t) for t in tasks) if tasks else empty


def format_project(p: Dict[str, Any]) -> str:
    line = (
        f"#{p['id']} {p['name']} — {p['status']}, "
        f"{p['completed_count']}/{p['task_count']} tasks completed"
    )
    if p.get("description"):
        line += f" ({p['description'][:120]})"
    return line


def format_week_summary(s: Dict[str, Any]) -> str:
    lines = [f"Activity from {s['start']} to {s['end']} (today is {s['today']})."]
    lines.append(f"\nTasks completed ({len(s['completed'])}):")
    lines.append(format_tasks(s["completed"], "  none"))
    lines.append(f"\nTasks created ({len(s['created'])}):")
    lines.append(format_tasks(s["created"], "  none"))
    lines.append(f"\nTasks still incomplete ({len(s['incomplete'])}), overdue: {len(s['overdue'])}:")
    lines.append(format_tasks(s["incomplete"], "  none"))
    lines.append(f"\nProjects with activity ({len(s['projects'])}):")
    if s["projects"]:
        for p in s["projects"]:
            lines.append(
                f"  {p['name']} ({p['status']}) — {p['tasks_touched']} tasks touched, "
                f"{p['tasks_completed']} completed"
            )
    else:
        lines.append("  none")
    lines.append(f"\nNotes created ({len(s['notes_created'])}):")
    if s["notes_created"]:
        lines.extend(f"  #{n['id']} {n['title']}" for n in s["notes_created"])
    else:
        lines.append("  none")
    lines.append(f"\nOlder notes edited this period: {s['notes_updated']}")
    return "\n".join(lines)


def _clean(value: Any) -> Any:
    """Models sometimes send 'null' / '' for missing values. Treat those as None."""
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "undefined"):
        return None
    return value


def _safe(func):
    """Turn any failure into an 'ERROR: ...' string instead of crashing the agent."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NomiError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001 - last-resort safety net for tools
            return f"ERROR: unexpected failure ({type(exc).__name__}: {exc})"

    return wrapper


def _resolve_project_id(project_id: Optional[int], project_name: Optional[str]) -> Optional[int]:
    project_id, project_name = _clean(project_id), _clean(project_name)
    if project_id is not None:
        return int(project_id)
    if project_name:
        project = db.find_project(project_name)
        if not project:
            raise NomiError(
                f"No single project matches '{project_name}'. Use get_projects to see the "
                "available projects, or create_project first."
            )
        return project["id"]
    return None


# --------------------------------------------------------------------------
# Task tools
# --------------------------------------------------------------------------
class CreateTaskInput(BaseModel):
    title: str = Field(description="Short task title, e.g. 'Finish biology assignment'.")
    description: Optional[str] = Field(default=None, description="Optional extra details.")
    priority: Optional[str] = Field(default="Medium", description="Low, Medium or High.")
    due_date: Optional[str] = Field(
        default=None,
        description="Due date as YYYY-MM-DD. Convert words like 'tomorrow' using today's date.",
    )
    status: Optional[str] = Field(default="Todo", description="Todo, In Progress or Completed.")
    project_name: Optional[str] = Field(default=None, description="Name of an existing project.")
    project_id: Optional[int] = Field(default=None, description="ID of an existing project.")


@tool("create_task", args_schema=CreateTaskInput)
@_safe
def create_task(
    title: str,
    description: Optional[str] = None,
    priority: Optional[str] = "Medium",
    due_date: Optional[str] = None,
    status: Optional[str] = "Todo",
    project_name: Optional[str] = None,
    project_id: Optional[int] = None,
) -> str:
    """Create and save a new task in the database."""
    task = db.create_task(
        title=title,
        description=_clean(description) or "",
        status=_clean(status) or "Todo",
        priority=_clean(priority) or "Medium",
        due_date=_clean(due_date),
        project_id=_resolve_project_id(project_id, project_name),
    )
    return "Task created: " + format_task(task)


class GetTasksInput(BaseModel):
    view: Optional[str] = Field(
        default="all",
        description="One of: all, today (due today, not done), overdue, incomplete, completed.",
    )
    status: Optional[str] = Field(default=None, description="Filter: Todo, In Progress or Completed.")
    priority: Optional[str] = Field(default=None, description="Filter: Low, Medium or High.")
    due_date: Optional[str] = Field(default=None, description="Only tasks due on this date (YYYY-MM-DD).")
    project_name: Optional[str] = Field(default=None, description="Only tasks in this project.")
    search: Optional[str] = Field(default=None, description="Keywords to look for in title/description.")


@tool("get_tasks", args_schema=GetTasksInput)
@_safe
def get_tasks(
    view: Optional[str] = "all",
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_date: Optional[str] = None,
    project_name: Optional[str] = None,
    search: Optional[str] = None,
) -> str:
    """Read tasks from the database. Use for any question about the user's tasks."""
    project_id = _resolve_project_id(None, project_name)
    tasks = db.list_tasks(
        view=_clean(view) or "all",
        status=_clean(status),
        priority=_clean(priority),
        project_id=project_id,
        due_on=_clean(due_date),
        search=_clean(search),
        limit=50,
    )
    header = f"{len(tasks)} task(s) (today is {db.today().isoformat()}):\n"
    return header + format_tasks(tasks)


class UpdateTaskInput(BaseModel):
    task_id: int = Field(description="ID of the task to change (find it with get_tasks).")
    title: Optional[str] = Field(default=None, description="New title.")
    description: Optional[str] = Field(default=None, description="New description.")
    status: Optional[str] = Field(default=None, description="Todo, In Progress or Completed.")
    priority: Optional[str] = Field(default=None, description="Low, Medium or High.")
    due_date: Optional[str] = Field(
        default=None, description="New due date YYYY-MM-DD, or 'none' to clear the date."
    )
    project_name: Optional[str] = Field(default=None, description="Move the task to this project.")
    project_id: Optional[int] = Field(
        default=None, description="Move the task to this project ID (0 removes it from its project)."
    )


@tool("update_task", args_schema=UpdateTaskInput)
@_safe
def update_task(
    task_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_date: Optional[str] = None,
    project_name: Optional[str] = None,
    project_id: Optional[int] = None,
) -> str:
    """Edit an existing task. Only the fields you provide are changed."""
    updates = {
        "title": _clean(title),
        "description": _clean(description),
        "status": _clean(status),
        "priority": _clean(priority),
        "due_date": _clean(due_date),
        "project_id": _resolve_project_id(project_id, project_name),
    }
    if updates["due_date"] is None and isinstance(due_date, str) and due_date.strip().lower() == "none":
        updates["due_date"] = "none"
    task = db.update_task(task_id, updates)
    return "Task updated: " + format_task(task)


class CompleteTaskInput(BaseModel):
    task_id: Optional[int] = Field(default=None, description="ID of the task to complete.")
    title_query: Optional[str] = Field(
        default=None,
        description="Keywords from the task title, used when the ID is not known (e.g. 'biology').",
    )


@tool("complete_task", args_schema=CompleteTaskInput)
@_safe
def complete_task(task_id: Optional[int] = None, title_query: Optional[str] = None) -> str:
    """Mark a task as Completed, by ID or by title keywords. If several tasks match, nothing
    is changed and the matches are listed so the user can choose."""
    task_id, title_query = _clean(task_id), _clean(title_query)
    if task_id is None and not title_query:
        raise NomiError("Provide task_id or title_query.")
    if task_id is None:
        matches = db.list_tasks(view="incomplete", search=title_query, limit=10)
        if not matches:
            raise NomiError(f"No incomplete task matches '{title_query}'.")
        if len(matches) > 1:
            return (
                f"Nothing changed. {len(matches)} incomplete tasks match '{title_query}'. "
                "Ask the user which one:\n" + format_tasks(matches)
            )
        task_id = matches[0]["id"]
    task = db.update_task(int(task_id), {"status": "Completed"})
    return "Task completed: " + format_task(task)


class DeleteTaskInput(BaseModel):
    task_id: int = Field(description="ID of the task to delete (find it with get_tasks first).")
    confirm: bool = Field(
        default=False,
        description="Set true ONLY after the user explicitly confirmed this exact deletion.",
    )


@tool("delete_task", args_schema=DeleteTaskInput)
@_safe
def delete_task(task_id: int, confirm: bool = False) -> str:
    """Permanently delete a task. Requires confirm=true after the user has confirmed."""
    task = db.get_task(task_id)
    if not task:
        raise NomiError(f"No task with id {task_id}.")
    if not confirm:
        return (
            "NOT DELETED - confirmation required. Ask the user to confirm deleting: "
            + format_task(task)
        )
    db.delete_task(task_id)
    return f"Task deleted: #{task['id']} {task['title']}"


# --------------------------------------------------------------------------
# Project tools
# --------------------------------------------------------------------------
class CreateProjectInput(BaseModel):
    name: str = Field(description="Project name, e.g. 'Nomi'.")
    description: Optional[str] = Field(default=None, description="Optional description.")
    status: Optional[str] = Field(default="Active", description="Active, On Hold or Completed.")


@tool("create_project", args_schema=CreateProjectInput)
@_safe
def create_project(name: str, description: Optional[str] = None, status: Optional[str] = "Active") -> str:
    """Create and save a new project."""
    project = db.create_project(name, _clean(description) or "", _clean(status) or "Active")
    return "Project created: " + format_project(project)


class GetProjectsInput(BaseModel):
    status: Optional[str] = Field(default=None, description="Filter: Active, On Hold or Completed.")
    project_name: Optional[str] = Field(
        default=None, description="If given, also list the tasks that belong to this project."
    )


@tool("get_projects", args_schema=GetProjectsInput)
@_safe
def get_projects(status: Optional[str] = None, project_name: Optional[str] = None) -> str:
    """Read projects (with task counts). Give project_name to see one project's tasks."""
    project_name = _clean(project_name)
    if project_name:
        project = db.find_project(project_name)
        if not project:
            raise NomiError(f"No single project matches '{project_name}'.")
        tasks = db.list_tasks(project_id=project["id"], limit=50)
        return format_project(project) + "\nTasks:\n" + format_tasks(tasks, "  (no tasks yet)")
    projects = db.list_projects(status=_clean(status))
    if not projects:
        return "No projects found."
    return f"{len(projects)} project(s):\n" + "\n".join(format_project(p) for p in projects)


class UpdateProjectInput(BaseModel):
    project_id: int = Field(description="ID of the project (find it with get_projects).")
    name: Optional[str] = Field(default=None, description="New name.")
    description: Optional[str] = Field(default=None, description="New description.")
    status: Optional[str] = Field(default=None, description="Active, On Hold or Completed.")


@tool("update_project", args_schema=UpdateProjectInput)
@_safe
def update_project(
    project_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Edit a project. Only the fields you provide are changed."""
    project = db.update_project(
        project_id,
        {"name": _clean(name), "description": _clean(description), "status": _clean(status)},
    )
    return "Project updated: " + format_project(project)


class DeleteProjectInput(BaseModel):
    project_id: int = Field(description="ID of the project to delete.")
    confirm: bool = Field(
        default=False,
        description="Set true ONLY after the user explicitly confirmed this exact deletion.",
    )


@tool("delete_project", args_schema=DeleteProjectInput)
@_safe
def delete_project(project_id: int, confirm: bool = False) -> str:
    """Delete a project. Its tasks are kept but unassigned. Requires confirm=true."""
    project = db.get_project(project_id)
    if not project:
        raise NomiError(f"No project with id {project_id}.")
    if not confirm:
        return (
            "NOT DELETED - confirmation required. Ask the user to confirm deleting project: "
            + format_project(project)
            + ". Its tasks will be kept but no longer belong to a project."
        )
    db.delete_project(project_id)
    return f"Project deleted: #{project['id']} {project['name']} (its tasks were kept, unassigned)."


# --------------------------------------------------------------------------
# Note tools
# --------------------------------------------------------------------------
class CreateNoteInput(BaseModel):
    title: Optional[str] = Field(default=None, description="Short title (optional; derived from content if empty).")
    content: str = Field(description="The note text, exactly what the user wants saved.")


@tool("create_note", args_schema=CreateNoteInput)
@_safe
def create_note(content: str, title: Optional[str] = None) -> str:
    """Save a new note (idea, thought, plan, brain dump)."""
    note = db.create_note(_clean(title), content)
    return f"Note saved: #{note['id']} {note['title']}"


class GetNotesInput(BaseModel):
    search: Optional[str] = Field(default=None, description="Keywords to search in title/content.")
    note_id: Optional[int] = Field(default=None, description="Read one note in full by ID.")
    limit: Optional[int] = Field(default=10, description="Max notes to list (default 10).")


@tool("get_notes", args_schema=GetNotesInput)
@_safe
def get_notes(search: Optional[str] = None, note_id: Optional[int] = None, limit: Optional[int] = 10) -> str:
    """Read notes. Search by keywords, or give note_id for the full text of one note."""
    note_id = _clean(note_id)
    if note_id is not None:
        note = db.get_note(int(note_id))
        if not note:
            raise NomiError(f"No note with id {note_id}.")
        return f"#{note['id']} {note['title']} (updated {note['updated_at']})\n{note['content']}"
    notes = db.list_notes(search=_clean(search), limit=int(_clean(limit) or 10))
    if not notes:
        return "No notes found."
    lines = []
    for n in notes:
        snippet = n["content"].replace("\n", " ")
        snippet = snippet[:200] + ("…" if len(snippet) > 200 else "")
        lines.append(f"#{n['id']} {n['title']} ({n['created_at'][:10]}): {snippet}")
    return f"{len(notes)} note(s):\n" + "\n".join(lines)


class UpdateNoteInput(BaseModel):
    note_id: int = Field(description="ID of the note to edit.")
    title: Optional[str] = Field(default=None, description="New title.")
    content: Optional[str] = Field(default=None, description="New full content (replaces the old content).")


@tool("update_note", args_schema=UpdateNoteInput)
@_safe
def update_note(note_id: int, title: Optional[str] = None, content: Optional[str] = None) -> str:
    """Edit a note. Content given here REPLACES the old content, so include everything to keep."""
    note = db.update_note(note_id, _clean(title), _clean(content))
    return f"Note updated: #{note['id']} {note['title']}"


class DeleteNoteInput(BaseModel):
    note_id: int = Field(description="ID of the note to delete.")
    confirm: bool = Field(
        default=False,
        description="Set true ONLY after the user explicitly confirmed this exact deletion.",
    )


@tool("delete_note", args_schema=DeleteNoteInput)
@_safe
def delete_note(note_id: int, confirm: bool = False) -> str:
    """Permanently delete a note. Requires confirm=true after the user has confirmed."""
    note = db.get_note(note_id)
    if not note:
        raise NomiError(f"No note with id {note_id}.")
    if not confirm:
        return f"NOT DELETED - confirmation required. Ask the user to confirm deleting note #{note['id']} '{note['title']}'."
    db.delete_note(note_id)
    return f"Note deleted: #{note['id']} {note['title']}"


# --------------------------------------------------------------------------
# Memory tools
# --------------------------------------------------------------------------
class SaveMemoryInput(BaseModel):
    content: str = Field(
        description="The fact to remember, written as a short statement, e.g. 'I prefer studying in the morning.'"
    )


@tool("save_memory", args_schema=SaveMemoryInput)
@_safe
def save_memory(content: str) -> str:
    """Save something the user explicitly asked Nomi to remember (preference, goal, habit, fact)."""
    memory = db.save_memory(content)
    if memory["duplicate"]:
        return f"Already remembered (#{memory['id']}): {memory['content']}"
    return f"Memory saved (#{memory['id']}): {memory['content']}"


class GetMemoriesInput(BaseModel):
    search: Optional[str] = Field(default=None, description="Optional keywords. Leave empty to list all.")


@tool("get_memories", args_schema=GetMemoriesInput)
@_safe
def get_memories(search: Optional[str] = None) -> str:
    """Read saved memories about the user."""
    memories = db.list_memories(search=_clean(search), limit=50)
    if not memories:
        return "No memories saved yet."
    return f"{len(memories)} memory item(s):\n" + "\n".join(
        f"#{m['id']} {m['content']}" for m in memories
    )


class DeleteMemoryInput(BaseModel):
    memory_id: int = Field(description="ID of the memory to delete (find it with get_memories).")
    confirm: bool = Field(
        default=False,
        description="Set true ONLY after the user explicitly confirmed this exact deletion.",
    )


@tool("delete_memory", args_schema=DeleteMemoryInput)
@_safe
def delete_memory(memory_id: int, confirm: bool = False) -> str:
    """Forget a saved memory. Requires confirm=true after the user has confirmed."""
    memory = db.get_memory(memory_id)
    if not memory:
        raise NomiError(f"No memory with id {memory_id}.")
    if not confirm:
        return f"NOT DELETED - confirmation required. Ask the user to confirm forgetting #{memory['id']}: {memory['content']}"
    db.delete_memory(memory_id)
    return f"Memory deleted: #{memory['id']} {memory['content']}"


# --------------------------------------------------------------------------
# Weekly summary tool
# --------------------------------------------------------------------------
class WeeklySummaryInput(BaseModel):
    period: Optional[str] = Field(
        default="this_week", description="'this_week' (Monday to today) or 'last_7_days'."
    )


@tool("get_weekly_summary", args_schema=WeeklySummaryInput)
@_safe
def get_weekly_summary(period: Optional[str] = "this_week") -> str:
    """Read real activity for the week: completed/incomplete tasks, projects, notes."""
    start, end = db.week_range(_clean(period) or "this_week")
    return format_week_summary(db.week_summary(start, end))


ALL_TOOLS = [
    create_task,
    get_tasks,
    update_task,
    complete_task,
    delete_task,
    create_project,
    get_projects,
    update_project,
    delete_project,
    create_note,
    get_notes,
    update_note,
    delete_note,
    save_memory,
    get_memories,
    delete_memory,
    get_weekly_summary,
]
