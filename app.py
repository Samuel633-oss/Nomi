"""Nomi - Your Personal AI Assistant (Streamlit app).

Run locally:  streamlit run app.py
"""

from datetime import date

import streamlit as st

import ai
import database as db
from database import NomiError

st.set_page_config(page_title="Nomi", page_icon="🧠", layout="wide")

PAGES = [
    "💬 Chat",
    "✅ Tasks",
    "📁 Projects",
    "📝 Notes",
    "🧠 Memory",
    "📅 Daily Plan",
    "📊 Weekly Review",
]
STATUS_ICON = {"Todo": "⬜", "In Progress": "🔄", "Completed": "✅"}
PRIORITY_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


# --------------------------------------------------------------------------
# Startup helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_database() -> bool:
    db.init_db()
    return True


@st.cache_resource(show_spinner="Starting Nomi...")
def get_agent():
    """Build the LangGraph agent once. Raises MissingAPIKeyError if no key is set."""
    return ai.build_agent()


def flash(message: str) -> None:
    st.session_state["flash"] = message


def show_flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        st.success(message)


def run_db(action, *args, success: str = "", **kwargs):
    """Run a database action, showing friendly errors. Returns (ok, result)."""
    try:
        result = action(*args, **kwargs)
    except NomiError as exc:
        st.error(str(exc))
        return False, None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong: {exc}")
        return False, None
    if success:
        flash(success)
    return True, result


def delete_button(label: str, key: str, action, *args, what: str = "this item") -> None:
    """A delete button that asks for confirmation inside a popover."""
    with st.popover("🗑️ Delete"):
        st.warning(f"Delete {what}? This cannot be undone.")
        if st.button(f"Yes, delete", key=f"del_{key}", type="primary"):
            ok, _ = run_db(action, *args, success=f"Deleted {label}.")
            if ok:
                st.rerun()


def project_options():
    projects = db.list_projects()
    ids = [None] + [p["id"] for p in projects]
    names = {None: "— No project —", **{p["id"]: p["name"] for p in projects}}
    return ids, names


def due_input(label_prefix: str, key: str, current: str | None = None):
    """Checkbox + date picker. Returns an ISO date string or None."""
    has_due = st.checkbox("Set due date", value=bool(current), key=f"{key}_has")
    default = date.fromisoformat(current) if current else db.today()
    chosen = st.date_input("Due date", value=default, key=f"{key}_date")
    return chosen.isoformat() if has_due else None


def is_overdue(task) -> bool:
    return bool(task["due_date"]) and task["status"] != "Completed" and task["due_date"] < db.today().isoformat()


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
def page_chat() -> None:
    st.header("💬 Chat with Nomi")
    history = st.session_state.setdefault("chat_history", [])

    try:
        agent = get_agent()
    except ai.MissingAPIKeyError as exc:
        st.error(str(exc))
        st.info("Add your key, then reload the page. See the README for the exact steps.")
        st.chat_input("Chat is disabled until GROQ_API_KEY is set", disabled=True)
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not start the AI agent: {exc}")
        return

    if not history:
        st.caption(
            "Try: “Create a high priority task called Finish biology assignment for tomorrow” · "
            "“What do I need to do today?” · “Remember that I prefer studying in the morning” · "
            "“Save this as a note: …” · “Plan my day” · “What did I accomplish this week?”"
        )

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("tools"):
                st.caption("🔧 Tools used: " + ", ".join(turn["tools"]))

    prompt = st.chat_input("Message Nomi...")
    if prompt is None:
        return
    if not prompt.strip():
        st.warning("Please type a message.")
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Nomi is thinking..."):
                reply, tools_used = ai.run_agent(agent, history, prompt)
        except ai.NomiAIError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")
            return
        st.markdown(reply)
        if tools_used:
            st.caption("🔧 Tools used: " + ", ".join(tools_used))

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": reply, "tools": tools_used})


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
def render_task(task, ids, names) -> None:
    with st.container(border=True):
        top, actions = st.columns([5, 2])
        overdue = " · 🚨 **OVERDUE**" if is_overdue(task) else ""
        due = f" · 📅 {task['due_date']}" if task["due_date"] else ""
        project = f" · 📁 {task['project_name']}" if task["project_name"] else ""
        top.markdown(
            f"{STATUS_ICON[task['status']]} **{task['title']}**  \n"
            f"{PRIORITY_ICON[task['priority']]} {task['priority']} · {task['status']}"
            f"{due}{overdue}{project} · `#{task['id']}`"
        )
        if task["description"]:
            top.caption(task["description"])
        with actions:
            if task["status"] != "Completed":
                if st.button("✅ Complete", key=f"done_{task['id']}"):
                    ok, _ = run_db(db.update_task, task["id"], {"status": "Completed"}, success="Task completed.")
                    if ok:
                        st.rerun()
            else:
                if st.button("↩️ Reopen", key=f"reopen_{task['id']}"):
                    ok, _ = run_db(db.update_task, task["id"], {"status": "Todo"}, success="Task reopened.")
                    if ok:
                        st.rerun()
            delete_button(f"task “{task['title']}”", f"task_{task['id']}", db.delete_task, task["id"], what="this task")

        with st.expander("✏️ Edit"):
            key = f"edit_task_{task['id']}"
            title = st.text_input("Title", value=task["title"], key=f"{key}_title")
            desc = st.text_area("Description", value=task["description"], key=f"{key}_desc")
            c1, c2, c3 = st.columns(3)
            status = c1.selectbox("Status", db.STATUSES, index=db.STATUSES.index(task["status"]), key=f"{key}_status")
            priority = c2.selectbox("Priority", db.PRIORITIES, index=db.PRIORITIES.index(task["priority"]), key=f"{key}_prio")
            current_index = ids.index(task["project_id"]) if task["project_id"] in ids else 0
            project_id = c3.selectbox("Project", ids, index=current_index, format_func=names.get, key=f"{key}_proj")
            due = due_input("Due", key, task["due_date"])
            if st.button("Save changes", key=f"{key}_save", type="primary"):
                updates = {
                    "title": title,
                    "description": desc,
                    "status": status,
                    "priority": priority,
                    "due_date": due or "none",
                    "project_id": project_id or 0,
                }
                ok, _ = run_db(db.update_task, task["id"], updates, success="Task updated.")
                if ok:
                    st.rerun()


def page_tasks() -> None:
    st.header("✅ Tasks")
    show_flash()
    ids, names = project_options()

    with st.expander("➕ New task"):
        with st.form("new_task", clear_on_submit=True):
            title = st.text_input("Title")
            desc = st.text_area("Description (optional)")
            c1, c2, c3 = st.columns(3)
            priority = c1.selectbox("Priority", db.PRIORITIES, index=1)
            status = c2.selectbox("Status", db.STATUSES)
            project_id = c3.selectbox("Project", ids, format_func=names.get)
            has_due = st.checkbox("Set due date")
            due = st.date_input("Due date", value=db.today())
            if st.form_submit_button("Create task", type="primary"):
                ok, _ = run_db(
                    db.create_task,
                    title=title,
                    description=desc,
                    status=status,
                    priority=priority,
                    due_date=due.isoformat() if has_due else None,
                    project_id=project_id,
                    success="Task created.",
                )
                if ok:
                    st.rerun()

    f1, f2, f3, f4, f5 = st.columns(5)
    view = f1.selectbox("View", ["All", "Today", "Overdue", "Incomplete", "Completed"])
    status = f2.selectbox("Status", ["Any"] + db.STATUSES)
    priority = f3.selectbox("Priority", ["Any"] + db.PRIORITIES)
    project_filter = f4.selectbox("Project", ids, format_func=names.get, key="task_project_filter")
    search = f5.text_input("Search")

    ok, tasks = run_db(
        db.list_tasks,
        view=view.lower(),
        status=None if status == "Any" else status,
        priority=None if priority == "Any" else priority,
        project_id=project_filter,
        search=search or None,
    )
    if not ok:
        return
    st.caption(f"{len(tasks)} task(s)")
    if not tasks:
        st.info("No tasks match. Create one above or ask Nomi in the chat.")
    for task in tasks:
        render_task(task, ids, names)


# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------
def page_projects() -> None:
    st.header("📁 Projects")
    show_flash()

    with st.expander("➕ New project"):
        with st.form("new_project", clear_on_submit=True):
            name = st.text_input("Name")
            desc = st.text_area("Description (optional)")
            status = st.selectbox("Status", db.PROJECT_STATUSES)
            if st.form_submit_button("Create project", type="primary"):
                ok, _ = run_db(db.create_project, name, desc, status, success="Project created.")
                if ok:
                    st.rerun()

    ok, projects = run_db(db.list_projects)
    if not ok:
        return
    if not projects:
        st.info("No projects yet. Create one above or ask Nomi: “Create a project called Nomi.”")
    for project in projects:
        with st.container(border=True):
            total, done = project["task_count"], project["completed_count"]
            st.markdown(f"### {project['name']}  `#{project['id']}`")
            st.caption(f"{project['status']} · {done}/{total} tasks completed")
            if project["description"]:
                st.write(project["description"])
            if total:
                st.progress(done / total)

            tasks = db.list_tasks(project_id=project["id"])
            with st.expander(f"Tasks in this project ({len(tasks)})"):
                for t in tasks:
                    flag = " 🚨" if is_overdue(t) else ""
                    due = f" · due {t['due_date']}" if t["due_date"] else ""
                    st.markdown(f"{STATUS_ICON[t['status']]} {t['title']} — {t['priority']}{due}{flag}")
                if not tasks:
                    st.caption("No tasks yet.")
                new_title = st.text_input("Quick add a task", key=f"quick_{project['id']}")
                if st.button("Add task", key=f"quick_add_{project['id']}"):
                    ok, _ = run_db(
                        db.create_task, title=new_title, project_id=project["id"], success="Task added to project."
                    )
                    if ok:
                        st.rerun()

            with st.expander("✏️ Edit project"):
                key = f"edit_project_{project['id']}"
                name = st.text_input("Name", value=project["name"], key=f"{key}_name")
                desc = st.text_area("Description", value=project["description"], key=f"{key}_desc")
                status = st.selectbox(
                    "Status", db.PROJECT_STATUSES, index=db.PROJECT_STATUSES.index(project["status"]), key=f"{key}_status"
                )
                if st.button("Save changes", key=f"{key}_save", type="primary"):
                    ok, _ = run_db(
                        db.update_project,
                        project["id"],
                        {"name": name, "description": desc, "status": status},
                        success="Project updated.",
                    )
                    if ok:
                        st.rerun()
            delete_button(
                f"project “{project['name']}”",
                f"project_{project['id']}",
                db.delete_project,
                project["id"],
                what="this project (its tasks are kept but unassigned)",
            )


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
def page_notes() -> None:
    st.header("📝 Notes")
    show_flash()

    with st.expander("➕ New note"):
        with st.form("new_note", clear_on_submit=True):
            title = st.text_input("Title (optional)")
            content = st.text_area("Content", height=150)
            if st.form_submit_button("Save note", type="primary"):
                ok, _ = run_db(db.create_note, title, content, success="Note saved.")
                if ok:
                    st.rerun()

    search = st.text_input("🔍 Search notes")
    ok, notes = run_db(db.list_notes, search=search or None)
    if not ok:
        return
    st.caption(f"{len(notes)} note(s)")
    if not notes:
        st.info("No notes found.")
    for note in notes:
        with st.expander(f"{note['title']}  ·  {note['updated_at'][:10]}"):
            st.write(note["content"] or "_(empty)_")
            st.caption(f"Created {note['created_at']} · Updated {note['updated_at']} · #{note['id']}")
            key = f"edit_note_{note['id']}"
            new_title = st.text_input("Title", value=note["title"], key=f"{key}_title")
            new_content = st.text_area("Content", value=note["content"], key=f"{key}_content", height=150)
            c1, c2 = st.columns([1, 1])
            if c1.button("Save changes", key=f"{key}_save", type="primary"):
                ok, _ = run_db(db.update_note, note["id"], new_title, new_content, success="Note updated.")
                if ok:
                    st.rerun()
            with c2:
                delete_button(f"note “{note['title']}”", f"note_{note['id']}", db.delete_note, note["id"], what="this note")


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
def page_memory() -> None:
    st.header("🧠 Memory")
    st.caption("Things Nomi remembers about you. Tell Nomi “Remember that …” in the chat, or add one here.")
    show_flash()

    with st.form("new_memory", clear_on_submit=True):
        content = st.text_input("Add a memory", placeholder="I prefer studying in the morning.")
        if st.form_submit_button("Save memory", type="primary"):
            ok, result = run_db(db.save_memory, content)
            if ok:
                flash("Already remembered." if result["duplicate"] else "Memory saved.")
                st.rerun()

    ok, memories = run_db(db.list_memories)
    if not ok:
        return
    if not memories:
        st.info("Nothing saved yet.")
    for memory in memories:
        with st.container(border=True):
            left, right = st.columns([6, 1])
            left.markdown(f"{memory['content']}")
            left.caption(f"Saved {memory['created_at']} · #{memory['id']}")
            with right:
                delete_button("memory", f"memory_{memory['id']}", db.delete_memory, memory["id"], what="this memory")


# --------------------------------------------------------------------------
# Daily plan
# --------------------------------------------------------------------------
def page_daily_plan() -> None:
    st.header("📅 Daily Plan")
    st.caption(f"Today is {db.now().strftime('%A, %d %B %Y')}")

    data = ai.gather_daily_data()
    m1, m2, m3 = st.columns(3)
    m1.metric("Due today", len(data["today"]))
    m2.metric("Overdue", len(data["overdue"]))
    m3.metric("Other open tasks", len(data["other"]))

    extra = st.text_input("Anything I should know about today? (optional)", placeholder="I only have 3 hours after lunch")
    if st.button("✨ Plan my day", type="primary"):
        try:
            with st.spinner("Nomi is planning your day..."):
                st.session_state["daily_plan"] = ai.generate_daily_plan(extra)
        except ai.MissingAPIKeyError as exc:
            st.error(str(exc))
        except ai.NomiAIError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not create the plan: {exc}")

    if st.session_state.get("daily_plan"):
        st.markdown(st.session_state["daily_plan"])

    with st.expander("Tasks used for this plan"):
        for title, tasks in (("Due today", data["today"]), ("Overdue", data["overdue"]), ("Other open", data["other"])):
            st.markdown(f"**{title}**")
            if tasks:
                for t in tasks:
                    st.markdown(f"- {STATUS_ICON[t['status']]} {t['title']} ({t['priority']})")
            else:
                st.caption("None")


# --------------------------------------------------------------------------
# Weekly review
# --------------------------------------------------------------------------
def page_weekly_review() -> None:
    st.header("📊 Weekly Review")
    period_label = st.radio("Period", ["This week (Mon–today)", "Last 7 days"], horizontal=True)
    period = "this_week" if period_label.startswith("This") else "last_7_days"
    start, end = db.week_range(period)
    st.caption(f"{start} → {end}")

    try:
        summary = db.week_summary(start, end)
    except NomiError as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasks completed", len(summary["completed"]))
    c2.metric("Still incomplete", len(summary["incomplete"]))
    c3.metric("Projects with activity", len(summary["projects"]))
    c4.metric("Notes created", len(summary["notes_created"]))

    if st.button("✨ Generate weekly review", type="primary"):
        try:
            with st.spinner("Nomi is reviewing your week..."):
                text, _ = ai.generate_weekly_review(period)
            st.session_state["weekly_review"] = text
        except ai.MissingAPIKeyError as exc:
            st.error(str(exc))
        except ai.NomiAIError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not create the review: {exc}")

    if st.session_state.get("weekly_review"):
        st.markdown(st.session_state["weekly_review"])

    with st.expander("Raw activity data"):
        from tools import format_week_summary

        st.text(format_week_summary(summary))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    try:
        init_database()
    except NomiError as exc:
        st.error(f"Database problem: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not start the database: {exc}")
        st.stop()

    with st.sidebar:
        st.title("Nomi")
        st.caption("Your Personal AI Assistant")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        if page == "💬 Chat" and st.button("Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()
        st.caption(f"Model: `{ai.get_model_name()}`")

    {
        "💬 Chat": page_chat,
        "✅ Tasks": page_tasks,
        "📁 Projects": page_projects,
        "📝 Notes": page_notes,
        "🧠 Memory": page_memory,
        "📅 Daily Plan": page_daily_plan,
        "📊 Weekly Review": page_weekly_review,
    }[page]()


main()
