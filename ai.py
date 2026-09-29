"""Nomi's brain: one LangGraph agent powered by a Groq model through LangChain.

Graph:   START -> agent -> (tools -> agent)* -> END

* "agent" calls the Groq model (with the tools bound to it).
* If the model asks for tools, the "tools" node runs them against SQLite and
  the result goes back to the model.
* When the model answers without asking for a tool, the graph ends.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import database as db
from tools import ALL_TOOLS, format_tasks, format_week_summary

# The Groq model name lives here (and can be overridden with GROQ_MODEL).
DEFAULT_MODEL = "openai/gpt-oss-120b"

MAX_HISTORY_MESSAGES = 12  # earlier chat turns sent to the model
RECURSION_LIMIT = 16  # safety cap on agent <-> tool loops


class MissingAPIKeyError(Exception):
    """GROQ_API_KEY is not configured."""


class NomiAIError(Exception):
    """A model/API problem, worded so it can be shown to the user."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting from environment variables (.env) or Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        secret = st.secrets.get(name)
        if secret:
            return str(secret)
    except Exception:  # no secrets file / not running inside Streamlit
        pass
    return default


def get_model_name() -> str:
    return get_setting("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def get_api_key() -> str:
    key = get_setting("GROQ_API_KEY")
    if not key or key.strip() in ("", "your-groq-api-key"):
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Locally: put GROQ_API_KEY=... in a .env file or in "
            ".streamlit/secrets.toml. On Streamlit Community Cloud: App settings → Secrets."
        )
    return key.strip()


@lru_cache(maxsize=4)
def _make_llm(api_key: str, model: str) -> ChatGroq:
    return ChatGroq(model=model, api_key=api_key, temperature=0.2, max_retries=2, timeout=60)


def get_llm() -> ChatGroq:
    return _make_llm(get_api_key(), get_model_name())


def friendly_error(exc: Exception) -> str:
    """Translate API/library errors into a message a person can act on."""
    text = str(exc)
    low = text.lower()
    name = type(exc).__name__
    if "authenticationerror" in name.lower() or "invalid api key" in low or "401" in low:
        return "Groq rejected the API key. Check that GROQ_API_KEY is correct and still active."
    if "ratelimit" in name.lower() or "rate limit" in low or "429" in low:
        return "Groq rate limit reached. Wait a little and try again (free tiers have per-minute and daily limits)."
    if "decommissioned" in low or "model_not_found" in low or "does not exist" in low:
        return (
            f"The Groq model '{get_model_name()}' is not available. "
            "Set GROQ_MODEL to a current tool-capable model (see console.groq.com/docs/models)."
        )
    if "tool_use_failed" in low or "failed to call a function" in low:
        return "The model produced a badly formatted tool call. Please try rephrasing your request."
    if "connection" in name.lower() or "timeout" in name.lower() or "connect" in low:
        return "Could not reach Groq. Check your internet connection and try again."
    return f"The AI request failed ({name}): {text[:300]}"


# --------------------------------------------------------------------------
# System prompt (rebuilt for every model call; only relevant memories included)
# --------------------------------------------------------------------------
def build_system_prompt(user_text: str) -> str:
    now = db.now()
    memories = db.relevant_memories(user_text, limit=5)
    memory_block = (
        "\n".join(f"- {m['content']}" for m in memories) if memories else "(none relevant)"
    )
    return f"""You are Nomi, a friendly, practical personal assistant. You help the user manage tasks, projects, notes, personal memories, daily planning and weekly reviews. All of that data lives in a real SQLite database that you access ONLY through your tools.

Today is {now.strftime('%A')}, {now.date().isoformat()} (time zone {os.environ.get('NOMI_TIMEZONE') or 'Africa/Lagos'}).

RULES
1. Anything that reads or changes stored data must use a tool. Never answer questions about the user's tasks, projects, notes or memories from imagination - call get_tasks / get_projects / get_notes / get_memories first.
2. Never claim you created, changed, completed, deleted or saved anything unless the tool result says so. If a tool returns "ERROR", tell the user honestly that it failed and why.
3. Normal conversation, advice and brainstorming need no tools.
4. Convert relative dates ("tomorrow", "Friday") into YYYY-MM-DD using today's date before calling tools.
5. To finish a task by name use complete_task with title_query. If it says several tasks match, ask the user which one.
6. Deleting: first identify the exact item with a get_* tool, then ask the user to confirm, naming the item and its ID (e.g. "Delete task #4 'Study biology'?"). Only call a delete tool with confirm=true after the user has clearly said yes in their latest message. If several items could match, ask which one. Never delete more than the user asked for.
7. "Remember that ..." means call save_memory with a short statement such as "I prefer studying in the morning."
8. "Save this as a note" means call create_note with the user's text.
9. To plan the day: call get_tasks with view="today", view="overdue" and view="incomplete", and get_projects if useful. Build Morning / Afternoon / Evening sections using ONLY those real tasks, respecting the user's memories. Do not invent tasks.
10. For "what did I accomplish this week" style questions call get_weekly_summary and report only what it returns.
11. Be concise. After using a tool, confirm plainly what happened (include the task/note/project ID and due date where relevant).

RELEVANT MEMORIES ABOUT THE USER
{memory_block}
"""


# --------------------------------------------------------------------------
# The LangGraph workflow
# --------------------------------------------------------------------------
def _last_human_text(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def build_agent():
    """Compile the LangGraph agent. Raises MissingAPIKeyError if no key is set."""
    llm_with_tools = get_llm().bind_tools(ALL_TOOLS)

    def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        system = SystemMessage(content=build_system_prompt(_last_human_text(state["messages"])))
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    return str(content)


def run_agent(agent, history: List[Dict[str, str]], user_message: str) -> Tuple[str, List[str]]:
    """Send one user message through the graph.

    history: earlier turns as [{"role": "user"|"assistant", "content": "..."}].
    Returns (assistant_reply, names_of_tools_that_ran).
    """
    user_message = (user_message or "").strip()
    if not user_message:
        raise NomiAIError("Please type a message first.")

    messages: List[BaseMessage] = []
    for turn in history[-MAX_HISTORY_MESSAGES:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))
    messages.append(HumanMessage(content=user_message))
    start_len = len(messages)

    for attempt in (1, 2):
        seen: List[BaseMessage] = messages
        try:
            for state in agent.stream(
                {"messages": messages},
                {"recursion_limit": RECURSION_LIMIT},
                stream_mode="values",
            ):
                seen = state["messages"]
            break
        except Exception as exc:  # noqa: BLE001
            tools_ran = [m.name for m in seen[start_len:] if isinstance(m, ToolMessage)]
            malformed_call = "tool_use_failed" in str(exc).lower()
            if malformed_call and not tools_ran and attempt == 1:
                continue  # nothing executed yet, so a retry is safe
            message = friendly_error(exc)
            if tools_ran:
                message += (
                    " Some actions already ran before the error (" + ", ".join(tools_ran)
                    + "), so check your Tasks/Notes pages before repeating the request."
                )
            raise NomiAIError(message) from exc

    new_messages = seen[start_len:]
    tools_used = [m.name for m in new_messages if isinstance(m, ToolMessage)]
    final = next((m for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content), None)
    reply = _to_text(final.content).strip() if final else ""
    if not reply:
        reply = "I finished, but I have nothing more to add." if tools_used else "I didn't get a response - please try again."
    return reply, tools_used


# --------------------------------------------------------------------------
# Daily plan and weekly review (single model calls over real SQLite data)
# --------------------------------------------------------------------------
def _ask_llm(system: str, prompt: str) -> str:
    try:
        response = get_llm().invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    except MissingAPIKeyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise NomiAIError(friendly_error(exc)) from exc
    text = _to_text(response.content).strip()
    if not text:
        raise NomiAIError("The model returned an empty answer. Please try again.")
    return text


def gather_daily_data() -> Dict[str, Any]:
    today_tasks = db.list_tasks(view="today")
    overdue = db.list_tasks(view="overdue")
    incomplete = db.list_tasks(view="incomplete")
    listed = {t["id"] for t in today_tasks + overdue}
    other = [t for t in incomplete if t["id"] not in listed]
    return {
        "today": today_tasks,
        "overdue": overdue,
        "other": other,
        "projects": db.list_projects(status="Active"),
        "memories": db.list_memories(limit=8),
    }


def generate_daily_plan(extra_context: str = "") -> str:
    data = gather_daily_data()
    if not (data["today"] or data["overdue"] or data["other"]):
        return (
            "You have no open tasks right now, so there is nothing to plan yet. "
            "Add a few tasks and I'll organise them into a plan."
        )
    memories = "\n".join(f"- {m['content']}" for m in data["memories"]) or "(none)"
    projects = "\n".join(
        f"- {p['name']} ({p['completed_count']}/{p['task_count']} tasks done)" for p in data["projects"]
    ) or "(none)"
    prompt = f"""Today is {db.now().strftime('%A')}, {db.today().isoformat()}.

TASKS DUE TODAY:
{format_tasks(data['today'], '(none)')}

OVERDUE TASKS:
{format_tasks(data['overdue'], '(none)')}

OTHER INCOMPLETE TASKS (no date or due later):
{format_tasks(data['other'], '(none)')}

ACTIVE PROJECTS:
{projects}

THINGS THE USER ASKED ME TO REMEMBER:
{memories}

EXTRA CONTEXT FROM THE USER TODAY: {extra_context.strip() or '(none)'}

Create a practical plan for today with these sections: Morning, Afternoon, Evening.
Rules: use ONLY the tasks listed above (mention their #IDs), never invent tasks, put overdue and high-priority work first, respect the remembered preferences, keep it realistic (not everything must fit today), and end with one short tip. If a section has nothing sensible, write a light suggestion such as "review unfinished tasks"."""
    return _ask_llm(
        "You are Nomi, a concise and practical personal planning assistant. Reply in Markdown.",
        prompt,
    )


def generate_weekly_review(period: str = "this_week") -> Tuple[str, Dict[str, Any]]:
    start, end = db.week_range(period)
    summary = db.week_summary(start, end)
    nothing = not (summary["completed"] or summary["created"] or summary["notes_created"] or summary["projects"])
    if nothing and not summary["incomplete"]:
        return (
            "There is no recorded activity for this period yet - no tasks, projects or notes. "
            "Once you use Nomi for a few days, your review will appear here.",
            summary,
        )
    prompt = f"""Here is the user's REAL recorded activity from the database.

{format_week_summary(summary)}

Write a weekly review in Markdown with exactly these sections:
1. **What you accomplished**
2. **What remains unfinished**
3. **Projects you worked on**
4. **Things that may need attention** (overdue items, stale work, tasks without dates)
5. **Suggested focus for next week**

Rules: base everything ONLY on the data above; do not invent accomplishments, tasks, projects or numbers; if a section has no data say so plainly; be encouraging but honest and concise. Note that 'completed' means marked Completed with activity during this period."""
    text = _ask_llm(
        "You are Nomi, a thoughtful and honest personal review assistant. Reply in Markdown.",
        prompt,
    )
    return text, summary
