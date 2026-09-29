# Nomi — Your Personal AI Assistant

Nomi is a small personal assistant that can **actually read and change your data**: tasks, projects, notes and personal memories are stored in SQLite, and the AI uses tools to work with them. It also builds a daily plan and a weekly review from your real data.

**Stack:** Python · Streamlit · LangChain · LangGraph · Groq API · SQLite

## Files

```
nomi/
├── app.py              # Streamlit UI (all pages)
├── database.py         # SQLite tables + all queries
├── tools.py            # LangChain tools (the AI's actions)
├── ai.py               # LangGraph agent, Groq model, plan/review generators
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## How it works

```
User message → LangGraph "agent" node → Groq model (via LangChain)
                    │  needs a tool?
                    ├── yes → "tools" node → SQLite → result back to the agent → repeat
                    └── no  → final answer → Streamlit
```

- One assistant, one graph: `START → agent → (tools → agent)* → END`.
- Tools (`tools.py`) call functions in `database.py`, so every action really hits SQLite.
- Deleting requires a confirmation: the AI must show you the exact item, and the delete tool refuses to run until `confirm=true` after you say yes. In the UI, deletes use a confirm popover.
- Only *relevant* memories (simple keyword match) are added to the prompt. No vector database.
- Below each answer, the chat shows which tools ran, so you can always see what Nomi really did.

## 1. Run locally

Requires Python 3.10+.

```bash
cd nomi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a key at <https://console.groq.com/keys>, then give it to Nomi in **one** of these ways:

**Option A — `.env` file** (in the `nomi/` folder):

```
GROQ_API_KEY=your-groq-api-key
```

**Option B — Streamlit secrets:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put your key in it.

Start the app:

```bash
streamlit run app.py
```

Optional settings (same places as the key): `GROQ_MODEL` (default `openai/gpt-oss-120b`, defined once in `ai.py`), `NOMI_TIMEZONE` (default `Africa/Lagos`, used for "today"/"tomorrow"), `NOMI_DB_PATH` (database file location).

## 2. Put the project on GitHub

`.gitignore` already keeps your key and database out of Git.

```bash
cd nomi
git init
git add .
git commit -m "Nomi personal assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/nomi.git
git push -u origin main
```

Create the empty repository on GitHub first (github.com → New repository). A private repo works with Streamlit Community Cloud.

## 3. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub** (or choose your private repo).
3. Pick your repository, branch `main`, and set **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   ```

   Optional: add `GROQ_MODEL = "..."` or `NOMI_TIMEZONE = "..."` on new lines.
5. Click **Deploy**. To change secrets later: app menu → **Settings → Secrets**, save, then reboot the app.

No separate backend is needed. Streamlit runs everything.

## ⚠️ SQLite on Streamlit Community Cloud

Nomi keeps its data in a local SQLite file (`nomi.db`). Streamlit Community Cloud's local filesystem is **not guaranteed permanent storage**: the file can be lost when the app reboots, is redeployed or moves to a new container. Treat the cloud copy as convenient, not durable. Practical advice:

- Locally, `nomi.db` is a normal file. Back it up by copying it.
- On Streamlit Cloud, don't rely on it for data you can't afford to lose.
- If you later need permanent hosted storage, switch the database layer (`database.py`) to a hosted database. That is intentionally out of scope for now.

## Things worth knowing

- **Tasks "completed this week"** means tasks currently marked Completed that were last updated during the period. Nomi has no separate completion timestamp, so editing an old completed task can make it appear in the current week.
- **Chat history** lives in Streamlit session state only; it resets when you refresh or restart the app. Your tasks, notes, projects and memories persist in SQLite.
- **Free Groq limits:** if you hit rate limits, Nomi shows a message. Wait a moment and retry.
- **Model changes:** Groq retires models over time. If you see a "model not available" message, set `GROQ_MODEL` to a current tool-capable model from <https://console.groq.com/docs/models>.

## Try these in the chat

- "Create a high priority task called Finish biology assignment for tomorrow."
- "What do I need to do today?" / "Show me my unfinished tasks."
- "Mark my biology task complete."
- "Create a project called Nomi." / "Add a task Build database to the Nomi project."
- "Save this as a note: I want to build an AI learning platform."
- "Remember that I prefer studying in the morning." / "What do you remember about me?"
- "Plan my day." / "What did I accomplish this week?"
- "Delete the task Old thing." (Nomi asks you to confirm first.)
