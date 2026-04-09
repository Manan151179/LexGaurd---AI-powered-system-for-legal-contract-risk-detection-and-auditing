"""
LexGuard — Chat History Persistence (SQLite)
=============================================
Saves and loads chat sessions to/from a local SQLite database.
Zero configuration, no external services required.

Tables:
  - chat_sessions: session_id, title, created_at, updated_at
  - chat_messages: message_id, session_id, role, content, risk_level, metadata, created_at
"""

import os
import uuid
import json
import sqlite3
import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Database path — stored alongside project data
# ──────────────────────────────────────────────
_DB_DIR = Path(__file__).resolve().parent / "project_data_store"
_DB_DIR.mkdir(exist_ok=True)
_DB_PATH = _DB_DIR / "chat_history.db"


def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_tables():
    """Create chat history tables if they don't exist."""
    try:
        conn = _get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id  TEXT PRIMARY KEY,
                title       TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id  TEXT PRIMARY KEY,
                session_id  TEXT REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT,
                risk_level  TEXT,
                metadata    TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id);
        """)
        conn.close()
        print("✅ Chat history tables ready (SQLite).")
    except Exception as e:
        print(f"⚠️ Could not init chat tables: {e}")


def new_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())[:12]


def save_message(
    session_id: str,
    role: str,
    content: str,
    risk_level: str = None,
    title: str = None,
    annotations: list = None,
):
    """Save a single message to SQLite. Creates session if needed."""
    try:
        conn = _get_connection()
        now = datetime.datetime.utcnow().isoformat()

        # Upsert session
        session_title = title or (content[:80] if role == "user" else None)
        conn.execute(
            """
            INSERT INTO chat_sessions (session_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = ?
            """,
            (session_id, session_title or "New Chat", now, now, now),
        )

        # Update title if first user message
        if session_title:
            conn.execute(
                """
                UPDATE chat_sessions SET title = ?
                WHERE session_id = ? AND title = 'New Chat'
                """,
                (session_title, session_id),
            )

        # Serialize annotations to JSON
        metadata_json = None
        if annotations:
            serializable = [
                {"clause_name": c, "info": info, "ref_num": r}
                for c, info, r in annotations
            ]
            metadata_json = json.dumps(serializable)

        # Insert message
        msg_id = str(uuid.uuid4())[:12]
        conn.execute(
            """
            INSERT INTO chat_messages
                (message_id, session_id, role, content, risk_level, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, session_id, role, content, risk_level or "N/A", metadata_json, now),
        )

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Chat save error: {e}")


def list_sessions(limit: int = 20) -> list[dict]:
    """List recent chat sessions."""
    try:
        conn = _get_connection()
        rows = conn.execute(
            """
            SELECT session_id, title, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {"id": r["session_id"], "title": r["title"],
             "created": r["created_at"], "updated": r["updated_at"]}
            for r in rows
        ]
    except Exception as e:
        print(f"⚠️ Chat list error: {e}")
        return []


def load_session(session_id: str) -> list[dict]:
    """Load all messages for a given session, including annotations."""
    try:
        conn = _get_connection()
        rows = conn.execute(
            """
            SELECT role, content, risk_level, created_at, metadata
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        ).fetchall()
        conn.close()

        messages = []
        for r in rows:
            msg = {
                "role": r["role"],
                "content": r["content"],
                "risk": r["risk_level"],
                "trace": None,
                "latency": None,
            }
            metadata_str = r["metadata"]
            if metadata_str and metadata_str != "null":
                try:
                    parsed = json.loads(metadata_str)
                    annotations = [
                        (item["clause_name"], item["info"], item["ref_num"])
                        for item in parsed
                    ]
                    msg["annotations"] = annotations
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            messages.append(msg)
        return messages
    except Exception as e:
        print(f"⚠️ Chat load error: {e}")
        return []


def delete_session(session_id: str):
    """Delete a session and all its messages."""
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Chat delete error: {e}")


def update_title(session_id: str, title: str):
    """Update the title of a chat session."""
    try:
        conn = _get_connection()
        conn.execute(
            "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
            (title[:100], session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Title update error: {e}")


def generate_title(user_message: str) -> str:
    """Use Gemini to generate a concise chat title from the user's first message."""
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""Generate a very short title (3-6 words max) for a legal contract audit conversation that starts with this message:

"{user_message[:500]}"

Reply with ONLY the title, no quotes, no punctuation at the end. Examples:
- Non-Compete Clause Review
- License Agreement Risk Scan
- Termination Rights Analysis""",
        )
        title = response.text.strip().strip('"').strip("'")[:100]
        return title if title else user_message[:40]
    except Exception as e:
        print(f"⚠️ Title generation error: {e}")
        return user_message[:40]
