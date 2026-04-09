"""
LexGuard — Chat History Persistence (Snowflake)
================================================
Saves and loads chat sessions to/from Snowflake for persistent history.

Tables:
  - CHAT_SESSIONS: session_id, title, created_at, updated_at
  - CHAT_MESSAGES: message_id, session_id, role, content, risk_level, created_at
"""

import os
import uuid
import json
import datetime
from dotenv import load_dotenv

load_dotenv()

def _get_connection():
    """Get a Snowflake connection using env variables."""
    import snowflake.connector
    return snowflake.connector.connect(
        user=os.getenv("SNOW_USER"),
        password=os.getenv("SNOW_PASS"),
        account=os.getenv("SNOW_ACCOUNT"),
        role=os.getenv("SNOW_ROLE", "TRAINING_ROLE"),
        warehouse=os.getenv("SNOW_WH", "COMPUTE_WH"),
        database=os.getenv("SNOW_DB", "LEXGUARD_DB"),
        schema=os.getenv("SNOW_SCHEMA", "CONTRACT_DATA")
    )

def init_tables():
    """Create chat history tables if they don't exist."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS CHAT_SESSIONS (
                SESSION_ID    VARCHAR(64) PRIMARY KEY,
                TITLE         VARCHAR(500),
                CREATED_AT    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                UPDATED_AT    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS CHAT_MESSAGES (
                MESSAGE_ID    VARCHAR(64) PRIMARY KEY,
                SESSION_ID    VARCHAR(64) REFERENCES CHAT_SESSIONS(SESSION_ID),
                ROLE          VARCHAR(20),
                CONTENT       VARCHAR(16777216),
                RISK_LEVEL    VARCHAR(20),
                METADATA      VARCHAR(16777216),
                CREATED_AT    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        # Add METADATA column if table already exists without it
        try:
            cur.execute("ALTER TABLE CHAT_MESSAGES ADD COLUMN IF NOT EXISTS METADATA VARCHAR(16777216)")
        except Exception:
            pass
        conn.close()
        print("✅ Chat history tables ready.")
    except Exception as e:
        print(f"⚠️ Could not init chat tables: {e}")

def new_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())[:12]

def save_message(session_id: str, role: str, content: str, risk_level: str = None, title: str = None, annotations: list = None):
    """Save a single message to Snowflake. Creates session if needed."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        
        # Upsert session
        now = datetime.datetime.utcnow().isoformat()
        session_title = title or content[:80].replace("'", "''") if role == "user" else None
        
        cur.execute(f"""
            MERGE INTO CHAT_SESSIONS t
            USING (SELECT '{session_id}' AS SID) s
            ON t.SESSION_ID = s.SID
            WHEN MATCHED THEN UPDATE SET UPDATED_AT = '{now}'
            WHEN NOT MATCHED THEN INSERT (SESSION_ID, TITLE, CREATED_AT, UPDATED_AT)
                VALUES ('{session_id}', '{session_title or "New Chat"}', '{now}', '{now}')
        """)
        
        # Update title if this is the first user message
        if session_title:
            cur.execute(f"""
                UPDATE CHAT_SESSIONS SET TITLE = '{session_title}' 
                WHERE SESSION_ID = '{session_id}' AND TITLE = 'New Chat'
            """)
        
        # Serialize annotations to JSON if present
        metadata_json = "null"
        if annotations:
            # annotations is a list of (clause_name, info_dict, ref_num) tuples
            serializable = [
                {"clause_name": c, "info": info, "ref_num": r}
                for c, info, r in annotations
            ]
            metadata_json = json.dumps(serializable).replace("'", "''")
        
        # Insert message
        msg_id = str(uuid.uuid4())[:12]
        safe_content = content.replace("'", "''")[:16000000]
        risk = risk_level or "N/A"
        cur.execute(f"""
            INSERT INTO CHAT_MESSAGES (MESSAGE_ID, SESSION_ID, ROLE, CONTENT, RISK_LEVEL, METADATA, CREATED_AT)
            VALUES ('{msg_id}', '{session_id}', '{role}', '{safe_content}', '{risk}', '{metadata_json}', '{now}')
        """)
        
        conn.close()
    except Exception as e:
        print(f"⚠️ Chat save error: {e}")

def list_sessions(limit: int = 20):
    """List recent chat sessions from Snowflake."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT SESSION_ID, TITLE, CREATED_AT, UPDATED_AT
            FROM CHAT_SESSIONS
            ORDER BY UPDATED_AT DESC
            LIMIT {limit}
        """)
        results = cur.fetchall()
        conn.close()
        return [{"id": r[0], "title": r[1], "created": str(r[2]), "updated": str(r[3])} for r in results]
    except Exception as e:
        print(f"⚠️ Chat list error: {e}")
        return []

def load_session(session_id: str):
    """Load all messages for a given session, including annotations."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT ROLE, CONTENT, RISK_LEVEL, CREATED_AT, METADATA
            FROM CHAT_MESSAGES
            WHERE SESSION_ID = '{session_id}'
            ORDER BY CREATED_AT ASC
        """)
        results = cur.fetchall()
        conn.close()
        
        messages = []
        for r in results:
            msg = {"role": r[0], "content": r[1], "risk": r[2], "trace": None, "latency": None}
            # Restore annotations from METADATA JSON
            metadata_str = r[4] if len(r) > 4 else None
            if metadata_str and metadata_str != "null":
                try:
                    parsed = json.loads(metadata_str)
                    # Convert back to list of (clause_name, info_dict, ref_num) tuples
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
        cur = conn.cursor()
        cur.execute(f"DELETE FROM CHAT_MESSAGES WHERE SESSION_ID = '{session_id}'")
        cur.execute(f"DELETE FROM CHAT_SESSIONS WHERE SESSION_ID = '{session_id}'")
        conn.close()
    except Exception as e:
        print(f"⚠️ Chat delete error: {e}")

def update_title(session_id: str, title: str):
    """Update the title of a chat session."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        safe_title = title.replace("'", "''")[:100]
        cur.execute(f"""
            UPDATE CHAT_SESSIONS SET TITLE = '{safe_title}'
            WHERE SESSION_ID = '{session_id}'
        """)
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
- Termination Rights Analysis"""
        )
        title = response.text.strip().strip('"').strip("'")[:100]
        return title if title else user_message[:40]
    except Exception as e:
        print(f"⚠️ Title generation error: {e}")
        return user_message[:40]
