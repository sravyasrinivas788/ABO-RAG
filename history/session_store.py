import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")

def get_postgres_connection():
    return psycopg2.connect(POSTGRES_URL)


def init_db():
    conn = get_postgres_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id ON conversation_history (session_id);
    """)
    conn.commit()
    cursor.close()
    conn.close()

def save_conversation(session_id, question, answer):
    conn = get_postgres_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversation_history (session_id, question, answer)
        VALUES (%s, %s, %s);
    """, (session_id, question, answer))
    conn.commit()
    cursor.close()
    conn.close()

def get_conversation_history(session_id,limit:int=5)->list[dict]:
    conn = get_postgres_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT question, answer, created_at
        FROM conversation_history
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT %s;
    """, (session_id, limit))
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"question": q, "answer": a, "created_at": c} for q, a, c in history]


if __name__ == "__main__":
    init_db()