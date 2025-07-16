import sqlite3
import datetime
import logging
from typing import List, Dict
from src.config import HISTORY_DB_PATH, HISTORY_MESSAGES_TO_KEEP

logger = logging.getLogger(__name__)

def init_db():
    """Инициализирует базу данных и создает таблицу диалогов, если она не существует."""
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Индекс для ускорения поиска по session_id
        cursor.execute("CREATE INDEX IF NOT EXISTS session_id_idx ON conversations (session_id)")
        conn.commit()
    logger.info("База данных для истории диалогов инициализирована.")

def add_message(session_id: str, role: str, content: str):
    """Добавляет сообщение в историю диалога."""
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()

def get_history(session_id: str, limit: int = 14) -> List[Dict[str, str]]:
    """
    Извлекает последние `limit` сообщений из истории диалога для указанного session_id.
    По умолчанию лимит 14 (7 вопросов + 7 ответов).
    """
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        # Сообщения извлекаются в обратном порядке (DESC), поэтому их нужно перевернуть
        history = [{"role": row["role"], "content": row["content"]} for row in reversed(cursor.fetchall())]
        return history

def prune_history():
    """
    Для каждой сессии оставляет только последние `messages_to_keep` сообщений, удаляя более старые.
    """
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        cursor = conn.cursor()
        # Сначала получаем все уникальные session_id
        cursor.execute("SELECT DISTINCT session_id FROM conversations")
        sessions = [row[0] for row in cursor.fetchall()]
        
        total_deleted = 0
        for session_id in sessions:
            # Для каждой сессии находим ID сообщений, которые нужно удалить
            # (все, кроме последних N)
            cursor.execute("""
                DELETE FROM conversations 
                WHERE id IN (
                    SELECT id FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT -1 OFFSET ?
                )
            """, (session_id, HISTORY_MESSAGES_TO_KEEP))
            total_deleted += cursor.rowcount
        
        conn.commit()

    if total_deleted > 0:
        logger.info(f"Очистка истории: удалено {total_deleted} старых записей (оставлено по {HISTORY_MESSAGES_TO_KEEP} в каждой сессии).")
    return total_deleted 