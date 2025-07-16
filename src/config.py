import os

# --- Пути ---
# Определяем корневую директорию проекта
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Путь к папке с документами для RAG
DOCS_DIR = os.path.join(ROOT_DIR, 'docs')
# Путь к базе данных ChromaDB
CHROMA_DB_PATH = os.path.join(ROOT_DIR, 'chroma_db')
# Путь к базе данных для хранения истории диалогов
HISTORY_DB_PATH = os.path.join(ROOT_DIR, 'history.db')

# --- Настройки LLM ---
LLM_MODEL_NAME = 'qwen3:latest'
LLM_MODEL_SERVER = 'http://localhost:11434/v1'

# --- Настройки Эмбеддинг-модели ---
EMBEDDING_MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'

# --- Настройки ChromaDB ---
CHROMA_COLLECTION_NAME = "svo_rag_docs"

# --- Настройки RAG ---
# Количество извлекаемых чанков
K_RETRIEVED_CHUNKS = 10
# Размер чанка при нарезке документов
CHUNK_SIZE = 1000
# Перекрытие чанков
CHUNK_OVERLAP = 150

# --- Настройки истории диалогов ---
# Количество последних сообщений для хранения в контексте
HISTORY_MESSAGES_TO_KEEP = 2 