import sys
import os
import asyncio
import logging
import re

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
from src.agent_config import get_llm_config, get_system_instruction, KnowledgeBaseRetriever
from src.data_processor import initialize_embedding_model, load_and_chunk_pdfs
from src.vector_store import get_chroma_collection, populate_collection
from src.history_manager import init_db, get_history, add_message, prune_history
from src.config import DOCS_DIR, HISTORY_MESSAGES_TO_KEEP
from src.logger_config import setup_logging

# --- Настройка логирования ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Модели данных для API (Pydantic) ---

class AskRequest(BaseModel):
    query: str
    session_id: str

class AskResponse(BaseModel):
    answer: str

def _strip_think_content(text: str) -> str:
    """
    Принудительно удаляет блоки <think>...</think> из ответа LLM.
    """
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# --- Инициализация FastAPI и состояние приложения ---

app = FastAPI(
    title="SVO RAG AI Assistant",
    description="API для чат-бота-справочника на базе Qwen-Agent и RAG.",
    version="1.0.0",
)

# Мы будем использовать state FastAPI для хранения этих объектов
app_state = {} 

# --- Логика инициализации и зависимостей FastAPI ---

@app.on_event("startup")
async def startup_event():
    """
    Асинхронная инициализация всех необходимых компонентов при старте сервера.
    """
    logger.info("Инициализация сервера...")
    
    # Инициализация и очистка истории
    init_db()
    prune_history()

    loop = asyncio.get_event_loop()
    
    # 1. Инициализация моделей
    app_state["embedding_model"] = await loop.run_in_executor(None, initialize_embedding_model)
    logger.info("Модель для эмбеддингов загружена.")
    
    # 2. Инициализация ChromaDB
    app_state["chroma_collection"] = await loop.run_in_executor(None, get_chroma_collection)
    logger.info("База векторов ChromaDB инициализирована.")
    
    # 3. Загрузка документов в базу, если это необходимо
    collection = app_state["chroma_collection"]
    if await loop.run_in_executor(None, collection.count) == 0:
        logger.info("База знаний пуста. Начинается обработка и заполнение...")
        if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
            error_msg = f"Папка '{DOCS_DIR}' пуста или не существует. Невозможно заполнить базу знаний."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        document_chunks = await loop.run_in_executor(None, load_and_chunk_pdfs, DOCS_DIR)
        await loop.run_in_executor(None, populate_collection, collection, document_chunks, app_state["embedding_model"])
    else:
        logger.info("База знаний уже заполнена, пропуск обработки документов.")

    # 4. Настройка и создание экземпляра агента (бота)
    llm_cfg = get_llm_config()
    system_instruction = get_system_instruction()
    
    # Инициализация и настройка кастомного инструмента
    knowledge_retriever = KnowledgeBaseRetriever(
        embedding_model=app_state["embedding_model"],
        chroma_collection=app_state["chroma_collection"],
        cfg=llm_cfg
    )

    tools = [knowledge_retriever]
    
    app_state["bot"] = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools
    )
    logger.info("Агент (бот) успешно создан и настроен с KnowledgeBaseRetriever.")
    logger.info("--- Сервер готов к работе ---")

def get_bot() -> Assistant:
    """Зависимость (dependency) для получения экземпляра бота в эндпоинтах."""
    bot = app_state.get("bot")
    if not bot:
        raise HTTPException(status_code=503, detail="Бот не инициализирован. Попробуйте позже.")
    return bot

# --- API эндпоинты ---

@app.get("/", summary="Проверка работы сервера")
def read_root():
    return {"status": "SVO RAG AI Assistant API is running"}

@app.post("/api/v1/ask", response_model=AskResponse, summary="Задать вопрос ассистенту")
async def ask(request: AskRequest, bot: Assistant = Depends(get_bot)):
    """
    Принимает вопрос, обрабатывает его с помощью RAG-агента и возвращает полный ответ.
    Учитывает историю диалога по session_id.
    """
    query = request.query
    session_id = request.session_id

    if not query or not session_id:
        raise HTTPException(status_code=400, detail="Поля 'query' и 'session_id' не могут быть пустыми")

    try:
        # 1. Получаем историю диалога
        messages = get_history(session_id, limit=HISTORY_MESSAGES_TO_KEEP)
        
        # 2. Добавляем текущий вопрос пользователя
        messages.append({'role': 'user', 'content': query})
        
        # 3. Сохраняем вопрос пользователя в БД
        add_message(session_id, 'user', query)

        final_content = ""
        assistant_responses = []

        # Запускаем генератор и дожидаемся, пока он полностью отработает.
        # Переменная assistant_responses будет содержать итоговый список сообщений от ассистента.
        for responses in bot.run(messages=messages):
            assistant_responses = responses

        # Извлекаем контент из последнего сообщения ассистента.
        # Qwen-Agent с `thought_in_content=False` возвращает ответ в `content` без "мыслей".
        if assistant_responses and assistant_responses[-1].get('role') == 'assistant':
            raw_content = assistant_responses[-1].get('content', '')
            final_content = _strip_think_content(raw_content)
        
        if final_content:
            # 5. Сохраняем ответ ассистента в БД
            add_message(session_id, 'assistant', final_content)

        return AskResponse(answer=final_content)

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке запроса: {e}", exc_info=True)
        # Возвращаем общее сообщение об ошибке, но в логах будет видно детальное исключение
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# --- Запуск сервера (для локальной отладки) ---

if __name__ == "__main__":
    import uvicorn
    # Добавляем корневую папку проекта в sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run(app, host="0.0.0.0", port=8000) 