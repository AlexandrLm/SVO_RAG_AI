import sys
import os
import asyncio

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from qwen_agent.agents import Assistant
from src.agent_config import get_llm_config, get_system_instruction, HelpdeskRetriever
from src.data_processor import initialize_embedding_model, load_and_chunk_pdfs
from src.vector_store import get_chroma_collection, populate_collection
from src.history_manager import init_db, get_history, add_message, prune_history

# --- Модели данных для API (Pydantic) ---

class AskRequest(BaseModel):
    query: str
    session_id: str

class AskResponse(BaseModel):
    answer: str

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
    print("Инициализация сервера...")
    
    # Инициализация и очистка истории
    init_db()
    prune_history(messages_to_keep=14)

    loop = asyncio.get_event_loop()
    
    # 1. Инициализация моделей
    app_state["embedding_model"] = await loop.run_in_executor(None, initialize_embedding_model)
    print("Модель для эмбеддингов загружена.")
    
    # 2. Инициализация ChromaDB
    app_state["chroma_collection"] = await loop.run_in_executor(None, get_chroma_collection)
    print("База векторов ChromaDB инициализирована.")
    
    # 3. Загрузка документов в базу, если это необходимо
    collection = app_state["chroma_collection"]
    if await loop.run_in_executor(None, collection.count) == 0:
        print("База знаний пуста. Начинается обработка и заполнение...")
        docs_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
        if not os.path.exists(docs_folder) or not os.listdir(docs_folder):
            raise RuntimeError(f"Папка '{docs_folder}' пуста или не существует. Невозможно заполнить базу знаний.")
        
        document_chunks = await loop.run_in_executor(None, load_and_chunk_pdfs, docs_folder)
        await loop.run_in_executor(None, populate_collection, collection, document_chunks, app_state["embedding_model"])
    else:
        print("База знаний уже заполнена, пропуск обработки документов.")

    # 4. Настройка и создание экземпляра агента (бота)
    llm_cfg = get_llm_config()
    system_instruction = get_system_instruction()
    
    # Инициализация и настройка кастомного инструмента
    helpdesk_tool = HelpdeskRetriever()
    helpdesk_tool.embedding_model = app_state["embedding_model"]
    helpdesk_tool.chroma_collection = app_state["chroma_collection"]
    
    tools = [helpdesk_tool]
    
    app_state["bot"] = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools
    )
    print("Агент (бот) успешно создан и настроен.")
    print("--- Сервер готов к работе ---")

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
        messages = get_history(session_id)
        
        # 2. Добавляем текущий вопрос пользователя
        messages.append({'role': 'user', 'content': query})
        
        # 3. Сохраняем вопрос пользователя в БД
        add_message(session_id, 'user', query)

        # Переменная для хранения полного ответа
        full_response_content = ""

        # Запускаем генератор и получаем финальный ответ
        for response in bot.run(messages=messages):
            # Нас интересует самое последнее сообщение в истории
            last_message = response[-1]
            if last_message.get('role') == 'assistant':
                full_response_content = last_message.get('content', '')

        # Теперь, когда у нас есть полный ответ, убираем из него "мысли"
        final_content = full_response_content
        think_end_tag = '</think>'
        end_tag_pos = full_response_content.find(think_end_tag)
        if end_tag_pos != -1:
            # Отсекаем <think>...</think> и берем только чистый ответ
            final_content = full_response_content[end_tag_pos + len(think_end_tag):].strip()
        
        if final_content:
            # 5. Сохраняем ответ ассистента в БД
            add_message(session_id, 'assistant', final_content)

        return AskResponse(answer=final_content)

    except Exception as e:
        print(f"Критическая ошибка при обработке запроса: {e}")
        # Возвращаем общее сообщение об ошибке, но в логах будет видно детальное исключение
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# --- Запуск сервера (для локальной отладки) ---

if __name__ == "__main__":
    import uvicorn
    # Добавляем корневую папку проекта в sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run(app, host="0.0.0.0", port=8000) 