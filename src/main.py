import os
import sys
import torch
from sentence_transformers import SentenceTransformer
from qwen_agent.agents import Assistant

# Добавляем корневую папку проекта в sys.path, чтобы работали импорты из src
# Это нужно, если вы запускаете main.py напрямую
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import load_and_chunk_pdfs
from src.vector_store import create_vector_store
from src.agent_config import get_llm_config, get_system_instruction, HelpdeskRetriever

def initialize_models():
    """Загружает и инициализирует эмбеддинг-модель."""
    print("Загрузка эмбеддинг-модели Qwen3-Embedding-0.6B...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство для эмбеддинг-модели: {device.upper()}")
    embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device=device)
    return embedding_model

def setup_knowledge_base(docs_folder, model):
    """Подготавливает базу знаний: чанкит документы и создает векторное хранилище."""
    if not os.path.exists(docs_folder) or not os.listdir(docs_folder):
        print(f"Папка '{docs_folder}' пуста или не существует.")
        print("Пожалуйста, создайте папку 'docs' в корне проекта и поместите в нее ваши PDF-файлы.")
        return None, None
        
    chunks = load_and_chunk_pdfs(docs_folder)
    index = create_vector_store(chunks, model)
    return chunks, index

def main():
    """Основная функция для запуска бота."""
    print("--- Запуск бота-справочника ---")
    
    # 1. Инициализация моделей
    embedding_model = initialize_models()
    
    # 2. Настройка базы знаний
    docs_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
    document_chunks, vector_index = setup_knowledge_base(docs_folder, embedding_model)
    
    if not document_chunks or vector_index is None:
        print("Не удалось создать базу знаний. Завершение работы.")
        return

    # 3. Настройка и создание агента
    llm_config = get_llm_config()
    system_instruction = get_system_instruction()
    
    # Создаем экземпляр нашего инструмента
    retriever_tool = HelpdeskRetriever()
    # "Прокидываем" в него нужные компоненты
    retriever_tool.embedding_model = embedding_model
    retriever_tool.vector_index = vector_index
    retriever_tool.document_chunks = document_chunks
    
    bot = Assistant(llm=llm_config,
                    system_message=system_instruction,
                    function_list=[retriever_tool])

    print("\n--- Бот-справочник готов к работе ---")
    
    messages = []
    try:
        while True:
            query = input("\nВаш вопрос: ")
            if query.lower() in ['exit', 'quit', 'выход']:
                break
            
            messages.append({'role': 'user', 'content': query})
            
            full_response = ""
            print("\nОтвет бота: ", end="")
            for response in bot.run(messages):
                if response['role'] == 'assistant' and 'content' in response and response['content']:
                    content = response.get('content', '')
                    print(content, end="", flush=True)
                    full_response += content

            if full_response:
                messages.append({'role': 'assistant', 'content': full_response})
            print()

    except KeyboardInterrupt:
        print("\n\nРабота бота прервана. До свидания!")
    except Exception as e:
        print(f"\nПроизошла критическая ошибка: {e}")

if __name__ == "__main__":
    main() 