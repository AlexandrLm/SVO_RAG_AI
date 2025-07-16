import os
import logging
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch
from src.config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def initialize_embedding_model():
    """
    Инициализирует и возвращает эмбеддинг-модель.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    logger.info(f"Эмбеддинг-модель '{EMBEDDING_MODEL_NAME}' загружена на {device.upper()}.")
    return model

def load_and_chunk_pdfs(folder_path):
    """
    Загружает все PDF из папки, читает текст и разбивает на чанки
    с помощью рекурсивного сплиттера.
    """
    all_texts = []
    logger.info("─" * 50)
    logger.info(f"Начало обработки документов из папки: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"  - Обработка файла: {file_path}")
            try:
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
                if text.strip():  # Убедимся, что текст не пустой
                    all_texts.append(text)
                else:
                    logger.warning(f"    ! Предупреждение: Файл {filename} пуст или не удалось извлечь текст.")
            except Exception as e:
                logger.error(f"    ! Ошибка: Не удалось прочитать файл {filename}: {e}", exc_info=True)

    if not all_texts:
        logger.warning("Не найдено текстов для обработки.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Добавил точку для лучшего разбиения
        length_function=len
    )
    
    documents = text_splitter.create_documents(all_texts)
    chunk_texts = [doc.page_content for doc in documents]
    
    logger.info(f"Обработка завершена. Всего создано {len(chunk_texts)} чанков.")
    logger.info("─" * 50)
    return chunk_texts 