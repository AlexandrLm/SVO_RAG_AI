import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

def initialize_embedding_model():
    """
    Инициализирует и возвращает эмбеддинг-модель.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'Qwen/Qwen3-Embedding-0.6B'
    model = SentenceTransformer(model_name, device=device)
    print(f"Эмбеддинг-модель '{model_name}' загружена на {device.upper()}.")
    return model

def load_and_chunk_pdfs(folder_path, chunk_size=1000, chunk_overlap=150):
    """
    Загружает все PDF из папки, читает текст и разбивает на чанки
    с помощью рекурсивного сплиттера.
    """
    all_texts = []
    print("─" * 50)
    print(f"Начало обработки документов из папки: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"  - Обработка файла: {file_path}")
            try:
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
                if text.strip():  # Убедимся, что текст не пустой
                    all_texts.append(text)
                else:
                    print(f"    ! Предупреждение: Файл {filename} пуст или не удалось извлечь текст.")
            except Exception as e:
                print(f"    ! Ошибка: Не удалось прочитать файл {filename}: {e}")

    if not all_texts:
        print("Не найдено текстов для обработки.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # Добавил точку для лучшего разбиения
        length_function=len
    )
    
    documents = text_splitter.create_documents(all_texts)
    chunk_texts = [doc.page_content for doc in documents]
    
    print(f"Обработка завершена. Всего создано {len(chunk_texts)} чанков.")
    print("─" * 50)
    return chunk_texts 