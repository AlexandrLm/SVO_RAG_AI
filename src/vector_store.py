import chromadb
import logging
from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)

def get_chroma_collection():
    """
    Инициализирует персистентный клиент ChromaDB и возвращает коллекцию.
    Данные будут храниться на диске в папке 'chroma_db'.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    logger.info(f"ChromaDB коллекция '{CHROMA_COLLECTION_NAME}' готова. Текущее количество документов: {collection.count()}.")
    return collection

def populate_collection(collection, chunks, model):
    """
    Заполняет коллекцию ChromaDB чанками документов.
    """
    if not chunks:
        logger.warning("Нет чанков для добавления в коллекцию.")
        return

    logger.info(f"Начинается заполнение коллекции... Всего чанков: {len(chunks)}")
    
    # Создаем эмбеддинги для чанков
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Создаем уникальные ID для каждого чанка, это требование ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Добавляем данные в коллекцию батчами для эффективности
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end_i = min(i + batch_size, len(chunks))
        logger.info(f"Добавление батча в ChromaDB: {i+1}-{end_i}")
        collection.add(
            embeddings=embeddings[i:end_i].tolist(), # ChromaDB ожидает list
            documents=chunks[i:end_i],
            ids=ids[i:end_i]
        )
    
    logger.info("Заполнение коллекции успешно завершено.")
    logger.info(f"Новое количество документов в коллекции: {collection.count()}.")


def search_in_store(query, model, collection, k=3) -> list[str]:
    """
    Ищет в коллекции ChromaDB k наиболее релевантных чанков и возвращает их как список строк.
    """
    if collection is None:
        logger.error("Коллекция ChromaDB не инициализирована.")
        return ["Коллекция ChromaDB не инициализирована."]

    logger.info(f"Поиск информации по запросу: '{query}'")
    # Создаем эмбеддинг для запроса
    query_embedding = model.encode([query]).tolist()
    
    # Выполняем поиск в ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    
    # Извлекаем найденные документы для формирования контекста
    documents = results.get('documents', [[]])[0]
    logger.info(f"Найдено {len(documents)} релевантн(ых) чанков.")
    return documents 