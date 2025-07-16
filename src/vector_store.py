import chromadb

def get_chroma_collection(path="./chroma_db", collection_name="svo_rag_docs"):
    """
    Инициализирует персистентный клиент ChromaDB и возвращает коллекцию.
    Данные будут храниться на диске в папке 'chroma_db'.
    """
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name)
    print(f"ChromaDB коллекция '{collection_name}' готова. Текущее количество документов: {collection.count()}.")
    return collection

def populate_collection(collection, chunks, model):
    """
    Заполняет коллекцию ChromaDB чанками документов.
    """
    if not chunks:
        print("Нет чанков для добавления в коллекцию.")
        return

    print(f"Начинается заполнение коллекции... Всего чанков: {len(chunks)}")
    
    # Создаем эмбеддинги для чанков
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Создаем уникальные ID для каждого чанка, это требование ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Добавляем данные в коллекцию батчами для эффективности
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end_i = min(i + batch_size, len(chunks))
        print(f"Добавление батча в ChromaDB: {i+1}-{end_i}")
        collection.add(
            embeddings=embeddings[i:end_i].tolist(), # ChromaDB ожидает list
            documents=chunks[i:end_i],
            ids=ids[i:end_i]
        )
    
    print("Заполнение коллекции успешно завершено.")
    print(f"Новое количество документов в коллекции: {collection.count()}.")


def search_in_store(query, model, collection, k=5):
    """
    Ищет в коллекции ChromaDB k наиболее релевантных чанков.
    """
    if collection is None:
        return "Коллекция ChromaDB не инициализирована."

    print(f"\nПоиск информации по запросу: '{query}'")
    # Создаем эмбеддинг для запроса
    query_embedding = model.encode([query]).tolist()
    
    # Выполняем поиск в ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    
    # Извлекаем найденные документы для формирования контекста
    context = "\n---\n".join(results['documents'][0])
    print(f"Найденный контекст для генерации ответа:\n{context[:400]}...")
    return context 