import json5
import logging
from qwen_agent.tools.base import BaseTool, register_tool
from src.vector_store import search_in_store
from src.config import LLM_MODEL_NAME, LLM_MODEL_SERVER, K_RETRIEVED_CHUNKS

logger = logging.getLogger(__name__)

def get_llm_config():
    """Возвращает конфигурацию для LLM."""
    return {
        'model': LLM_MODEL_NAME,
        'model_server': LLM_MODEL_SERVER,
        'api_key': 'EMPTY',
        'generate_cfg': {
            'thought_in_content': False,
        }
    }

def get_system_instruction():
    """Возвращает системную инструкцию для агента."""
    return '''Ты — вежливый и точный ассистент-справочник. 
Твоя задача — отвечать на вопросы пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленной информации из документов.

Правила работы:
1. Для поиска информации всегда вызывай инструмент `knowledge_base_retriever`.
2. Внимательно изучи полученный из инструмента контекст.
3. Сформулируй четкий и ясный ответ на основе этого контекста.
4. Если в контексте нет информации для ответа или инструмент вернул ошибку/пустой результат, твой ЕДИНСТВЕННЫЙ ответ должен быть: "К сожалению, я не нашел информации по вашему вопросу в документах."
НИКОГДА не придумывай информацию и не используй свои общие знания.
'''

@register_tool('knowledge_base_retriever')
class KnowledgeBaseRetriever(BaseTool):
    """
    Инструмент для прямого поиска информации в базе знаний по документам.
    """
    description = (
        "Ищет и извлекает релевантную информацию из базы знаний по документам для ответа на вопрос пользователя. "
        "Используй этот инструмент всегда, когда нужно получить информацию из документов."
    )
    parameters = [{'name': 'query', 'type': 'string', 'description': 'Поисковый запрос, сформулированный на основе вопроса пользователя', 'required': True}]

    def __init__(self, embedding_model, chroma_collection, cfg=None):
        super().__init__(cfg)
        if embedding_model is None or chroma_collection is None:
            raise ValueError("embedding_model и chroma_collection должны быть предоставлены.")
        self.embedding_model = embedding_model
        self.chroma_collection = chroma_collection

    def call(self, params: str, **kwargs) -> str:
        query = ""
        try:
            parsed_params = json5.loads(params)
            query = parsed_params.get('query', '') if isinstance(parsed_params, dict) else str(parsed_params)
        except Exception:
            query = params
        
        query = query.strip().strip('"').strip("'")
        if not query:
            logger.warning("Получен пустой поисковый запрос.")
            return "Поиск не дал результатов."

        try:
            docs = search_in_store(query, self.embedding_model, self.chroma_collection, k=K_RETRIEVED_CHUNKS)
            
            if not docs:
                logger.info("В базе знаний не найдено релевантных чанков.")
                return "Поиск не дал результатов."

            final_context = "\n\n---\n\n".join(docs)
            logger.info(f"Найдено {len(docs)} релевантн(ых) чанков.")
            return final_context
            
        except Exception as e:
            logger.error(f"Произошла ошибка при поиске в базе знаний: {e}", exc_info=True)
            return "Ошибка при поиске в базе знаний." 