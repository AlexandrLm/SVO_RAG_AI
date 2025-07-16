import json5
from qwen_agent.tools.base import BaseTool, register_tool
from src.vector_store import search_in_store

def get_llm_config(model_name='qwen3:latest', model_server='http://localhost:11434/v1'):
    """Возвращает конфигурацию для LLM."""
    return {
        'model': model_name,
        'model_server': model_server,
        'api_key': 'EMPTY',
        'generate_cfg': {
            'thought_in_content': True,
        }
    }

def get_system_instruction():
    """Возвращает системную инструкцию для агента."""
    return '''Ты — вежливый и точный ассистент-справочник. 
Твоя задача — отвечать на вопросы пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленной информации из документов.

Правила работы:
1. Внимательно проанализируй вопрос пользователя. Если он кажется тебе неполным или двусмысленным, задай уточняющий вопрос, прежде чем использовать инструменты.
2. Для поиска информации всегда вызывай инструмент `helpdesk_retriever`.
3. Внимательно изучи полученный из инструмента контекст.
4. Сформулируй четкий и ясный ответ на основе этого контекста.
5. Если в контексте нет информации для ответа, честно скажи: "К сожалению, я не нашел информации по вашему вопросу в документах."
НИКОГДА не придумывай информацию.
'''

@register_tool('helpdesk_retriever')
class HelpdeskRetriever(BaseTool):
    """
    Кастомный инструмент для поиска информации в базе знаний по документам.
    """
    description = "Ищет релевантную информацию в базе знаний по документам для ответа на вопрос пользователя."
    parameters = [{'name': 'query', 'type': 'string', 'description': 'Вопрос пользователя', 'required': True}]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # Эти атрибуты будут установлены из main.py или server.py после инициализации
        self.embedding_model = None
        self.chroma_collection = None

    def call(self, params: str, **kwargs) -> str:
        query = ""
        try:
            # Пытаемся распарсить JSON, который сгенерировала LLM
            parsed_params = json5.loads(params)
            
            if isinstance(parsed_params, dict):
                # Идеальный случай: LLM вернула словарь {'query': '...'}.
                query = parsed_params.get('query', '')
            elif isinstance(parsed_params, list) and parsed_params:
                # Случай, когда LLM вернула список: ['...']. Берем первый элемент.
                query = str(parsed_params[0])
            elif isinstance(parsed_params, str):
                # Случай, когда LLM вернула просто строку в JSON: '"..."'
                query = parsed_params
            else:
                # Если что-то еще, используем исходную строку как есть
                query = params

        except Exception:
            # Если json5 не смог распарсить (например, LLM вернула невалидный JSON),
            # считаем, что вся строка `params` и есть наш поисковый запрос.
            print(f"Предупреждение: не удалось распарсить JSON от LLM: '{params}'. Используется вся строка как запрос.")
            query = params

        # На всякий случай убираем лишние кавычки по краям, если они есть
        query = query.strip().strip('"').strip("'")

        if not query:
            return "Ошибка: не удалось извлечь поисковый запрос из параметров LLM."
        
        if not all([self.embedding_model, self.chroma_collection]):
            return "Ошибка: Компоненты для поиска (модель, коллекция ChromaDB) не инициализированы."

        return search_in_store(
            query=query,
            model=self.embedding_model,
            collection=self.chroma_collection
        ) 