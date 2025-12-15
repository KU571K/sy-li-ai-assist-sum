import os
from typing import List, Optional
from openai import OpenAI
from hybrid_search import SearchEngine, FaissStore

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


class RAGChain:
    """
    RAG цепочка для генерации ответов на основе найденного контекста.
    Использует гибридный поиск для получения релевантных чанков и LLM для генерации ответа.
    """
    
    def __init__(
        self,
        search_engine: SearchEngine,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Инициализация RAG цепочки.
        
        Args:
            search_engine: Экземпляр SearchEngine для поиска релевантных документов
            model: Модель OpenAI для генерации ответов
            temperature: Температура для генерации (0.0-1.0)
            max_tokens: Максимальное количество токенов в ответе
        """
        self.search_engine = search_engine
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Инициализация OpenAI клиента
        # Поддерживаем как OpenRouter, так и OpenAI
        if HAS_STREAMLIT:
            try:
                api_key = st.secrets.get("OPENROUTER_API_KEY") or st.secrets.get("OPENAI_API_KEY")
            except:
                api_key = None
        else:
            api_key = None
        
        if not api_key:
            api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError(
                'API ключ не установлен. Установите OPENROUTER_API_KEY или OPENAI_API_KEY в переменных окружения или Streamlit secrets'
            )
        
        # Определяем, используется ли OpenRouter (модель содержит "/")
        # Для OpenRouter нужно указать base_url
        if '/' in model:
            # OpenRouter использует стандартный OpenAI API формат, но другой endpoint
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            # Стандартный OpenAI
            self.client = OpenAI(api_key=api_key)
    
    def _format_context(self, context_items: List[dict]) -> str:
        """
        Форматирует найденные чанки в контекст для промпта.
        
        Args:
            context_items: Список словарей с информацией о найденных чанках
            
        Returns:
            Отформатированная строка с контекстом
        """
        formatted_parts = []
        for item in context_items:
            chunk = item.get('chunk', '')
            section = item.get('section', '')
            doc_id = item.get('doc_id', '')
            
            part = f"Документ: {doc_id}\n"
            if section:
                part += f"Раздел: {section}\n"
            part += f"Содержание: {chunk}\n"
            formatted_parts.append(part)
        
        return "\n---\n".join(formatted_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Строит промпт для LLM на основе запроса и контекста.
        
        Args:
            query: Вопрос пользователя
            context: Найденный контекст из документов
            
        Returns:
            Полный промпт для LLM
        """
        system_prompt = """Ты - AI ассистент личного кабинета студента. 
Твоя задача - отвечать на вопросы студентов на основе предоставленных документов (законы, приказы, постановления).

ВАЖНО: Вопросы студентов могут быть сформулированы простым языком, но в документах используется формальная терминология.
Ты должен находить релевантную информацию ПО СМЫСЛУ, а не только по точным словам.

Примеры соответствий формулировок:
- "Как поступить?" → "правила приема", "порядок приема", "процедура поступления", "условия приема"
- "Что нужно для поступления?" → "требования", "документы", "условия приема"
- "Как подать документы?" → "порядок подачи", "сроки подачи", "документы для приема"
- "Когда начинается прием?" → "сроки приема", "период приема", "даты приема"

Инструкции:
1. Внимательно проанализируй вопрос студента и определи, какая информация ему нужна
2. Ищи в контексте информацию ПО СМЫСЛУ, используя синонимы и связанные термины
3. Если вопрос начинается с "как", "что", "когда", "где" - ищи соответствующие правила, порядки, процедуры, сроки, требования
4. Внимательно проанализируй весь предоставленный контекст, даже если формулировки отличаются
5. Собери информацию из всех релевантных чанков, даже если она разбита на несколько частей
6. Отвечай только на основе предоставленного контекста, но используй всю доступную информацию
7. Если информация разбросана по нескольким чанкам, объедини её в полный ответ
8. Если в контексте нет информации для ответа, честно скажи об этом
9. Отвечай на русском языке, четко и структурированно
10. При необходимости ссылайся на конкретные документы или разделы
11. Будь вежливым и профессиональным"""
        
        user_prompt = f"""Контекст из документов:

{context}

Вопрос студента: {query}

Внимательно проанализируй вопрос студента. Даже если в контексте используется другая формулировка (например, "правила приема" вместо "как поступить"), найди релевантную информацию по смыслу и ответь на вопрос, используя только информацию из предоставленного контекста."""
        
        return system_prompt, user_prompt
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = False
    ) -> dict:
        """
        Генерирует ответ на вопрос пользователя используя RAG подход.
        
        Args:
            query: Вопрос пользователя
            top_k: Количество релевантных чанков для использования
            use_reranker: Использовать ли реранкер для улучшения результатов
            
        Returns:
            Словарь с ответом и метаданными:
            {
                'answer': str - сгенерированный ответ,
                'sources': List[dict] - источники использованные для ответа,
                'context_used': str - использованный контекст
            }
        """
        # Получаем релевантный контекст через гибридный поиск
        context_items = self.search_engine.retrieve_context(query, top_k=top_k)
        
        if not context_items:
            return {
                'answer': 'Извините, не удалось найти релевантную информацию для ответа на ваш вопрос.',
                'sources': [],
                'context_used': ''
            }
        
        # Форматируем контекст
        formatted_context = self._format_context(context_items)
        
        # Строим промпт
        system_prompt, user_prompt = self._build_prompt(query, formatted_context)
        
        # Генерируем ответ через OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Формируем список источников
            sources = [
                {
                    'doc_id': item.get('doc_id', ''),
                    'section': item.get('section', ''),
                    'score': item.get('score', 0.0),
                    'rank': item.get('rank', 0)
                }
                for item in context_items
            ]
            
            return {
                'answer': answer,
                'sources': sources,
                'context_used': formatted_context
            }
            
        except Exception as e:
            error_str = str(e)
            
            # Определяем тип ошибки и формируем понятное сообщение
            if "401" in error_str or "invalid_api_key" in error_str.lower() or "incorrect api key" in error_str.lower():
                error_msg = """❌ **Ошибка аутентификации API ключа**

Ваш API ключ недействителен или неверен. Пожалуйста, проверьте:

1. **Правильность ключа** - убедитесь, что вы скопировали ключ полностью без пробелов
2. **Переменную окружения** - ключ должен быть установлен как `OPENAI_API_KEY` или `OPENROUTER_API_KEY`
3. **Файл .env** - если используете .env файл, убедитесь, что он находится в корне проекта
4. **Доступ к сервису** - проверьте, что ключ имеет доступ к используемой модели

Получить новый ключ можно на:
- OpenAI: https://platform.openai.com/account/api-keys
- OpenRouter: https://openrouter.ai/keys"""
            elif "429" in error_str or "rate limit" in error_str.lower():
                error_msg = """⚠️ **Превышен лимит запросов**

Вы превысили лимит запросов к API. Подождите немного и попробуйте снова."""
            elif "model" in error_str.lower() and ("not found" in error_str.lower() or "invalid" in error_str.lower()):
                error_msg = f"""❌ **Ошибка выбора модели**

Указанная модель недоступна или не найдена. Ошибка: {error_str}

Проверьте название модели в настройках приложения."""
            else:
                error_msg = f"""❌ **Произошла ошибка при генерации ответа**

{error_str}

Если ошибка повторяется, пожалуйста, проверьте:
- Статус API сервиса
- Корректность настроек подключения
- Интернет-соединение"""
            
            return {
                'answer': error_msg,
                'sources': [],
                'context_used': formatted_context,
                'error': True
            }

