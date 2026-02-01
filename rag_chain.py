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
    def __init__(
        self,
        search_engine: SearchEngine,
        model: str = "gpt-5-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        #Инициализация RAG цепочки.
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
        
        base_url = os.getenv("OPENROUTER_URL") or "https://api.artemox.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # Artemox и аналоги принимают короткие имена (gpt-4o, gpt-4o-mini), без "openai/"
        self.model = model.split("/")[-1] if "/" in model else model
    
    def _format_context(self, context_items: List[dict]) -> str:
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
    
    def _build_clarification_prompt(self, query: str, context: str) -> tuple:
        """Промпт для уточнения общего вопроса (первый этап диалога)."""
        system_prompt = """Ты — AI-ассистент личного кабинета студента в стиле портала Госуслуги.

ТВОЯ ЗАДАЧА: Проанализировать вопрос пользователя и предложить уточняющие варианты.

АЛГОРИТМ:
1. Определи, является ли вопрос ОБЩИМ (требует уточнения) или КОНКРЕТНЫМ (можно сразу ответить)
2. Если вопрос ОБЩИЙ — предложи 3-5 конкретных направлений на основе контекста
3. Если вопрос КОНКРЕТНЫЙ — верни КОНКРЕТНЫЙ: да

ПРИМЕРЫ ОБЩИХ ВОПРОСОВ:
- "Как поступить?" → требует уточнения (правила, документы, сроки, льготы)
- "Расскажи про стипендию" → требует уточнения (виды, размер, условия получения)
- "Что делать если..." → требует уточнения (разные сценарии)

ПРИМЕРЫ КОНКРЕТНЫХ ВОПРОСОВ:
- "Какие документы нужны для поступления на бюджет?" → можно ответить сразу
- "Размер социальной стипендии" → можно ответить сразу
- "Сроки подачи документов на магистратуру" → можно ответить сразу"""

        user_prompt = f"""КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}

Проанализируй вопрос. Если он ОБЩИЙ, предложи уточняющие варианты в формате:
УТОЧНЕНИЕ: Вы хотите узнать:
ВАРИАНТЫ: вариант1|вариант2|вариант3|вариант4

Если вопрос КОНКРЕТНЫЙ, верни:
КОНКРЕТНЫЙ: да"""

        return system_prompt, user_prompt

    def _build_answer_prompt(self, query: str, context: str) -> tuple:
        """Промпт для генерации ответа (второй этап или прямой ответ)."""
        system_prompt = """Ты — AI-ассистент личного кабинета студента в стиле портала Госуслуги.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если информации нет — честно скажи об этом
3. Ссылайся на конкретные документы

ФОРМАТ ОТВЕТА:
- Начни с краткого резюме (1-2 предложения)
- Используй маркированные списки
- Выделяй важное: сроки, суммы, требования
- Структурируй длинные ответы подзаголовками"""

        user_prompt = f"""КОНТЕКСТ:
{context}

ВОПРОС: {query}

Дай полный и структурированный ответ на вопрос."""

        return system_prompt, user_prompt
    
    def clarify_question(self, query: str, top_k: int = 10) -> dict:
        """
        Первый этап: определяет, нужно ли уточнение, и возвращает варианты.
        
        Returns:
            {
                'needs_clarification': bool,
                'clarification_text': str,  # "Вы хотите узнать:"
                'options': list[str],       # варианты для кнопок
                'context_items': list,      # сохраняем для второго этапа
            }
        """
        context_items = self.search_engine.retrieve_context(query, top_k=top_k)
        
        if not context_items:
            return {
                'needs_clarification': False,
                'clarification_text': '',
                'options': [],
                'context_items': []
            }
        
        formatted_context = self._format_context(context_items)
        system_prompt, user_prompt = self._build_clarification_prompt(query, formatted_context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Низкая температура для стабильности
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            # Парсим ответ
            if "КОНКРЕТНЫЙ:" in result:
                return {
                    'needs_clarification': False,
                    'clarification_text': '',
                    'options': [],
                    'context_items': context_items
                }
            
            # Извлекаем уточнение и варианты
            clarification_text = "Вы хотите узнать:"
            options = []
            
            if "УТОЧНЕНИЕ:" in result:
                parts = result.split("УТОЧНЕНИЕ:")
                if len(parts) > 1:
                    clarification_part = parts[1].split("ВАРИАНТЫ:")[0].strip()
                    if clarification_part:
                        clarification_text = clarification_part
            
            if "ВАРИАНТЫ:" in result:
                variants_part = result.split("ВАРИАНТЫ:")[-1].strip()
                options = [v.strip() for v in variants_part.split("|") if v.strip()]
                options = options[:5]  # Максимум 5 вариантов
            
            return {
                'needs_clarification': len(options) > 0,
                'clarification_text': clarification_text,
                'options': options,
                'context_items': context_items
            }
            
        except Exception as e:
            # При ошибке — не уточняем, сразу отвечаем
            return {
                'needs_clarification': False,
                'clarification_text': '',
                'options': [],
                'context_items': context_items
            }

    def generate_answer(
        self,
        query: str,
        top_k: int = 10,
        context_items: Optional[List[dict]] = None
    ) -> dict:
        """
        Генерирует ответ на вопрос.
        
        Args:
            query: Вопрос пользователя
            top_k: Количество чанков для поиска
            context_items: Предварительно загруженный контекст (из clarify_question)
        """
        # Используем переданный контекст или получаем новый
        if context_items is None:
            context_items = self.search_engine.retrieve_context(query, top_k=top_k)
        
        if not context_items:
            return {
                'answer': 'Извините, не удалось найти релевантную информацию для ответа на ваш вопрос.',
                'sources': [],
                'context_used': ''
            }
        
        formatted_context = self._format_context(context_items)
        system_prompt, user_prompt = self._build_answer_prompt(query, formatted_context)
        
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
            
            if "401" in error_str or "invalid_api_key" in error_str.lower() or "incorrect api key" in error_str.lower():
                error_msg = "❌ **Ошибка аутентификации API ключа.** Проверьте переменную окружения OPENAI_API_KEY."
            elif "429" in error_str or "rate limit" in error_str.lower():
                error_msg = "⚠️ **Превышен лимит запросов.** Подождите немного и попробуйте снова."
            elif "model" in error_str.lower() and ("not found" in error_str.lower() or "invalid" in error_str.lower()):
                error_msg = f"❌ **Модель недоступна.** {error_str}"
            else:
                error_msg = f"❌ **Ошибка:** {error_str}"
            
            return {
                'answer': error_msg,
                'sources': [],
                'context_used': formatted_context,
                'error': True
            }

