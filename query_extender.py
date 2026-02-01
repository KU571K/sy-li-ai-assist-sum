from openai import OpenAI
from typing import List, Optional
import re
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

class QueryExpander:
    def __init__(self, client: OpenAI, model: str = "gpt-5-mini", enable_expansion: bool = True):
        """
        Инициализация QueryExpander для расширения запросов пользователя.
        
        Args:
            client: OpenAI клиент
            model: Модель для расширения запросов
            enable_expansion: Включить/выключить расширение запросов
        """
        self.client = client
        self.model = model
        self.enable_expansion = enable_expansion

    def expand(self, query: str) -> List[str]:
        """
        Расширяет запрос пользователя, переводя его в юридическую терминологию.
        
        Args:
            query: Оригинальный запрос пользователя
            
        Returns:
            Список запросов: оригинальный + расширенные
        """
        if not self.enable_expansion or not query.strip():
            return [query]
        
        try:
            prompt = f"""Ты помощник по юридическим документам в сфере образования.

Переформулируй вопрос студента в 3–5 формальных поисковых запросов,
используя юридическую и нормативную терминологию, которая используется в нормативных актах, приказах и постановлениях.

Примеры переформулировки:
- "За что меня могут отчислить?" → "основания для отчисления", "причины отчисления", "порядок отчисления студентов"
- "Если я не сдам практику, допустят ли меня к защите диплома?" → "требования для допуска к защите диплома", "условия допуска к защите ВКР", "сдача практики и защита диплома"
- "Как поступить?" → "правила приема в вуз", "порядок приема", "процедура поступления"
- "Экзамен" → "промежуточная аттестация", "формы промежуточной аттестации", "порядок промежуточной аттестации"

❗ Не отвечай на вопрос.
❗ Не объясняй.
❗ Только список поисковых запросов, каждый с новой строки.

Вопрос студента:
{query}

Переформулированные поисковые запросы:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()
            
            # Улучшенный парсинг: убираем нумерацию, маркеры, лишние символы
            lines = content.split("\n")
            queries = []
            for line in lines:
                # Убираем нумерацию (1. 2. 3. и т.д.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                # Убираем маркеры списков (• - * и т.д.)
                line = re.sub(r'^[•\-\*\+]\s*', '', line)
                # Убираем лишние пробелы
                line = line.strip()
                # Оставляем только значимые строки (минимум 10 символов)
                if len(line) > 10:
                    queries.append(line)
            
            # Если не удалось распарсить, возвращаем оригинальный запрос
            if not queries:
                return [query]
            
            # Возвращаем оригинальный запрос + расширенные (уникальные)
            all_queries = [query]
            for q in queries:
                if q.lower() not in [q2.lower() for q2 in all_queries]:
                    all_queries.append(q)
            
            return all_queries[:6]  # Максимум 6 запросов (оригинал + 5 расширенных)
            
        except Exception as e:
            # В случае ошибки возвращаем только оригинальный запрос
            logger.warning(f"Ошибка при расширении запроса: {e}", exc_info=True)
            return [query]
    
    def expand_for_search(self, query: str) -> str:
        """
        Расширяет запрос для поиска, объединяя все варианты в один строку поиска.
        Полезно для гибридного поиска.
        
        Args:
            query: Оригинальный запрос пользователя
            
        Returns:
            Расширенный запрос для поиска
        """
        expanded_queries = self.expand(query)
        # Объединяем все запросы в один для более широкого поиска
        return " ".join(expanded_queries)
