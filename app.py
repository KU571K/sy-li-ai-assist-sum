import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from hybrid_search import SearchEngine, FaissStore
from rag_chain import RAGChain
from query_extender import QueryExpander

# Загрузка переменных окружения
load_dotenv()

# Настройка страницы
st.set_page_config(
    page_title="AI Ассистент личного кабинета студента",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Кастомные стили в стиле Синергии
st.markdown("""
<style>
    /* Основные цвета: белый фон, красный текст */
    .main {
        background-color: #ffffff;
    }
    
    /* Заголовки красного цвета */
    h1, h2, h3, h4, h5, h6 {
        color: #C8102E !important;
        font-weight: 600;
    }
    
    /* Обычный текст темно-серый для читаемости */
    .stMarkdown, .stMarkdown p {
        color: #333333;
    }
    
    /* Чат-сообщения */
    .stChatMessage {
        background-color: #ffffff;
    }
    
    /* Кнопки */
    .stButton > button {
        background-color: #C8102E;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #a00e26;
        color: white;
    }
    
    /* Кнопки типовых вопросов и уточнений */
    button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #C8102E !important;
        border: 1px solid #C8102E !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    button[kind="secondary"]:hover {
        background-color: #C8102E !important;
        color: white !important;
        border-color: #C8102E !important;
    }
    
    /* Поле ввода */
    .stChatInput > div > div > input {
        border: 1px solid #C8102E;
        border-radius: 4px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #C8102E;
        font-weight: 500;
    }
    
    /* Содержимое expander - черный текст */
    .streamlit-expanderContent {
        color: #000000 !important;
    }
    
    .streamlit-expanderContent .stMarkdown,
    .streamlit-expanderContent .stMarkdown p,
    .streamlit-expanderContent p {
        color: #000000 !important;
    }
    
    /* Скрываем sidebar */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem;
    }
    
    /* Скрываем предупреждения о secrets и другие alert сообщения */
    .stAlert {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stAlert"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    div[data-baseweb="notification"] {
        display: none !important;
    }
    /* Скрываем любые warning/error сообщения */
    .element-container:has(.stAlert) {
        display: none !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    div:has(> .stAlert) {
        display: none !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    /* Дополнительные селекторы для скрытия предупреждений */
    [data-testid="stException"] {
        display: none !important;
    }
    .stException {
        display: none !important;
    }
    /* Скрываем элементы с классами, содержащими alert/warning */
    [class*="stAlert"],
    [class*="alert"],
    [class*="warning"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    
    /* Центрирование и отступы */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }
    
    /* Разделители */
    hr {
        border-color: #C8102E;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация сессии
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.top_k = 8
    st.session_state.pending_query = None
    st.session_state.pending_clarification = None  # Хранит данные для уточнения
    st.session_state.original_query = None  # Исходный вопрос пользователя  


def _create_query_expander():
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    except:
        api_key = None
    
    if not api_key:
        api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
    
    if api_key:
        client = OpenAI(api_key=api_key, base_url="https://api.artemox.com/v1")
        return QueryExpander(client, model="gpt-5-mini", enable_expansion=True)
    return None


@st.cache_resource
def load_search_engine():
    """Загружает поисковый движок с кэшированием."""
    try:
        store = FaissStore(index_path="faiss.index", meta_path="faiss_meta.npy")
        query_expander = None 
        search_engine = SearchEngine(store, use_reranker=False, query_expander=None)
        return search_engine
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки индекса: {e}")
        st.info("""
        **Инструкция по подготовке индекса:**
        
        1. Убедитесь, что файлы `faiss.index` и `faiss_meta.npy` существуют
        2. Если их нет, запустите векторизацию:
        
        ```bash
        python faiss_vectorization.py --folder files --embedder sbert
        ```
        """)
        return None


@st.cache_resource
def load_rag_chain(_search_engine):
    """Создает RAG цепочку с кэшированием."""
    if _search_engine is None:
        return None
    
    try:
        rag_chain = RAGChain(
            search_engine=_search_engine,
            model="gpt-5-mini",
            temperature=0.7,
            max_tokens=1000
        )
        return rag_chain
    except RuntimeError as e:
        st.error(f"Ошибка инициализации RAG цепочки: {e}")
        st.info("""
        **Убедитесь, что:**
        
        1. Установлена переменная окружения `OPENROUTER_API_KEY`
        2. API ключ валиден и имеет доступ к OpenRouter
        3. Получить ключ можно на https://openrouter.ai
        """)
        return None


def main():
    # Проверка наличия API ключа OpenRouter
    api_key = None
    try:
        if hasattr(st, 'secrets'):
            api_key = st.secrets.get("OPENROUTER_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    except (AttributeError, KeyError, FileNotFoundError, Exception):
        pass
    
    if not api_key:
        api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
    
    # Заголовок
    st.title("AI Ассистент личного кабинета студента")
    st.markdown("""
    <p style='color: #666666; font-size: 1.1em; margin-bottom: 2rem;'>
    Помогаю находить ответы на вопросы на основе нормативных документов
    </p>
    """, unsafe_allow_html=True)
    
    # Компактная индикация статуса API ключа
    if not api_key:
        st.markdown("""
        <div style='background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; margin-bottom: 1.5rem; border-radius: 4px;'>
            <strong style='color: #856404;'>⚠️ API ключ не установлен</strong><br>
            <span style='color: #856404; font-size: 0.9em;'>
            Установите переменную окружения OPENROUTER_API_KEY или создайте файл .env с ключом. 
            Получить ключ можно на <a href='https://openrouter.ai' style='color: #C8102E;'>openrouter.ai</a>
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Загрузка компонентов
    search_engine = load_search_engine()
    rag_chain = load_rag_chain(search_engine)
    
    if search_engine is None or rag_chain is None:
        st.stop()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Типовые вопросы (отображаются только если нет истории сообщений)
    if not st.session_state.messages:
        st.markdown("""
        <div style='margin-bottom: 2rem;'>
            <h3 style='color: #333333; font-size: 1.1em; margin-bottom: 1rem; font-weight: 500;'>Популярные вопросы:</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Список типовых вопросов
        typical_questions = [
            "Как поступить в университет?",
            "Какие документы нужны для поступления?",
            "Правила приема в вуз",
            "Стипендии для студентов",
            "Академический отпуск",
            "Перевод из одного вуза в другой"
        ]
        
        # Размещаем кнопки в 3 колонки (по 2 вопроса в ряд)
        cols = st.columns(3)
        for idx, question in enumerate(typical_questions):
            col_idx = idx % 3
            with cols[col_idx]:
                if st.button(
                    question,
                    key=f"quick_question_{idx}",
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.pending_query = question
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Отображение истории сообщений
    if st.session_state.messages:
        for msg_idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Показываем источники для ответов ассистента
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    source = message["sources"][0]
                    with st.expander("📚 Источники информации", expanded=False):
                        st.markdown(f"""
                        Документ: `{source.get('doc_id', 'N/A')}`  
                        Раздел: {source.get('section', 'N/A')}
                        """)
                
                # Показываем варианты уточнения (стиль Госуслуг)
                if message["role"] == "assistant" and "clarification_options" in message:
                    options = message.get("clarification_options", [])
                    if options:
                        st.markdown("<br>", unsafe_allow_html=True)
                        cols = st.columns(min(len(options), 3))
                        
                        for q_idx, option in enumerate(options):
                            col_idx = q_idx % 3
                            with cols[col_idx]:
                                if st.button(
                                    option,
                                    key=f"clarify_hist_{msg_idx}_{q_idx}",
                                    use_container_width=True,
                                    type="secondary"
                                ):
                                    st.session_state.pending_query = option
                                    st.rerun()
    
    # Обработка запроса из кнопки типового вопроса
    prompt_from_button = None
    if st.session_state.pending_query:
        prompt_from_button = st.session_state.pending_query
        st.session_state.pending_query = None  # Сбрасываем флаг
    
    # Поле ввода вопроса
    user_input = st.chat_input("Введите ваш вопрос...")
    
    # Определяем, какой запрос обрабатывать
    prompt = prompt_from_button or user_input
    
    if prompt:
        # Проверяем, это выбор из уточняющих вариантов или новый вопрос
        is_clarification_choice = st.session_state.pending_clarification is not None
        
        # Добавляем вопрос пользователя в историю
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                if is_clarification_choice:
                    # Это выбор уточнения — сразу генерируем ответ
                    with st.spinner("🔍 Генерация ответа..."):
                        context_items = st.session_state.pending_clarification.get('context_items', [])
                        result = rag_chain.generate_answer(
                            query=prompt,
                            top_k=st.session_state.top_k,
                            context_items=context_items
                        )
                        st.session_state.pending_clarification = None
                        st.session_state.original_query = None
                        
                        answer = result['answer']
                        sources = result['sources']
                        
                        st.markdown(answer)
                        
                        if sources:
                            source = sources[0]
                            with st.expander("📚 Источники информации", expanded=False):
                                st.markdown(f"""
                                Документ: `{source.get('doc_id', 'N/A')}`  
                                Раздел: {source.get('section', 'N/A')}
                                """)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                else:
                    # Новый вопрос — сначала проверяем, нужно ли уточнение
                    with st.spinner("🔍 Анализ вопроса..."):
                        clarification = rag_chain.clarify_question(
                            query=prompt,
                            top_k=st.session_state.top_k
                        )
                    
                    if clarification['needs_clarification'] and clarification['options']:
                        # Нужно уточнение — показываем варианты
                        clarification_text = clarification['clarification_text']
                        options = clarification['options']
                        
                        st.markdown(f"**{clarification_text}**")
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Сохраняем контекст для следующего этапа
                        st.session_state.pending_clarification = clarification
                        st.session_state.original_query = prompt
                        
                        # Показываем варианты в кнопках
                        cols = st.columns(min(len(options), 3))
                        message_idx = len(st.session_state.messages)
                        
                        for idx, option in enumerate(options):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                if st.button(
                                    option,
                                    key=f"clarify_{message_idx}_{idx}",
                                    use_container_width=True,
                                    type="secondary"
                                ):
                                    st.session_state.pending_query = option
                                    st.rerun()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**{clarification_text}**",
                            "clarification_options": options
                        })
                    else:
                        # Вопрос конкретный — сразу отвечаем
                        with st.spinner("🔍 Генерация ответа..."):
                            result = rag_chain.generate_answer(
                                query=prompt,
                                top_k=st.session_state.top_k,
                                context_items=clarification.get('context_items')
                            )
                        
                        answer = result['answer']
                        sources = result['sources']
                        
                        st.markdown(answer)
                        
                        if sources:
                            source = sources[0]
                            with st.expander("📚 Источники информации", expanded=False):
                                st.markdown(f"""
                                Документ: `{source.get('doc_id', 'N/A')}`  
                                Раздел: {source.get('section', 'N/A')}
                                """)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
            except Exception as e:
                error_msg = f"Произошла ошибка: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
    
    # Кнопка очистки истории
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Очистить историю", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()

