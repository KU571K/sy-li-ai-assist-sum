import os
import streamlit as st
from dotenv import load_dotenv
from hybrid_search import SearchEngine, FaissStore
from rag_chain import RAGChain

# Загрузка переменных окружения
load_dotenv()

# Настройка страницы
st.set_page_config(
    page_title="AI Ассистент личного кабинета студента",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация сессии
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []


@st.cache_resource
def load_search_engine():
    """Загружает поисковый движок с кэшированием."""
    try:
        store = FaissStore(index_path="faiss.index", meta_path="faiss_meta.npy")
        search_engine = SearchEngine(store, use_reranker=False)
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
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000
        )
        return rag_chain
    except RuntimeError as e:
        st.error(f"Ошибка инициализации RAG цепочки: {e}")
        st.info("""
        **Убедитесь, что:**
        
        1. Установлена переменная окружения `OPENAI_API_KEY`
        2. API ключ валиден и имеет доступ к модели gpt-4o-mini
        """)
        return None


def main():
    """Основная функция приложения."""
    
    # Заголовок
    st.title("🎓 AI Ассистент личного кабинета студента")
    st.markdown("---")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Проверка наличия API ключа
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.warning("⚠️ OPENAI_API_KEY не установлен")
            st.info("Установите переменную окружения OPENAI_API_KEY или создайте файл .env")
        else:
            st.success("✅ API ключ найден")
        
        st.markdown("---")
        
        # Настройки поиска
        st.subheader("🔍 Параметры поиска")
        top_k = st.slider(
            "Количество релевантных фрагментов (top_k)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Количество фрагментов документов, используемых для генерации ответа"
        )
        
        st.markdown("---")
        
        # Информация о системе
        st.subheader("ℹ️ О системе")
        st.markdown("""
        Этот AI ассистент помогает студентам находить ответы на вопросы 
        на основе нормативных документов (законы, приказы, постановления).
        
        **Технологии:**
        - Гибридный поиск (FAISS + BM25)
        - RAG (Retrieval-Augmented Generation)
        - GPT-3.5-turbo
        """)
    
    # Загрузка компонентов
    search_engine = load_search_engine()
    rag_chain = load_rag_chain(search_engine)
    
    if search_engine is None or rag_chain is None:
        st.stop()
    
    # Основная область чата
    st.subheader("💬 Задайте вопрос")
    
    # Отображение истории сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Показываем источники для ответов ассистента
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Источники"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        **Источник {i}:**
                        - Документ: `{source.get('doc_id', 'N/A')}`
                        - Раздел: {source.get('section', 'N/A')}
                        - Релевантность: {source.get('score', 0):.4f}
                        """)
    
    # Поле ввода вопроса
    if prompt := st.chat_input("Введите ваш вопрос..."):
        # Добавляем вопрос пользователя в историю
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Генерируем ответ
        with st.chat_message("assistant"):
            with st.spinner("🔍 Поиск информации и генерация ответа..."):
                try:
                    result = rag_chain.generate_answer(
                        query=prompt,
                        top_k=top_k,
                        use_reranker=False
                    )
                    
                    answer = result['answer']
                    sources = result['sources']
                    
                    # Отображаем ответ
                    st.markdown(answer)
                    
                    # Отображаем источники
                    if sources:
                        with st.expander("📚 Источники информации"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                **Источник {i}:**
                                - Документ: `{source.get('doc_id', 'N/A')}`
                                - Раздел: {source.get('section', 'N/A')}
                                - Релевантность: {source.get('score', 0):.4f}
                                """)
                    
                    # Сохраняем ответ в историю
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
                        "content": error_msg
                    })
    
    # Кнопка очистки истории
    if st.session_state.messages:
        if st.button("🗑️ Очистить историю", type="secondary"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()

