import os
import re
import argparse
import numpy as np
from typing import Callable, List, Optional, Tuple
import faiss

# модель HF
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# модель openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Парсинг id из названия файла
def extract_doc_id(filename: str, text: Optional[str] = None) -> str:
    """
    Пытаемся извлечь номер и дату из имени файла.
    Если не получается, используем имя файла без расширения.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    num_match = re.search(r'[N№]\s*(\d+)(?:-ФЗ)?', name, flags=re.IGNORECASE)
    if num_match:
        num = num_match.group(1)
        date_match = re.search(r'от\s*(\d{2})\.(\d{2})\.(\d{4})', name)
        if date_match:
            day, month, year = date_match.groups()
            return f"{num}_{year}_{month}_{day}"
        return f"{num}_0000_00_00"

    # Если имя UUID или не содержит номера, пробуем взять первую непустую строку как идентификатор
    if text:
        for line in text.splitlines():
            if line.strip():
                safe = re.sub(r'\s+', '_', line.strip())[:64]
                if safe:
                    return safe
                break
    return name

# Нормализация разметки для имеющихся файлов (убирает лишние табуляции и переносы строк)
def parse_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        t = f.read()
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t

SECTION_RX = re.compile(
    r'''(?mx)
    ^\s*(
        \#{1,6}\s+.+                                
        
        |(Глава|Статья|Раздел|Параграф|Часть)\s+    
         ([IVXLC]+|\d+)\.?                          
         (?:\s*[-–.]?\s*.+)?                         

        |Раздел\s+[IVXLC]+(?:\s*[-–.]?\s*.+)?        
    )\s*$
    ''',
)

def normalize_text(text: str) -> str:
    """Нормализация правовых текстов перед парсингом."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # Ситуация: "Статья 5.\nНазвание"
    text = re.sub(
        r'((Глава|Статья|Раздел|Параграф|Часть)\s+[IVXLC\d.]+)\s*\n\s*(\S)',
        r'\1 \3',
        text
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# Разбитине документа на секции по статьям или главам
def split_into_sections(text: str) -> List[Tuple[str, str]]:
    text = normalize_text(text)

    matches = list(SECTION_RX.finditer(text))
    if not matches:
        return [("document", text.strip())]

    sections = []
    for i, m in enumerate(matches):
        title = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))

    return sections

# Формировние чанков
SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')

def chunk_text(section: str, max_chars=1200, overlap=200):
    sents = [s.strip() for s in SENT_SPLIT.split(section) if s.strip()]
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) + 1 <= max_chars or not cur:
            cur.append(s); cur_len += len(s) + 1
        else:
            chunks.append(' '.join(cur))
            ov, ov_len = [], 0
            for sent in reversed(cur):
                if ov_len + len(sent) <= overlap:
                    ov.insert(0, sent); ov_len += len(sent)
                else:
                    break
            cur, cur_len = ov.copy(), ov_len
            cur.append(s); cur_len += len(s) + 1
    if cur:
        chunks.append(' '.join(cur))
    return chunks

def build_embedder(method: str) -> tuple[int, Callable[[List[str]], np.ndarray]]:
    """
    Возвращает размерность и функцию эмбеддера.
    Модели и клиенты создаются один раз, чтобы не тратить время при батчинге.
    """
    if method == 'sbert':
        if SentenceTransformer is None:
            raise RuntimeError('sentence-transformers not installed')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dim = model.get_sentence_embedding_dimension()

        def encode(texts: List[str]) -> np.ndarray:
            return model.encode(texts, convert_to_numpy=True)

        return dim, encode

    if method == 'openai':
        if OpenAI is None:
            raise RuntimeError('openai not installed')
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY not set')
        client = OpenAI(api_key=key)
        model_name = 'text-embedding-3-small'

        def encode(texts: List[str]) -> np.ndarray:
            batch = 32
            out = []
            for i in range(0, len(texts), batch):
                b = texts[i:i + batch]
                resp = client.embeddings.create(model=model_name, input=b)
                out.extend(np.array(item.embedding, dtype=np.float32) for item in resp.data)
            return np.vstack(out)

        # Получаем размерность из одиночного вызова
        dim = len(encode(["test vector"])[0])
        return dim, encode

    raise ValueError(f"Unknown embedder: {method}")

# FAISSDB
class FaissStore:
    def __init__(self, dim: int, index_path='faiss.index', meta_path='faiss_meta.npy'):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        for path in [index_path, meta_path]:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)

        if os.path.exists(meta_path):
            self.meta = np.load(meta_path, allow_pickle=True).tolist()
        else:
            self.meta = []

    def add_vectors(self, vectors: np.ndarray, metas: List[dict]):
        self.index.add(vectors.astype(np.float32))
        self.meta.extend(metas)
        faiss.write_index(self.index, self.index_path)
        np.save(self.meta_path, np.array(self.meta, dtype=object))

    def search(self, qvec: np.ndarray, k=5):
        D, I = self.index.search(qvec.reshape(1, -1).astype(np.float32), k)
        res = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0: continue
            res.append({"score": float(dist), "meta": self.meta[idx]})
        return res

def process_file(path: str, embed_fn: Callable[[List[str]], np.ndarray], store: FaissStore, max_chars: int = 1200, overlap: int = 200):
    text = parse_text_file(path)
    doc_id = extract_doc_id(os.path.basename(path), text=text)
    sections = split_into_sections(text)

    for title, body in sections:
        chunks = chunk_text(body, max_chars=max_chars, overlap=overlap)
        if not chunks:
            continue
        vecs = embed_fn(chunks)

        metas = [{
            "doc_id": doc_id,
            "section_title": title,
            "chunk_index": i,
            "chunk_text": chunks[i],
            "source_path": path,
        } for i in range(len(chunks))]

        store.add_vectors(vecs, metas)
        print(f"Stored {len(chunks)} chunks for {doc_id} — {title[:60]}")


def process_folder(folder: str, embed_fn: Callable[[List[str]], np.ndarray], store: FaissStore, max_chars: int = 1200, overlap: int = 200):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.md', '.txt'))
    ]
    if not files:
        print(f"Нет .md или .txt файлов в {folder}")
        return
    print(f"Найдено файлов: {len(files)}")
    for path in files:
        print(f"➡ Обработка: {os.path.basename(path)}")
        process_file(path, embed_fn, store, max_chars=max_chars, overlap=overlap)

# CLI
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', type=str, help='Один MD файл')
    ap.add_argument('--folder', type=str, help='Папка с MD/TXT файлами')
    ap.add_argument('--embedder', choices=['sbert','openai'], default='sbert')
    ap.add_argument('--index', default='faiss.index')
    ap.add_argument('--meta', default='faiss_meta.npy')
    ap.add_argument('--max-chars', type=int, default=1200, help='Макс. символов в чанке')
    ap.add_argument('--overlap', type=int, default=200, help='Перекрытие чанков в символах')
    args = ap.parse_args()

    dim, embed_fn = build_embedder(args.embedder)
    store = FaissStore(dim, args.index, args.meta)

    if args.file:
        process_file(args.file, embed_fn, store, max_chars=args.max_chars, overlap=args.overlap)
    elif args.folder:
        process_folder(args.folder, embed_fn, store, max_chars=args.max_chars, overlap=args.overlap)
    else:
        print("Укажите --file или --folder")
