import os
import re
import numpy as np
from typing import Optional, List, TYPE_CHECKING
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from query_extender import QueryExpander

# Определяем device: CUDA если доступен, иначе CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Быстрый токенизатор на регулярке (вместо NLTK word_tokenize)
_TOKEN_PATTERN = re.compile(r'[а-яёa-z0-9]+', re.IGNORECASE)

def fast_tokenize(text: str) -> List[str]:
    """Быстрая токенизация через регулярку."""
    return _TOKEN_PATTERN.findall(text.lower())

# FaissStore
class FaissStore:
    def __init__(self, index_path="faiss.index", meta_path="faiss_meta.npy"):
        if not os.path.exists(index_path):
            raise FileNotFoundError("faiss.index не найден")

        if not os.path.exists(meta_path):
            raise FileNotFoundError("faiss_meta.npy не найден")

        self.index = faiss.read_index(index_path)
        self.meta = np.load(meta_path, allow_pickle=True).tolist()

def build_bm25(meta):
    corpus = [m["chunk_text"] for m in meta]
    tokenized = [fast_tokenize(t) for t in corpus]
    return BM25Okapi(tokenized)


# Поисковый движок
class SearchEngine:
    def __init__(
        self, 
        store: FaissStore, 
        use_reranker: bool = False,
        query_expander: Optional["QueryExpander"] = None
    ):
        self.store = store
        self.encoder = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
        self.reranker = None
        self.use_reranker = use_reranker
        self.query_expander = query_expander
        if use_reranker:
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)
        self.bm25 = build_bm25(store.meta)
        self.default_top_k = 5
    
    # Dense search
    def search_dense(self, query, top_k=10):
        qvec = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        distances, indices = self.store.index.search(qvec, top_k)

        out = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue

            out.append({
                "idx": int(idx),
                "score": float(-dist),  
                "meta": self.store.meta[idx],
            })
        return out

    # BM25 search
    def search_bm25(self, query, top_k=10):
        tokens = fast_tokenize(query)
        scores = self.bm25.get_scores(tokens)

        idxs = np.argsort(scores)[::-1][:top_k]

        return [{
            "idx": int(i),
            "score": float(scores[i]),
            "meta": self.store.meta[i]
        } for i in idxs]

    # Reranker
    def rerank(self, query, candidates, max_candidates_for_rerank=20):
        """Оптимизированный реранкинг с ограничением кандидатов и батчингом."""
        if self.use_reranker and self.reranker:
            # Ограничиваем количество кандидатов - обрабатываем только топ по предварительному score
            candidates_sorted = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
            candidates_to_rerank = candidates_sorted[:max_candidates_for_rerank]
            
            # Подготавливаем пары для реранкера
            pairs = [(query, c["meta"]["chunk_text"]) for c in candidates_to_rerank]
            
            # Используем батчинг для параллельной обработки
            scores = self.reranker.predict(
                pairs, 
                batch_size=32,  # Батчинг ускоряет обработку
                show_progress_bar=False  # Отключаем прогресс-бар для скорости
            )
            
            reranked = [
                {"score": float(s), "meta": c["meta"]}
                for s, c in zip(scores, candidates_to_rerank)
            ]
            
            # Добавляем остальные кандидаты (не прошедшие реранкинг) в конец
            if len(candidates_sorted) > max_candidates_for_rerank:
                remaining = [
                    {"score": c["score"], "meta": c["meta"]} 
                    for c in candidates_sorted[max_candidates_for_rerank:]
                ]
                reranked.extend(remaining)
        else:
            # Без реранкера просто сортируем по имеющемуся score
            reranked = [{"score": c["score"], "meta": c["meta"]} for c in candidates]

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    # Reciprocal Rank Fusion
    def _rrf_score(self, dense_results, bm25_results, k=60):
        """Объединяет результаты dense и BM25 через RRF."""
        rrf_scores = {}
        
        # RRF для dense результатов
        for rank, item in enumerate(dense_results, start=1):
            idx = item["idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # RRF для BM25 результатов
        for rank, item in enumerate(bm25_results, start=1):
            idx = item["idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        return rrf_scores

    def _multi_query_rrf(self, all_results: List[List[dict]], k=60):
        """Объединяет результаты из нескольких запросов через RRF."""
        rrf_scores = {}
        all_items = {}
        
        for results in all_results:
            for rank, item in enumerate(results, start=1):
                idx = item["idx"]
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
                if idx not in all_items:
                    all_items[idx] = item
        
        return rrf_scores, all_items

    # Hybrid search 
    def hybrid_search(self, query, top_k_dense=10, top_k_bm25=10, final_k=None):
        final_k = final_k or self.default_top_k
        
        # Расширяем запрос если доступен expander
        if self.query_expander:
            queries = self.query_expander.expand(query)
        else:
            queries = [query]
        
        # Multi-query поиск: собираем результаты по всем запросам
        all_dense = []
        all_bm25 = []
        for q in queries:
            all_dense.append(self.search_dense(q, top_k_dense))
            all_bm25.append(self.search_bm25(q, top_k_bm25))
        
        # RRF по всем результатам (dense + BM25 для каждого запроса)
        all_results = all_dense + all_bm25
        rrf_scores, all_items = self._multi_query_rrf(all_results)
        
        # Собираем кандидатов с RRF score
        candidates = [
            {"idx": idx, "score": rrf_scores[idx], "meta": item["meta"]}
            for idx, item in all_items.items()
        ]
        
        # Реранкинг по оригинальному запросу
        return self.rerank(query, candidates)[:final_k]

    def retrieve_context(self, query: str, top_k: Optional[int] = None):
        """Возвращает лучшие чанки с метаданными для построения ответа."""
        results = self.hybrid_search(query, final_k=top_k or self.default_top_k)
        context = []
        for i, item in enumerate(results, 1):
            meta = item["meta"]
            context.append({
                "rank": i,
                "score": item["score"],
                "chunk": meta.get("chunk_text", ""),
                "section": meta.get("section_title", ""),
                "doc_id": meta.get("doc_id", ""),
                "source_path": meta.get("source_path", ""),
            })
        return context