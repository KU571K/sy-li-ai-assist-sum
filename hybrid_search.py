import os
import re
import numpy as np
from typing import Optional
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt", quiet=True)

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
    tokenized = [word_tokenize(t.lower()) for t in corpus]
    return BM25Okapi(tokenized)


# Поисковый движок
class SearchEngine:
    def __init__(self, store: FaissStore, use_reranker: bool = False):
        self.store = store
        # Используем модель BGE-M3 для лучшего качества эмбеддингов
        self.encoder = SentenceTransformer("BAAI/bge-m3")
        self.reranker = None
        self.use_reranker = use_reranker
        if use_reranker:
            # Реранкер грузится долго и может тянуть большие веса — включаем только при явной необходимости
            # Используем BGE-Reranker v2-m3 для согласованности с BGE-M3 эмбеддингами и лучшей работы с русским языком
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        self.bm25 = build_bm25(store.meta)
        self.default_top_k = 5
    
    # Dense search
    def search_dense(self, query, top_k=10):
        # BGE-M3 нормализует эмбеддинги для лучшего качества поиска
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
        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)

        idxs = np.argsort(scores)[::-1][:top_k]

        return [{
            "idx": int(i),
            "score": float(scores[i]),
            "meta": self.store.meta[i]
        } for i in idxs]

    # Reranker
    def rerank(self, query, candidates):
        if self.use_reranker and self.reranker:
            pairs = [(query, c["meta"]["chunk_text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            reranked = [
                {"score": float(s), "meta": c["meta"]}
                for s, c in zip(scores, candidates)
            ]
        else:
            # Без реранкера просто сортируем по имеющемуся score
            reranked = [{"score": c["score"], "meta": c["meta"]} for c in candidates]

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    # Hybryd search 
    def hybrid_search(self, query, top_k_dense=15, top_k_bm25=15, final_k=None):
        final_k = final_k or self.default_top_k
        dense = self.search_dense(query, top_k_dense)
        bm = self.search_bm25(query, top_k_bm25)
        combined = {c["idx"]: c for c in dense}
        for c in bm:
            combined.setdefault(c["idx"], c)

        candidates = list(combined.values())
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