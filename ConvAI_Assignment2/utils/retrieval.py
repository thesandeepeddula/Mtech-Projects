import json
import os
from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

class HybridRetriever:
    def __init__(self, emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(emb_model)
        self.faiss_index = None
        self.texts: List[str] = []
        self.meta: List[dict] = []
        self.bm25 = None

    def build(self, records: List[dict], faiss_dir: str, bm25_dir: str):
        self.texts = [r["text"] for r in records]
        self.meta = [{k: v for k, v in r.items() if k != "text"} for r in records]
        # Dense index
        X = self.embedder.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
        X = X.astype("float32")
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        self.faiss_index = index
        # Sparse BM25
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        # Persist
        os.makedirs(faiss_dir, exist_ok=True)
        os.makedirs(bm25_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(faiss_dir, "index.faiss"))
        with open(os.path.join(bm25_dir, "bm25.json"), "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "meta": self.meta}, f)

    def load(self, faiss_dir: str, bm25_dir: str):
        self.faiss_index = faiss.read_index(os.path.join(faiss_dir, "index.faiss"))
        with open(os.path.join(bm25_dir, "bm25.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.texts, self.meta = data["texts"], data["meta"]
        self.bm25 = BM25Okapi([t.split() for t in self.texts])

    def search(self, query: str, top_k: int = 5, alpha: float = 0.6) -> List[Tuple[float, str, dict]]:
        # alpha weights dense vs sparse (Hybrid Search)
        # Dense
        qv = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qv)
        D, I = self.faiss_index.search(qv, top_k)
        dense_hits = [(float(D[0][i]), self.texts[idx], self.meta[idx]) for i, idx in enumerate(I[0])]
        # Sparse
        scores = self.bm25.get_scores(query.split())
        idxs = np.argsort(scores)[::-1][:top_k]
        sparse_hits = [(float(scores[i]), self.texts[i], self.meta[i]) for i in idxs]
        # Normalize + fuse
        def norm(hits):
            if not hits:
                return []
            sc = np.array([h[0] for h in hits])
            mx = sc.max() if sc.size else 1.0
            if mx > 0:
                sc = sc / mx
            return [(float(s), t, m) for s, (_, t, m) in zip(sc, hits)]
        dense_n = norm(dense_hits)
        sparse_n = norm(sparse_hits)
        combined = {}
        for s, t, m in dense_n:
            combined[t] = (alpha * s, m)
        for s, t, m in sparse_n:
            prev = combined.get(t, (0.0, m))[0]
            combined[t] = (prev + (1 - alpha) * s, m)
        out = sorted([(sc, t, m) for t, (sc, m) in combined.items()], key=lambda x: x[0], reverse=True)[:top_k]
        return out