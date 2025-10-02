import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class MemoryIndex:
    """
    Lightweight FAISS-backed memory index.
    - Uses sentence-transformers for embeddings.
    - IndexFlatL2 for CPU portability.
    - Stores parallel list of texts in a jsonl file.
    """

    def __init__(self, embeddings_model_name="all-MiniLM-L6-v2", index_path="data/faiss", dim=None):
        self.index_path = index_path
        self.emb_model = SentenceTransformer(embeddings_model_name)
        self.dim = dim or self.emb_model.get_sentence_embedding_dimension()
        self.texts = []
        self.index = faiss.IndexFlatL2(self.dim)

    def add(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return
        embs = self.emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)
        self.index.add(embs.astype("float32"))
        self.texts.extend(texts)

    def query(self, text, k=4):
        if len(self.texts) == 0:
            return []
        emb = self.emb_model.encode([text], convert_to_numpy=True)
        k = min(k, len(self.texts))
        D, I = self.index.search(emb.astype("float32"), k)
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])
        return results

    def save(self, path=None):
        path = path or self.index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".index")
        with open(path + ".texts.jsonl", "w", encoding="utf-8") as f:
            for t in self.texts:
                json.dump({"text": t}, f, ensure_ascii=False)
                f.write("\n")

    def load(self, path=None):
        path = path or self.index_path
        if os.path.exists(path + ".index") and os.path.exists(path + ".texts.jsonl"):
            self.index = faiss.read_index(path + ".index")
            self.texts = []
            with open(path + ".texts.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self.texts.append(obj.get("text", ""))
