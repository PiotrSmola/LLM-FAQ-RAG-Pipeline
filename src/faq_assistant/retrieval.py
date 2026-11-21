"""Etap 2 pipeline'u: indeksowanie i wyszukiwanie fragmentów bazy wiedzy.

Metody wyszukiwania:
- "tfidf"  – TF-IDF + podobieństwo cosinusowe (punkt odniesienia z Lab 3, przykład 4),
- "bm25"   – Okapi BM25 (rank-bm25, Lab 3 zad. 3.3) z prostym stemmingiem prefiksowym dla polskiego,
- "dense"  – wyszukiwanie semantyczne: embeddingi bge-m3 (LM Studio) w bazie wektorowej ChromaDB,
- "hybrid" – połączenie BM25 i dense metodą Reciprocal Rank Fusion (RRF).

Filtr kategorii (routing z etapu 1): wpisy FAQ są zawężane do rozpoznanej kategorii, dokumenty firmowe
(regulaminy itp.) przeszukiwane są zawsze, bo dotyczą wielu kategorii naraz.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CHROMA_DIR
from .ingest import Chunk

METHODS = ("tfidf", "bm25", "dense", "hybrid")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    method: str
    details: dict = field(default_factory=dict)


_PL_STOP = set("""a aby ale albo bo by być czy dla do go i ich ile im jak jaki jaka jakie je jest jestem już
ma mam mi mnie może mogę mój moje na nie nic niż o od on ona oraz po pod przez przy się są ta tak te
tego tej ten też to tu tym w we wy z za ze że czym co kiedy gdzie""".split())


def _strip_accents(s: str) -> str:
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def tokenize_pl(text: str, stem: int = 6) -> list[str]:
    """Tokenizacja dla BM25: małe litery, bez diakrytyków, bez stop-słów, stemming = pierwsze `stem` znaków.

    Prefiksowy stemming to tani sposób na polską fleksję (np. „zwrotu”, „zwrotem”, „zwroty” -> „zwrot”).
    """
    words = re.findall(r"\w+", _strip_accents(text.lower()))
    return [w[:stem] for w in words if w not in _PL_STOP and (len(w) > 1 or w.isdigit())]


class KnowledgeBase:
    """Indeks bazy wiedzy z czterema metodami wyszukiwania.

    persist=True  – kolekcja ChromaDB zapisana na dysku (baza firmowa, budowana raz: scripts/build_index.py),
    persist=False – kolekcja w pamięci (pliki przesłane przez użytkownika w trakcie sesji).
    """

    def __init__(self, name: str = "faq_kb", llm=None, persist: bool = True,
                 path: str | Path = CHROMA_DIR) -> None:
        import chromadb

        from .llm import LMStudioClient

        self.name = name
        self.llm = llm or LMStudioClient()
        self.path = Path(path)
        self.client = chromadb.PersistentClient(path=str(self.path)) if persist else chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self.persist = persist
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, int] = {}
        self._bm25 = None
        self._tfidf = None
        if persist:
            self._load_chunks()

    # ---------------------------------------------------------------------------------------- budowa
    @property
    def _chunks_file(self) -> Path:
        return self.path / f"{self.name}_chunks.jsonl"

    def _load_chunks(self) -> None:
        if self._chunks_file.exists():
            with open(self._chunks_file, encoding="utf-8") as f:
                self.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
            self._rebuild_lexical()

    def _rebuild_lexical(self) -> None:
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._by_id = {c.id: i for i, c in enumerate(self.chunks)}
        if not self.chunks:
            self._bm25 = self._tfidf = None
            return
        self._bm25 = BM25Okapi([tokenize_pl(c.text) for c in self.chunks])
        vec = TfidfVectorizer(preprocessor=lambda s: _strip_accents(s.lower()), sublinear_tf=True)
        self._tfidf = (vec, vec.fit_transform([c.text for c in self.chunks]))

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.collection = self.client.get_or_create_collection(
            name=self.name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self.chunks = []
        self._rebuild_lexical()
        if self.persist and self._chunks_file.exists():
            self._chunks_file.unlink()

    def add(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        new = [c for c in chunks if c.id not in self._by_id]
        if not new:
            return 0
        embeddings = self.llm.embed([c.text for c in new], batch_size=batch_size)
        self.collection.add(
            ids=[c.id for c in new],
            embeddings=embeddings,
            documents=[c.text for c in new],
            metadatas=[{"kind": c.kind, "category": c.category, "source": c.source} for c in new],
        )
        self.chunks.extend(new)
        self._rebuild_lexical()
        if self.persist:
            self.path.mkdir(parents=True, exist_ok=True)
            with open(self._chunks_file, "w", encoding="utf-8") as f:
                for c in self.chunks:
                    f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        return len(new)

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------------------------- wyszukiwanie
    def _allowed_mask(self, category: str | None) -> np.ndarray:
        if category is None:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([c.kind == "doc" or c.category == category for c in self.chunks])

    def _rank_scores(self, scores: np.ndarray, mask: np.ndarray, k: int, method: str) -> list[Hit]:
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), r + 1, method)
                for r, i in enumerate(order) if np.isfinite(scores[i])]

    def search_bm25(self, query: str, k: int, category: str | None = None) -> list[Hit]:
        scores = np.asarray(self._bm25.get_scores(tokenize_pl(query)))
        return self._rank_scores(scores, self._allowed_mask(category), k, "bm25")

    def search_tfidf(self, query: str, k: int, category: str | None = None) -> list[Hit]:
        vec, mat = self._tfidf
        scores = (mat @ vec.transform([query]).T).toarray().ravel()
        return self._rank_scores(scores, self._allowed_mask(category), k, "tfidf")

    def search_dense(self, query: str, k: int, category: str | None = None) -> list[Hit]:
        where = None if category is None else {"$or": [{"kind": "doc"}, {"category": category}]}
        n = min(k, len(self.chunks))
        res = self.collection.query(query_embeddings=self.llm.embed([query]), n_results=n, where=where,
                                    include=["distances"])
        hits = []
        for r, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0])):
            if cid in self._by_id:
                hits.append(Hit(self.chunks[self._by_id[cid]], 1.0 - float(dist), r + 1, "dense"))
        return hits

    def search_hybrid(self, query: str, k: int, category: str | None = None, pool: int = 20,
                      rrf_k: int = 60) -> list[Hit]:
        """Reciprocal Rank Fusion: score = Σ 1/(rrf_k + rank) po metodach BM25 i dense."""
        lists = {"bm25": self.search_bm25(query, pool, category), "dense": self.search_dense(query, pool, category)}
        fused: dict[str, dict] = {}
        for method, hits in lists.items():
            for h in hits:
                e = fused.setdefault(h.chunk.id, {"chunk": h.chunk, "rrf": 0.0, "details": {}})
                e["rrf"] += 1.0 / (rrf_k + h.rank)
                e["details"][f"{method}_rank"] = h.rank
                e["details"][f"{method}_score"] = round(h.score, 4)
        ranked = sorted(fused.values(), key=lambda e: -e["rrf"])[:k]
        return [Hit(e["chunk"], e["rrf"], r + 1, "hybrid", e["details"]) for r, e in enumerate(ranked)]

    def search(self, query: str, k: int = 4, method: str = "hybrid", category: str | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        fn = {"tfidf": self.search_tfidf, "bm25": self.search_bm25,
              "dense": self.search_dense, "hybrid": self.search_hybrid}[method]
        return fn(query, k, category)
