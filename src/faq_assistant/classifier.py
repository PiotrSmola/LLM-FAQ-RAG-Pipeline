"""Etap 1 pipeline'u: klasyfikacja (routing) zapytania klienta.

Trzy wymienne implementacje o wspólnym interfejsie `predict(text) -> Prediction`:
- BertClassifier: dostrojony (fine-tuning) HerBERT / DistilBERT (główny router),
- TfidfClassifier: baseline TF-IDF + regresja logistyczna (jak w Lab 2),
- LLMClassifier: klasyfikacja zero/few-shot przez LLM (Gemma 4, porównanie).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CATEGORIES, OUT_OF_SCOPE, GenerationParams


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)
    latency_s: float = 0.0
    method: str = ""


class BertClassifier:
    method = "bert"

    def __init__(self, path: str | Path, device: str | None = None, max_length: int = 64) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.max_length = max_length
        self.method = f"bert:{Path(path).name}"

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            out.append(self.torch.softmax(logits, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    def predict(self, text: str) -> Prediction:
        t = time.perf_counter()
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return Prediction(self.labels[i], float(p[i]), dict(zip(self.labels, map(float, p))),
                          time.perf_counter() - t, self.method)


class TfidfClassifier:
    method = "tfidf_lr"

    def __init__(self, analyzer: str = "char_wb") -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        ngrams = (2, 5) if analyzer.startswith("char") else (1, 2)
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, lowercase=True, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=10.0),
        )
        self.method = f"tfidf_{analyzer}_lr"

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfClassifier":
        self.pipe.fit(texts, labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = self.pipe.predict_proba(texts)
        order = [list(self.pipe.classes_).index(c) for c in CATEGORIES]
        return proba[:, order]

    def predict(self, text: str) -> Prediction:
        t = time.perf_counter()
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return Prediction(CATEGORIES[i], float(p[i]), dict(zip(CATEGORIES, map(float, p))),
                          time.perf_counter() - t, self.method)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self.pipe, path)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfClassifier":
        import joblib

        obj = cls.__new__(cls)
        obj.pipe = joblib.load(path)
        obj.method = "tfidf_lr"
        return obj


class LLMClassifier:
    method = "llm_zero_shot"

    def __init__(self, llm=None) -> None:
        from .llm import LLMClient

        self.llm = llm or LLMClient()

    def predict(self, text: str) -> Prediction:
        from .prompts import CLASSIFY_SYSTEM
        from .utils import parse_json

        t = time.perf_counter()
        raw = self.llm.complete(f'Pytanie: "{text}"', system=CLASSIFY_SYSTEM,
                                params=GenerationParams(temperature=0.0, max_tokens=64, seed=0))
        try:
            label = parse_json(raw).get("category", "")
        except (ValueError, AttributeError):
            label = next((c for c in CATEGORIES if c in raw), "")
        if label not in CATEGORIES:
            label = OUT_OF_SCOPE
        probs = {c: float(c == label) for c in CATEGORIES}
        return Prediction(label, 1.0, probs, time.perf_counter() - t, self.method)
