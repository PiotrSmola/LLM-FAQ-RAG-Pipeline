"""Centralna konfiguracja projektu (wartości z pliku .env lub domyślne)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CHROMA_DIR = ROOT / "chroma_db"

# Kategorie routingu (etap 1 pipeline'u). Kolejność = indeksy etykiet klasyfikatora.
CATEGORIES: list[str] = [
    "platnosci",
    "dostawa",
    "zwroty_reklamacje",
    "konto",
    "produkty_gwarancja",
    "poza_zakresem",
]
CATEGORY_LABELS_PL: dict[str, str] = {
    "platnosci": "Płatności",
    "dostawa": "Dostawa",
    "zwroty_reklamacje": "Zwroty i reklamacje",
    "konto": "Konto i logowanie",
    "produkty_gwarancja": "Produkty i gwarancja",
    "poza_zakresem": "Poza zakresem",
}
OUT_OF_SCOPE = "poza_zakresem"


def _env(name: str, default: str, *aliases: str) -> str:
    """Wartość zmiennej środowiskowej; `aliases` to starsze nazwy (LMSTUDIO_*) sprawdzane, gdy nowej brak."""
    for key in (name, *aliases):
        value = os.getenv(key)
        if value:
            return value
    return default


@dataclass
class GenerationParams:
    """Parametry generowania przekazywane do serwera LLM (opis w docs/01_konfiguracja_docker_llama_cpp.md)."""

    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    seed: int | None = 42
    # "none" wyłącza rozumowanie Gemmy 4, "low"/"medium"/"high" włącza
    reasoning_effort: str = "none"


@dataclass
class Settings:
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:8080/v1",
                                                       "LMSTUDIO_BASE_URL"))
    # osobny serwer embeddingów (w Dockerze kontener embeddings), domyślnie ten sam adres
    embedding_base_url: str = field(default_factory=lambda: _env("EMBEDDING_BASE_URL", "")
                                    or _env("LLM_BASE_URL", "http://localhost:8080/v1", "LMSTUDIO_BASE_URL"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "local", "LMSTUDIO_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "google/gemma-4-12b-qat"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-bge-m3"))
    classifier_path: Path = field(default_factory=lambda: ROOT / _env("CLASSIFIER_PATH", "models/herbert-faq"))
    classifier_threshold: float = field(default_factory=lambda: float(_env("CLASSIFIER_THRESHOLD", "0.55")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))


settings = Settings()
