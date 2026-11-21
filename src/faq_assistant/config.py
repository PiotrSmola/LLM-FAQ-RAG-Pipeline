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


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class GenerationParams:
    """Parametry generowania przekazywane do LM Studio (opis w docs/01_konfiguracja_lm_studio.md)."""

    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 700
    seed: int | None = 42


@dataclass
class Settings:
    base_url: str = field(default_factory=lambda: _env("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))
    api_key: str = field(default_factory=lambda: _env("LMSTUDIO_API_KEY", "lm-studio"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "google/gemma-4-12b-qat"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-bge-m3"))
    classifier_path: Path = field(default_factory=lambda: ROOT / _env("CLASSIFIER_PATH", "models/herbert-faq"))
    classifier_threshold: float = field(default_factory=lambda: float(_env("CLASSIFIER_THRESHOLD", "0.55")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))


settings = Settings()
