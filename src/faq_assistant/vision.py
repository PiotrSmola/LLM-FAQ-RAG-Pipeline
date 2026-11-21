"""Obraz od klienta -> odczytany tekst i pytanie do pipeline'u."""

from __future__ import annotations

from pathlib import Path

from .config import GenerationParams
from .prompts import VISION_EXTRACT
from .utils import parse_json


def image_to_question(llm, image: str | Path | bytes, user_text: str = "") -> dict:
    """Zwraca {"ocr": ..., "question": ...}. Tekst wpisany przez użytkownika jest dołączany jako kontekst."""
    prompt = VISION_EXTRACT
    if user_text.strip():
        prompt += f"\n\nKlient dopisał do obrazu wiadomość: \"{user_text.strip()}\", uwzględnij ją w pytaniu."
    raw = llm.vision(prompt, image, params=GenerationParams(temperature=0.1, max_tokens=1500, seed=0))
    try:
        data = parse_json(raw)
        return {"ocr": str(data.get("ocr", "")).strip(), "question": str(data.get("question", "")).strip()}
    except (ValueError, AttributeError):
        return {"ocr": raw.strip(), "question": user_text.strip() or raw.strip()[:300]}
