"""Drobne narzędzia wspólne dla modułów."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json(text: str) -> Any:
    """Wyciąga pierwszy obiekt/tablicę JSON z odpowiedzi modelu (toleruje bloki ```json i dodatkowy tekst)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Nie udało się odczytać JSON z odpowiedzi modelu: {text[:200]!r}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
