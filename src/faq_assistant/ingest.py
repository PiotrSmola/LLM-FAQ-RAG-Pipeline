"""Wczytywanie bazy wiedzy: wpisy FAQ i dokumenty (txt/md/pdf) -> fragmenty (chunki) z metadanymi."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DATA_DIR
from .utils import read_jsonl

DOC_CATEGORY = "dokument"


@dataclass
class Chunk:
    id: str
    text: str
    source: str            # id wpisu FAQ albo nazwa pliku
    title: str             # pytanie FAQ albo ścieżka nagłówków w dokumencie
    kind: str              # "faq" | "doc"
    category: str          # kategoria FAQ albo "dokument"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return self.source if self.kind == "faq" else f"{self.source} – {self.title}"


# ---------------------------------------------------------------------------------------------------- FAQ
def load_faq(path: str | Path = DATA_DIR / "faq" / "faq.jsonl") -> list[Chunk]:
    chunks = []
    for row in read_jsonl(path):
        chunks.append(Chunk(
            id=row["id"], text=f"Pytanie: {row['question']}\nOdpowiedź: {row['answer']}",
            source=row["id"], title=row["question"], kind="faq", category=row["category"],
            meta={"answer": row["answer"]},
        ))
    return chunks


# ---------------------------------------------------------------------------------------------- dokumenty
def read_document(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Dzieli tekst na sekcje według nagłówków markdown; zwraca (ścieżka_nagłówków, treść)."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            # pomijamy nagłówek H1 (tytuł całego dokumentu – jest w nazwie pliku), zostawiamy sekcje
            path = stack[1:] if len(stack) > 1 else stack
            sections.append((" > ".join(path) or "(początek)", body))
        buf.clear()

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = stack[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Dzieli długi tekst na okna ~size znaków z nakładką, tnąc na granicach akapitów/zdań."""
    if len(text) <= size:
        return [text]
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > size:
            out.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += (" " if cur else "") + p
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_document(path: str | Path, chunk_size: int = 1000, overlap: int = 150,
                   display_name: str | None = None) -> list[Chunk]:
    """Dokument -> fragmenty. Rozmiar w znakach (~1000 znaków ≈ 250 tokenów dla polskiego tekstu)."""
    path = Path(path)
    name = display_name or path.name
    text = read_document(path)
    sections = _split_markdown_sections(text) if path.suffix.lower() in {".md", ".txt"} else [("", text)]
    doc_title = next((t for t, _ in sections), name)
    chunks = []
    for heading, body in sections:
        for j, piece in enumerate(_window(body, chunk_size, overlap)):
            uid = hashlib.md5(f"{name}|{heading}|{j}|{piece[:50]}".encode()).hexdigest()[:10]
            title = heading or doc_title
            chunks.append(Chunk(
                id=f"{name}#{uid}", text=f"[{title}]\n{piece}", source=name, title=title,
                kind="doc", category=DOC_CATEGORY, meta={"part": j},
            ))
    return chunks


def load_documents(folder: str | Path = DATA_DIR / "docs", **kw) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            chunks += chunk_document(path, **kw)
    return chunks


def load_knowledge_base() -> list[Chunk]:
    return load_faq() + load_documents()
