"""Buduje indeks bazy wiedzy (FAQ + dokumenty firmowe) w ChromaDB i pokazuje przykładowe wyszukiwanie.

Użycie:
    python scripts/build_index.py            # przebudowa indeksu od zera
    python scripts/build_index.py --query "ile kosztuje paczkomat"
"""

from __future__ import annotations

import argparse
import time

from faq_assistant.ingest import load_documents, load_faq
from faq_assistant.retrieval import METHODS, KnowledgeBase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--query", default="Ile mam czasu na zwrot towaru kupionego online?")
    ap.add_argument("--no-rebuild", action="store_true")
    a = ap.parse_args()

    kb = KnowledgeBase()
    if not a.no_rebuild:
        kb.reset()
        faq = load_faq()
        docs = load_documents(chunk_size=a.chunk_size, overlap=a.overlap)
        t = time.perf_counter()
        kb.add(faq + docs)
        print(f"Zindeksowano {len(faq)} wpisów FAQ + {len(docs)} fragmentów dokumentów "
              f"w {time.perf_counter() - t:.1f}s")

    print(f"\nZapytanie: {a.query}")
    for m in METHODS:
        print(f"\n--- {m} ---")
        for h in kb.search(a.query, k=3, method=m):
            print(f"{h.rank}. [{h.score:.3f}] {h.chunk.citation}: {h.chunk.text[:90].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
