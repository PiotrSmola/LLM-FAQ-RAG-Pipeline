"""Generowanie danych treningowych klasyfikatora przez LLM.

Dla każdej kategorii model w LM Studio generuje parafrazy/nowe pytania klientów, inspirowane wpisami FAQ.
Wynik: data/classifier/train.jsonl ({"text", "label", "source"}).
Zbiór testowy (data/classifier/test.jsonl) jest pisany ręcznie i usuwamy z treningu ewentualne duplikaty,
żeby nie było przecieku danych.

Użycie:
    python scripts/generate_training_data.py --per-category 120 --batch 30
"""

from __future__ import annotations

import argparse
import random
import time

from faq_assistant.config import CATEGORIES, DATA_DIR, OUT_OF_SCOPE, GenerationParams
from faq_assistant.llm import LMStudioClient
from faq_assistant.prompts import (
    CATEGORY_DESCRIPTIONS,
    GENERATE_OUT_OF_SCOPE_USER,
    GENERATE_TRAINING_SYSTEM,
    GENERATE_TRAINING_USER,
)
from faq_assistant.utils import normalize, parse_json, read_jsonl, write_jsonl


def generate_batch(llm: LMStudioClient, category: str, faq: list[dict], n: int, seed: int) -> list[str]:
    if category == OUT_OF_SCOPE:
        prompt = GENERATE_OUT_OF_SCOPE_USER.format(n=n)
    else:
        rng = random.Random(seed)
        pool = [f["question"] for f in faq if f["category"] == category]
        seeds = "\n".join(f"- {q}" for q in rng.sample(pool, min(8, len(pool))))
        prompt = GENERATE_TRAINING_USER.format(
            category=category, description=CATEGORY_DESCRIPTIONS[category], seeds=seeds, n=n
        )
    # wysoka temperatura = większa różnorodność; inny seed w każdej partii
    params = GenerationParams(temperature=1.0, top_p=0.95, top_k=60, max_tokens=4096, seed=seed)
    raw = llm.complete(prompt, system=GENERATE_TRAINING_SYSTEM, params=params)
    items = parse_json(raw)
    return [s.strip() for s in items if isinstance(s, str) and len(s.strip()) >= 3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-category", type=int, default=120)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--out", default=str(DATA_DIR / "classifier" / "train.jsonl"))
    args = ap.parse_args()

    llm = LMStudioClient()
    faq = read_jsonl(DATA_DIR / "faq" / "faq.jsonl")
    test_texts = {normalize(r["text"]) for r in read_jsonl(DATA_DIR / "classifier" / "test.jsonl")}

    rows: list[dict] = []
    seen: set[str] = set()
    for f in faq:
        key = normalize(f["question"])
        if key not in test_texts:
            rows.append({"text": f["question"], "label": f["category"], "source": "faq"})
            seen.add(key)

    for ci, cat in enumerate(CATEGORIES):
        collected: list[str] = []
        attempt = 0
        t0 = time.perf_counter()
        while len(collected) < args.per_category and attempt < args.per_category // args.batch + 4:
            seed = 1000 * ci + attempt
            attempt += 1
            try:
                batch = generate_batch(llm, cat, faq, args.batch, seed)
            except ValueError as e:
                print(f"  [{cat}] partia {attempt}: błąd parsowania, pomijam ({e})")
                continue
            for text in batch:
                key = normalize(text)
                if key in seen or key in test_texts:
                    continue
                seen.add(key)
                collected.append(text)
        collected = collected[: args.per_category]
        rows.extend({"text": t, "label": cat, "source": "llm"} for t in collected)
        print(f"[{cat}] {len(collected)} pytań w {time.perf_counter() - t0:.0f}s ({attempt} partii)")

    random.Random(42).shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"\nZapisano {len(rows)} przykładów -> {args.out}")


if __name__ == "__main__":
    main()
