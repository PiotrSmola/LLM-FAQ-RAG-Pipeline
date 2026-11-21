"""Przetwarzanie wsadowe: lista pytań z pliku -> odpowiedzi ze źródłami.

Wejście: plik .txt (jedno pytanie w linii), .csv (kolumna "question"/"pytanie") lub .jsonl (pole "question").
Wyjście (katalog --out): wyniki.csv, wyniki.xlsx, wyniki.jsonl, raport.txt oraz opcjonalnie SVG dla każdego pytania.

Przykład:
    python scripts/batch.py data/examples/pytania.csv --out results/batch --svg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from faq_assistant.config import GenerationParams
from faq_assistant.export import conversation_to_text, save_svg, save_text
from faq_assistant.pipeline import FAQPipeline
from faq_assistant.prompts import category_label


def read_questions(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".jsonl":
        return [json.loads(line)["question"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            # jednokolumnowy plik bez cudzysłowów (pytania z przecinkami), czytamy linia po linii
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            header = lines[0].lower() in {"question", "pytanie", "pytania"}
            return lines[1:] if header else lines
    else:
        df = pd.read_excel(path)
    col = next((c for c in df.columns if c.lower() in {"question", "pytanie", "pytania"}), df.columns[0])
    return [str(q).strip() for q in df[col].dropna() if str(q).strip()]


def run_batch(pipe: FAQPipeline, questions: list[str], out_dir: Path, method: str = "hybrid", top_k: int = 4,
              svg: bool = False, params: GenerationParams | None = None, progress=None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, traces = [], []
    for i, q in enumerate(questions, 1):
        t = time.perf_counter()
        tr = pipe.run(q, method=method, top_k=top_k, params=params)
        traces.append(tr)
        rows.append({
            "nr": i, "pytanie": q, "kategoria": category_label(tr.category) if tr.category else "",
            "pewnosc": round(tr.confidence, 3), "routing": tr.route,
            "zrodla": "; ".join(s["source"] for s in tr.sources()), "odpowiedz": tr.answer,
            "czas_s": round(time.perf_counter() - t, 2),
        })
        if svg:
            save_svg(tr, out_dir / "svg" / f"pytanie_{i:03d}.svg")
        if progress:
            progress(i, len(questions), q)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "wyniki.csv", index=False, encoding="utf-8-sig")  # BOM -> poprawne polskie znaki w Excelu
    df.to_excel(out_dir / "wyniki.xlsx", index=False)
    with open(out_dir / "wyniki.jsonl", "w", encoding="utf-8") as f:
        for tr in traces:
            f.write(json.dumps(tr.to_dict(), ensure_ascii=False, default=str) + "\n")
    save_text(conversation_to_text(traces), out_dir / "raport.txt")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/batch"))
    ap.add_argument("--method", default="hybrid", choices=["tfidf", "bm25", "dense", "hybrid"])
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--svg", action="store_true", help="zapisz wizualizację SVG dla każdego pytania")
    a = ap.parse_args()

    questions = read_questions(a.input)
    print(f"Wczytano {len(questions)} pytań z {a.input}")
    pipe = FAQPipeline()
    t = time.perf_counter()
    df = run_batch(pipe, questions, a.out, a.method, a.top_k, a.svg,
                   progress=lambda i, n, q: print(f"[{i}/{n}] {q[:70]}"))
    print(f"\nGotowe w {time.perf_counter() - t:.0f}s. Wyniki w {a.out}/ (wyniki.csv, wyniki.xlsx, raport.txt)")
    print(df["kategoria"].value_counts().to_string())


if __name__ == "__main__":
    main()
