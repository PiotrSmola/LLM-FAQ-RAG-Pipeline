"""Ewaluacja pipeline'u na zbiorze data/eval/qa_eval.jsonl (40 pytań, w tym 3 poza zakresem).

Części:
  retrieval: Recall@1/@3/@k i MRR dla TF-IDF, BM25, dense, hybrid; bez filtra i z filtrem kategorii (routing)
  qa: ablacja (synergia komponentów) + porównanie z modelami z laboratoriów:
                llm_bez_rag: sam LLM, bez dokumentów (Lab 5: halucynacje)
                rag_bez_klasyfikatora: retrieval w całej bazie, bez routingu
                pelny_pipeline: klasyfikacja -> retrieval z filtrem -> LLM
                extractive_xlmr: extractive QA (span) na tych samych fragmentach (Lab 3)
                flan_t5: FLAN-T5-base z Lab 3 na tych samych fragmentach
              metryki: Token-F1 (Lab 3), ROUGE-L, BERTScore-F1 (Lab 4), LLM-as-judge: faithfulness/correctness,
              poprawność odmowy dla pytań poza zakresem.

Użycie:
    python scripts/evaluate.py                 # wszystko
    python scripts/evaluate.py --part retrieval
    python scripts/evaluate.py --part qa --skip flan_t5
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from faq_assistant.config import DATA_DIR, OUT_OF_SCOPE, RESULTS_DIR, GenerationParams
from faq_assistant.pipeline import FAQPipeline
from faq_assistant.retrieval import METHODS, tokenize_pl
from faq_assistant.utils import parse_json, read_jsonl

REFUSAL_MARKERS = ("nie odpowiem", "nie mam tej informacji", "nie posiadam", "nie mogę pomóc", "poza zakresem",
                   "nie dotyczy", "mogę pomóc w sprawach", "nie jestem w stanie")


def is_hit(chunk, gold: set[str]) -> bool:
    return chunk.source in gold


def eval_retrieval(pipe: FAQPipeline, data: list[dict], k: int = 4) -> pd.DataFrame:
    out = RESULTS_DIR / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    items = [d for d in data if d["gold_sources"]]
    preds = {d["id"]: pipe.classifier.predict(d["question"]) for d in items}
    rows = []
    for method in METHODS:
        for routing in (False, True):
            r1 = r3 = rk = mrr = 0.0
            t0 = time.perf_counter()
            for d in items:
                p = preds[d["id"]]
                cat = p.label if routing and p.label != OUT_OF_SCOPE and p.confidence >= pipe.threshold else None
                hits = pipe.kb.search(d["question"], k=10, method=method, category=cat)
                gold = set(d["gold_sources"])
                ranks = [i for i, h in enumerate(hits, 1) if is_hit(h.chunk, gold)]
                first = ranks[0] if ranks else None
                r1 += first == 1
                r3 += first is not None and first <= 3
                rk += first is not None and first <= k
                mrr += 1 / first if first else 0
            n = len(items)
            rows.append({"metoda": method, "routing": "z filtrem kategorii" if routing else "cała baza",
                         "Recall@1": r1 / n, "Recall@3": r3 / n, f"Recall@{k}": rk / n, "MRR": mrr / n,
                         "ms/zapytanie": 1000 * (time.perf_counter() - t0) / n})
            print(f"{method:7s} {'routing' if routing else 'all    '}  R@1={r1 / n:.3f} R@3={r3 / n:.3f} "
                  f"R@{k}={rk / n:.3f} MRR={mrr / n:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "retrieval_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv = df.pivot(index="metoda", columns="routing", values="MRR").loc[list(METHODS)]
    piv.plot.bar(ax=ax, rot=0, color=["#8a94a6", "#2b4c7e"])
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 1)
    ax.set_title(f"Jakość wyszukiwania (MRR, {len(items)} pytań), wpływ metody i routingu klasyfikatorem")
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"retrieval_mrr.{ext}", dpi=150)
    plt.close(fig)
    return df


def token_f1(pred: str, gold: str) -> float:
    """Token-F1 jak w Lab 3, ale z tokenizacją dla polskiego (bez diakrytyków, stemming prefiksowy)."""
    p, g = Counter(tokenize_pl(pred, stem=5)), Counter(tokenize_pl(gold, stem=5))
    common = sum((p & g).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


class _PLTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return tokenize_pl(text, stem=5)


def judge(llm, question: str, reference: str, answer: str, context: str) -> dict:
    from faq_assistant.prompts import JUDGE_SYSTEM, JUDGE_USER

    raw = llm.complete(JUDGE_USER.format(context=context, question=question, reference=reference, answer=answer),
                       system=JUDGE_SYSTEM, params=GenerationParams(temperature=0.0, max_tokens=600, seed=0))
    try:
        d = parse_json(raw)
        return {"faithfulness": float(d["faithfulness"]), "correctness": float(d["correctness"]),
                "judge_comment": d.get("comment", "")}
    except (ValueError, KeyError, TypeError):
        return {"faithfulness": None, "correctness": None, "judge_comment": raw[:200]}


def summarize(df: pd.DataFrame, out) -> pd.DataFrame:
    df.to_csv(out / "qa_details.csv", index=False, encoding="utf-8-sig")
    metric_cols = [c for c in ("token_f1", "rougeL", "bertscore_f1", "faithfulness", "correctness",
                               "trafienie_zrodla", "odmowa_poprawna") if c in df.columns]
    summary = df.groupby("wariant")[metric_cols].mean().round(3)
    order = [v for v in ("llm_bez_rag", "flan_t5", "extractive_xlmr", "rag_bez_klasyfikatora", "pelny_pipeline")
             if v in summary.index]
    summary = summary.loc[order].reset_index()
    summary.to_csv(out / "qa_summary.csv", index=False)
    print("\n", summary.to_string(index=False))

    plot_cols = [c for c in ("token_f1", "rougeL", "bertscore_f1") if c in summary.columns]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    summary.set_index("wariant")[plot_cols].plot.bar(ax=axes[0], rot=20)
    axes[0].set_title("Metryki automatyczne vs odpowiedź wzorcowa")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)
    jcols = [c for c in ("faithfulness", "correctness") if c in summary.columns]
    if jcols:
        summary.set_index("wariant")[jcols].dropna(how="all").plot.bar(ax=axes[1], rot=20, color=["#2e8b57", "#e07a1f"])
        axes[1].set_ylim(1, 5)
        axes[1].set_title("LLM-as-judge (1-5)")
        axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"qa_ablacja.{ext}", dpi=150)
    plt.close(fig)
    return summary


def bertscore_f1(cands: list[str], refs: list[str], model_name: str = "google-bert/bert-base-multilingual-cased",
                 layer: int = 9, batch_size: int = 32) -> list[float]:
    """BERTScore-F1 (Zhang i in., 2020), własna implementacja, bo pakiet bert_score 0.3.13 nie działa z transformers 5.

    Dla każdej pary: embeddingi tokenów z warstwy `layer` (9. warstwa mBERT, zalecenie autorów dla modeli
    wielojęzycznych), macierz podobieństw cosinusowych, precision = średnie max po tokenach kandydata,
    recall = średnie max po tokenach referencji, F1 = średnia harmoniczna. Bez wag IDF i bez przeskalowania.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()

    def embed(texts: list[str]) -> list[torch.Tensor]:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(dev)
            with torch.no_grad():
                h = model(**enc, output_hidden_states=True).hidden_states[layer]
            h = torch.nn.functional.normalize(h, dim=-1)
            for j in range(h.shape[0]):
                keep = enc["attention_mask"][j].bool() & ~torch.isin(
                    enc["input_ids"][j], torch.tensor(tok.all_special_ids, device=dev))
                out.append(h[j][keep])
        return out

    ec, er = embed(cands), embed(refs)
    scores = []
    for c, r in zip(ec, er):
        if len(c) == 0 or len(r) == 0:
            scores.append(0.0)
            continue
        sim = c @ r.T
        prec, rec = sim.max(1).values.mean().item(), sim.max(0).values.mean().item()
        scores.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return scores


def add_bertscore(df: pd.DataFrame) -> None:
    mask = df["kategoria"] != OUT_OF_SCOPE
    cand = df.loc[mask, "odpowiedz"].fillna("").astype(str).replace("", "-").tolist()
    refs = df.loc[mask, "referencja"].astype(str).tolist()
    df.loc[mask, "bertscore_f1"] = bertscore_f1(cand, refs)


def eval_qa(pipe: FAQPipeline, data: list[dict], skip: set[str]) -> pd.DataFrame:
    from rouge_score import rouge_scorer

    from faq_assistant.qa import build_context

    out = RESULTS_DIR / "qa"
    out.mkdir(parents=True, exist_ok=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=_PLTokenizer())
    params = GenerationParams(temperature=0.0, seed=0)
    extractive = flan = None
    if "extractive_xlmr" not in skip:
        from faq_assistant.qa import ExtractiveQA
        extractive = ExtractiveQA()
    if "flan_t5" not in skip:
        from faq_assistant.qa import FlanT5QA
        flan = FlanT5QA()

    rows = []
    for i, d in enumerate(data, 1):
        q, ref, oos = d["question"], d["reference"], d["category"] == OUT_OF_SCOPE
        full = pipe.run(q, params=params)
        no_cls = pipe.run(q, params=params, use_classifier=False)
        context = build_context(full.hits) if full.hits else build_context(no_cls.hits)
        variants = {"pelny_pipeline": (full.answer, full), "rag_bez_klasyfikatora": (no_cls.answer, no_cls)}
        if "llm_bez_rag" not in skip:
            from faq_assistant.qa import answer_without_context
            variants["llm_bez_rag"] = (answer_without_context(pipe.llm, q, params), None)
        hits = full.hits or no_cls.hits
        if extractive is not None:
            variants["extractive_xlmr"] = (extractive.answer(q, hits)[0] if hits else "", None)
        if flan is not None:
            variants["flan_t5"] = (flan.answer(q, hits) if hits else "", None)

        for name, (ans, tr) in variants.items():
            row = {"id": d["id"], "pytanie": q, "kategoria": d["category"], "wariant": name, "odpowiedz": ans,
                   "referencja": ref}
            if oos:
                refused = (tr is not None and tr.answered_by == "out_of_scope") or \
                          any(m in ans.lower() for m in REFUSAL_MARKERS)
                row["odmowa_poprawna"] = float(refused)
            else:
                row["token_f1"] = token_f1(ans, ref)
                row["rougeL"] = scorer.score(ref, ans)["rougeL"].fmeasure
                if tr is not None:
                    gold = set(d["gold_sources"])
                    row["trafienie_zrodla"] = float(any(is_hit(h.chunk, gold) for h in tr.hits))
                if name in {"pelny_pipeline", "rag_bez_klasyfikatora", "llm_bez_rag"} or name == "extractive_xlmr":
                    row.update(judge(pipe.llm, q, ref, ans, context))
            rows.append(row)
        print(f"[{i}/{len(data)}] {q[:60]}")

    df = pd.DataFrame(rows)
    add_bertscore(df)
    return summarize(df, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "retrieval", "qa", "bertscore"], default="all",
                    help="bertscore = przelicz tylko BERTScore i podsumowanie z zapisanego qa_details.csv")
    ap.add_argument("--skip", nargs="*", default=[], help="pomiń warianty QA, np. flan_t5 extractive_xlmr")
    ap.add_argument("--limit", type=int, default=None, help="tylko pierwsze N pytań (szybki test)")
    a = ap.parse_args()

    if a.part == "bertscore":
        out = RESULTS_DIR / "qa"
        df = pd.read_csv(out / "qa_details.csv")
        add_bertscore(df)
        summarize(df, out)
        return
    data = read_jsonl(DATA_DIR / "eval" / "qa_eval.jsonl")[: a.limit]
    pipe = FAQPipeline()
    print(f"Klasyfikator: {getattr(pipe.classifier, 'method', '-')}, baza: {len(pipe.kb)} fragmentów\n")
    if a.part in {"all", "retrieval"}:
        eval_retrieval(pipe, data)
    if a.part in {"all", "qa"}:
        eval_qa(pipe, data, set(a.skip))


if __name__ == "__main__":
    main()
