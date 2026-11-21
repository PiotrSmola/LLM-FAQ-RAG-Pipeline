"""Etap 1: fine-tuning klasyfikatora BERT do routingu zapytań + porównanie z baseline'ami.

Tryby:
    train   – fine-tuning modelu BERT (domyślnie polski HerBERT) na data/classifier/train.jsonl
    compare – ewaluacja na ręcznym zbiorze testowym: BERT(y) vs TF-IDF+LR vs LLM zero-shot
    curve   – learning curve: F1 od liczby przykładów treningowych (jak zad. 2.4 z Lab 2)

Przykłady:
    python scripts/train_classifier.py train
    python scripts/train_classifier.py train --model distilbert-base-multilingual-cased --out models/distilbert-faq
    python scripts/train_classifier.py compare --bert models/herbert-faq models/distilbert-faq
    python scripts/train_classifier.py curve --sizes 50 100 200 400 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from faq_assistant.config import CATEGORIES, DATA_DIR, MODELS_DIR, RESULTS_DIR
from faq_assistant.utils import read_jsonl

OUT_DIR = RESULTS_DIR / "classifier"


def load_split() -> tuple[list[dict], list[dict]]:
    return read_jsonl(DATA_DIR / "classifier" / "train.jsonl"), read_jsonl(DATA_DIR / "classifier" / "test.jsonl")


# ------------------------------------------------------------------------------------------------ train
def finetune(model_name: str, train_rows: list[dict], out_dir: Path | None, epochs: int = 4, lr: float = 3e-5,
             batch_size: int = 16, seed: int = 42, val_frac: float = 0.1, quiet: bool = False):
    """Fine-tuning AutoModelForSequenceClassification (schemat z Lab 2, przykład 2). Zwraca (model, tokenizer)."""
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)

    set_seed(seed)
    label2id = {c: i for i, c in enumerate(CATEGORIES)}
    rows = train_rows[:]
    random.Random(seed).shuffle(rows)
    n_val = max(len(CATEGORIES), int(len(rows) * val_frac)) if val_frac > 0 else 0
    val_rows, tr_rows = rows[:n_val], rows[n_val:]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_ds(rs):
        ds = Dataset.from_dict({"text": [r["text"] for r in rs], "label": [label2id[r["label"]] for r in rs]})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=64), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CATEGORIES), id2label=dict(enumerate(CATEGORIES)), label2id=label2id
    )

    def metrics(ev):
        preds = np.argmax(ev.predictions, axis=-1)
        return {"accuracy": accuracy_score(ev.label_ids, preds),
                "f1_macro": f1_score(ev.label_ids, preds, average="macro")}

    tmp_dir = MODELS_DIR / "_trainer_tmp"
    args = TrainingArguments(
        output_dir=str(tmp_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch" if n_val else "no",
        save_strategy="epoch" if n_val else "no",
        load_best_model_at_end=bool(n_val),
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        logging_steps=20,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=seed,
        disable_tqdm=quiet,
    )
    trainer = Trainer(model=model, args=args, train_dataset=to_ds(tr_rows),
                      eval_dataset=to_ds(val_rows) if n_val else None,
                      processing_class=tokenizer, compute_metrics=metrics)
    trainer.train()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        log = [h for h in trainer.state.log_history]
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return trainer.model, tokenizer


def cmd_train(a: argparse.Namespace) -> None:
    train_rows, _ = load_split()
    out = Path(a.out) if a.out else MODELS_DIR / "herbert-faq"
    t = time.perf_counter()
    finetune(a.model, train_rows, out, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size)
    print(f"Model zapisany w {out} (trening {time.perf_counter() - t:.0f}s, {len(train_rows)} przykładów)")


# ---------------------------------------------------------------------------------------------- compare
def evaluate_predictions(name: str, y_true: list[str], y_pred: list[str], latency: float) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {"model": name, "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", labels=CATEGORIES, zero_division=0),
            "latency_ms_per_query": 1000 * latency}


def save_confusion(name: str, y_true: list[str], y_pred: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=CATEGORIES, ax=ax, cmap="Blues",
                                            xticks_rotation=35, colorbar=False)
    ax.set_title(f"Macierz pomyłek – {name}")
    fig.tight_layout()
    safe = name.replace("/", "_").replace(":", "_")
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"confusion_{safe}.{ext}", dpi=150)
    plt.close(fig)


def cmd_compare(a: argparse.Namespace) -> None:
    import pandas as pd

    from faq_assistant.classifier import BertClassifier, LLMClassifier, TfidfClassifier

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows, test_rows = load_split()
    texts = [r["text"] for r in test_rows]
    y_true = [r["label"] for r in test_rows]
    results, errors = [], []

    def run(name: str, predict_fn) -> None:
        t = time.perf_counter()
        y_pred = predict_fn()
        lat = (time.perf_counter() - t) / len(texts)
        res = evaluate_predictions(name, y_true, y_pred, lat)
        results.append(res)
        save_confusion(name, y_true, y_pred)
        errors.extend({"model": name, "text": x, "true": yt, "pred": yp}
                      for x, yt, yp in zip(texts, y_true, y_pred) if yt != yp)
        print(f"{name:32s} acc={res['accuracy']:.3f}  F1-macro={res['f1_macro']:.3f}  "
              f"{res['latency_ms_per_query']:.1f} ms/zapytanie")

    for analyzer in ("word", "char_wb"):
        clf = TfidfClassifier(analyzer).fit([r["text"] for r in train_rows], [r["label"] for r in train_rows])
        if analyzer == "char_wb":
            MODELS_DIR.mkdir(exist_ok=True)
            clf.save(MODELS_DIR / "tfidf_lr.joblib")
        run(clf.method, lambda c=clf: [CATEGORIES[i] for i in c.predict_proba(texts).argmax(1)])

    for path in a.bert:
        bert = BertClassifier(path)
        bert.predict_proba(texts[:4])  # rozgrzewka GPU (nie liczymy do czasu)
        run(bert.method, lambda b=bert: [b.labels[i] for i in b.predict_proba(texts, batch_size=1).argmax(1)])

    if not a.skip_llm:
        llm_clf = LLMClassifier()
        run("llm_zero_shot", lambda: [llm_clf.predict(x).label for x in texts])

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "comparison.csv", index=False)
    pd.DataFrame(errors).to_csv(OUT_DIR / "errors.csv", index=False)
    print(f"\nWyniki: {OUT_DIR / 'comparison.csv'}, błędy: {OUT_DIR / 'errors.csv'}")


# ------------------------------------------------------------------------------------------------ curve
def cmd_curve(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from sklearn.metrics import f1_score

    from faq_assistant.classifier import TfidfClassifier

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows, test_rows = load_split()
    texts = [r["text"] for r in test_rows]
    y_true = [r["label"] for r in test_rows]
    rows = []
    for size in a.sizes:
        for seed in a.seeds:
            rng = random.Random(seed)
            # próbkowanie warstwowe – każda klasa reprezentowana proporcjonalnie
            by_cls = {c: [r for r in train_rows if r["label"] == c] for c in CATEGORIES}
            subset = []
            for c in CATEGORIES:
                subset += rng.sample(by_cls[c], min(len(by_cls[c]), max(1, size // len(CATEGORIES))))
            model, tok = finetune(a.model, subset, None, epochs=a.epochs, seed=seed, val_frac=0, quiet=True)
            model.eval()
            enc = tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                pred = model(**enc).logits.argmax(-1).cpu().tolist()
            f1_bert = f1_score(y_true, [CATEGORIES[i] for i in pred], average="macro")
            tf = TfidfClassifier("char_wb").fit([r["text"] for r in subset], [r["label"] for r in subset])
            f1_tfidf = f1_score(y_true, [CATEGORIES[i] for i in tf.predict_proba(texts).argmax(1)], average="macro")
            rows.append({"size": len(subset), "seed": seed, "f1_bert": f1_bert, "f1_tfidf": f1_tfidf})
            print(f"n={len(subset):4d} seed={seed}  F1 BERT={f1_bert:.3f}  F1 TF-IDF={f1_tfidf:.3f}")
            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "learning_curve.csv", index=False)
    agg = df.groupby("size").agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, label in (("f1_bert", Path(a.model).name + " (fine-tuning)"), ("f1_tfidf", "TF-IDF + LR")):
        m, s = agg[(col, "mean")], agg[(col, "std")].fillna(0)
        ax.plot(agg.index, m, marker="o", label=label)
        ax.fill_between(agg.index, m - s, m + s, alpha=0.2)
    ax.set_xlabel("Liczba przykładów treningowych")
    ax.set_ylabel("F1-macro (zbiór testowy)")
    ax.set_title("Learning curve klasyfikatora kategorii")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"learning_curve.{ext}", dpi=150)
    print(f"\nZapisano {OUT_DIR / 'learning_curve.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", default="allegro/herbert-base-cased")
    t.add_argument("--out", default=None)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--batch-size", type=int, default=16)
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("compare")
    c.add_argument("--bert", nargs="*", default=[str(MODELS_DIR / "herbert-faq")])
    c.add_argument("--skip-llm", action="store_true")
    c.set_defaults(fn=cmd_compare)

    lc = sub.add_parser("curve")
    lc.add_argument("--model", default="allegro/herbert-base-cased")
    lc.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200, 400])
    lc.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    lc.add_argument("--epochs", type=int, default=6)
    lc.set_defaults(fn=cmd_curve)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
