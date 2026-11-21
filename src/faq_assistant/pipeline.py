"""Orkiestracja pipeline'u: (obraz ->) przepisanie pytania -> klasyfikacja -> retrieval -> odpowiedź RAG.

Każde wywołanie zwraca obiekt `PipelineTrace` z wynikami i czasami wszystkich etapów, na jego podstawie
UI pokazuje „wizualizację etapów”, a moduł export tworzy pliki .txt i .svg.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import OUT_OF_SCOPE, GenerationParams, settings
from .prompts import CATEGORY_HINTS, DECOMPOSE_SYSTEM, OUT_OF_SCOPE_ANSWER, REWRITE_SYSTEM, category_label
from .retrieval import Hit, KnowledgeBase


@dataclass
class StageTiming:
    name: str
    seconds: float


@dataclass
class PipelineTrace:
    user_input: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    image_name: str | None = None
    image_ocr: str | None = None
    question: str = ""                  # pytanie po ewentualnym odczycie z obrazu i przepisaniu
    rewritten: bool = False
    sub_questions: list[dict] = field(default_factory=list)  # pytanie złożone -> podpytania z routingiem
    category: str | None = None
    confidence: float = 0.0
    probs: dict[str, float] = field(default_factory=dict)
    classifier: str = ""
    route: str = ""                     # opis decyzji routingu
    search_category: str | None = None  # kategoria użyta jako filtr (None = cała baza)
    retrieval_method: str = ""
    hits: list[Hit] = field(default_factory=list)
    answer: str = ""
    answered_by: str = ""               # "rag" | "out_of_scope" | "empty"
    params: dict = field(default_factory=dict)
    timings: list[StageTiming] = field(default_factory=list)

    def add_timing(self, name: str, t0: float) -> None:
        self.timings.append(StageTiming(name, time.perf_counter() - t0))

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.timings)

    def sources(self) -> list[dict]:
        return [{"n": i, "source": h.chunk.citation, "kind": h.chunk.kind, "score": round(h.score, 4),
                 "text": h.chunk.text} for i, h in enumerate(self.hits, 1)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hits"] = self.sources()
        return d


def is_compound(question: str) -> bool:
    """Heurystyka: długie pytanie z kilkoma pytajnikami lub numerowaną listą spraw = pytanie złożone."""
    numbered = len(re.findall(r"(?m)^\s*\d+[.)]\s", question))
    return len(question) > 250 and (question.count("?") >= 3 or numbered >= 2)


def strip_long_dashes(text: str) -> str:
    """Usuwa długie myślniki (– —) z odpowiedzi: wtrącenie „ – ” zamienia na przecinek, zakres „5–6” na łącznik."""
    text = re.sub(r"^([ \t]*)[–—][ \t]*", r"\1- ", text, flags=re.M)  # punktor na początku linii
    text = re.sub(r"[ \t]+[–—][ \t]+", ", ", text)                      # wtrącenie w zdaniu
    text = re.sub(r"(?<=\S)[–—](?=\S)", "-", text)                      # zakres, np. 9:00–16:00
    return text.replace("–", "-").replace("—", "-")


def load_default_classifier():
    """HerBERT (jeśli wytrenowany) -> TF-IDF (zapisany albo trenowany od razu z train.jsonl) -> LLM zero-shot.

    Dzięki temu aplikacja działa także na komputerze bez folderu models/ (np. po sklonowaniu repozytorium):
    TF-IDF uczy się w ułamku sekundy, a HerBERT można odtworzyć: python scripts/train_classifier.py train
    """
    from .classifier import BertClassifier, LLMClassifier, TfidfClassifier
    from .config import DATA_DIR, MODELS_DIR
    from .utils import read_jsonl

    if (settings.classifier_path / "config.json").exists():
        return BertClassifier(settings.classifier_path)
    tfidf_path = MODELS_DIR / "tfidf_lr.joblib"
    if tfidf_path.exists():
        return TfidfClassifier.load(tfidf_path)
    train_file = DATA_DIR / "classifier" / "train.jsonl"
    if train_file.exists():
        rows = read_jsonl(train_file)
        clf = TfidfClassifier("char_wb").fit([r["text"] for r in rows], [r["label"] for r in rows])
        MODELS_DIR.mkdir(exist_ok=True)
        clf.save(tfidf_path)
        print(f"[setup] Brak HerBERT, wytrenowano klasyfikator zastępczy TF-IDF ({len(rows)} przykładów)")
        return clf
    return LLMClassifier()


class FAQPipeline:
    def __init__(self, kb: KnowledgeBase | None = None, classifier=None, llm=None,
                 threshold: float | None = None) -> None:
        from .llm import LMStudioClient

        self.llm = llm or LMStudioClient()
        self.kb = kb or KnowledgeBase(llm=self.llm)
        if kb is None and len(self.kb) == 0:
            # pierwsze uruchomienie (brak folderu chroma_db/), budujemy indeks bazy wiedzy automatycznie
            from .ingest import load_knowledge_base

            n = self.kb.add(load_knowledge_base())
            print(f"[setup] Zbudowano indeks bazy wiedzy: {n} fragmentów")
        self.classifier = classifier if classifier is not None else load_default_classifier()
        self.threshold = settings.classifier_threshold if threshold is None else threshold

    def rewrite(self, question: str, history: list[dict]) -> str:
        convo = "\n".join(f"{'Klient' if t['role'] == 'user' else 'Konsultant'}: {t['content'][:400]}"
                          for t in history[-4:])
        prompt = f"Historia rozmowy:\n{convo}\n\nOstatnie pytanie klienta: {question}\n\nSamodzielne pytanie:"
        out = self.llm.complete(prompt, system=REWRITE_SYSTEM,
                                params=GenerationParams(temperature=0.0, max_tokens=400, seed=0))
        return out.strip().strip('"') or question

    def prepare(self, user_input: str, history: list[dict] | None = None, image: str | Path | None = None,
                user_kb: KnowledgeBase | None = None, use_classifier: bool = True, method: str = "hybrid",
                top_k: int | None = None, params: GenerationParams | None = None,
                classifier=None) -> PipelineTrace:
        """Etapy przed generowaniem odpowiedzi. Gdy pytanie jest poza zakresem, od razu ustawia odpowiedź."""
        top_k = top_k or settings.top_k
        params = params or GenerationParams()
        tr = PipelineTrace(user_input=user_input, retrieval_method=method, params=asdict(params))
        question = user_input.strip()

        if image:
            from .vision import image_to_question

            t0 = time.perf_counter()
            res = image_to_question(self.llm, image, user_input)
            tr.image_name, tr.image_ocr = Path(image).name, res["ocr"]
            question = res["question"] or question
            tr.add_timing("Obraz → pytanie (vision)", t0)

        # pytanie uzupełniające (np. „a w sobotę?”) -> samodzielne
        if history:
            t0 = time.perf_counter()
            new_q = self.rewrite(question, history)
            tr.rewritten = new_q.strip().lower() != question.strip().lower()
            question = new_q
            tr.add_timing("Przepisanie pytania", t0)
        tr.question = question
        clf = classifier or self.classifier

        # kilka spraw w jednej wiadomości -> podpytania, każde z własnym routingiem i wyszukiwaniem
        if is_compound(question):
            t0 = time.perf_counter()
            subs = self.decompose(question)
            tr.add_timing("Podział na podpytania", t0)
            if len(subs) > 1:
                self._prepare_compound(tr, subs, clf if use_classifier else None, user_kb, method, top_k)
                return tr

        if use_classifier and clf is not None:
            t0 = time.perf_counter()
            pred = clf.predict(question)
            tr.category, tr.confidence, tr.probs, tr.classifier = pred.label, pred.confidence, pred.probs, pred.method
            tr.add_timing("Klasyfikacja (routing)", t0)
            if pred.label == OUT_OF_SCOPE and pred.confidence >= self.threshold and not user_kb:
                tr.route = f"Poza zakresem (pewność {pred.confidence:.0%}), odmowa bez wywołania LLM"
                tr.answer, tr.answered_by = OUT_OF_SCOPE_ANSWER, "out_of_scope"
                return tr
            if pred.label != OUT_OF_SCOPE and pred.confidence >= self.threshold:
                tr.search_category = pred.label
                tr.route = (f"Kategoria „{category_label(pred.label)}” (pewność {pred.confidence:.0%}): "
                            "FAQ zawężone do kategorii + dokumenty firmowe")
            else:
                tr.route = (f"Niska pewność ({pred.confidence:.0%} < {self.threshold:.0%}): "
                            "wyszukiwanie w całej bazie wiedzy")
        else:
            tr.route = "Klasyfikator wyłączony: wyszukiwanie w całej bazie wiedzy"

        t0 = time.perf_counter()
        hits = self.kb.search(question, k=top_k, method=method, category=tr.search_category)
        if user_kb is not None and len(user_kb):
            user_hits = user_kb.search(question, k=top_k, method=method)
            hits = user_hits + hits[: max(2, top_k // 2)]  # pliki użytkownika mają pierwszeństwo
        tr.hits = hits
        tr.add_timing(f"Wyszukiwanie ({method})", t0)
        return tr

    def decompose(self, question: str) -> list[str]:
        """Dzieli pytanie złożone na samodzielne podpytania (LLM, wynik w JSON)."""
        from .utils import parse_json

        raw = self.llm.complete(question, system=DECOMPOSE_SYSTEM,
                                params=GenerationParams(temperature=0.0, max_tokens=2048, seed=0))
        try:
            subs = [s.strip() for s in parse_json(raw) if isinstance(s, str) and s.strip()]
        except (ValueError, TypeError):
            return [question]
        return subs[:12] or [question]

    def _prepare_compound(self, tr: PipelineTrace, subs: list[str], clf, user_kb, method: str, top_k: int) -> None:
        """Routing i wyszukiwanie osobno dla każdego podpytania; fragmenty łączone naprzemiennie, bez powtórzeń."""
        per_sub = 3
        lists, cats, probs_sum = [], [], {}
        t_cls = t_ret = 0.0
        for sq in subs:
            entry = {"question": sq, "category": None, "confidence": 0.0}
            cat = None
            if clf is not None:
                t0 = time.perf_counter()
                pred = clf.predict(sq)
                t_cls += time.perf_counter() - t0
                entry.update(category=pred.label, confidence=pred.confidence)
                for c, p in pred.probs.items():
                    probs_sum[c] = probs_sum.get(c, 0.0) + p / len(subs)
                tr.classifier = pred.method
                if pred.label == OUT_OF_SCOPE and pred.confidence >= self.threshold:
                    entry["route"] = "poza zakresem, pominięte"
                    tr.sub_questions.append(entry)
                    continue
                if pred.confidence >= self.threshold:
                    cat = pred.label
                    cats.append(cat)
            t0 = time.perf_counter()
            hits = self.kb.search(sq, k=per_sub, method=method, category=cat)
            if user_kb is not None and len(user_kb):
                hits = user_kb.search(sq, k=2, method=method) + hits
            t_ret += time.perf_counter() - t0
            entry["route"] = f"kategoria {category_label(cat)}" if cat else "cała baza"
            tr.sub_questions.append(entry)
            lists.append(hits)
        # łączenie naprzemienne: najpierw najlepsze trafienie każdego podpytania, potem kolejne
        merged, seen = [], set()
        for rank in range(max((len(h) for h in lists), default=0)):
            for hits in lists:
                if rank < len(hits) and hits[rank].chunk.id not in seen:
                    seen.add(hits[rank].chunk.id)
                    merged.append(hits[rank])
        tr.hits = merged[:24]
        tr.probs = probs_sum
        if cats:
            tr.category = max(set(cats), key=cats.count)
            tr.confidence = cats.count(tr.category) / len(subs)
        tr.timings.append(StageTiming("Klasyfikacja (routing)", t_cls))
        tr.timings.append(StageTiming(f"Wyszukiwanie ({method})", t_ret))
        if not lists:
            tr.route = "Wszystkie podpytania poza zakresem, odmowa bez wywołania LLM"
            tr.answer, tr.answered_by = OUT_OF_SCOPE_ANSWER, "out_of_scope"
            return
        tr.route = (f"Pytanie złożone: {len(subs)} podpytań, osobny routing i wyszukiwanie dla każdego "
                    f"({len(tr.hits)} fragmentów łącznie)")
        # dłuższa odpowiedź na wiele spraw: podnosimy limit, jeśli użytkownik zostawił mały
        tr.params["max_tokens"] = max(tr.params.get("max_tokens", 2048), 500 * len(subs))

    def generate_stream(self, tr: PipelineTrace, history: list[dict] | None = None) -> Iterator[str]:
        """Etap 4: generowanie odpowiedzi RAG (strumieniowo). Uzupełnia tr.answer."""
        from .qa import answer_stream

        if tr.answered_by == "out_of_scope":
            yield tr.answer
            return
        t0 = time.perf_counter()
        params = GenerationParams(**tr.params)
        parts: list[str] = []
        for token in answer_stream(self.llm, tr.question, tr.hits, history, params, tr.image_ocr):
            parts.append(token)
            yield token
        text = strip_long_dashes("".join(parts).strip())
        if not text:
            # model zużył cały limit tokenów (np. na ukryte rozumowanie) i nie zwrócił treści
            text = ("Nie udało się wygenerować odpowiedzi: model wykorzystał cały limit tokenów, zanim zaczął "
                    "odpowiadać. Zwiększ max_tokens albo ustaw reasoning_effort na „none” i zadaj pytanie ponownie.")
            yield text
            tr.answer, tr.answered_by = text, "empty"
            tr.add_timing("Generowanie odpowiedzi (LLM)", t0)
            return
        hint = CATEGORY_HINTS.get(tr.category or "")
        if hint and hint not in text and not tr.sub_questions:
            text += f"\n\nWskazówka: {hint}"
            yield f"\n\nWskazówka: {hint}"
        tr.answer, tr.answered_by = text, "rag"
        tr.add_timing("Generowanie odpowiedzi (LLM)", t0)

    def run(self, user_input: str, history: list[dict] | None = None, **kw) -> PipelineTrace:
        tr = self.prepare(user_input, history=history, **kw)
        for _ in self.generate_stream(tr, history):
            pass
        return tr
