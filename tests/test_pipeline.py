"""Testy jednostkowe – działają bez LM Studio (atrapa klienta LLM z deterministycznymi embeddingami)."""

from __future__ import annotations

import hashlib
import uuid

import numpy as np
import pytest

from faq_assistant.classifier import Prediction
from faq_assistant.config import CATEGORIES
from faq_assistant.ingest import _split_markdown_sections, _window, chunk_document, load_documents, load_faq
from faq_assistant.retrieval import KnowledgeBase, tokenize_pl
from faq_assistant.utils import parse_json


class FakeLLM:
    """Embedding = znormalizowany „worek słów” po haszowaniu tokenów; czat zwraca stały tekst."""

    model = "fake"

    def embed(self, texts, batch_size=32):
        out = []
        for t in texts:
            v = np.zeros(256)
            for tok in tokenize_pl(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
            out.append((v / (np.linalg.norm(v) or 1)).tolist())
        return out

    def chat_stream(self, messages, params=None):
        yield from ["Odpowiedź ", "testowa [1]."]

    def chat(self, messages, params=None):
        return "Odpowiedź testowa [1]."

    def complete(self, prompt, system=None, params=None):
        return "przepisane pytanie"


class FakeClassifier:
    method = "fake"

    def __init__(self, label, conf=0.9):
        self.label, self.conf = label, conf

    def predict(self, text):
        probs = {c: (self.conf if c == self.label else (1 - self.conf) / 5) for c in CATEGORIES}
        return Prediction(self.label, self.conf, probs, 0.0, self.method)


@pytest.fixture(scope="module")
def kb():
    k = KnowledgeBase(name=f"test_{uuid.uuid4().hex[:6]}", llm=FakeLLM(), persist=False)
    k.add(load_faq() + load_documents())
    return k


# ------------------------------------------------------------------------------------------------ ingest
def test_faq_loaded():
    faq = load_faq()
    assert len(faq) >= 70
    assert {c.category for c in faq} == set(CATEGORIES) - {"poza_zakresem"}


def test_markdown_sections_drop_h1():
    secs = _split_markdown_sections("# Tytuł\n\n## A\ntekst a\n### A1\ntekst a1\n## B\ntekst b")
    assert [s[0] for s in secs] == ["A", "A > A1", "B"]


def test_window_overlap_and_size():
    text = " ".join(f"Zdanie numer {i}." for i in range(200))
    parts = _window(text, size=300, overlap=50)
    assert len(parts) > 5
    assert all(len(p) <= 360 for p in parts)


def test_chunk_document_has_metadata():
    chunks = chunk_document("data/docs/cennik_dostaw.md")
    assert chunks and all(c.kind == "doc" and c.source == "cennik_dostaw.md" for c in chunks)


# --------------------------------------------------------------------------------------------- retrieval
def test_tokenize_pl_stemming_and_accents():
    assert tokenize_pl("Zwroty zwrotów ZWROTEM") == ["zwroty", "zwroto", "zwrote"]
    assert "lodowk" in tokenize_pl("lodówka")


@pytest.mark.parametrize("method", ["tfidf", "bm25", "dense", "hybrid"])
def test_search_finds_faq(kb, method):
    hits = kb.search("Ile kosztuje dostawa do paczkomatu?", k=5, method=method)
    assert any(h.chunk.source in {"DOS-01", "cennik_dostaw.md"} for h in hits)


def test_category_filter_keeps_docs_and_category(kb):
    hits = kb.search("ile kosztuje dostawa", k=10, method="bm25", category="konto")
    assert hits and all(h.chunk.kind == "doc" or h.chunk.category == "konto" for h in hits)


def test_hybrid_rrf_scores_sorted(kb):
    hits = kb.search("reklamacja telefonu", k=6, method="hybrid")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 2 / 61 for s in scores)  # RRF z k=60 i dwiema listami


# ---------------------------------------------------------------------------------------------- pipeline
def test_pipeline_out_of_scope_skips_llm(kb):
    from faq_assistant.pipeline import FAQPipeline

    p = FAQPipeline(kb=kb, classifier=FakeClassifier("poza_zakresem"), llm=FakeLLM())
    tr = p.run("jaka jest pogoda")
    assert tr.answered_by == "out_of_scope" and not tr.hits


def test_pipeline_routes_and_answers(kb):
    from faq_assistant.pipeline import FAQPipeline

    p = FAQPipeline(kb=kb, classifier=FakeClassifier("dostawa"), llm=FakeLLM())
    tr = p.run("Ile kosztuje kurier?", top_k=3)
    assert tr.search_category == "dostawa"
    assert tr.answer.startswith("Odpowiedź testowa") and tr.answered_by == "rag"
    assert {t.name.split(" (")[0] for t in tr.timings} >= {"Klasyfikacja", "Wyszukiwanie"}


def test_pipeline_low_confidence_searches_everything(kb):
    from faq_assistant.pipeline import FAQPipeline

    p = FAQPipeline(kb=kb, classifier=FakeClassifier("konto", conf=0.3), llm=FakeLLM())
    tr = p.prepare("cokolwiek")
    assert tr.search_category is None and "Niska pewność" in tr.route


def test_pipeline_rewrites_with_history(kb):
    from faq_assistant.pipeline import FAQPipeline

    p = FAQPipeline(kb=kb, classifier=FakeClassifier("dostawa"), llm=FakeLLM())
    tr = p.prepare("a w sobotę?", history=[{"role": "user", "content": "ile kosztuje kurier"}])
    assert tr.question == "przepisane pytanie" and tr.rewritten


# ------------------------------------------------------------------------------------------------ export
def test_export_txt_and_svg(kb, tmp_path):
    from faq_assistant.export import save_svg, trace_to_text
    from faq_assistant.pipeline import FAQPipeline

    p = FAQPipeline(kb=kb, classifier=FakeClassifier("dostawa"), llm=FakeLLM())
    tr = p.run("Ile kosztuje kurier?")
    txt = trace_to_text(tr)
    assert "ETAP 1" in txt and "ETAP 2" in txt and "Odpowiedź testowa" in txt
    svg = save_svg(tr, tmp_path / "t.svg").read_text(encoding="utf-8")
    assert svg.lstrip().startswith("<?xml") and "<svg" in svg


# ------------------------------------------------------------------------------------------------- utils
def test_parse_json_variants():
    assert parse_json('```json\n{"category": "dostawa"}\n```') == {"category": "dostawa"}
    assert parse_json('Oto wynik: ["a", "b"] koniec') == ["a", "b"]
    with pytest.raises(ValueError):
        parse_json("brak jsona")
