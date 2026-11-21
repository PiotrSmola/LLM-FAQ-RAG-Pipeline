"""Etap 3 pipeline'u: odpowiedź na pytanie na podstawie znalezionych fragmentów.

- generatywnie: LLM (Gemma w LM Studio) z numerowanym kontekstem i cytowaniem [n] – tryb główny (RAG),
- ekstrakcyjnie: model QA typu SQuAD wskazuje fragment tekstu (span) – porównanie, jak w Lab 3,
- FLAN-T5 – generatywny model z Lab 3 (angielski), tylko do porównania w ewaluacji.
"""

from __future__ import annotations

from collections.abc import Iterator

from .config import GenerationParams
from .prompts import QA_IMAGE_BLOCK, QA_SYSTEM, QA_USER
from .retrieval import Hit


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (źródło: {h.chunk.citation})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None,
                   image_text: str | None = None) -> list[dict]:
    from datetime import date

    # data jest potrzebna, by liczyć terminy (zwrot 30 dni, reklamacja DOA 7 dni) na podstawie dat z dokumentów
    system = f"{QA_SYSTEM}\n\nDzisiejsza data: {date.today():%d.%m.%Y}."
    messages: list[dict] = [{"role": "system", "content": system}]
    for turn in (history or [])[-6:]:  # krótka pamięć rozmowy (ostatnie 3 wymiany)
        messages.append({"role": turn["role"], "content": turn["content"]})
    context = build_context(hits) if hits else "(brak pasujących fragmentów w bazie wiedzy)"
    user = QA_USER.format(context=context, question=question)
    if image_text:
        user = QA_IMAGE_BLOCK.format(ocr=image_text[:3000]) + user
    messages.append({"role": "user", "content": user})
    return messages


def answer_stream(llm, question: str, hits: list[Hit], history: list[dict] | None = None,
                  params: GenerationParams | None = None, image_text: str | None = None) -> Iterator[str]:
    yield from llm.chat_stream(build_messages(question, hits, history, image_text), params)


def answer(llm, question: str, hits: list[Hit], history: list[dict] | None = None,
           params: GenerationParams | None = None) -> str:
    return llm.chat(build_messages(question, hits, history), params)


def answer_without_context(llm, question: str, params: GenerationParams | None = None) -> str:
    """Wariant ablacyjny „sam LLM” – bez RAG, model odpowiada z własnej wiedzy (ryzyko halucynacji)."""
    system = ("Jesteś konsultantem obsługi klienta sklepu internetowego Zorza Elektronika. "
              "Odpowiadasz po polsku, konkretnie i zwięźle.")
    return llm.complete(question, system=system, params=params)


class ExtractiveQA:
    """Extractive QA (span) – kod jak w Lab 3, ale model wielojęzyczny (XLM-RoBERTa fine-tuned na SQuAD 2.0)."""

    def __init__(self, model_name: str = "deepset/xlm-roberta-base-squad2", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        self.name = model_name

    def _best_span(self, question: str, context: str, max_len: int = 384, max_answer: int = 40):
        enc = self.tokenizer(question, context, return_tensors="pt", truncation="only_second",
                             max_length=max_len).to(self.device)
        with self.torch.no_grad():
            out = self.model(**enc)
        start_l, end_l = out.start_logits[0], out.end_logits[0]
        seq_ids = enc.sequence_ids(0)
        ctx = self.torch.tensor([s == 1 for s in seq_ids], device=self.device)
        start_l = start_l.masked_fill(~ctx, -1e4)
        end_l = end_l.masked_fill(~ctx, -1e4)
        # najlepsza para (start <= end, długość <= max_answer)
        scores = start_l[:, None] + end_l[None, :]
        n = scores.shape[0]
        valid = self.torch.triu(self.torch.ones(n, n, dtype=self.torch.bool, device=self.device))
        valid &= ~self.torch.triu(self.torch.ones(n, n, dtype=self.torch.bool, device=self.device), max_answer)
        scores = scores.masked_fill(~valid, -1e9)
        idx = int(scores.argmax())
        s, e = divmod(idx, n)
        prob = float(self.torch.softmax(start_l, -1)[s] * self.torch.softmax(end_l, -1)[e])
        text = self.tokenizer.decode(enc["input_ids"][0][s:e + 1], skip_special_tokens=True).strip()
        return text, prob

    def answer(self, question: str, hits: list[Hit]) -> tuple[str, float, int]:
        """Zwraca (odpowiedź, pewność, numer źródła) – najlepszy span spośród znalezionych fragmentów."""
        best = ("", 0.0, 0)
        for i, h in enumerate(hits, 1):
            text, prob = self._best_span(question, h.chunk.text)
            if text and prob > best[1]:
                best = (text, prob, i)
        return best


class FlanT5QA:
    """Generatywny FLAN-T5 z Lab 3 – punkt odniesienia (model angielski, słaby dla polskiego)."""

    def __init__(self, model_name: str = "google/flan-t5-base", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def answer(self, question: str, hits: list[Hit]) -> str:
        context = " ".join(h.chunk.text for h in hits)
        prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
