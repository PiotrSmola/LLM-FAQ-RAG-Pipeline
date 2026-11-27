"""Klient serwerów modeli (API zgodne z OpenAI): czat, obraz, embeddingi.

Test: python -m faq_assistant.llm [obraz.png]
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI

from .config import GenerationParams, settings

Message = dict[str, object]
REASONING_BUDGET = {"none": 0, "low": 2048, "medium": 4096, "high": 8192}


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, embedding_model: str | None = None,
                 embedding_base_url: str | None = None) -> None:
        key = api_key or settings.api_key
        self.base_url = base_url or settings.base_url
        self.embedding_base_url = embedding_base_url or (base_url if base_url else settings.embedding_base_url)
        self.client = OpenAI(base_url=self.base_url, api_key=key)
        self.emb_client = (self.client if self.embedding_base_url == self.base_url
                           else OpenAI(base_url=self.embedding_base_url, api_key=key))
        self.model = model or settings.llm_model
        self.embedding_model = embedding_model or settings.embedding_model

    @staticmethod
    def _kwargs(params: GenerationParams | None) -> dict:
        p = asdict(params or GenerationParams())
        # parametry spoza API OpenAI idą w extra_body
        extra = {"top_k": p.pop("top_k"), "repeat_penalty": p.pop("repeat_penalty"),
                 "reasoning_effort": p.pop("reasoning_effort")}
        if p["seed"] is None:
            p.pop("seed")
        # myślenie wlicza się do max_tokens, więc doliczamy budżet (inaczej odpowiedź bywa pusta)
        budget = REASONING_BUDGET.get(extra["reasoning_effort"], 0)
        if budget:
            extra["thinking_budget_tokens"] = budget
        p["max_tokens"] += budget
        return {**p, "extra_body": extra}

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    def list_embedding_models(self) -> list[str]:
        return [m.id for m in self.emb_client.models.list().data]

    def chat(self, messages: list[Message], params: GenerationParams | None = None) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **self._kwargs(params))
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: list[Message], params: GenerationParams | None = None) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **self._kwargs(params)
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def complete(self, prompt: str, system: str | None = None, params: GenerationParams | None = None) -> str:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, params)

    @staticmethod
    def image_part(image: str | Path | bytes, mime: str = "image/png") -> dict:
        """Zamienia obraz (ścieżka lub bajty) na fragment wiadomości w formacie data-URL."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            mime = mimetypes.guess_type(path.name)[0] or mime
            data = path.read_bytes()
        else:
            data = image
        b64 = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    def vision(self, prompt: str, image: str | Path | bytes, system: str | None = None,
               params: GenerationParams | None = None) -> str:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}, self.image_part(image)]})
        return self.chat(messages, params)

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            resp = self.emb_client.embeddings.create(model=self.embedding_model, input=texts[i:i + batch_size])
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out



def _smoke_test() -> None:
    import sys
    import time

    llm = LLMClient()
    print(f"Serwer LLM ({llm.base_url}):", llm.list_models())
    print(f"Serwer embeddingów ({llm.embedding_base_url}):", llm.list_embedding_models())

    t = time.perf_counter()
    answer = llm.complete("Odpowiedz jednym zdaniem: czym jest RAG?")
    print(f"\n[chat] ({time.perf_counter() - t:.1f}s)\n{answer}")

    t = time.perf_counter()
    vecs = llm.embed(["Jak zwrócić towar?", "Zwrot produktu w ciągu 14 dni", "Pogoda w Rzeszowie"])
    import numpy as np

    v = np.array(vecs)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    print(f"\n[embeddings] ({time.perf_counter() - t:.1f}s) wymiar={v.shape[1]}, "
          f"cos(zwrot,zwrot)={v[0] @ v[1]:.3f}, cos(zwrot,pogoda)={v[0] @ v[2]:.3f}")

    if len(sys.argv) > 1:
        t = time.perf_counter()
        desc = llm.vision("Opisz krótko, co jest na obrazie.", sys.argv[1])
        print(f"\n[vision] ({time.perf_counter() - t:.1f}s)\n{desc}")


if __name__ == "__main__":
    _smoke_test()
