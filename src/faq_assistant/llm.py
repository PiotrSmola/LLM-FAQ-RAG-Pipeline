"""Klient lokalnego serwera LM Studio (API zgodne z OpenAI).

Obsługuje: czat (także strumieniowo), zapytania z obrazem (model z mmproj) oraz embeddingi.
Uruchomienie `python -m faq_assistant.llm` wykonuje test dymny wszystkich trzech funkcji.
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


class LMStudioClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, embedding_model: str | None = None) -> None:
        self.client = OpenAI(base_url=base_url or settings.base_url, api_key=api_key or settings.api_key)
        self.model = model or settings.llm_model
        self.embedding_model = embedding_model or settings.embedding_model

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _kwargs(params: GenerationParams | None) -> dict:
        p = asdict(params or GenerationParams())
        # top_k i repeat_penalty nie należą do specyfikacji OpenAI - LM Studio przyjmuje je w extra_body
        extra = {"top_k": p.pop("top_k"), "repeat_penalty": p.pop("repeat_penalty"),
                 "reasoning_effort": p.pop("reasoning_effort")}
        if p["seed"] is None:
            p.pop("seed")
        return {**p, "extra_body": extra}

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    # --------------------------------------------------------------------- chat
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

    # ------------------------------------------------------------------- vision
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

    # --------------------------------------------------------------- embeddings
    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(model=self.embedding_model, input=texts[i:i + batch_size])
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out


def _smoke_test() -> None:
    import sys
    import time

    llm = LMStudioClient()
    print("Modele w LM Studio:", llm.list_models())

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
