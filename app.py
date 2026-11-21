"""Aplikacja Gradio: Asystent FAQ sklepu Zorza Elektronika.

Zakładki:
  1. Asystent: czat (tekst + obraz), podgląd etapów pipeline'u, źródła, eksport .txt/.svg
  2. Twoje dokumenty: przesyłanie plików .txt/.md/.pdf i zadawanie do nich pytań (RAG na plikach sesji)
  3. Wsadowo: hurtowe przetwarzanie pytań z pliku CSV/TXT
  4. Parametry modelu: eksperymenty z temperature/top_p/top_k/seed/max_tokens/reasoning
  5. Ewaluacja: wyniki porównań (klasyfikacja, retrieval, QA, ablacja)

Uruchomienie:  python app.py   (wymaga działającego serwera LM Studio z modelami z pliku .env)
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd

from faq_assistant.config import CATEGORIES, RESULTS_DIR, GenerationParams, settings
from faq_assistant.export import conversation_to_text, save_svg, save_text, trace_to_figure, trace_to_text
from faq_assistant.ingest import chunk_document
from faq_assistant.llm import LMStudioClient
from faq_assistant.pipeline import FAQPipeline, PipelineTrace, strip_long_dashes
from faq_assistant.prompts import category_label
from faq_assistant.retrieval import KnowledgeBase

TMP = Path(tempfile.gettempdir()) / "faq_assistant_exports"
TMP.mkdir(exist_ok=True)

llm = LMStudioClient()
pipe = FAQPipeline(llm=llm)


def server_status() -> str:
    try:
        models = llm.list_models()
        ok_llm = settings.llm_model in models
        ok_emb = settings.embedding_model in models
        def state(ok: bool) -> str:
            return "" if ok else " (niezaładowany)"

        return (f"LM Studio: połączono z {settings.base_url} · LLM: {settings.llm_model}{state(ok_llm)} · "
                f"embeddingi: {settings.embedding_model}{state(ok_emb)} · "
                f"klasyfikator: {getattr(pipe.classifier, 'method', '-')} · baza wiedzy: {len(pipe.kb)} fragmentów")
    except Exception as e:
        return f"Brak połączenia z LM Studio ({settings.base_url}): {e}"


def params_from(temperature, top_p, top_k, max_tokens, seed, reasoning) -> GenerationParams:
    return GenerationParams(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
                            max_tokens=int(max_tokens), seed=int(seed) if seed is not None and seed >= 0 else None,
                            reasoning_effort=reasoning)


def stages_markdown(tr: PipelineTrace | None) -> str:
    if tr is None:
        return "_Zadaj pytanie, aby zobaczyć przebieg pipeline'u._"
    md = []
    if tr.image_name:
        md.append(f"**0. Obraz → pytanie** (`{tr.image_name}`)\n\n> {(tr.image_ocr or '')[:400]}")
    md.append(f"**Pytanie:** {tr.question}" + (" _(przepisane z kontekstu rozmowy)_" if tr.rewritten else ""))
    if tr.sub_questions:
        rows = "\n".join(
            f"{i}. {s['question']}  \n<small>"
            f"{category_label(s['category']) if s['category'] else 'bez klasyfikacji'} ({s['confidence']:.0%}), "
            f"{s.get('route', '')}</small>" for i, s in enumerate(tr.sub_questions, 1))
        md.append(f"**Pytanie złożone, podział na {len(tr.sub_questions)} podpytań:**\n\n{rows}")
        md.append(f"**1-2. Routing i wyszukiwanie:** {tr.route}")
    elif tr.category:
        md.append(f"**1. Klasyfikacja:** `{tr.classifier}` → **{category_label(tr.category)}** "
                  f"({tr.confidence:.0%})\n\nDecyzja: {tr.route}")
    else:
        md.append(f"**1. Klasyfikacja:** {tr.route}")
    if tr.hits and not tr.sub_questions:
        md.append(f"**2. Retrieval** (`{tr.retrieval_method}`): {len(tr.hits)} fragmentów, lista poniżej")
    md.append("**3. Odpowiedź:** " + {"rag": "wygenerowana przez LLM na podstawie źródeł",
                                      "empty": "pusta, model wyczerpał limit tokenów",
                                      "out_of_scope": "odmowa (pytanie poza zakresem, LLM nie był wywołany)"}
              .get(tr.answered_by, "w trakcie…"))
    if tr.timings:
        md.append("**Czasy:** " + " · ".join(f"{t.name}: {t.seconds:.2f}s" for t in tr.timings)
                  + f" · **razem {tr.total_seconds:.2f}s**")
    return strip_long_dashes("\n\n".join(md))


def sources_md(tr: PipelineTrace | None) -> str:
    """Lista źródeł jako zwykły tekst: numer jak w cytowaniu [n], źródło, wynik i początek fragmentu."""
    if tr is None or not tr.hits:
        return "_Brak źródeł._" if tr is not None else ""
    items = []
    for s in tr.sources():
        kind = "FAQ" if s["kind"] == "faq" else "dokument"
        body = s["text"].replace("\n", " ")
        body = body.split("] ", 1)[-1] if body.startswith("[") else body  # bez powtórzonego nagłówka sekcji
        items.append(f"**[{s['n']}] {s['source']}**  \n<small>{kind} · wynik {s['score']:.3f}</small>  \n"
                     f"{body[:260]}{'…' if len(body) > 260 else ''}")
    return strip_long_dashes("\n\n".join(items))


_CLASSIFIERS: dict = {}


def get_classifier(name: str):
    """Klasyfikator wybrany w UI (ładowany leniwie i zapamiętywany)."""
    from faq_assistant.classifier import LLMClassifier, TfidfClassifier
    from faq_assistant.config import MODELS_DIR

    if name == "wyłączony":
        return None
    if name not in _CLASSIFIERS:
        if name.startswith("HerBERT"):
            _CLASSIFIERS[name] = pipe.classifier
        elif name.startswith("LLM"):
            _CLASSIFIERS[name] = LLMClassifier(llm)
        else:
            _CLASSIFIERS[name] = TfidfClassifier.load(MODELS_DIR / "tfidf_lr.joblib")
    return _CLASSIFIERS[name]


def chat_submit(message, chat, history, traces, user_kb, use_user_docs, method, top_k, classifier_name,
                temperature, top_p, tk, max_tokens, seed, reasoning):
    text = (message or {}).get("text", "") or ""
    files = (message or {}).get("files", []) or []
    image = next((f for f in files if str(f).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))), None)
    if not text.strip() and not image:
        yield chat, history, traces, gr.update(), gr.update(), gr.update(), gr.update(value=None)
        return
    chat = chat + ([{"role": "user", "content": {"path": image}}] if image else [])
    chat = chat + [{"role": "user", "content": text or "(pytanie na obrazie)"},
                   {"role": "assistant", "content": "Analizuję pytanie…"}]
    yield chat, history, traces, gr.update(), gr.update(), gr.update(), gr.update(value=None)

    params = params_from(temperature, top_p, tk, max_tokens, seed, reasoning)
    clf = get_classifier(classifier_name)
    tr = pipe.prepare(text, history=history, image=image, user_kb=user_kb if use_user_docs else None,
                      use_classifier=clf is not None, classifier=clf, method=method, top_k=int(top_k), params=params)
    chat[-1]["content"] = "Generuję odpowiedź…"
    yield chat, history, traces, stages_markdown(tr), sources_md(tr), gr.update(), gr.update(value=None)

    answer = ""
    for token in pipe.generate_stream(tr, history):
        answer += token
        chat[-1]["content"] = answer
        yield chat, history, traces, gr.update(), gr.update(), gr.update(), gr.update(value=None)

    chat[-1]["content"] = tr.answer  # wersja po oczyszczeniu (bez długich myślników)
    history = history + [{"role": "user", "content": tr.question}, {"role": "assistant", "content": tr.answer}]
    traces = traces + [tr]
    fig = trace_to_figure(tr)
    yield chat, history, traces, stages_markdown(tr), sources_md(tr), fig, gr.update(value=None)


def export_last(traces, fmt: str):
    if not traces:
        gr.Warning("Brak odpowiedzi w tej sesji. Zadaj pytanie (po restarcie aplikacji odśwież stronę).")
        return gr.update(value=None, visible=False)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if fmt == "svg":
        path = save_svg(traces[-1], TMP / f"odpowiedz_{stamp}.svg")
    elif fmt == "txt":
        path = save_text(trace_to_text(traces[-1]), TMP / f"odpowiedz_{stamp}.txt")
    else:
        path = save_text(conversation_to_text(traces), TMP / f"rozmowa_{stamp}.txt")
    return gr.update(value=str(path), visible=True)


def clear_chat():
    return [], [], [], stages_markdown(None), sources_md(None), None


def upload_docs(files, user_kb, chunk_size, overlap):
    if user_kb is None:
        user_kb = KnowledgeBase(name=f"user_{uuid.uuid4().hex[:8]}", llm=llm, persist=False)
    added, rows = 0, []
    for f in files or []:
        path = Path(f)
        try:
            chunks = chunk_document(path, chunk_size=int(chunk_size), overlap=int(overlap), display_name=path.name)
            n = user_kb.add(chunks)
            added += n
            rows.append({"plik": path.name, "fragmenty": len(chunks), "status": "zindeksowano" if n else "już był"})
        except Exception as e:
            rows.append({"plik": path.name, "fragmenty": 0, "status": f"błąd: {e}"})
    info = f"Dodano {added} fragmentów. W bazie sesji: {len(user_kb)} fragmentów z {len({c.source for c in user_kb.chunks})} plików."
    return user_kb, pd.DataFrame(rows), info, gr.update(value=True)


def ask_docs(question, user_kb, method, top_k, temperature, reasoning):
    if user_kb is None or not len(user_kb):
        return "Najpierw prześlij przynajmniej jeden plik.", sources_md(None), None
    if not question.strip():
        return "Wpisz pytanie.", sources_md(None), None
    from faq_assistant.qa import answer as qa_answer

    t0 = time.perf_counter()
    hits = user_kb.search(question, k=int(top_k), method=method)
    params = GenerationParams(temperature=float(temperature), reasoning_effort=reasoning)
    tr = PipelineTrace(user_input=question, question=question, retrieval_method=method, hits=hits,
                       route="Pytanie do przesłanych plików, bez klasyfikacji", params=params.__dict__)
    tr.add_timing(f"Wyszukiwanie ({method})", t0)
    t0 = time.perf_counter()
    tr.answer, tr.answered_by = qa_answer(llm, question, hits, params=params), "rag"
    tr.add_timing("Generowanie odpowiedzi (LLM)", t0)
    txt = save_text(trace_to_text(tr), TMP / f"dokumenty_{time.strftime('%H%M%S')}.txt")
    return tr.answer, sources_md(tr), str(txt)


def batch_run(file, method, top_k, make_svg, progress=gr.Progress()):
    import shutil
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from batch import read_questions, run_batch

    if file is None:
        raise gr.Error("Wybierz plik z pytaniami (.csv, .txt lub .xlsx).")
    questions = read_questions(Path(file))
    out = TMP / f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    df = run_batch(pipe, questions, out, method, int(top_k), bool(make_svg),
                   progress=lambda i, n, q: progress(i / n, desc=f"[{i}/{n}] {q[:50]}"))
    files = [str(out / f) for f in ("wyniki.csv", "wyniki.xlsx", "raport.txt")]
    if make_svg:
        files.append(shutil.make_archive(str(out / "svg"), "zip", out / "svg"))
    return df, files


PARAM_HELP = """
| Parametr | Znaczenie | Typowe wartości |
|---|---|---|
| **temperature** | „Ostrość” rozkładu prawdopodobieństwa tokenów: logity dzielone przez T. Niska → deterministycznie, wysoka → kreatywnie i ryzykownie. | 0-0.3 fakty/RAG, 0.7-1.0 kreatywne |
| **top_p** (nucleus) | Losujemy tylko z najmniejszego zbioru tokenów o łącznym prawdopodobieństwie ≥ p. | 0.9-0.95 |
| **top_k** | Losujemy tylko spośród k najbardziej prawdopodobnych tokenów (LM Studio, `extra_body`). | 20-64 |
| **repeat_penalty** | Kara za powtarzanie tokenów już użytych (ogranicza degenerację i pętle). | 1.0-1.2 |
| **max_tokens** | Limit długości odpowiedzi (w tokenach); po jego osiągnięciu tekst jest ucinany (`finish_reason=length`). | 256-1024 |
| **seed** | Ziarno generatora losowego: ten sam seed + te same parametry = powtarzalny wynik. -1 = losowo. | dowolna liczba |
| **reasoning_effort** | Gemma 4 potrafi „myśleć” przed odpowiedzią (ukryty tok rozumowania). `none` = szybko, `low/medium/high` = dokładniej, ale wolniej. | `none` w czacie |
"""


def param_playground(prompt, n, temperature, top_p, top_k, repeat_penalty, max_tokens, seed, reasoning):
    rows = []
    for i in range(int(n)):
        s = None if seed < 0 else int(seed)
        p = GenerationParams(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
                             repeat_penalty=float(repeat_penalty), max_tokens=int(max_tokens), seed=s,
                             reasoning_effort=reasoning)
        t0 = time.perf_counter()
        r = llm.client.chat.completions.create(model=llm.model, messages=[{"role": "user", "content": prompt}],
                                               **llm._kwargs(p))
        out = r.choices[0].message.content or ""
        rows.append({"wariant": i + 1, "odpowiedź": out, "tokeny": r.usage.completion_tokens,
                     "finish_reason": r.choices[0].finish_reason, "czas_s": round(time.perf_counter() - t0, 2)})
    df = pd.DataFrame(rows)
    uniq = df["odpowiedź"].nunique()
    return df, f"Unikalnych odpowiedzi: **{uniq} / {len(df)}**" + (", wynik deterministyczny." if uniq == 1 else "")


def load_eval():
    def csv(p):
        p = RESULTS_DIR / p
        return pd.read_csv(p) if p.exists() else pd.DataFrame({"info": [f"Brak pliku {p.name}, uruchom ewaluację"]})

    imgs = [str(p) for p in sorted(RESULTS_DIR.rglob("*.png")) if "examples" not in p.parts]
    return (csv("classifier/comparison.csv"), csv("retrieval/retrieval_metrics.csv"),
            csv("qa/qa_summary.csv"), imgs)


HELP_MD = """
Asystent obsługuje klientów **fikcyjnego sklepu internetowego Zorza Elektronika**. Odpowiada na podstawie bazy FAQ
(74 wpisy) i dokumentów firmowych: regulaminu, polityki zwrotów i reklamacji, cennika dostaw oraz warunków usługi
Zorza Ochrona+. Każda odpowiedź ma numery źródeł [n], a lista źródeł jest w prawym panelu.

**O co pytać:** płatności (BLIK, raty, faktury, „Zapłać później”), dostawa (ceny, terminy, paczkomaty, wniesienie),
zwroty i reklamacje (terminy, procedura, wyjątki), konto (hasło, punkty lojalnościowe, usuwanie danych),
gwarancja i Zorza Ochrona+ (ceny, udział własny, wyłączenia). Można dopytywać („a w sobotę?”), asystent
uzupełni pytanie z kontekstu rozmowy. Pytania niezwiązane ze sklepem (np. pogoda, przepisy) dostają odmowę.

**Obrazy** (spinacz w polu pytania): zrzut e-maila od klienta, zdjęcie paragonu lub faktury, zdjęcie notatki z pytaniem.
Model odczytuje tekst z obrazu, formułuje pytanie i wykorzystuje dane z dokumentu (np. datę zakupu do policzenia
terminu zwrotu). Do obrazu można dopisać własne pytanie. Dokument z innego sklepu zostanie odczytany, ale
odpowiedź będzie dotyczyć zasad Zorza Elektronika.

**Eksport:** przyciski pod listą źródeł zapisują ostatnią odpowiedź (.txt), wizualizację jej etapów (.svg)
lub całą rozmowę (.txt). Po restarcie aplikacji odśwież stronę, bo historia sesji jest wtedy czyszczona.
"""

# Font systemowy zamiast fontu Google z domyślnego motywu Gradio.
SYSTEM_FONT = [gr.themes.Font("Arial"), gr.themes.Font("Helvetica"), gr.themes.Font("sans-serif")]
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=SYSTEM_FONT,
    font_mono=[gr.themes.Font("Consolas"), gr.themes.Font("Courier New"), gr.themes.Font("monospace")],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#f6f7f9",
    body_background_fill_dark="#0f141b",
    block_background_fill="#ffffff",
    block_background_fill_dark="#171d26",
    block_border_width="1px",
    block_shadow="none",
    button_primary_background_fill="#2b4c7e",
    button_primary_background_fill_hover="#223d66",
    button_primary_background_fill_dark="#3a6db5",
    button_primary_background_fill_hover_dark="#4a7dc5",
    button_primary_text_color="#ffffff",
)
CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto; }
#app-header { padding: 18px 22px 12px; margin-bottom: 4px;
              background: #ffffff; border-radius: 4px; }
#app-header h1 { font-size: 22px; font-weight: 700; margin: 0 0 4px; color: #1d2b3a; }
#app-header p { margin: 0; color: #556270; font-size: 14px; }
#status { font-size: 12.5px; color: #667085; padding: 2px 4px 8px; }
#status p { margin: 0; }
.section-title h3 { font-size: 15px; margin: 6px 0 2px; color: #1d2b3a; }
#sources, #doc-sources { max-height: 330px; overflow-y: auto; font-size: 13px; line-height: 1.45;
                         padding: 8px 12px; background: #ffffff; border: 1px solid #e3e7ec; border-radius: 4px; }
#sources p, #doc-sources p { margin: 0 0 10px; }
#sources small, #doc-sources small { color: #6b7685; }
/* tryb ciemny (gdy przeglądarka/system używa ciemnego motywu) */
.dark #app-header { background: #171d26; }
.dark #app-header h1 { color: #e8edf3; }
.dark #app-header p, .dark #status { color: #9aa6b5; }
.dark .section-title h3 { color: #e8edf3; }
.dark #sources, .dark #doc-sources { background: #171d26; border-color: #2a3340; }
.dark #sources small, .dark #doc-sources small { color: #8a96a6; }
"""

with gr.Blocks(title="Asystent FAQ Zorza Elektronika") as demo:
    gr.Markdown("# Asystent FAQ: Zorza Elektronika\n"
                "Klasyfikacja zapytania (HerBERT) → wyszukiwanie w bazie wiedzy (BM25 + bge-m3) → "
                "odpowiedź RAG ze źródłami (Gemma 4, LM Studio)", elem_id="app-header")
    status = gr.Markdown(server_status(), elem_id="status")

    history_st = gr.State([])
    traces_st = gr.State([])
    user_kb_st = gr.State(None)

    with gr.Tab("Asystent"):
        with gr.Accordion("Jak korzystać", open=False):
            gr.Markdown(HELP_MD)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Rozmowa", height=520, layout="bubble")
                msg = gr.MultimodalTextbox(placeholder="Zadaj pytanie lub dołącz zrzut ekranu / zdjęcie…",
                                           file_types=["image"], file_count="single", sources=["upload"],
                                           show_label=False)
                gr.Examples([
                    {"text": "Ile kosztuje dostawa kurierem i od jakiej kwoty jest darmowa?"},
                    {"text": "Czy Ochrona+ pokryje zbity ekran i ile wtedy dopłacę?"},
                    {"text": "Laptop nie włącza się zaraz po rozpakowaniu, co robić?"},
                    {"text": "", "files": ["data/examples/paragon_pytanie.png"]},
                    {"text": "Czy zdążę jeszcze zwrócić ten laptop?", "files": ["data/examples/faktura_laptop.png"]},
                    {"text": "Napisz mi wiersz o kotach"},
                ], inputs=msg, label="Przykłady")
                with gr.Row():
                    use_user_docs = gr.Checkbox(label="Uwzględnij moje przesłane dokumenty", value=False)
                    clear_btn = gr.Button("Nowa rozmowa", size="sm")
            with gr.Column(scale=2):
                with gr.Accordion("Ustawienia pipeline'u", open=False):
                    method = gr.Radio(["hybrid", "dense", "bm25", "tfidf"], value="hybrid", label="Metoda wyszukiwania")
                    top_k = gr.Slider(1, 8, value=settings.top_k, step=1, label="Liczba fragmentów (top-k)")
                    use_cls = gr.Radio(["HerBERT (fine-tuning)", "LLM zero-shot", "TF-IDF + LR", "wyłączony"],
                                       value="HerBERT (fine-tuning)", label="Klasyfikator (etap 1, routing)")
                    temperature = gr.Slider(0, 1.5, value=0.2, step=0.05, label="temperature")
                    top_p = gr.Slider(0.1, 1, value=0.9, step=0.05, label="top_p")
                    tk = gr.Slider(1, 100, value=40, step=1, label="top_k (sampling)")
                    max_tokens = gr.Slider(128, 8192, value=2048, step=64, label="max_tokens")
                    seed = gr.Number(value=42, label="seed (-1 = losowy)", precision=0)
                    reasoning = gr.Dropdown(["none", "low", "medium", "high"], value="none", label="reasoning_effort")
                gr.Markdown("### Etapy pipeline'u", elem_classes="section-title")
                stages = gr.Markdown(stages_markdown(None))
                gr.Markdown("### Źródła", elem_classes="section-title")
                sources = gr.Markdown(sources_md(None), elem_id="sources")
                with gr.Row():
                    b_txt = gr.Button("Odpowiedź .txt", size="sm")
                    b_svg = gr.Button("Etapy .svg", size="sm")
                    b_conv = gr.Button("Cała rozmowa .txt", size="sm")
                export_file = gr.File(label="Plik do pobrania", interactive=False, visible=False)
        plot = gr.Plot(label="Wizualizacja etapów (ostatnie pytanie)")

        inputs = [msg, chat, history_st, traces_st, user_kb_st, use_user_docs, method, top_k, use_cls,
                  temperature, top_p, tk, max_tokens, seed, reasoning]
        msg.submit(chat_submit, inputs, [chat, history_st, traces_st, stages, sources, plot, msg])
        b_txt.click(lambda t: export_last(t, "txt"), traces_st, export_file, api_name="export_txt")
        b_svg.click(lambda t: export_last(t, "svg"), traces_st, export_file, api_name="export_svg")
        b_conv.click(lambda t: export_last(t, "conv"), traces_st, export_file, api_name="export_conv")
        clear_btn.click(clear_chat, None, [chat, history_st, traces_st, stages, sources, plot])

    with gr.Tab("Twoje dokumenty"):
        gr.Markdown("Prześlij pliki **.txt / .md / .pdf**. Zostaną podzielone na fragmenty, zindeksowane "
                    "(embeddingi bge-m3 + BM25) w bazie tej sesji. Pytaj tutaj albo w zakładce *Asystent* "
                    "(zaznacz „Uwzględnij moje przesłane dokumenty”).")
        with gr.Row():
            with gr.Column():
                up = gr.File(file_count="multiple", file_types=[".txt", ".md", ".pdf"], label="Pliki")
                with gr.Row():
                    csize = gr.Slider(300, 2000, value=1000, step=100, label="Rozmiar fragmentu (znaki)")
                    cover = gr.Slider(0, 400, value=150, step=50, label="Nakładka (znaki)")
                up_btn = gr.Button("Indeksuj pliki", variant="primary")
                up_info = gr.Markdown()
                up_table = gr.Dataframe(label="Zindeksowane pliki")
            with gr.Column():
                dq = gr.Textbox(label="Pytanie do przesłanych plików", lines=2)
                with gr.Row():
                    d_method = gr.Radio(["hybrid", "dense", "bm25"], value="hybrid", label="Wyszukiwanie")
                    d_k = gr.Slider(1, 8, value=4, step=1, label="top-k")
                    d_temp = gr.Slider(0, 1, value=0.2, step=0.05, label="temperature")
                    d_reason = gr.Dropdown(["none", "low", "medium", "high"], value="none", label="reasoning")
                d_btn = gr.Button("Zapytaj", variant="primary")
                d_ans = gr.Markdown()
                d_src = gr.Markdown(elem_id="doc-sources")
                d_file = gr.File(label="Raport .txt", interactive=False)
        up_btn.click(upload_docs, [up, user_kb_st, csize, cover], [user_kb_st, up_table, up_info, use_user_docs])
        d_btn.click(ask_docs, [dq, user_kb_st, d_method, d_k, d_temp, d_reason], [d_ans, d_src, d_file])

    with gr.Tab("Wsadowo"):
        gr.Markdown("Automatyczne przetwarzanie listy pytań: plik **.csv** (kolumna `pytanie`), **.txt** "
                    "(pytanie w linii) lub **.xlsx**. Wynik: CSV, XLSX, raport TXT (+ SVG dla każdego pytania).")
        with gr.Row():
            bfile = gr.File(file_types=[".csv", ".txt", ".xlsx", ".jsonl"], label="Plik z pytaniami",
                            value="data/examples/pytania.csv")
            with gr.Column():
                b_method = gr.Radio(["hybrid", "dense", "bm25", "tfidf"], value="hybrid", label="Wyszukiwanie")
                b_k = gr.Slider(1, 8, value=4, step=1, label="top-k")
                b_svg_cb = gr.Checkbox(value=False, label="Generuj SVG dla każdego pytania")
                b_run = gr.Button("Uruchom", variant="primary")
        b_df = gr.Dataframe(label="Wyniki", wrap=True)
        b_files = gr.File(label="Pliki wynikowe", file_count="multiple", interactive=False)
        b_run.click(batch_run, [bfile, b_method, b_k, b_svg_cb], [b_df, b_files])

    with gr.Tab("Parametry modelu"):
        gr.Markdown("Eksperymenty z parametrami generowania (wywołania API LM Studio). "
                    "Ustaw np. temperature=1.2 i seed=-1, a potem temperature=0 i porównaj różnorodność.")
        gr.Markdown(PARAM_HELP)
        with gr.Row():
            with gr.Column(scale=2):
                pp = gr.Textbox(value="Wymyśl krótkie hasło reklamowe dla sklepu z elektroniką Zorza.", lines=3,
                                label="Prompt")
                pn = gr.Slider(1, 6, value=3, step=1, label="Liczba wariantów")
            with gr.Column(scale=3):
                with gr.Row():
                    p_t = gr.Slider(0, 2, value=1.0, step=0.05, label="temperature")
                    p_p = gr.Slider(0.05, 1, value=0.95, step=0.05, label="top_p")
                    p_k = gr.Slider(1, 100, value=40, step=1, label="top_k")
                with gr.Row():
                    p_r = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="repeat_penalty")
                    p_m = gr.Slider(16, 4096, value=256, step=16, label="max_tokens")
                    p_s = gr.Number(value=-1, precision=0, label="seed (-1 = losowy)")
                    p_re = gr.Dropdown(["none", "low", "medium", "high"], value="none", label="reasoning_effort")
        p_btn = gr.Button("Generuj", variant="primary")
        p_info = gr.Markdown()
        p_df = gr.Dataframe(wrap=True, label="Wyniki")
        p_btn.click(param_playground, [pp, pn, p_t, p_p, p_k, p_r, p_m, p_s, p_re], [p_df, p_info])

    with gr.Tab("Ewaluacja"):
        gr.Markdown("Wyniki z `scripts/train_classifier.py compare` i `scripts/evaluate.py` (katalog `results/`).")
        e_btn = gr.Button("Odśwież")
        e_cls = gr.Dataframe(label="Klasyfikacja: porównanie modeli")
        e_ret = gr.Dataframe(label="Retrieval: Recall@k / MRR")
        e_qa = gr.Dataframe(label="QA: jakość odpowiedzi i ablacja")
        e_gal = gr.Gallery(label="Wykresy", columns=3, height=420)
        e_btn.click(load_eval, None, [e_cls, e_ret, e_qa, e_gal])
        demo.load(load_eval, None, [e_cls, e_ret, e_qa, e_gal])

    demo.load(server_status, None, status)


if __name__ == "__main__":
    demo.queue().launch(theme=THEME, css=CSS, inbrowser=False, footer_links=[], allowed_paths=[str(TMP), "data/examples"])
