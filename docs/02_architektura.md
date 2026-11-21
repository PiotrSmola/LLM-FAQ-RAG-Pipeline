# Architektura i decyzje projektowe

## Przepływ danych

```
app.py (Gradio) / scripts/batch.py
        │  FAQPipeline.prepare(...)  →  PipelineTrace
        ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ 0. vision.image_to_question      obraz → {ocr, question}      (Gemma + mmproj)       │
│ 1. FAQPipeline.rewrite           pytanie uzupełniające → samodzielne   (gdy historia) │
│ 2. classifier.predict            kategoria + pewność → decyzja routingu              │
│      poza_zakresem ≥ próg  → OUT_OF_SCOPE_ANSWER (koniec, bez LLM)                   │
│      kategoria ≥ próg      → filtr {kind=doc} ∪ {category=kat}                       │
│      < próg                → bez filtra                                              │
│ 3. KnowledgeBase.search          hybrid = RRF(BM25, dense)  → top-k Hit               │
│      + KnowledgeBase sesji (pliki użytkownika)                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │  FAQPipeline.generate_stream(trace)
        ▼
│ 4. qa.answer_stream              system prompt + data, [obraz od klienta], ŹRÓDŁA [n], │
│                                  historia (3 wymiany) → tokeny → trace.answer          │
        ▼
export.trace_to_text / save_svg   ·   UI: etapy, źródła, wykres
```

## Moduły

| Moduł | Odpowiedzialność | Najważniejsze elementy |
|---|---|---|
| `config.py` | ustawienia z `.env`, kategorie, parametry generowania | `Settings`, `GenerationParams`, `CATEGORIES` |
| `llm.py` | jedyny punkt kontaktu z LM Studio | `LMStudioClient.chat / chat_stream / vision / embed`; `top_k`, `repeat_penalty`, `reasoning_effort` w `extra_body` |
| `prompts.py` | wszystkie prompty w jednym miejscu | klasyfikacja few-shot (JSON), generowanie danych, przepisanie pytania, OCR obrazu, odpowiedź RAG, LLM-as-judge |
| `classifier.py` | etap 1 – wspólny interfejs `predict(text) → Prediction` | `BertClassifier`, `TfidfClassifier`, `LLMClassifier` |
| `ingest.py` | podział bazy wiedzy na fragmenty | FAQ = 1 wpis → 1 fragment; dokumenty: sekcje markdown → okna ~1000 znaków z nakładką 150 |
| `retrieval.py` | etap 2 – indeks i 4 metody wyszukiwania | `KnowledgeBase` (ChromaDB + BM25 + TF-IDF), `tokenize_pl` (bez diakrytyków, stop-słowa, stemming prefiksowy), RRF |
| `qa.py` | etap 3 | `build_messages` (numerowane źródła), `ExtractiveQA` (XLM-R), `FlanT5QA` |
| `vision.py` | wielomodalność | obraz → JSON `{ocr, question}` |
| `pipeline.py` | orkiestracja i zapis przebiegu | `FAQPipeline.prepare / generate_stream / run`, `PipelineTrace`, automatyczny setup |
| `export.py` | pliki wynikowe | raport `.txt`, wizualizacja `.svg` (schemat etapów + wykresy) |

## Kluczowe decyzje

| Decyzja | Uzasadnienie |
|---|---|
| **LM Studio** zamiast HF Transformers dla LLM | gotowe narzędzie (próg 3.0), kwantyzacja GGUF – model 12B mieści się w 16 GB VRAM, API zgodne z OpenAI, obsługa obrazów i embeddingów w jednym serwerze |
| **Gemma 4 12B QAT** | dobra jakość po polsku, wbudowana obsługa obrazu (mmproj), QAT = mała strata jakości przy Q4 |
| **bge-m3** do embeddingów | jeden z najlepszych otwartych modeli wielojęzycznych; ten sam serwer co LLM |
| **HerBERT** jako router | polski BERT – wymaganie tematu (fine-tuning BERT); 10 ms/zapytanie, działa bez serwera LLM |
| **Dane treningowe z LLM** | brak realnych zgłoszeń klientów; LLM generuje różnorodne parafrazy (literówki, styl potoczny); test pisany ręcznie, aby nie oceniać modelu na danych z tego samego źródła |
| **Filtr kategorii tylko dla FAQ** | dokumenty firmowe (regulamin itp.) dotyczą wielu kategorii naraz; filtr ograniczałby trafność |
| **Hybryda RRF jako domyślna metoda** | najlepszy Recall@3/@4 – w RAG ważne, by właściwy fragment trafił do kontekstu; RRF nie wymaga normalizacji wyników różnych metod |
| **Stemming prefiksowy w BM25** | prosty sposób na fleksję polską bez zewnętrznych narzędzi (np. „zwrotu/zwrotem/zwroty” → „zwrot”) |
| **Odmowa bez LLM dla „poza zakresem”** | zero ryzyka halucynacji, odpowiedź w ~0,4 s, brak obciążenia GPU |
| **Cytowania [n] + „(wiedza ogólna)”** | spełnia wymaganie 5.0 (wiedza modelu + dokumenty) przy zachowaniu przejrzystości, skąd pochodzi informacja |
| **`reasoning_effort="none"`** | Gemma 4 domyślnie „myśli”; dla FAQ rozumowanie nie poprawia odpowiedzi, a wydłuża czas ~10× |
| **Własny pipeline zamiast LangChain/Haystack** | temat traktuje je jako opcjonalne; ~300 linii kodu, pełna kontrola i przejrzystość etapów do wizualizacji |
| **Własna implementacja BERTScore** | pakiet `bert_score` 0.3.13 jest niezgodny z transformers 5 |

## Format danych

* `data/faq/faq.jsonl` – `{"id": "DOS-01", "category": "dostawa", "question": "...", "answer": "..."}`
* `data/classifier/*.jsonl` – `{"text": "...", "label": "dostawa", "source": "llm|faq"}`
* `data/eval/qa_eval.jsonl` – `{"id", "question", "category", "gold_sources": ["DOS-01", "cennik_dostaw.md"], "reference"}`
  (trafienie w retrievalu = fragment, którego `source` jest na liście `gold_sources`)
