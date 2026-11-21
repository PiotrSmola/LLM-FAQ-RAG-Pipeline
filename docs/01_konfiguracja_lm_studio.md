# Konfiguracja LM Studio i modeli – krok po kroku

Projekt korzysta z gotowego narzędzia **LM Studio** (silnik llama.cpp), które uruchamia lokalne modele
w formacie GGUF na karcie graficznej i udostępnia je przez **API zgodne z OpenAI**
(`http://localhost:1234/v1`). Aplikacja w Pythonie łączy się z tym API biblioteką `openai`.

Sprzęt testowy: NVIDIA RTX 5070 Ti (16 GB VRAM), Windows 11, LM Studio 0.4.28.

## 1. Instalacja i pobranie modeli

1. Pobierz i zainstaluj LM Studio ze strony producenta (lmstudio.ai), uruchom aplikację.
2. W zakładce **Discover** (lupa) wyszukaj i pobierz:

| Model | Plik | Rozmiar | Rola w projekcie |
|---|---|---|---|
| **Gemma 4 12B Instruct QAT** (`lmstudio-community/gemma-4-12B-it-QAT-GGUF`) | `gemma-4-12B-it-QAT-Q4_0.gguf` + `mmproj-…-BF16.gguf` | 7,0 GB + 0,18 GB | LLM: odpowiedzi RAG, klasyfikacja zero-shot, przepisywanie pytań, odczyt obrazów, LLM-as-judge |
| **BGE-M3** (`gpustack/bge-m3-GGUF`) | `bge-m3-Q8_0.gguf` | 0,6 GB | model embeddingów (wektory 1024-wymiarowe, wielojęzyczny) |

   *QAT (Quantization-Aware Training)* – model był douczany ze świadomością kwantyzacji 4-bitowej, więc
   wersja Q4_0 traci znacznie mniej jakości niż zwykła kwantyzacja, a zajmuje ~4× mniej pamięci niż FP16.
   Plik **mmproj** to projektor wizyjny – bez niego model nie przyjmuje obrazów (funkcja wielomodalna).

3. Alternatywnie z wiersza poleceń (CLI `lms` instaluje się razem z LM Studio):
   ```bash
   lms get lmstudio-community/gemma-4-12B-it-QAT-GGUF
   lms get gpustack/bge-m3-GGUF
   lms ls                       # lista pobranych modeli i ich identyfikatorów API
   ```

## 2. Załadowanie modeli i parametry ładowania

W GUI: zakładka **Developer** → *Select a model to load* → ustaw parametry → *Load*.
To samo w CLI (tak robi się to w projekcie):

```bash
lms server start                                                  # serwer API na porcie 1234
lms load google/gemma-4-12b-qat --gpu max --context-length 32768 --parallel 1 -y
lms load text-embedding-bge-m3 --gpu max -y
lms ps                                                            # co jest załadowane
```

| Parametr ładowania | Wartość | Znaczenie |
|---|---|---|
| **GPU offload** (`--gpu max`) | wszystkie warstwy | ile warstw modelu trafia do VRAM; „max” = całość na GPU (najszybciej). Przy braku pamięci część warstw liczy CPU – generowanie zwalnia kilkukrotnie |
| **Context length** (`--context-length`) | 32768 | maksymalna liczba tokenów w oknie (prompt + historia + źródła RAG + odpowiedź + ewentualne rozumowanie). Większy kontekst = więcej VRAM na KV-cache. 32k mieści do 16 fragmentów dla pytań złożonych, historię rozmowy i długą odpowiedź |
| **Flash Attention** | włączone (domyślnie) | wydajniejsza implementacja mechanizmu uwagi, mniejsze zużycie pamięci przy długim kontekście |
| **KV cache quantization** | F16 (domyślnie) | kwantyzacja pamięci kluczy/wartości uwagi; Q8 zmniejsza zużycie VRAM kosztem minimalnej jakości |
| **Parallel** (`--parallel`) | 1 | liczba równoległych zapytań obsługiwanych przez serwer. Każde miejsce rezerwuje własne bufory, więc wartość 4 (domyślna) podwajała zużycie RAM przez `llama-server` (14,4 GB wobec 7,5 GB przy 1). Aplikacja i tak wysyła zapytania po kolei |
| **TTL** | brak | czas bezczynności, po którym model jest automatycznie wyładowywany |
| **Keep model in memory / Try mmap()** (GUI: ustawienia ładowania modelu) | wyłączone, jeśli brakuje RAM | przy pełnym offloadzie na GPU LM Studio i tak trzyma kopię wag w RAM (ok. 7 GB dla Gemmy 12B). Wyłączenie obu opcji zwalnia tę pamięć kosztem nieco wolniejszego ponownego ładowania modelu |

Zużycie VRAM w projekcie: Gemma 4 12B ≈ 6,7 GB + KV-cache, bge-m3 ≈ 0,6 GB, klasyfikator HerBERT ≈ 0,5 GB
(w trakcie fine-tuningu ~3 GB) – całość mieści się w 16 GB.

## 3. Serwer API i test połączenia

```bash
curl http://localhost:1234/v1/models
curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"google/gemma-4-12b-qat\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Czym jest RAG? Jedno zdanie.\"}],
  \"temperature\": 0.2, \"max_tokens\": 100, \"reasoning_effort\": \"none\" }"
curl http://localhost:1234/v1/embeddings -H "Content-Type: application/json" -d "{
  \"model\": \"text-embedding-bge-m3\", \"input\": \"Jak zwrócić towar?\" }"
```

W projekcie test wszystkich trzech funkcji (czat, embeddingi, obraz) wykonuje:
```bash
python -m faq_assistant.llm data/examples/mail_reklamacja.png
```

Identyfikatory modeli i adres serwera ustawia się w pliku `.env` (szablon: `.env.example`).

## 4. Parametry generowania (wysyłane w każdym zapytaniu)

| Parametr | Znaczenie | Wartość w projekcie |
|---|---|---|
| **temperature** | Skaluje logity przed softmaxem: `p_i ∝ exp(z_i / T)`. T→0 – zawsze najbardziej prawdopodobny token (deterministycznie); T>1 – rozkład spłaszczony, większa kreatywność i ryzyko błędów | **0.2** w odpowiedziach RAG (fakty), **0.0** w klasyfikacji, ocenie i przepisywaniu pytań, **1.0** w generowaniu danych treningowych (różnorodność) |
| **top_p** (nucleus sampling) | Losowanie tylko z najmniejszego zbioru tokenów, których łączne prawdopodobieństwo ≥ p – odcina „długi ogon” mało prawdopodobnych tokenów | 0.9 (0.95 przy generowaniu danych) |
| **top_k** | Losowanie tylko z k najbardziej prawdopodobnych tokenów. Nie należy do specyfikacji OpenAI – w LM Studio przekazywany w `extra_body` | 40 (60 przy generowaniu danych) |
| **repeat_penalty** | Kara (dzielnik logitów) dla tokenów, które już wystąpiły – przeciwdziała pętlom i powtórzeniom | 1.1 |
| **max_tokens** | Maksymalna długość odpowiedzi w tokenach. Po przekroczeniu `finish_reason = "length"` i tekst jest ucięty | 2048 (odpowiedzi; przy pytaniach złożonych min. 500 na podpytanie), 4096 (generowanie danych). Przy włączonym rozumowaniu aplikacja dolicza budżet: +2048 / +4096 / +8192 dla low / medium / high |
| **seed** | Ziarno losowania – identyczne parametry + seed = identyczna odpowiedź (powtarzalność eksperymentów) | 42 (czat), 0 (ewaluacja) |
| **reasoning_effort** | Gemma 4 jest modelem „rozumującym”: przed odpowiedzią generuje ukryty tok myślenia (`reasoning_content`). Zużywa on `max_tokens` i czas. `"none"` wyłącza rozumowanie | **none** (bez tego klasyfikacja z `max_tokens=30` zwracała pustą odpowiedź – patrz niżej) |
| **stream** | Odpowiedź przesyłana token po tokenie (efekt „pisania” w czacie) | true w czacie |

### Obserwacja: wpływ rozumowania (reasoning) na wyniki

Dla zapytania klasyfikującego („Ile kosztuje dostawa?”) z `max_tokens=30`:

| reasoning_effort | tokeny rozumowania | czas | treść odpowiedzi |
|---|---|---|---|
| domyślnie (włączone) | 138–344 | 2,0–4,2 s | **pusta** (limit zużyty na myślenie, `finish_reason=length`) |
| `none` | 0 | **0,3 s** | `{"category": "dostawa"}` |

Wniosek: przy modelach rozumujących `max_tokens` obejmuje także ukryte tokeny myślenia – trzeba je wyłączyć
lub istotnie podnieść limit. Interaktywne porównanie parametrów: zakładka **„Parametry modelu”** w aplikacji
oraz notebook `notebooks/01_lm_studio_parametry.ipynb`.

## 5. Typowe problemy

| Objaw | Przyczyna / rozwiązanie |
|---|---|
| `Connection refused` na porcie 1234 | serwer nie działa – `lms server start` lub *Developer → Start Server* |
| `model not found` | inny identyfikator – sprawdź `lms ls` i popraw `LLM_MODEL` / `EMBEDDING_MODEL` w `.env` |
| pusta odpowiedź, `finish_reason=length` | zbyt małe `max_tokens` przy włączonym rozumowaniu – ustaw `reasoning_effort="none"` |
| model nie widzi obrazu | brak pliku `mmproj` w folderze modelu |
| bardzo wolne generowanie | model nie mieści się w VRAM (częściowy offload na CPU) – zmniejsz kontekst lub wybierz mniejszą kwantyzację |
