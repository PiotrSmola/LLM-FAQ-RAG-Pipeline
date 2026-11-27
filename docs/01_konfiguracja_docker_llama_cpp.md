# Konfiguracja: Docker, llama.cpp i modele krok po kroku

Cały projekt działa w kontenerach Dockera. Modele językowe uruchamia gotowe narzędzie **llama.cpp**
(`llama-server`, oficjalny obraz `ghcr.io/ggml-org/llama.cpp`). Wczytuje ono modele w formacie GGUF na kartę
graficzną i udostępnia je przez **API zgodne z OpenAI**. Aplikacja w Pythonie łączy się z tym API biblioteką
`openai`, tak samo jak z każdym serwerem zgodnym z OpenAI (także z LM Studio, które korzysta z tego samego silnika).

Sprzęt testowy: NVIDIA RTX 5070 Ti (16 GB VRAM), Windows 11, Docker Desktop (silnik 29.6, backend WSL2),
llama.cpp build b12808 (CUDA 12.8).

## 1. Co działa w kontenerach

| Kontener | Obraz | Zawartość | Port na komputerze |
|---|---|---|---|
| `llm` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12808` | Gemma 4 12B QAT + projektor obrazu `mmproj`, GPU | 8080 (`/v1/chat/completions`) |
| `embeddings` | ten sam obraz | bge-m3, tryb embeddingów, GPU | 8081 (`/v1/embeddings`) |
| `app` | `faq-assistant-app` (budowany z `Dockerfile`) | aplikacja Gradio, pipeline, HerBERT (CPU), skrypty, notebooki | 7860 (aplikacja) |
| `models` | `faq-assistant-app` | jednorazowe pobranie plików GGUF (profil `setup`, nie startuje z resztą) | brak |

Aplikacja widzi serwery pod nazwami usług z `docker-compose.yml`: `http://llm:8080/v1` i
`http://embeddings:8080/v1`. Porty 8080 i 8081 są wystawione na komputer, żeby dało się wywołać API ręcznie
(`curl`, notebook, skrypty uruchamiane poza Dockerem).

Na dysku, poza kontenerami, zostają:

| Miejsce | Zawartość |
|---|---|
| `models/gguf/` albo katalog z `GGUF_DIR` | pliki GGUF modeli (7,8 GB), montowane do serwerów tylko do odczytu |
| `models/herbert-faq/`, `models/tfidf_lr.joblib` | wytrenowane klasyfikatory |
| `results/`, `notebooks/` | wyniki ewaluacji i eksporty, wykonane notebooki |
| wolumen `chroma` | indeks wektorowy bazy wiedzy (budowany przy pierwszym starcie) |
| wolumen `hf_cache` | modele z Hugging Face pobierane przez skrypty (HerBERT bazowy, XLM-R, mBERT) |

## 2. Wymagania

* **Windows 10/11** z **Docker Desktop** na backendzie WSL2 (*Settings > General > Use the WSL 2 based engine*).
  **Linux**: Docker Engine z wtyczką `docker compose` i **NVIDIA Container Toolkit**.
* **Karta NVIDIA** z aktualnym sterownikiem (CUDA 12.8, czyli sterownik 570 lub nowszy) i co najmniej 12 GB VRAM.
  Projekt zajmuje około 10,4 GB VRAM (szczegóły w punkcie 6).
* Około 20 GB miejsca na dysku: modele 7,8 GB, obraz llama.cpp 7,1 GB, obraz aplikacji 3,4 GB.
* Przy pierwszym uruchomieniu: dostęp do internetu (obrazy z GitHub Container Registry i Docker Hub,
  modele z Hugging Face).

## 3. Uruchomienie jednym poleceniem

W katalogu projektu:

```bat
start.bat
```

Na Linuksie: `./start.sh`. Skrypt wykonuje kolejno:

1. sprawdza, czy działa Docker, a jeśli nie, uruchamia Docker Desktop i czeka na silnik;
2. tworzy `.env` z `.env.example`, jeśli go brak;
3. pobiera obraz llama.cpp i uruchamia w nim `nvidia-smi`, żeby sprawdzić, czy kontenery widzą kartę;
4. szuka modeli GGUF: w `GGUF_DIR` z `.env`, w `models/gguf/`, a potem w katalogu modeli LM Studio
   (`%USERPROFILE%\.lmstudio\models`). Jeśli znajdzie je w LM Studio, zapisuje tę ścieżkę w `.env` i niczego nie
   pobiera. Jeśli nie ma ich nigdzie, pobiera je z Hugging Face do `models/gguf/` (kontener `models`);
5. buduje obraz aplikacji i uruchamia kontenery (`docker compose up -d --build`);
6. czeka, aż oba serwery załadują modele na GPU, a aplikacja zbuduje indeks i odpowie na zapytanie HTTP;
7. otwiera `http://localhost:7860` w przeglądarce.

Pierwsze uruchomienie z pobieraniem trwa kilkanaście minut (zależnie od łącza). Kolejne trwają około 1,5 minuty,
z czego większość to ładowanie modeli na GPU.

| Polecenie | Działanie |
|---|---|
| `start.bat` | start całego projektu |
| `start.bat stop` | zatrzymanie i usunięcie kontenerów; modele, indeks i wyniki zostają |
| `start.bat status` | stan kontenerów |
| `start.bat logs` | logi wszystkich kontenerów na żywo (Ctrl+C kończy podgląd) |
| `start.bat test` | testy jednostkowe (pytest) w kontenerze |
| `start.bat train` | fine-tuning HerBERT w kontenerze, zapis do `models/herbert-faq`, restart aplikacji |

To samo ręcznie, bez skryptu:

```bash
docker compose --profile setup run --rm models    # jednorazowo: pobranie modeli do models/gguf
docker compose up -d --build                      # start
docker compose ps                                 # stan (kolumna STATUS: healthy)
docker compose down                               # zatrzymanie
```

## 4. Modele

| Model | Plik (repozytorium Hugging Face) | Rozmiar | Rola w projekcie |
|---|---|---|---|
| **Gemma 4 12B Instruct QAT** | `gemma-4-12B-it-QAT-Q4_0.gguf` i `mmproj-gemma-4-12B-it-QAT-BF16.gguf` (`lmstudio-community/gemma-4-12B-it-QAT-GGUF`) | 7,0 GB + 0,18 GB | LLM: odpowiedzi RAG, klasyfikacja zero-shot, przepisywanie pytań, odczyt obrazów, LLM-as-judge |
| **BGE-M3** | `bge-m3-Q8_0.gguf` (`gpustack/bge-m3-GGUF`) | 0,6 GB | model embeddingów (wektory 1024-wymiarowe, wielojęzyczny) |

*QAT (Quantization-Aware Training)*: model był douczany ze świadomością kwantyzacji 4-bitowej, więc wersja Q4_0
traci znacznie mniej jakości niż zwykła kwantyzacja, a zajmuje około 4 razy mniej pamięci niż FP16.
Plik **mmproj** to projektor wizyjny. Bez niego model nie przyjmuje obrazów (funkcja wielomodalna).

Pobieranie wykonuje `scripts/download_models.py` (biblioteka `huggingface_hub`). Pliki trafiają do
`<katalog>/<repozytorium>/<plik>`, czyli w tym samym układzie co w LM Studio, dlatego oba narzędzia mogą
korzystać z jednej kopii. Sprawdzenie bez pobierania: `python scripts/download_models.py --check`.

W API modele mają aliasy nadane flagą `--alias` (`google/gemma-4-12b-qat`, `text-embedding-bge-m3`). Są takie
same jak identyfikatory w LM Studio, więc kod i `.env` działają z oboma serwerami bez zmian.

## 5. Parametry serwera (flagi `llama-server` w `docker-compose.yml`)

| Flaga | Wartość | Znaczenie |
|---|---|---|
| `--n-gpu-layers` | 999 (wszystkie) | ile warstw modelu trafia do VRAM. Całość na GPU jest najszybsza. Gdy pamięci brakuje, część warstw liczy CPU i generowanie zwalnia kilkukrotnie |
| `--ctx-size` | 32768 (LLM), 8192 (bge-m3) | okno kontekstu w tokenach: prompt, historia, źródła RAG, odpowiedź i ewentualne rozumowanie. Większe okno oznacza większy KV cache w VRAM. 32k mieści do 16 fragmentów dla pytań złożonych, historię rozmowy i długą odpowiedź. bge-m3 przyjmuje najwyżej 8192 tokeny |
| `--parallel` | 1 | liczba slotów, czyli zapytań obsługiwanych równocześnie. Każdy slot dzieli okno kontekstu i ma własne bufory. Aplikacja wysyła zapytania po kolei, więc jeden slot daje całe okno jednemu zapytaniu |
| `--batch-size`, `--ubatch-size` | 2048 (LLM), 8192 (bge-m3) | ile tokenów promptu serwer przetwarza naraz. Obraz w Gemmie 4 to do 1120 tokenów i musi zmieścić się w jednym ubatchu, inaczej llama.cpp obniża budżet obrazu do 512 tokenów. Przy embeddingach cały tekst musi zmieścić się w jednym ubatchu |
| `--mmproj` | plik projektora | włącza obsługę obrazów |
| `--jinja` | | szablon czatu zapisany w pliku GGUF; potrzebny do obrazów i sterowania rozumowaniem |
| `--embeddings` | (tylko bge-m3) | serwer zwraca wektory zamiast generować tekst |
| `--load-mode none` | | wagi są czytane z pliku i kopiowane do VRAM bez mapowania pliku w pamięci (mmap), więc w RAM nie zostaje ich kopia |
| `--cache-ram` | 1024 MiB | pamięć podręczna stanów poprzednich promptów w RAM; przyspiesza zapytania o wspólnym początku (ten sam system prompt). Domyślnie llama.cpp rezerwuje do 8 GB i przy długiej ewaluacji kontener rósł do 10,2 GB RAM |
| `--alias` | `google/gemma-4-12b-qat` | identyfikator modelu w API |
| `--host`, `--port` | 0.0.0.0, 8080 | serwer nasłuchuje w sieci kontenerów; porty na komputerze ustawia `docker-compose.yml` |

Flash attention i KV cache w F16 to ustawienia domyślne llama.cpp, więc nie ma ich w pliku.

## 6. Zasoby (pomiar po pełnym starcie i serii zapytań)

| Kontener | VRAM | RAM |
|---|---|---|
| `llm` (Gemma 4 12B + mmproj, kontekst 32k) | 9,4 GB | 2,5 do 3,2 GB |
| `embeddings` (bge-m3) | 1,0 GB | 1,0 GB |
| `app` (Gradio, HerBERT na CPU, ChromaDB) | 0 | 0,7 do 0,9 GB |

Dla porównania: LM Studio z domyślnymi ustawieniami trzymało w RAM kopię wag Gemmy (około 7,5 GB), a przy
4 slotach (`parallel 4`, domyślnie) 14,4 GB. W kontenerach te koszty usuwają `--load-mode none`, `--parallel 1`
i `--cache-ram 1024`.

HerBERT klasyfikuje pytanie na CPU w kilkadziesiąt milisekund, dlatego obraz aplikacji zawiera PyTorch w wersji
CPU (obraz ma 3,4 GB, z PyTorch CUDA byłby o kilka GB większy). Wariant z CUDA:
`docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 app`.

## 7. API i test połączenia

```bash
curl http://localhost:8080/health                  # {"status":"ok"} po załadowaniu modelu
curl http://localhost:8080/v1/models
curl http://localhost:8080/props                   # ustawienia serwera: kontekst, sloty, domyślne próbkowanie
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"google/gemma-4-12b-qat\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Czym jest RAG? Jedno zdanie.\"}],
  \"temperature\": 0.2, \"max_tokens\": 100, \"reasoning_effort\": \"none\" }"
curl http://localhost:8081/v1/embeddings -H "Content-Type: application/json" -d "{
  \"model\": \"text-embedding-bge-m3\", \"input\": \"Jak zwrocic towar?\" }"
```

Konsola Windows (`cmd`) może przekłamać polskie znaki w treści JSON, dlatego przykłady są bez nich.
Test wszystkich trzech funkcji (czat, embeddingi, obraz) z kontenera aplikacji:

```bash
docker compose exec app python -m faq_assistant.llm data/examples/mail_reklamacja.png
```

## 8. Parametry generowania (wysyłane w każdym zapytaniu)

Bez parametrów w zapytaniu llama.cpp losuje z ustawieniami zapisanymi w pliku GGUF Gemmy (temperature 1.0,
top_k 64, top_p 0.95, repeat_penalty 1.0). Aplikacja zawsze przekazuje własne wartości (`GenerationParams`
w `src/faq_assistant/config.py`).

| Parametr | Znaczenie | Wartość w projekcie |
|---|---|---|
| **temperature** | Skaluje logity przed softmaxem: `p_i ∝ exp(z_i / T)`. Przy T→0 wybierany jest zawsze najbardziej prawdopodobny token (deterministycznie). Przy T>1 rozkład się spłaszcza: większa kreatywność i ryzyko błędów | **0.2** w odpowiedziach RAG (fakty), **0.0** w klasyfikacji, ocenie i przepisywaniu pytań, **1.0** w generowaniu danych treningowych (różnorodność) |
| **top_p** (nucleus sampling) | Losowanie tylko z najmniejszego zbioru tokenów, których łączne prawdopodobieństwo ≥ p. Odcina „długi ogon” mało prawdopodobnych tokenów | 0.9 (0.95 przy generowaniu danych) |
| **top_k** | Losowanie tylko z k najbardziej prawdopodobnych tokenów. Nie należy do specyfikacji OpenAI, llama.cpp odczytuje go z `extra_body` | 40 (60 przy generowaniu danych) |
| **repeat_penalty** | Kara (dzielnik logitów) dla tokenów, które już wystąpiły. Przeciwdziała pętlom i powtórzeniom | 1.1 |
| **max_tokens** | Maksymalna długość odpowiedzi w tokenach. Po jej osiągnięciu `finish_reason = "length"` i tekst jest ucięty | 2048 (odpowiedzi; przy pytaniach złożonych min. 500 na podpytanie), 4096 (generowanie danych). Przy włączonym rozumowaniu aplikacja dolicza budżet rozumowania |
| **seed** | Ziarno losowania: te same parametry i ten sam seed dają tę samą odpowiedź (powtarzalność eksperymentów) | 42 (czat), 0 (ewaluacja) |
| **reasoning_effort** | Gemma 4 jest modelem „rozumującym”: przed odpowiedzią może generować ukryty tok myślenia (`reasoning_content`), który zużywa `max_tokens` i czas. `"none"` wyłącza rozumowanie, każdy inny poziom je włącza | **none** (bez tego klasyfikacja z `max_tokens=30` zwraca pustą odpowiedź, patrz niżej) |
| **thinking_budget_tokens** | Górny limit tokenów rozumowania; po jego osiągnięciu llama.cpp kończy myślenie i model przechodzi do odpowiedzi | low 2048, medium 4096, high 8192; ten sam budżet jest doliczany do `max_tokens` |
| **stream** | Odpowiedź przesyłana token po tokenie (efekt „pisania” w czacie) | true w czacie |

### Obserwacja: wpływ rozumowania (reasoning) na wyniki

Zapytanie klasyfikujące („Ile kosztuje dostawa?”) z `max_tokens=30`:

| reasoning_effort | czas | treść odpowiedzi |
|---|---|---|
| `high` (rozumowanie włączone) | 0,6 do 1,1 s | **pusta**: cały limit zużyty na myślenie, `finish_reason=length` |
| `none` | **0,5 s** | `{"category": "dostawa"}` |

Proste obliczenie kosztu dostawy (notebook 01, sekcja 7), wynik we wszystkich wariantach 19,99 zł:

| reasoning_effort | thinking_budget_tokens | tokeny odpowiedzi | czas |
|---|---|---|---|
| `none` | brak | 7 | 0,24 s |
| `high` | 16 | 26 | 0,63 s |
| `high` | 64 | 185 | 2,37 s, myślenie ucięte w połowie i dokończone po angielsku w treści odpowiedzi |
| `high` | bez limitu | 202 | 2,64 s |

Pytanie klienta w aplikacji, `max_tokens=300`: `none` 2,0 s i 121 tokenów, `low` 4,2 s i 354 tokeny,
`high` 6,2 s i 495 tokenów. Odpowiedzi były poprawne we wszystkich wariantach.

Wnioski: przy modelach rozumujących `max_tokens` obejmuje także ukryte tokeny myślenia, więc rozumowanie trzeba
wyłączyć albo wyraźnie podnieść limit. Budżet rozumowania nie może być zbyt mały, bo ucięte myślenie przechodzi
do odpowiedzi. W pipeline'ie FAQ rozumowanie nie poprawia odpowiedzi, a wydłuża czas, dlatego jest wyłączone.
Interaktywne porównanie parametrów: zakładka **„Parametry modelu”** w aplikacji oraz notebook
`notebooks/01_parametry_llm.ipynb`.

## 9. Zgodność z LM Studio

Pierwsza wersja projektu używała LM Studio 0.4.28. Po przejściu na llama.cpp w Dockerze sprawdziłem, czy wyniki
się nie zmieniły:

* embeddingi bge-m3 z obu serwerów dla 40 fragmentów bazy wiedzy: podobieństwo cosinusowe minimum 0,99887,
  średnio 0,99981; najbliższy sąsiad każdego fragmentu był ten sam w obu wersjach;
* ewaluacja wyszukiwania (`scripts/evaluate.py --part retrieval`) dała identyczne Recall@k i MRR dla wszystkich
  metod, różniły się tylko czasy.

Jedyna różnica w działaniu: LM Studio nie obsługuje `thinking_budget_tokens`, więc poziomy `low` i `high`
dawały tam identyczny wynik (191 tokenów rozumowania). llama.cpp ogranicza rozumowanie zgodnie z budżetem.

Projekt nadal działa z LM Studio zamiast kontenerów z modelami. W `.env` wystarczy ustawić
`LLM_BASE_URL=http://localhost:1234/v1` i `EMBEDDING_BASE_URL=http://localhost:1234/v1`, a następnie załadować modele:

```bash
lms server start
lms load google/gemma-4-12b-qat --gpu max --context-length 32768 --parallel 1 -y
lms load text-embedding-bge-m3 --gpu max -y
python app.py                    # aplikacja z lokalnego środowiska Pythona (README, punkt 3.4)
```

## 10. Skrypty i notebooki w kontenerach

Kod jest kopiowany do obrazu przy budowie, a katalogi `models/`, `results/` i `notebooks/` są montowane
z dysku, więc wyniki zapisują się w repozytorium. Po zmianie kodu trzeba przebudować obraz (`--build`).

| Polecenie | Działanie |
|---|---|
| `docker compose exec app python scripts/batch.py data/examples/pytania.csv --out results/batch --svg` | przetwarzanie wsadowe |
| `docker compose exec app python scripts/evaluate.py` | ewaluacja retrievalu, QA i ablacji |
| `docker compose exec app python scripts/build_index.py` | przebudowa indeksu |
| `docker compose run --rm --build --no-deps app python scripts/build_notebooks.py` | zbudowanie i wykonanie notebooków |
| `docker compose run --rm --build --no-deps app python scripts/train_classifier.py train` | fine-tuning HerBERT na CPU (to samo robi `start.bat train`) |
| `docker compose run --rm --build --no-deps app python -m pytest -q` | testy (to samo robi `start.bat test`) |
| `docker compose exec app bash` | powłoka w kontenerze aplikacji |

`exec` uruchamia polecenie w działającym kontenerze aplikacji. `run --rm` tworzy nowy, tymczasowy kontener
z aktualnym kodem; `--no-deps` nie uruchamia ponownie serwerów, ale jeśli działają, kontener łączy się z nimi
w tej samej sieci.

Fine-tuning HerBERT w kontenerze na CPU trwa około 4,5 minuty (8 epok, 794 przykłady). Na GPU, w lokalnym
środowisku Pythona z PyTorch CUDA, około 30 s.

## 11. Typowe problemy

| Objaw | Przyczyna i rozwiązanie |
|---|---|
| `start.bat` kończy się komunikatem, że kontenery nie widzą karty NVIDIA | Docker Desktop nie korzysta z WSL2 albo sterownik NVIDIA jest za stary. Włącz *Use the WSL 2 based engine*, zaktualizuj sterownik, sprawdź `docker run --rm --gpus all --entrypoint nvidia-smi ghcr.io/ggml-org/llama.cpp:server-cuda-b12808` |
| kontener `llm` w stanie `exited`, w logach `failed to load model` | brak lub niepełny plik GGUF. `python scripts/download_models.py --check`, a potem ponowne pobranie: `docker compose --profile setup run --rm models` |
| w logach `cudaMalloc failed: out of memory` | za mało VRAM, bo inne programy zajmują kartę. Zamknij je albo zmniejsz `--ctx-size` w `docker-compose.yml` (np. 16384) |
| `Error code: 503 - Loading model` | serwer jeszcze ładuje model; `docker compose ps` pokaże `healthy`, gdy będzie gotowy |
| port 7860, 8080 lub 8081 zajęty | zmień `APP_PORT`, `LLM_PORT` lub `EMBEDDING_PORT` w `.env` (np. gdy działa LM Studio albo inna aplikacja) |
| pusta odpowiedź, `finish_reason=length` | zbyt małe `max_tokens` przy włączonym rozumowaniu; ustaw `reasoning_effort="none"` |
| aplikacja używa TF-IDF zamiast HerBERT | brak `models/herbert-faq`; `start.bat train` albo skopiowanie modelu z archiwum projektu |
| skrypt PowerShell zablokowany przez zasady wykonywania | `start.bat` uruchamia `start.ps1` z `-ExecutionPolicy Bypass` tylko dla tego procesu; uruchamiaj projekt przez `start.bat` |
