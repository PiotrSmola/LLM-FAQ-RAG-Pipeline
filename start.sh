#!/usr/bin/env bash
# Uruchamia cały projekt w Dockerze (Linux z kartą NVIDIA i NVIDIA Container Toolkit). Odpowiednik start.bat.
#   ./start.sh [start|stop|status|logs|train|test]
set -u
cd "$(dirname "$0")"

LLAMA_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda-b12808"
GGUF_FILES=(
  "lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf:6975879008"
  "lmstudio-community/gemma-4-12B-it-QAT-GGUF/mmproj-gemma-4-12B-it-QAT-BF16.gguf:175115328"
  "gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf:634553760"
)

say() { echo "[projekt] $*"; }
fail() { echo "[projekt] BŁĄD: $*" >&2; exit 1; }
dotenv() { [ -f .env ] && grep -E "^\s*$1\s*=" .env | tail -1 | cut -d= -f2- | xargs; }
env_or() { local v; v=$(dotenv "$1"); echo "${v:-$2}"; }

gguf_ok() {
  local dir=$1 entry path size
  for entry in "${GGUF_FILES[@]}"; do
    path="$dir/${entry%%:*}"; size="${entry##*:}"
    [ -f "$path" ] && [ "$(stat -c %s "$path" 2>/dev/null || stat -f %z "$path")" = "$size" ] || return 1
  done
}

ensure_docker() {
  command -v docker >/dev/null || fail "Nie znaleziono Dockera (https://docs.docker.com/engine/install/)."
  docker info >/dev/null 2>&1 || fail "Silnik Dockera nie działa (sudo systemctl start docker)."
}

ensure_gpu() {
  say "Sprawdzam dostęp do GPU w kontenerach..."
  docker image inspect "$LLAMA_IMAGE" >/dev/null 2>&1 || docker pull "$LLAMA_IMAGE" || fail "Nie udało się pobrać $LLAMA_IMAGE."
  local gpu
  gpu=$(docker run --rm --gpus all --entrypoint nvidia-smi "$LLAMA_IMAGE" -L 2>&1) \
    || fail "Kontenery nie widzą karty NVIDIA (potrzebny sterownik i NVIDIA Container Toolkit). $gpu"
  say "GPU: $(echo "$gpu" | head -1)"
}

ensure_models() {
  local dir="${GGUF_DIR:-$(dotenv GGUF_DIR)}"
  if [ -z "$dir" ]; then
    dir="./models/gguf"
    if ! gguf_ok "$dir" && gguf_ok "$HOME/.lmstudio/models"; then
      dir="$HOME/.lmstudio/models"
      printf '\nGGUF_DIR=%s\n' "$dir" >> .env
      say "Używam modeli z LM Studio: $dir (zapisane w .env jako GGUF_DIR)"
    fi
  fi
  export GGUF_DIR="$dir"
  if gguf_ok "$dir"; then say "Modele GGUF: $dir"; return; fi
  say "Brak modeli GGUF w $dir. Pobieram z Hugging Face (około 7,8 GB, jednorazowo)..."
  mkdir -p "$dir"
  docker compose --profile setup run --rm --build models && gguf_ok "$dir" || fail "Pobieranie modeli nie powiodło się."
}

health() {
  local id; id=$(docker compose ps -q "$1" 2>/dev/null)
  [ -z "$id" ] && { echo "brak"; return; }
  docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "$id" | xargs
}

wait_ready() {
  say "Czekam, aż modele załadują się na GPU i aplikacja wystartuje (pierwszy start: 1-3 min)..."
  local last="" states end=$((SECONDS + 900))
  while [ $SECONDS -lt $end ]; do
    states="llm: $(health llm) | embeddings: $(health embeddings) | app: $(health app)"
    [ "$states" != "$last" ] && { say "$states"; last=$states; }
    if [[ "$states" =~ exited|dead ]]; then docker compose logs --tail 40; fail "Któryś kontener się zatrzymał."; fi
    [ "$(health app)" = "running healthy" ] && return
    sleep 3
  done
  docker compose logs --tail 40; fail "Aplikacja nie wystartowała w ciągu 15 minut."
}

start_project() {
  ensure_docker
  [ -f .env ] || { cp .env.example .env; say "Utworzono .env z .env.example"; }
  ensure_gpu
  ensure_models
  if [ ! -f models/herbert-faq/config.json ]; then
    say "Brak wytrenowanego HerBERT (models/herbert-faq): aplikacja użyje klasyfikatora TF-IDF."
    say "Fine-tuning HerBERT w kontenerze: ./start.sh train"
  fi
  say "Buduję obraz aplikacji i uruchamiam kontenery..."
  docker compose up -d --build || fail "docker compose up nie powiodło się."
  wait_ready
  local url="http://localhost:$(env_or APP_PORT 7860)"
  say "Gotowe: $url"
  say "API modeli: LLM http://localhost:$(env_or LLM_PORT 8080)/v1, embeddingi http://localhost:$(env_or EMBEDDING_PORT 8081)/v1"
  say "Zatrzymanie: ./start.sh stop"
  (xdg-open "$url" || open "$url") >/dev/null 2>&1 || true
}

case "${1:-start}" in
  start)  start_project ;;
  stop)   ensure_docker; docker compose down; say "Zatrzymano. Modele, indeks i wyniki zostają na dysku." ;;
  status) ensure_docker; docker compose ps ;;
  logs)   ensure_docker; docker compose logs -f --tail 50 ;;
  train)
    ensure_docker
    say "Fine-tuning HerBERT (8 epok) w kontenerze aplikacji, na CPU trwa to około 5 minut..."
    docker compose run --rm --build --no-deps app python scripts/train_classifier.py train || fail "Trening nie powiódł się."
    [[ "$(health app)" == running* ]] && { docker compose restart app; say "Aplikacja uruchomiona ponownie z HerBERT."; }
    ;;
  test)   ensure_docker; docker compose run --rm --build --no-deps app python -m pytest -q ;;
  *) echo "Użycie: ./start.sh [start|stop|status|logs|train|test]"; exit 1 ;;
esac
