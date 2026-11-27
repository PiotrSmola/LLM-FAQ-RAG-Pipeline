# Obraz aplikacji: Gradio + pipeline (klasyfikator HerBERT, retrieval, QA) + skrypty i notebooki.
# Modele językowe działają w osobnych kontenerach llama.cpp (docker-compose.yml), tu jest tylko klient API.
FROM python:3.14-slim

# PyTorch w wersji CPU: HerBERT klasyfikuje pytanie w kilkadziesiąt ms, a obraz jest o kilka GB mniejszy.
# Wariant z CUDA (np. do szybkiego treningu): --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/cache/huggingface \
    MPLBACKEND=Agg \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

# fonts-liberation: Liberation Sans ma metryki Arial, którego używają wykresy SVG
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-liberation curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt docker/constraints.txt ./
RUN pip install torch -c constraints.txt --index-url "$TORCH_INDEX" \
    && pip install -r requirements.txt -c constraints.txt

COPY . .

EXPOSE 7860
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=60 \
    CMD curl -fs http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "app.py"]
