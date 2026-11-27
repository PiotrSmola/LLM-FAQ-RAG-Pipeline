"""Pobiera pliki GGUF modeli z Hugging Face do katalogu używanego przez kontenery llama.cpp.

Układ katalogów jest taki sam jak w LM Studio (<wydawca>/<repozytorium>/<plik>), więc zamiast pobierać
można wskazać istniejący katalog modeli LM Studio (GGUF_DIR=~/.lmstudio/models w pliku .env).

Użycie (robi to start.bat / start.sh, gdy plików brakuje):
    docker compose --profile setup run --rm models
    python scripts/download_models.py --dest models/gguf          # bez Dockera
    python scripts/download_models.py --dest models/gguf --check  # tylko sprawdzenie
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (repozytorium na Hugging Face, plik, rozmiar w bajtach)
FILES = [
    ("lmstudio-community/gemma-4-12B-it-QAT-GGUF", "gemma-4-12B-it-QAT-Q4_0.gguf", 6_975_879_008),
    ("lmstudio-community/gemma-4-12B-it-QAT-GGUF", "mmproj-gemma-4-12B-it-QAT-BF16.gguf", 175_115_328),
    ("gpustack/bge-m3-GGUF", "bge-m3-Q8_0.gguf", 634_553_760),
]


def missing(dest: Path) -> list[tuple[str, str, int]]:
    return [(repo, name, size) for repo, name, size in FILES
            if not (dest / repo / name).exists() or (dest / repo / name).stat().st_size != size]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("models/gguf"))
    ap.add_argument("--check", action="store_true", help="tylko sprawdź, czy pliki są na miejscu")
    args = ap.parse_args()

    todo = missing(args.dest)
    if not todo:
        print(f"Wszystkie modele GGUF są w {args.dest}")
        return 0
    if args.check:
        for repo, name, size in todo:
            print(f"Brak: {repo}/{name} ({size / 1e9:.2f} GB)")
        return 1

    from huggingface_hub import hf_hub_download

    print(f"Do pobrania: {sum(s for *_, s in todo) / 1e9:.1f} GB")
    for repo, name, size in todo:
        print(f"\n{repo}/{name} ({size / 1e9:.2f} GB)")
        hf_hub_download(repo_id=repo, filename=name, local_dir=args.dest / repo)
    still = missing(args.dest)
    if still:
        print("Nie udało się pobrać:", ", ".join(n for _, n, _ in still))
        return 1
    print(f"\nGotowe, modele zapisane w {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
