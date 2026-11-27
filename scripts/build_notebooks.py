"""Buduje notebooki z definicji komórek (scripts/notebooks/nbXX_*.py) i wykonuje je (nbclient).

Użycie:
    python scripts/build_notebooks.py               # wszystkie
    python scripts/build_notebooks.py nb01_parametry
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).parent / "notebooks"
OUT = ROOT / "notebooks"
NAMES = {"nb01_parametry": "01_parametry_llm.ipynb", "nb02_klasyfikator": "02_klasyfikator.ipynb",
         "nb03_ewaluacja": "03_ewaluacja_pipeline.ipynb"}


def build(name: str, execute: bool = True) -> Path:
    spec = importlib.util.spec_from_file_location(name, SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
                for kind, src in mod.CELLS]
    path = OUT / NAMES[name]
    OUT.mkdir(exist_ok=True)
    if execute:
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
    nbformat.write(nb, path)
    print("Zapisano", path)
    return path


if __name__ == "__main__":
    for n in sys.argv[1:] or list(NAMES):
        if (SRC / f"{n}.py").exists():
            build(n)
