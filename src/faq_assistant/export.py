"""Eksport wyników (próg 4.5): raport tekstowy .txt i wizualizacja etapów pipeline'u .svg."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]  # prosty font systemowy, spójny z UI
plt.rcParams["svg.fonttype"] = "none"                   # tekst w SVG jako tekst (da się zaznaczyć/kopiować)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .config import CATEGORIES  # noqa: E402
from .pipeline import PipelineTrace  # noqa: E402
from .prompts import category_label  # noqa: E402

COLORS = {"stage": "#2b4c7e", "accent": "#e07a1f", "muted": "#8a94a6", "ok": "#2e8b57"}


# --------------------------------------------------------------------------------------------------- TXT
def trace_to_text(tr: PipelineTrace) -> str:
    w = textwrap.TextWrapper(width=100, subsequent_indent="    ")
    lines = [
        "=" * 100,
        f"ASYSTENT FAQ – Zorza Elektronika        {tr.timestamp}",
        "=" * 100,
        f"Wejście użytkownika : {tr.user_input or '(brak tekstu)'}",
    ]
    if tr.image_name:
        lines += [f"Obraz               : {tr.image_name}", w.fill(f"Odczyt z obrazu     : {tr.image_ocr}")]
    lines += [
        f"Pytanie (końcowe)   : {tr.question}" + ("  [przepisane z kontekstu rozmowy]" if tr.rewritten else ""),
        "",
        "ETAP 1 – KLASYFIKACJA / ROUTING",
        f"    Klasyfikator : {tr.classifier or '-'}",
        f"    Kategoria    : {category_label(tr.category) if tr.category else '-'} (pewność {tr.confidence:.1%})",
        f"    Decyzja      : {tr.route}",
    ]
    if tr.probs:
        lines += [f"        {category_label(c):24s} {tr.probs.get(c, 0):6.1%}" for c in CATEGORIES]
    lines += ["", f"ETAP 2 – WYSZUKIWANIE ({tr.retrieval_method}), znalezione fragmenty: {len(tr.hits)}"]
    for s in tr.sources():
        lines.append(f"    [{s['n']}] {s['source']}  (wynik {s['score']:.4f})")
        lines.append(w.fill("        " + s["text"].replace("\n", " ")[:300] + "…"))
    lines += ["", "ETAP 3 – ODPOWIEDŹ", *[w.fill("    " + p) for p in tr.answer.splitlines() if p.strip()], ""]
    lines.append("CZASY ETAPÓW")
    lines += [f"    {t.name:38s} {t.seconds:6.2f} s" for t in tr.timings]
    lines.append(f"    {'RAZEM':38s} {tr.total_seconds:6.2f} s")
    p = tr.params
    lines += ["", "PARAMETRY GENEROWANIA",
              f"    temperature={p.get('temperature')}  top_p={p.get('top_p')}  top_k={p.get('top_k')}  "
              f"repeat_penalty={p.get('repeat_penalty')}  max_tokens={p.get('max_tokens')}  seed={p.get('seed')}  "
              f"reasoning={p.get('reasoning_effort')}", "=" * 100]
    return "\n".join(lines)


def conversation_to_text(traces: list[PipelineTrace]) -> str:
    return "\n\n".join(trace_to_text(t) for t in traces)


def save_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------------- SVG
def _box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="white", ec=color, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.075), w, 0.075, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=color, ec=color, lw=2))
    ax.text(x + w / 2, y + h - 0.038, title, ha="center", va="center", color="white", fontsize=9.5,
            fontweight="bold")
    ax.text(x + 0.012, y + h - 0.1, body, ha="left", va="top", fontsize=8, color="#222", wrap=True,
            linespacing=1.35)


def short_citation(chunk, n: int = 34) -> str:
    """FAQ -> id wpisu; dokument -> skrócona nazwa pliku + ostatni nagłówek sekcji."""
    if chunk.kind == "faq":
        return chunk.source
    section = chunk.title.split(" > ")[-1]
    return _short(f"{Path(chunk.source).stem[:14]}… § {section}", n)


def _short(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def trace_to_figure(tr: PipelineTrace) -> plt.Figure:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.28, wspace=0.6)
    fig.suptitle(f"Przebieg pipeline'u: „{_short(tr.question, 90)}”", fontsize=12.5, fontweight="bold")

    # --- schemat etapów
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.02, 1.0)
    ax.axis("off")
    timing = {t.name.split(" (")[0].split(" →")[0]: t.seconds for t in tr.timings}

    def t_of(prefix: str) -> str:
        v = next((s for k, s in timing.items() if k.startswith(prefix)), None)
        return f"{v:.2f} s" if v is not None else "—"

    q_body = textwrap.fill(_short(tr.question, 120), 30)
    if tr.image_name:
        q_body = f"[obraz: {tr.image_name}]\n" + q_body
    if tr.rewritten:
        q_body += "\n(przepisane z historii)"
    cls_body = (f"{category_label(tr.category) if tr.category else '-'}\npewność: {tr.confidence:.0%}\n"
                f"{_short(tr.classifier, 28)}\nczas: {t_of('Klasyfikacja')}")
    ret_body = "\n".join(f"[{i}] {short_citation(h.chunk, 30)}" for i, h in enumerate(tr.hits[:5], 1)) or "—"
    ret_body += f"\nczas: {t_of('Wyszukiwanie')}"
    ans_body = textwrap.fill(_short(tr.answer, 230), 32) + f"\nczas: {t_of('Generowanie')}"

    stages = [("Pytanie", q_body, COLORS["muted"]), ("1. Klasyfikacja", cls_body, COLORS["stage"]),
              (f"2. Retrieval ({tr.retrieval_method})", ret_body, COLORS["stage"]),
              ("3. Odpowiedź (LLM)", ans_body, COLORS["ok"] if tr.answered_by == "rag" else COLORS["accent"])]
    w, h, gap = 0.215, 0.82, (1 - 4 * 0.215) / 3
    for i, (title, body, color) in enumerate(stages):
        x = i * (w + gap)
        _box(ax, x, 0.08, w, h, title, body, color)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + w + 0.004, 0.5), (x + w + gap - 0.004, 0.5),
                                         arrowstyle="-|>", mutation_scale=18, lw=2, color="#555"))
    ax.text(0.5, 0.0, f"Routing: {_short(tr.route, 150)}   |   łączny czas: {tr.total_seconds:.2f} s",
            ha="center", va="bottom", fontsize=8.5, style="italic", color="#444")

    # --- prawdopodobieństwa kategorii
    ax1 = fig.add_subplot(gs[1, 0])
    labels = [category_label(c) for c in CATEGORIES]
    vals = [tr.probs.get(c, 0.0) for c in CATEGORIES]
    colors = [COLORS["accent"] if c == tr.category else COLORS["stage"] for c in CATEGORIES]
    ax1.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("prawdopodobieństwo")
    ax1.set_title("Etap 1: rozkład kategorii", fontsize=10)
    for i, v in enumerate(vals[::-1]):
        inside = v > 0.85
        ax1.text(v - 0.02 if inside else v + 0.01, i, f"{v:.0%}", va="center", fontsize=8,
                 ha="right" if inside else "left", color="white" if inside else "#222")

    # --- wyniki retrievalu
    ax2 = fig.add_subplot(gs[1, 1])
    if tr.hits:
        names = [f"[{i}] {short_citation(h.chunk, 40)}" for i, h in enumerate(tr.hits, 1)]
        scores = [h.score for h in tr.hits]
        ax2.barh(names[::-1], scores[::-1],
                 color=[COLORS["ok"] if h.chunk.kind == "faq" else COLORS["muted"] for h in tr.hits][::-1])
        ax2.set_xlabel("wynik dopasowania" + (" (RRF)" if tr.retrieval_method == "hybrid" else ""))
        ax2.tick_params(axis="y", labelsize=8)
    else:
        ax2.text(0.5, 0.5, "brak wyszukiwania\n(pytanie poza zakresem)", ha="center", va="center")
        ax2.set_xticks([])
        ax2.set_yticks([])
    ax2.set_title("Etap 2: znalezione fragmenty (zielone = FAQ, szare = dokumenty)", fontsize=10)
    return fig


def save_svg(tr: PipelineTrace, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = trace_to_figure(tr)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path
