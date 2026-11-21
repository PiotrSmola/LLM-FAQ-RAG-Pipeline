"""Zrzuty ekranu aplikacji Gradio do dokumentacji (Playwright + Chromium).

Wymaga działającej aplikacji (python app.py) i LM Studio. Scenariusz: pytanie tekstowe, pytanie z obrazem,
pytanie poza zakresem, zakładki pomocnicze. Wynik: docs/screenshots/*.png

    pip install playwright && python -m playwright install chromium
    python scripts/screenshots.py [--only asystent]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
URL = "http://127.0.0.1:7860/"


def ask(page: Page, text: str = "", image: str | None = None, wait_s: int = 90) -> None:
    box = page.locator("textarea").first
    if image:
        page.locator("input[type=file]").first.set_input_files(str(ROOT / image))
        page.wait_for_timeout(1500)
    box.fill(text)
    box.press("Enter")
    # koniec przetwarzania = zniknęły komunikaty pośrednie handlera (start, generowanie, etap 3 „w trakcie”)
    page.wait_for_timeout(3000)
    page.wait_for_function("() => { const t = document.body.innerText; return !t.includes('Analizuję pytanie') "
                           "&& !t.includes('Generuję odpowiedź') && !t.includes('w trakcie…'); }",
                           timeout=wait_s * 1000)
    page.wait_for_timeout(2500)


def open_tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(1200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(URL)
        page.wait_for_selector("textarea", timeout=60000)
        page.wait_for_timeout(2000)
        page.screenshot(path=OUT / "00_start.png")
        if a.only in (None, "asystent"):
            ask(page, "Ile kosztuje pełna ochrona Zorza Ochrona+ na 2 lata dla laptopa za 4000 zł?")
            page.screenshot(path=OUT / "01_asystent_odpowiedz.png", full_page=True)
            ask(page, "a ile wynosi udział własny przy zbitym ekranie?")
            page.screenshot(path=OUT / "02_asystent_dopytanie.png", full_page=True)
            page.get_by_role("button", name="Nowa rozmowa").click()
            page.wait_for_timeout(800)
            ask(page, "", image="data/examples/mail_reklamacja.png")
            page.screenshot(path=OUT / "03_asystent_obraz.png", full_page=True)
            ask(page, "Napisz mi przepis na sernik")
            page.screenshot(path=OUT / "04_poza_zakresem.png", full_page=True)
        if a.only in (None, "zakladki"):
            for i, tab in enumerate(["Twoje dokumenty", "Wsadowo", "Parametry modelu", "Ewaluacja"], 5):
                open_tab(page, tab)
                page.screenshot(path=OUT / f"{i:02d}_{tab.lower().replace(' ', '_')}.png", full_page=True)
        browser.close()
    print("Zapisano zrzuty w", OUT)


if __name__ == "__main__":
    main()
