"""Generuje przykładowe obrazy do pytań zadawanych obrazem.

Obrazy imitują zrzut ekranu wiadomości od klienta i zdjęcie paragonu - model z obsługą obrazu
(Gemma + mmproj) ma z nich odczytać pytanie klienta.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "data" / "examples"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size)


def email_screenshot() -> Path:
    img = Image.new("RGB", (900, 520), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 900, 60], fill="#2b4c7e")
    d.text((24, 16), "Poczta: Odebrane", font=_font(24, True), fill="white")
    d.text((24, 80), "Od: Anna Kowalska <anna.k@example.com>", font=_font(20), fill="#333")
    d.text((24, 110), "Do: pomoc@zorza-elektronika.example", font=_font(20), fill="#333")
    d.text((24, 140), "Temat: Zamówienie nr 48213, telefon nie działa", font=_font(20, True), fill="#111")
    d.line([24, 180, 876, 180], fill="#ccc", width=2)
    body = (
        "Dzień dobry,\n\n"
        "wczoraj kurier dostarczył mi smartfon z zamówienia 48213. Po wyjęciu z pudełka\n"
        "telefon w ogóle się nie włącza: nie reaguje na ładowarkę ani na przycisk.\n\n"
        "Co mam teraz zrobić i czy dostanę nowy egzemplarz?\n\n"
        "Pozdrawiam,\nAnna Kowalska"
    )
    d.multiline_text((24, 200), body, font=_font(21), fill="#111", spacing=8)
    path = OUT / "mail_reklamacja.png"
    img.save(path)
    return path


def receipt_photo() -> Path:
    img = Image.new("RGB", (520, 760), "#f4f1e8")
    d = ImageDraw.Draw(img)
    mono = _font(20)
    lines = [
        "ZORZA ELEKTRONIKA sp. z o.o.",
        "ul. Kwiatowa 12, 35-001 Rzeszów",
        "NIP 000-000-00-00",
        "",
        "PARAGON FISKALNY",
        "2026-11-14            nr 1187/0042",
        "--------------------------------",
        "Słuchawki BT X200    1 x 349,00",
        "Kabel USB-C 1m       1 x  29,99",
        "--------------------------------",
        "SUMA PLN                 378,99",
        "Płatność: karta",
        "",
        "Zakup w salonie stacjonarnym",
        "Dziękujemy za zakupy!",
    ]
    y = 30
    for line in lines:
        d.text((30, y), line, font=mono, fill="#222")
        y += 30
    d.text((30, y + 30), "Pytanie klienta (dopisane długopisem):", font=_font(18, True), fill="#1a3d8f")
    d.multiline_text((30, y + 60), "Czy mogę oddać te słuchawki\nprzez paczkomat?", font=_font(22), fill="#1a3d8f")
    path = OUT / "paragon_pytanie.png"
    img.save(path)
    return path


def invoice() -> Path:
    img = Image.new("RGB", (820, 620), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 25), "ZORZA ELEKTRONIKA sp. z o.o.", font=_font(24, True), fill="#1d2b3a")
    d.text((30, 58), "ul. Kwiatowa 12, 35-001 Rzeszów · NIP 000-000-00-00", font=_font(16), fill="#444")
    d.text((520, 25), "FAKTURA VAT", font=_font(24, True), fill="#2b4c7e")
    d.text((520, 58), "nr FV/2026/11/0815", font=_font(16), fill="#444")
    rows = [("Data sprzedaży:", "05.11.2026"), ("Data doręczenia:", "07.11.2026"),
            ("Sposób zakupu:", "sklep internetowy, zam. 51240"), ("Płatność:", "BLIK")]
    y = 110
    for k, v in rows:
        d.text((30, y), k, font=_font(17, True), fill="#222")
        d.text((210, y), v, font=_font(17), fill="#222")
        y += 30
    d.line([30, y + 10, 790, y + 10], fill="#2b4c7e", width=2)
    y += 25
    d.text((30, y), "Lp.  Nazwa                                   Ilość     Cena brutto", font=_font(16, True), fill="#222")
    y += 32
    for line in ["1.   Laptop Voltbook 14 i7/16GB/1TB            1         4 299,00 zł",
                 "2.   Zorza Ochrona+ Pełna ochrona 2 lata        1           644,85 zł"]:
        d.text((30, y), line, font=_font(16), fill="#222")
        y += 28
    d.line([30, y + 8, 790, y + 8], fill="#ccc", width=1)
    d.text((520, y + 22), "RAZEM: 4 943,85 zł", font=_font(20, True), fill="#1d2b3a")
    path = OUT / "faktura_laptop.png"
    img.save(path)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (email_screenshot(), receipt_photo(), invoice()):
        print("Zapisano", p)
