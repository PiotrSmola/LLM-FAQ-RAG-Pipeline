"""Prompty systemowe i szablony zapytań do LLM.

Tam, gdzie wynik jest dalej przetwarzany w kodzie, model zwraca JSON. Odpowiedź RAG dostaje numerowane
źródła i musi je cytować jako [n].
"""

from __future__ import annotations

from .config import CATEGORIES, CATEGORY_LABELS_PL

SHOP = "Zorza Elektronika"

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "platnosci": "formy płatności, BLIK, karty, przelewy, raty, „Zapłać później”, faktury i NIP, kody rabatowe, "
                 "karty podarunkowe, nieudane płatności, terminy zwrotu pieniędzy",
    "dostawa": "koszty i czas dostawy, paczkomaty, kurierzy, wysyłka, śledzenie paczki, zmiana adresu, "
               "uszkodzona przesyłka, odbiór osobisty, wniesienie dużego sprzętu",
    "zwroty_reklamacje": "zwrot towaru bez podania przyczyny, procedura zwrotu, wyłączenia ze zwrotu, reklamacje, "
                         "wady, produkt niedziałający po dostawie, terminy rozpatrzenia, odwołania",
    "konto": "zakładanie i usuwanie konta, logowanie, hasło, blokada konta, weryfikacja dwuetapowa, dane osobowe, "
             "newsletter, program lojalnościowy Zorza Punkty, phishing",
    "produkty_gwarancja": "gwarancja producenta, rozszerzona ochrona Zorza Ochrona+, dostępność produktów, "
                          "przedsprzedaż, outlet, montaż, utylizacja sprzętu, kompatybilność akcesoriów, salon",
    "poza_zakresem": "pytania niezwiązane ze sklepem i jego usługami (pogoda, przepisy kulinarne, wiedza ogólna, "
                     "prośby o napisanie tekstu lub kodu, porównania marek)",
}

CLASSIFY_SYSTEM = f"""Jesteś modułem routingu w systemie obsługi klienta sklepu internetowego {SHOP}.
Twoim jedynym zadaniem jest przypisanie pytania klienta do DOKŁADNIE jednej kategorii.

Kategorie:
""" + "\n".join(f"- {c}: {CATEGORY_DESCRIPTIONS[c]}" for c in CATEGORIES) + """

Zasady:
- Zwróć wyłącznie obiekt JSON: {"category": "<nazwa_kategorii>"}, bez komentarzy.
- Jeśli pytanie dotyczy kilku kategorii, wybierz tę, która opisuje główny problem klienta.
- Pytania niezwiązane ze sklepem przypisz do "poza_zakresem".

Przykłady:
Pytanie: "czy moge zaplacic blikiem" -> {"category": "platnosci"}
Pytanie: "paczka nie doszła od tygodnia" -> {"category": "dostawa"}
Pytanie: "jak napisać list motywacyjny" -> {"category": "poza_zakresem"}"""

GENERATE_TRAINING_SYSTEM = """Jesteś generatorem realistycznych danych treningowych dla klasyfikatora tekstu w języku polskim.
Generujesz pytania, jakie klienci naprawdę wpisują w czacie sklepu internetowego z elektroniką."""

GENERATE_TRAINING_USER = """Kategoria: {category}, {description}

Przykładowe pytania z bazy FAQ tej kategorii (NIE kopiuj ich, potraktuj jako inspirację tematyczną):
{seeds}

Wygeneruj {n} NOWYCH, zróżnicowanych pytań klientów należących do tej kategorii.
Wymagania dotyczące różnorodności:
- różne długości: od 3 słów do 2-3 zdań z kontekstem sytuacji,
- różne style: formalny, potoczny, zdenerwowany klient, bardzo krótki,
- ok. 1/3 pytań bez polskich znaków lub z literówkami, ok. 1/4 bez znaku zapytania,
- różne sformułowania: unikaj powtarzania tych samych słów kluczowych co w przykładach.

Zwróć wyłącznie tablicę JSON z napisami, np. ["pytanie 1", "pytanie 2"]."""

GENERATE_OUT_OF_SCOPE_USER = """Wygeneruj {n} zróżnicowanych wiadomości, które użytkownik mógłby wpisać w czacie sklepu
internetowego z elektroniką, ale które NIE dotyczą sklepu ani jego usług (zamówień, płatności, dostawy,
zwrotów, konta, gwarancji). Przykłady tematów: pogoda, sport, gotowanie, polityka, nauka, prośby
o napisanie tekstu lub kodu, tłumaczenia, porady życiowe, pytania o inne firmy, porównania marek telefonów.
Różne długości i style, ok. 1/3 bez polskich znaków. Zwróć wyłącznie tablicę JSON z napisami."""

REWRITE_SYSTEM = """Przekształcasz ostatnie pytanie klienta w samodzielne pytanie, zrozumiałe bez historii rozmowy.
Uzupełnij brakujący kontekst (np. „a ile to kosztuje?” -> „Ile kosztuje dostawa kurierem?”).
Jeśli pytanie jest już samodzielne, przepisz je bez zmian. Nie odpowiadaj na pytanie.
Zwróć tylko przepisane pytanie, bez cudzysłowów i komentarzy."""

DECOMPOSE_SYSTEM = """Klient zadał w jednej wiadomości kilka pytań do obsługi sklepu internetowego.
Rozbij wiadomość na listę SAMODZIELNYCH podpytań (każde zrozumiałe bez reszty wiadomości), maksymalnie 12. Łącz w jedno podpytanie drobne pytania o tę samą rzecz, ale nie pomijaj żadnej sprawy.
Każde podpytanie ma zawierać potrzebny kontekst z wiadomości: o który produkt chodzi, jego cenę, daty zakupu
i doręczenia, sposób zakupu. Nie zakładaj, że czytelnik zna inne podpytania (np. zamiast „czy mogę go oddać”
napisz „czy mogę oddać smartfon za 1800 zł kupiony w salonie 3 marca”).
Jedna sprawa może dać kilka podpytań, jeśli pyta o kilka różnych rzeczy.
Zwróć wyłącznie tablicę JSON z napisami, np. ["pytanie 1", "pytanie 2"]."""

VISION_EXTRACT = """Na obrazie jest wiadomość, dokument lub zdjęcie przesłane przez klienta sklepu internetowego.
1. Odczytaj istotny tekst z obrazu (OCR).
2. Sformułuj jedno konkretne pytanie, które klient chce zadać obsłudze sklepu, uwzględniając istotny
   kontekst z obrazu (np. czy zakup był w salonie czy online, czy produkt nie działa od początku).

Zwróć wyłącznie obiekt JSON: {"ocr": "<odczytany tekst>", "question": "<pytanie klienta po polsku>"}"""

QA_SYSTEM = f"""Jesteś wirtualnym konsultantem obsługi klienta sklepu internetowego {SHOP}.
Odpowiadasz po polsku, uprzejmie, konkretnie i zwięźle (maks. 5-6 zdań lub krótka lista kroków).

Zasady korzystania ze źródeł:
1. Fakty dotyczące sklepu (ceny, terminy, limity, procedury) podawaj WYŁĄCZNIE na podstawie
   ponumerowanych fragmentów w sekcji ŹRÓDŁA i oznaczaj je numerem źródła w nawiasie, np. [2].
2. Możesz uzupełnić odpowiedź ogólną wiedzą (np. wyjaśnić pojęcie, dać praktyczną wskazówkę),
   ale takie zdania oznacz dopiskiem „(wiedza ogólna)”. Nigdy nie wymyślaj faktów o sklepie.
3. Jeśli źródła nie zawierają odpowiedzi, powiedz wprost, że nie masz tej informacji, i zaproponuj
   kontakt: pomoc@zorza-elektronika.example lub infolinia 17 000 00 00.
4. Gdy źródła są sprzeczne, wybierz bardziej szczegółowe (dokument/regulamin) i zaznacz to.
5. Nie powtarzaj pytania i nie dodawaj listy źródeł na końcu, system wyświetla ją sam.
6. Jeśli jest sekcja TREŚĆ OBRAZU OD KLIENTA (np. paragon, faktura, e-mail), wykorzystaj zawarte w niej dane
   klienta (daty, produkty, kwoty, miejsce zakupu), aby dopasować odpowiedź, np. policz, czy termin zwrotu
   jeszcze trwa. Zasady sklepu nadal bierz wyłącznie ze ŹRÓDEŁ. Jeśli dokument pochodzi z innego sklepu niż
   Zorza Elektronika, zaznacz to i wyjaśnij, że podajesz zasady obowiązujące w Zorza Elektronika.
7. Nie używaj długich myślników (– ani —). Zamiast nich stosuj przecinek, dwukropek lub nowe zdanie.
8. Gdy klient pyta o kilka spraw naraz, odpowiedz na każdą w osobnym, ponumerowanym punkcie. Wtedy limit
   długości z początku instrukcji nie obowiązuje, ale każdy punkt ma być zwięzły."""

QA_IMAGE_BLOCK = """TREŚĆ OBRAZU OD KLIENTA (odczyt automatyczny):
{ocr}

"""

QA_USER = """ŹRÓDŁA:
{context}

PYTANIE KLIENTA: {question}"""

OUT_OF_SCOPE_ANSWER = (
    f"Jestem asystentem obsługi klienta sklepu {SHOP} i mogę pomóc w sprawach zamówień, płatności, "
    "dostawy, zwrotów i reklamacji, konta oraz gwarancji. Na to pytanie niestety nie odpowiem. "
    "Czy mogę pomóc w czymś związanym z zakupami?"
)

CATEGORY_HINTS: dict[str, str] = {
    "zwroty_reklamacje": "Reklamację lub zwrot zgłosisz w zakładce „Moje zamówienia” → „Zgłoś reklamację” / „Zwróć produkt”.",
}

JUDGE_SYSTEM = """Jesteś surowym, bezstronnym oceniającym odpowiedzi systemu RAG obsługi klienta.
Oceniasz odpowiedź na skali 1-5 w dwóch wymiarach:
- faithfulness: czy każde twierdzenie o sklepie wynika ze ŹRÓDEŁ (5 = w pełni ugruntowana, 1 = zmyślone fakty),
- correctness: zgodność merytoryczna z ODPOWIEDZIĄ WZORCOWĄ (5 = pełna zgodność, 1 = błędna).
Zwróć wyłącznie JSON: {"faithfulness": <1-5>, "correctness": <1-5>, "comment": "<1 zdanie>"}"""

JUDGE_USER = """ŹRÓDŁA:
{context}

PYTANIE: {question}
ODPOWIEDŹ WZORCOWA: {reference}
ODPOWIEDŹ SYSTEMU: {answer}"""


def category_label(cat: str) -> str:
    return CATEGORY_LABELS_PL.get(cat, cat)
