"""Definicja komórek notebooka 02, etap 1: klasyfikator routingu (fine-tuning BERT)."""

CELLS = [
    ("md", """# 02. Etap 1: klasyfikacja zapytań (routing), fine-tuning BERT

Temat 29 wymaga *„fine-tuningu BERT do kierowania zapytania”*. Klasyfikator przypisuje pytanie klienta do jednej
z 6 kategorii: 5 tematycznych (płatności, dostawa, zwroty i reklamacje, konto, produkty i gwarancja) oraz
**poza zakresem**. Kategoria steruje dalszym przebiegiem pipeline'u:
* zawęża wyszukiwanie wpisów FAQ do danej kategorii (dokumenty firmowe przeszukiwane są zawsze),
* „poza zakresem” → grzeczna odmowa **bez wywołania LLM** (brak ryzyka halucynacji, oszczędność czasu),
* niska pewność (< 0.55) → wyszukiwanie w całej bazie.

**Dane**
* trening: pytania z FAQ + **pytania wygenerowane automatycznie przez LLM** (Gemma, `scripts/generate_training_data.py`),
* test: **90 pytań napisanych ręcznie** (potoczny język, literówki, brak polskich znaków), rozłączne z treningiem.

**Porównywane modele** (schemat jak w Lab 2): HerBERT (polski BERT, fine-tuning), DistilBERT multilingual
(model z Lab 2 w wersji wielojęzycznej), TF-IDF + regresja logistyczna (baseline z Lab 2, słowa i n-gramy znakowe),
LLM zero-shot (Gemma 4 z promptem few-shot)."""),
    ("code", """import json, pandas as pd, matplotlib.pyplot as plt
from collections import Counter
from IPython.display import Image, display
R = "../results/classifier"
train = pd.read_json("../data/classifier/train.jsonl", lines=True)
test = pd.read_json("../data/classifier/test.jsonl", lines=True)
print(f"trening: {len(train)} przykładów ({(train.source=='llm').sum()} z LLM, {(train.source=='faq').sum()} z FAQ), test: {len(test)}")
pd.concat([train.label.value_counts().rename("trening"), test.label.value_counts().rename("test")], axis=1)"""),
    ("code", """# przykładowe pytania wygenerowane przez LLM (automatyzacja)
train[train.source == "llm"].groupby("label").head(3)[["label", "text"]]"""),
    ("md", "## Wyniki porównania na ręcznym zbiorze testowym"),
    ("code", """cmp = pd.read_csv(f"{R}/comparison.csv").sort_values("f1_macro", ascending=False)
cmp.style.format({"accuracy": "{:.3f}", "f1_macro": "{:.3f}", "latency_ms_per_query": "{:.1f}"})"""),
    ("code", """fig, ax = plt.subplots(1, 2, figsize=(12, 4))
cmp.plot.barh(x="model", y=["accuracy", "f1_macro"], ax=ax[0]); ax[0].set_xlim(0, 1); ax[0].set_title("Jakość")
cmp.plot.barh(x="model", y="latency_ms_per_query", ax=ax[1], logx=True, legend=False, color="tab:gray")
ax[1].set_title("Czas na zapytanie [ms] (skala log)"); plt.tight_layout(); plt.show()"""),
    ("md", "## Macierz pomyłek: HerBERT"),
    ("code", """display(Image(f"{R}/confusion_bert_herbert-faq.png", width=520))"""),
    ("md", "## Analiza błędów"),
    ("code", """err = pd.read_csv(f"{R}/errors.csv")
print(err.groupby("model").size().rename("liczba błędów").to_string())
err[err.model.str.contains("herbert")][["text", "true", "pred"]]"""),
    ("code", """# pytania, które myli każdy model (trudne przypadki)
hard = err.groupby("text").model.nunique()
err[err.text.isin(hard[hard >= 3].index)].drop_duplicates("text")[["text", "true", "pred"]]"""),
    ("md", """## Learning curve: ile danych potrzeba? (jak zad. 2.4 z Lab 2)
Fine-tuning HerBERT na ok. 50 / 100 / 200 / 400 / 800 przykładach (próbkowanie warstwowe, 3 seedy, średnia ± odchylenie)
w porównaniu z TF-IDF na tych samych podzbiorach."""),
    ("code", """lc = pd.read_csv(f"{R}/learning_curve.csv")
display(lc.groupby("size")[["f1_bert", "f1_tfidf"]].agg(["mean", "std"]).round(3))
display(Image(f"{R}/learning_curve.png", width=560))"""),
    ("md", """## Wnioski
* **HerBERT (fine-tuning)** osiąga F1-macro ≈ 0.89 przy ~10 ms na zapytanie, wystarczająco dobrze i szybko,
  by pełnić rolę routera. Polski model wyraźnie wygrywa z wielojęzycznym DistilBERT (≈ 0.77), znaczenie ma
  słownik i pre-trening na języku docelowym (porównaj Lab 2, gdzie angielski DistilBERT działał na angielskich danych).
* **TF-IDF na n-gramach znakowych** (≈ 0.84) jest zaskakująco mocnym baselinem, n-gramy znakowe są odporne na
  literówki, brak polskich znaków i fleksję. Klasyczny TF-IDF na słowach (jak w Lab 2) jest dużo słabszy (≈ 0.72).
* **LLM zero-shot** klasyfikuje bezbłędnie, ale jest ~30× wolniejszy od HerBERT i wymaga działającego serwera LLM.
  Warto pamiętać, że dane treningowe wygenerował ten sam LLM, a opisy kategorii w prompcie są bardzo precyzyjne.
* **Learning curve:** przy małej liczbie przykładów (≤ 200) fine-tuning przegrywa z TF-IDF, za mało kroków
  optymalizacji, by dostroić 124 mln parametrów. Od ~400 przykładów BERT wyprzedza baseline. Automatyczne
  generowanie danych przez LLM (720 przykładów w ~3,5 min) rozwiązało problem braku danych.
* **Kalibracja:** HerBERT myli się z wysoką pewnością (większość błędów > 0.85), więc próg pewności tylko częściowo
  chroni przed złym routingiem: dlatego dokumenty firmowe są przeszukiwane zawsze, niezależnie od kategorii.
* Większość błędów to pytania leżące na granicy kategorii (np. zwrot pieniędzy: *płatności* czy *zwroty*?)."""),
]
