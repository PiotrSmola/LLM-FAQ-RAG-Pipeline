"""Definicja komórek notebooka 03, ewaluacja pipeline'u (retrieval, QA, ablacja, przykłady)."""

CELLS = [
    ("md", """# 03. Ewaluacja pipeline'u: retrieval → QA → synergia komponentów

Zbiór ewaluacyjny `data/eval/qa_eval.jsonl`: **40 pytań**, 37 o sklepie (odpowiedź w FAQ, tylko w dokumentach
albo w obu) oraz 3 poza zakresem. Każde pytanie ma listę źródeł wzorcowych i odpowiedź wzorcową.
Wyniki liczy `scripts/evaluate.py` (tu tylko wczytujemy i interpretujemy).

Metryki:
* **retrieval**: Recall@k (czy źródło wzorcowe jest w top-k) i MRR (średnia odwrotność pozycji pierwszego trafienia),
* **QA**: Token-F1 (Lab 3), ROUGE-L, BERTScore-F1 (Lab 4), **LLM-as-judge** (Gemma ocenia 1-5 *faithfulness*
 : ugruntowanie w źródłach, i *correctness*, zgodność z wzorcem), poprawność odmowy poza zakresem."""),
    ("code", """import pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.max_colwidth", 160)
R = "../results\""""),
    ("md", "## 1. Retrieval: metoda wyszukiwania i wpływ routingu klasyfikatorem"),
    ("code", """ret = pd.read_csv(f"{R}/retrieval/retrieval_metrics.csv")
display(ret.round(3)); display(Image(f"{R}/retrieval/retrieval_mrr.png", width=700))"""),
    ("md", "## 2. QA: ablacja i porównanie z modelami z laboratoriów"),
    ("code", """qa = pd.read_csv(f"{R}/qa/qa_summary.csv"); display(qa)
display(Image(f"{R}/qa/qa_ablacja.png", width=900))"""),
    ("md", "## 3. Przykłady: to samo pytanie, różne warianty"),
    ("code", """det = pd.read_csv(f"{R}/qa/qa_details.csv")
for qid in ["Q11", "Q30", "Q18"]:
    sub = det[det.id == qid]
    print("=" * 100); print(sub.pytanie.iloc[0]); print("WZORZEC:", sub.referencja.iloc[0]); print()
    for _, r in sub.iterrows():
        print(f"[{r.wariant}] (F={r.get('faithfulness')}, C={r.get('correctness')})\\n   {str(r.odpowiedz)[:400]}\\n")"""),
    ("md", "## 4. Gdzie pełny pipeline się myli?"),
    ("code", """full = det[(det.wariant == "pelny_pipeline") & (det.kategoria != "poza_zakresem")]
full.sort_values("correctness")[["id", "pytanie", "odpowiedz", "correctness", "faithfulness", "trafienie_zrodla"]].head(6)"""),
    ("md", "## 5. Pytania poza zakresem"),
    ("code", """det[det.kategoria == "poza_zakresem"][["pytanie", "wariant", "odmowa_poprawna", "odpowiedz"]]"""),
    ("md", """## Wnioski: synergia komponentów
* **Retrieval:** wyszukiwanie semantyczne (bge-m3) bije metody leksykalne (MRR 0.88 vs 0.71-0.76), bo rozumie
  parafrazy i potoczne sformułowania. Hybryda RRF ma najlepszy Recall@3/@4 (1.00 bez filtra), w RAG liczy się,
  by właściwy fragment był *gdziekolwiek* w kontekście, dlatego jest domyślną metodą.
* **Routing klasyfikatorem** podnosi MRR każdej metody o 0.03-0.05: mniej konkurencyjnych wpisów z innych kategorii.
  Koszt: przy błędnej klasyfikacji filtr może odciąć właściwy wpis (hybryda: Recall@4 1.00 → 0.97).
* **RAG vs sam LLM:** bez dokumentów model płynnie zmyśla zasady sklepu (faithfulness 1.9/5, correctness 2.0/5)
 : to te same halucynacje, które badaliśmy w Lab 5. Z RAG: 4.6-4.9 / 4.7.
* **Klasyfikator w pipeline'ie** zwiększa ugruntowanie odpowiedzi (faithfulness 4.60 → 4.89) i pozwala odmawiać
  na pytania spoza zakresu bez wywołania LLM.
* **Metryki automatyczne:** BERTScore prawie nie odróżnia halucynacji od poprawnej odpowiedzi (0.71 vs 0.74),
  a ROUGE/Token-F1 karzą poprawne odpowiedzi sformułowane innymi słowami (Lab 4). Ocena ugruntowania
  (LLM-as-judge) lepiej oddaje jakość RAG, z zastrzeżeniem, że oceniający model jest tym samym, który odpowiada.
* **Modele z laboratoriów:** FLAN-T5 (angielski) i extractive QA nie radzą sobie z polskimi pytaniami wymagającymi
  łączenia faktów lub obliczeń (np. 15% z 4000 zł), uzasadnia to użycie lokalnego LLM jako generatora."""),
]
