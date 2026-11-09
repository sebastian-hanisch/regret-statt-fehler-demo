# Regret statt Fehler: die beste Prognose bestellt nicht die beste Menge – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-regret-statt-fehler-demo.streamlit.app/)**

Erstes Stück der Reihe **Entscheidungsorientiertes Lernen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es legt die Grundlage: den **Regret** (Kosten der gewählten Entscheidung minus Kosten der besten Entscheidung bei bekannter Verteilung) und die **zweistufige Pipeline** (erst eine Prognose, dann eine Regel, die daraus eine Menge macht), am **Newsvendor mit Merkmalen**. Vehikel ist eine einzige Bestellmenge je Merkmalszeile; Verfahren, die auf den Kosten statt auf dem Fehler trainieren, und Entscheidungen mit Löser sind nicht Teil dieser Demo.

Verwandte Demos: [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo) zeigt denselben Mechanismus an einem Hafen (Prognose, dann Einlagerungsregel); [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo) bestimmt Sicherheitsbestände aus der Prognoseunsicherheit bei Zeitreihen; [newsvendor-evpi-demo](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) kennt die Verteilung und fragt nach dem Wert der Information, hier ist sie unbekannt und wird aus Daten geschätzt. Zahlen aus diesen Demos werden hier nicht wiederholt.

## Warum dieses Problem

Eine Prognose wird meist danach bewertet, wie nah sie an der Wahrheit liegt (Quadratfehler). Die Entscheidung, die man daraus ableitet, hat aber eigene Kosten. Beim Newsvendor kostet jede fehlende Einheit **b**, jede überschüssige **h**; bei ungleichen Kosten ist die beste Menge im Allgemeinen nicht das Mittel der Nachfrage, sondern ihr **τ-Quantil** mit τ = b / (b + h). Die Demo fragt: Was kostet es, das Mittel zu bestellen, wenn das Modell **perfekt** ist, und wie verhält sich das mit mehr Daten?

## Modell

- **Kosten:** C(q, D) = b · max(D − q, 0) + h · max(q − D, 0), kritisches Verhältnis τ = b / (b + h). Das Handbeispiel hat b = 4, h = 1, also τ = 0,8.
- **Beste Menge:** das τ-Quantil der bedingten Nachfrage. Bei Gleichstand (F = τ auf einer Strecke) nennt die Demo die kleinste optimale Menge und zeigt die Strecke (Toleranz 1e-9 auf die Kosten).
- **Regret** einer Regel: erwartete Kosten der gewählten Menge minus erwartete Kosten der besten Menge, gemittelt über Testpunkte; die Demo nennt ihn in **Prozent Mehrkosten** gegen die beste Menge.
- **Zelle (Experiment 1):** feste Nachfragen, jede gleich wahrscheinlich: das **Handbeispiel** 2, 4, 6, 8, 10 und eine **schiefe Zelle** 3, 4, 4, 5, 6, 24.
- **Welt (Experimente 2 bis 4):** D = max(0, 60 + 10 · β′x + 8 · z), x standardnormal mit p ∈ {3, 10} Merkmalen, β gleichverteilt in [−1, 1]^p und auf Länge 1 normiert, z ~ N(0, 1). Das lineare Modell ist richtig spezifiziert, die Streuung konstant. Dieselbe Welt wie die Vormessung `eo_v2_news.py` (Welt „linear“).
- **Regeln:** *Mittel bestellen* (q = Prognose ŷ des Kleinste-Quadrate-Modells) und *KQ plus Normalquantil* (q = ŷ + z_τ · σ̂, σ̂ die gepoolte Reststreuung mit n − p − 1 Freiheitsgraden). Bestellmengen sind nicht negativ (q wird bei 0 abgeschnitten); das betrifft in den Läufen dieser README 20 von 14,4 Millionen Testmengen (6 Zellen (p, n) mit je 30 Seeds, 20 000 Testpunkten und vier Mengen: Prognose sowie Quantilregel bei drei τ), die kleinste ist −13,3.
- **Kosten der Entscheidung** werden je Testpunkt **exakt** aus der Normalverteilung gerechnet (geschlossene Form mit Stutzung bei 0), nicht an gezogenen Nachfragen; 20 000 Testpunkte je Seed. Das Orakel (wahre Quantilregel) bestellt max(0, μ(x) + 8 · z_τ).

Formeln und Herleitungen: Expander „📐 Mathematische Formulierung“ in der App.

## Methodik

Vier Experimente in der App:

1. **Eine Zelle von Hand:** Kostenkurve über der Menge; Mittel, τ-Quantil und beste Menge (Strecke) markiert. Voreinstellung ist das Handbeispiel.
2. **Perfektes Modell, teure Entscheidung (Kernfrage):** Balken „Mittel bestellen“ gegen „KQ plus Normalquantil“, Median und Minimum bis Maximum über **30 feste Seeds** (0 bis 29), für das τ aus b und h sowie die Referenzwerte 0,5 / 0,8 / 0,95; p und n wählbar. Dazu eine Einzelinstanz mit Zufalls-Seed und das „wahre Mittel bestellen“ (ohne Schätzfehler). Die Rechnung läuft live (je Zelle (p, n) einige Sekunden, danach zwischengespeichert).
3. **Prognosefehler gegen Regret:** Die Prognose wird um s Reststreuungen vom wahren Mittel verschoben; Quadratfehler (Minimum bei s = 0) und Kosten (Minimum bei s = z_τ) nebeneinander. Geschlossen für die Normalverteilung ohne Stutzung.
4. **Lernkurve:** Mehrkosten über n = 30 / 100 / 1000 für beide Regeln, mit der Untergrenze des Mittels als Satz im Verdikt.

Der Zufall kommt aus einem ganzzahligen SplitMix64-Strom (Zustand und Ausgabe nur aus Seed und Position, auf jeder Plattform gleich); `numpy.random` und Python-`hash()` kommen nicht vor.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieser Tabelle stehen in `tests/test_claims.py` (Spalte „Test“ der Tabelle weiter unten). „Median [Minimum; Maximum]“ meint 30 feste Seeds, Welt linear, 20 000 Testpunkte je Seed.

| Frage | Befund |
|---|---|
| Handbeispiel (b = 4, h = 1, Nachfragen 2, 4, 6, 8, 10) | Das Mittel 6 kostet 6,00; die Mengen 8 bis 10 kosten je 4,00 (F(8) = 0,8 = τ, jede Menge von 8 bis 10 ist optimal); **die Mitte kostet 50,0 % mehr**. Mit b = 19: Mittel 24,00, beste Menge 10 mit 4,00, **500,0 % mehr**. |
| Schiefe Zelle (3, 4, 4, 5, 6, 24; b = h = 1) | Mittel 7,67 kostet 5,44, die beste Menge (4 bis 5) kostet 4,00: **36,1 % mehr**, obwohl die Kosten symmetrisch sind (das Mittel ist hier nicht der Median). |
| Das exakte Mittel als Menge (Normalverteilung ohne Stutzung) | Mehrkosten exakt **e^(z²/2) − 1**: 0 / 42,5 / 286,8 % bei τ = 0,5 / 0,8 / 0,95; unabhängig von μ, σ und, bei festem τ, von b und h. In der Welt dieser Demo liegt das wahre Mittel in allen 18 Zellen (p, n, τ) und in jedem der 30 Seeds auf dieser Formel (Abweichung höchstens 0,01 Punkte). |
| Mittel bestellen, p = 3, n = 1000 | 0,1 [0,0; 0,5] / **42,5** [36,7; 47,5] / **286,7** [262,7; 306,6] % bei τ = 0,5 / 0,8 / 0,95 |
| KQ plus Normalquantil, p = 3, n = 1000 | 0,1 [0,0; 0,5] / 0,1 [0,0; 0,6] / 0,2 [0,0; 0,8] % |
| Dasselbe bei p = 10, n = 1000 | Mittel bestellen 0,5 [0,3; 1,3] / 42,9 [36,8; 48,2] / 287,5 [263,0; 308,0] %; KQ plus Normalquantil 0,5 / 0,5 / 0,6 % (Maximum 1,3) |
| Datenmenge, p = 3, τ = 0,8 | Mittel bestellen n = 30 / 100 / 1000: 50,2 / 47,3 / 42,5 %; KQ plus Normalquantil: 6,4 / 2,3 / 0,1 %. Bei τ = 0,95: Mittel 312,9 / 301,6 / 286,7 %, Quantilregel 10,0 / 3,0 / 0,2 %. |
| Wenig Daten, viele Merkmale (p = 10, n = 30) | KQ plus Normalquantil 21,2 / 29,0 / 38,2 % bei τ = 0,5 / 0,8 / 0,95, Mittel bestellen 21,2 / 78,9 / 390,8 %; Maximum der Bänder bei τ = 0,95: 100,7 % (Quantilregel) und 559,4 % (Mittel). |
| Verschiebung der Prognose (Experiment 3, Normalverteilung ohne Stutzung) | Quadratfehler σ² (1 + s²), Kosten minimal bei s = z_τ (0,84 bei τ = 0,8; 1,64 bei τ = 0,95). Die Prognose mit dem kleinsten Quadratfehler (s = 0) kostet **42,5 / 286,8 %** mehr als die beste Menge; die beste Menge hat einen um **70,8 / 270,6 %** größeren Quadratfehler. |

**Zählungen** (zu den Urteilen der App und dieser README; je ein Test):

- Die Quantilregel liegt bei n = 1000 im Median in **6 von 6** Zellen (p ∈ {3, 10}, τ ∈ {0,5; 0,8; 0,95}) unter 1 % Mehrkosten (größter Median 0,6 %); über die Seeds reicht das Band bis 1,3 %.
- Bei τ = 0,5 sind beide Regeln in **6 von 6** Zellen (p × n) gleich, weil z_0,5 = 0 ist.
- Bei τ = 0,8 und 0,95 liegt das Mittel in **12 von 12** Zellen (p × n × τ) im Median mindestens 42 Punkte über der Quantilregel (kleinster Abstand 42,3 Punkte bei p = 10, n = 1000, τ = 0,8).
- Bei n = 1000 liegt der Median des Mittels in den vier Zellen (p ∈ {3, 10}, τ ∈ {0,8; 0,95}) höchstens 0,8 Punkte über dem Wert e^(z²/2) − 1 des exakten Mittels; bei n = 30 und n = 100 liegt er in **8 von 8** Zellen darüber.
- Der Median der Quantilregel fällt mit n = 30, 100, 1000 in **6 von 6** Reihen (p × τ) streng.

**Vergleich mit der Vormessung** (`out_eo_v2_news.txt`, Zufallsstrom von numpy statt SplitMix64, Kosten an gezogenen Nachfragen, ebenfalls 30 Seeds): Für Mittel bestellen nennt sie bei p = 3, n = 1000: 0,2 / 44,2 [34,8; 48,8] / 292,9 [256,9; 312,5] %; KQ plus Normalquantil 0,2 / 0,2 / 0,2 %. In allen 18 Zellen (p, n, τ) und beiden Regeln, also 36 Vergleichen, liegt der Median jeder Messung im Minimum-Maximum-Band der anderen: 36 von 36 in beide Richtungen. Die Vormessung ist in der App im Expander „Vormessung mit anderem Zufallsstrom“ eingefroren abgelegt (`vormessung_eo_v2.json`), nicht live gerechnet.

### Zahl → Test

| Zahl in der README | Test (`tests/test_claims.py`, sonst genannt) | Art |
|---|---|---|
| 6,00 / 4,00 / 50,0 %; 8 bis 10 | `test_handbeispiel_zahlen_der_readme`, `test_algorithm.py` | exakt von Hand |
| 24,00 / 500,0 %; 2,40 | `test_zelle_bei_tau_095_und_tau_halb_zahlen_der_readme` | exakt von Hand |
| 5,44 / 36,1 % | `test_schiefe_zelle_zahlen_der_readme` | exakt von Hand (49/9, 13/36) |
| e^(z²/2) − 1 = 0 / 42,5 / 286,8 % | `test_mehrkosten_des_exakten_mittels_exp_z2_halbe`, Orakel `test_mehrkosten_des_exakten_mittels_gegen_aufzaehlung` | exakt, Aufzählung |
| wahres Mittel in 18 Zellen auf der Formel | `test_exaktes_mittel_in_der_welt_liegt_in_jedem_seed_auf_der_formel` | Messung, Toleranz 0,01 |
| Tabelle p, n, τ (Median, Minimum, Maximum) | `test_tabelle_der_messung_dieser_demo` (Toleranz 0,06 wegen Rundung) | Messung, deterministisch |
| 36 von 36, Vormessungszahlen | `test_vormessung_und_nachbau_…`, `test_vormessung_zahlen_der_readme` | Zählung, eingefrorene Datei |
| 6 von 6, 12 von 12, 8 von 8, höchstens 0,8 Punkte | die gleichnamigen `test_…`-Funktionen | Zählung |
| 21,2 / 29,0 / 38,2 | `test_wenig_daten_viele_merkmale_p10_n30_tau095` | Messung |
| Lernkurve 50,2 / 47,3 / 42,5 usw. | `test_lernkurve_zahlen_der_readme` | Messung |
| z = 0,84 / 1,64; 42,5 / 286,8 %; 70,8 / 270,6 % | `test_verschiebung_zahlen_der_readme` | exakt, geschlossene Form |

Zeiten stehen nirgends als feste Sekundenzahl.

## Befunde und Korrekturen gegenüber dem Plan

- **„Mittel bestellen hängt nicht an n“ stimmt nur für die Untergrenze.** Der Plan nannte 44,2 → 44,4 → 48,2 % (n = 1000 / 100 / 30). Der Nachbau misst bei p = 3, τ = 0,8: 42,5 / 47,3 / 50,2 %, bei τ = 0,95: 286,7 / 301,6 / 312,9 %. Der Anteil, der an der **Regel** hängt, ist eine feste Zahl, e^(z²/2) − 1 (42,5 beziehungsweise 286,8 %), die ein perfekt bekanntes Mittel im Erwartungswert nicht unterschreitet; mit mehr Daten nähert sich der Median dieser Zahl (einzelne Seeds liegen schon bei n = 1000 darunter oder darüber, der Median bei p = 3, n = 1000, τ = 0,95 mit 286,7 % knapp darunter). Dass der Median bei kleinem n darüber liegt, ist gemessen (8 von 8 Zellen); dass der Schätzfehler die Ursache ist, ist eine Deutung (die exakte Mittelregel liegt in allen Seeds auf der Formel).
- **Zahlen weichen von der Vormessung ab, Bänder überlappen.** 42,5 / 286,7 statt 44,2 / 292,9 bei p = 3, n = 1000; n = 30 und 100 für KQ plus Normalquantil 4,5 / 6,4 / 10,0 und 2,2 / 2,3 / 3,0 statt 5,4 / 7,0 / 10,6 und 2,0 / 2,0 / 2,4. Gründe (nicht weiter zerlegt): anderer Zufallsstrom, Kosten exakt statt an 20 000 gezogenen Nachfragen. Es gilt die neue Messung; 36 von 36 Mediane liegen im Band der jeweils anderen.
- **Mehrkosten sind nie negativ.** Die Vormessung zeigte Minima von −0,0 (Rauschen der gezogenen Nachfragen); mit exakten erwarteten Kosten ist jede Regel je Testpunkt mindestens so teuer wie die wahre Quantilregel (`tests/test_oracle_regret_statt_fehler.py`).
- **Experiment 2 wird live gerechnet statt vorberechnet** (Plan: „vorberechnet“): OLS und geschlossene Kosten sind billig, daher gelten die Regler p, n und τ frei; τ ist beliebig aus b und h, nicht nur 0,5 / 0,8 / 0,95. Die Vormessung bleibt als eingefrorene Vergleichstabelle.
- **Experiment 4** nennt im Verdikt die Untergrenze e^(z²/2) − 1 (neu gegenüber dem Plan; Herleitung im Expander, Aufzählung im Orakel-Test).
- **Gleichstand (Entscheidung Z4):** wie geplant die kleinste optimale Menge mit Anzeige der Strecke. Handbeispiel (Mitte 6,00, Mengen 8 und 10 je 4,00, Mehrkosten 50,0 %) aus dem Plan bestätigt.

## Ehrliche Grenzen

- **Eine Welt:** Normalrauschen, Stutzung bei 0, lineares und **richtig spezifiziertes** Modell, konstante Streuung, p ∈ {3, 10}, n ∈ {30, 100, 1000}, eine Periode ohne Kopplung zwischen Merkmalszeilen. Keine Messung mit Echtdaten.
- **„KQ plus Normalquantil“ ist hier die richtige Regel**, nicht die Regel schlechthin: In der Vormessung (Welt „Streuung linear“, p = 3, n = 1000, τ = 0,95, `out_eo_v2_news.txt`) kostet sie 33,7 % mehr als die wahre Quantilregel (eine Streuungsform, eine Messung); diese Welt ist nicht in der Demo nachgebaut.
- **Die Untergrenze e^(z²/2) − 1 gilt für Normalverteilung und nur im Erwartungswert (für das exakte Mittel).** Für andere Verteilungen ist sie nicht gemessen; einzelne Seeds mit geschätzter Prognose können darunter liegen.
- **Die Bänder sind breit.** Der Median sagt wenig über eine einzelne Instanz: p = 3, n = 30, τ = 0,95: Mittel bestellen 155,0 bis 490,6 %, Quantilregel 1,1 bis 35,6 %.
- **Zwei Regeln, kein vollständiger Methodenvergleich:** das Mittel und die Quantilregel mit Normalquantil; empirisches Residuenquantil und direktes Lernen der Menge sind nicht in der App.
- **Das Mittel in der schiefen Zelle** (Experiment 1) zeigt, dass „bei symmetrischen Kosten genügt das Mittel“ eine Aussage über symmetrische Nachfrageverteilungen ist (36,1 % Mehrkosten in der schiefen Zelle, 0,1 % bei p = 3, n = 1000 in der Welt).
- **Rechenzeit:** die erste Rechnung je Zelle (p, n) dauert einige Sekunden, danach zwischengespeichert; die Größenordnung hängt vom Rechner ab.

## Verifikation

- **Kern:** Zellkosten und beste Menge gegen exakte Bruchrechnung und Aufzählung über ein Mengenraster (je 60 Zellen); Handbeispiel und Zelle mit Ausreißer von Hand; Kleinste Quadrate gegen Normalgleichungen und Hand-Regression.
- **Orakel** (`tests/test_oracle_regret_statt_fehler.py`): erwartete Kosten mit und ohne Stutzung gegen Aufzählung über 200 000 gleich wahrscheinliche Nachfragen; wahres Quantil gegen Monte Carlo (400 000 Ziehungen, 1 %) und gegen Aufzählung der Kostenfunktion bei Stutzung; SplitMix64 gegen skalare Referenz und den veröffentlichten ersten Ausgabewert; Eigenschaften (Mehrkosten ≥ 0, Orakelregel billigste Menge je Testpunkt, Skalierung der Kosten, Verschiebung der Nachfrage, Verschiebung der Prognose).
- **Randfälle** (`tests/test_edge_cases.py`): n = 1 und n = 2, Gleichstand und Toleranz, τ nahe 0 und 1, σ gegen 0, Stutzung bei 0 aktiv, Merkmal ohne Wirkung, τ = 0,5.
- **Verdikte** als reine Funktionen (`tests/test_verdicts.py`): jede Schwelle (10 % Mehrkosten, 5 Punkte Unterschied) und jede Zahl im Text; je Preset die erwartete Verdikt-Zeile (`tests/test_presets.py`).
- **Claims:** `tests/test_claims.py` (Tabelle oben).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Zahlenregler am Minimum und Maximum), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf), Footer, Laufzeit-Importe.
- Die Tests prüfen Rechnungen und Bänder; sie ersetzen keine Prüfung im Browser.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, plotly |
| `rsf_algorithm.py` | Kern: Zellkosten, beste Menge, erwartete Kosten der Normalverteilung, Kleinste Quadrate, Regeln, Simulation, Verschiebung |
| `rsf_constants.py` | Regler-Grenzen, Welt, Zellen, Farben, Presets |
| `rsf_evaluation.py` | Analysen, Sweeps, Verdikte (reine Funktionen), Zahlenformat |
| `rsf_layout.py` | Fester Kopf (Stylesheet) |
| `rsf_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `rsf_rng.py` | SplitMix64-Zufallsstrom (Herkunft im Modulkopf) |
| `rsf_scenario.py` | Zelle und Welt, Ziehen von Merkmalen und Nachfragen |
| `rsf_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `vormessung_eo_v2.json` | eingefrorene Zeilen der Vormessung (Welten „linear“ und „Streuung linear“), nicht live |
| `tests/` | Handrechnung, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Direktes Lernen der Menge auf den Kosten** (Quantilregression, Pinball-Verlust), **Regularisierung** und **flexible Modelle** (Gewichte aus Nachbarn oder Wäldern): eigene Demos der Reihe; hier nur der Vergleich „Mittel“ gegen „KQ plus Quantil“.
- **Entscheidungen mit Löser** (Weg im Netz): hier keine Nebenbedingungen.
- **Merkmalsabhängige Streuung, nichtlineare Wahrheit, Fehlspezifikation:** die Vormessung hat sie gemessen (`out_eo_v2_news.txt`), die Demo bildet sie nicht ab.
- **PDF-Export und Echtdaten:** nicht enthalten.
- **Literaturaussagen:** die Demo stützt keine Aussage auf eine Quelle.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
