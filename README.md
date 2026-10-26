# Welches Modell erzeugt dieses Netz? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-modellwahl-netze-demo.streamlit.app/)** -->

Neuntes und letztes Stück der **Netzmodelle-Linie** (Graphentheorie) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, der Zusammenfluss der Linie. Die Linie erzeugt Netze aus Zufallsmodellen und prüft sie gegen Theorie und Daten;
dieses Stück fragt, **ob man einem Netz ansehen kann, aus welchem Modell es stammt, und was geschieht, wenn es aus keinem der bekannten stammt**. Die sechs Familien kommen aus [zufallsgraph-demo](https://github.com/sebastian-hanisch/zufallsgraph-demo) (Erdős–Rényi), [konfigurationsmodell-demo](https://github.com/sebastian-hanisch/konfigurationsmodell-demo) (Kantentausch),
[wachstumsmodelle-demo](https://github.com/sebastian-hanisch/wachstumsmodelle-demo) (Barabási–Albert) und [raeumliche-netze-demo](https://github.com/sebastian-hanisch/raeumliche-netze-demo) (geometrisch); das Blockmodell steht auch in [stochastic-block-demo](https://github.com/sebastian-hanisch/stochastic-block-demo), wo „Modularität ist nicht Wahrheit“ gezeigt wird.

Die Demo gibt **zwei Antworten nebeneinander**. Ein **Klassifikator** (die 15 nächsten von 360 Trainingsnetzen im Raum von sieben Zusammenfassungsgrößen) **muss immer eine der sechs Familien nennen**.
Eine **Anpassungsprüfung** passt jedes der sechs Modelle an das Netz an, zieht Netze daraus und fragt, ob das beobachtete Netz in dieser Wolke unauffällig liegt; sie darf „keines davon“ sagen. Getestet wird an erzeugten Netzen (Herkunft bekannt), an einem Netz aus einem **siebten Modell** (Holme–Kim, nicht auf der Liste) und an drei echten Netzen.
Der Kern ist eigener Code; `networkx` und `scikit-learn` kommen nur als Gegenprobe in die Tests.

## Warum dieses Problem

Fragt man ein Klassifikationsverfahren, welches Modell ein Netz erzeugt hat, antwortet es immer; die Antwort sagt nichts darüber, ob irgendein Modell der Liste passt. Ein Netz aus dem Alltag gehört fast nie genau zu einer der Familien, die man gerade zur Hand hat.
Die Frage der Demo: **Wie weit tragen sieben einfache Größen bei der Wahl der Familie, und woran erkennt man, dass keine passt?**

## Modell

**Sieben Größen.** Größter Grad / mittlerer Grad, Varianz / Mittel des Grades, Transitivität (3 · Dreiecke / verbundene Dreiergruppen), Grad-Assortativität (Korrelation der Grade an den zwei Enden einer Kante), mittlere Weglänge in der größten Komponente, Modularität der gierigen Zerlegung (Clauset, Newman, Moore), Anteil der größten Komponente.
**Sechs Familien.** Erdős–Rényi G(n, m), Konfigurationsmodell (die Grade eines Barabási–Albert-Netzes, durch Kantentausch neu verdrahtet), Barabási–Albert, Watts–Strogatz, geometrischer Zufallsgraph im Einheitsquadrat, Blockmodell mit drei gleich großen Blöcken.
**Klassifikator.** Je Familie 60 Trainingsnetze mit derselben Knotenzahl und demselben mittleren Grad (auf 0,1 gerundet) wie das untersuchte Netz, die Parameter von Watts–Strogatz (Umlegewahrscheinlichkeit 0,02 bis 0,30) und Blockmodell (Verhältnis innen zu außen 6 bis 40) zufällig; die sieben Größen werden standardisiert, die Antwort ist der Anteil der Familien unter den 15 nächsten Trainingsnetzen.
**Anpassungsprüfung.** Das Modell wird aus dem Netz geschätzt (Erdős–Rényi: Kantenzahl; Konfigurationsmodell: die Gradfolge, Kantentausch mit zehn Vorschlägen je Kante; Barabási–Albert: m = Grad / 2; Watts–Strogatz: Grad und Umlegewahrscheinlichkeit aus der Transitivität; geometrisch: der Radius, bei dem der erwartete Grad stimmt; Blockmodell: Zerlegung der gierigen Modularitätsmaximierung und die Dichten zwischen den Gruppen), daraus werden 60 Netze gezogen (einstellbar).
Der **Gesamt-p-Wert** ist (1 + Zahl der Wolkennetze, die von den übrigen Wolkennetzen mindestens so weit entfernt liegen wie das beobachtete Netz von der Wolke) / (Zahl der Wolkennetze + 1), mit dem Mahalanobis-Abstand in den standardisierten Größen; ein Modell besteht bei p ≥ 0,05. Je Größe gibt es zusätzlich den zweiseitigen empirischen p-Wert und die Heatmap der standardisierten Abstände.
Das ist eine Prüfung mit **eingesetzten Parametern** (parametrischer Bootstrap), keine bayessche Posterior-Predictive-Prüfung. Formeln, Lesehilfe und Literatur (aus dem Gedächtnis, nicht im Original geprüft) stehen im 📐-Expander der App.

## Methodik

- **Kern** (`mwn_algorithm.py`): Erzeuger der sechs Familien und des Holme–Kim-Netzes, die sieben Größen, gierige Modularitätsmaximierung (Zwischenwerte nur für die geänderte Zeile und Spalte neu gerechnet), k nächste Nachbarn, Anpassung, Ziehen aus dem angepassten Modell, Gesamt-p-Wert.
- **Szenario und Auswertung** (`mwn_scenario.py`, `mwn_evaluation.py`): erzeugte Netze, Trainingsmengen (zwischengespeichert je Knotenzahl und Grad), Analyse je Einstellung, drei Messreihen mit festem Programm, alle Verdikte als reine Funktionen.
- **Oberfläche** (`app.py`): fünf Schnellstart-Szenarien, Regler für Netz, Knotenzahl, mittleren Grad, Zahl der gezogenen Netze und Zufalls-Seed, Antwort des Klassifikators, Ergebnis der Prüfung, Heatmap der Abstände, Tabellen, Kernfrage, drei Messreihen auf Klick, Annahmen und Formeln.
- **Daten** (`data/netze.json`): Kantenlisten des Zachary-Karateclubs (34 Knoten, 78 Kanten), des Netzes der gemeinsamen Auftritte in „Les Misérables“ (77 Knoten, 254 Kanten) und der Florentiner Familien (15 Knoten, 20 Kanten), ohne Gewichte, aus `networkx` übernommen (wie in `netzmotive-demo`); die Quellenangaben (Zachary 1977, Knuth 1993, Padgett und Ansell) sind aus dem Gedächtnis.

## Befunde (gemessen, keine Behauptungen)

Alle Reihen mit festen Zufalls-Seeds; erzeugte Netze mit 150 Knoten und mittlerem Grad 3,5 (die Messreihen mit 100 Knoten), 60 gezogene Netze je Modell. Der kleinste p-Wert ist dann 1 / 61 = **0,016**. **Die Zufallszahlen eines `numpy`-Generators sind nicht über Versionen stabil**: die Tests prüfen deshalb Bänder um diese Zahlen.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der Kern? | Transitivität, Assortativität, Grade, größte Komponente und mittlere Weglänge gleich `networkx` auf 18 Netzen aller Familien; die gierige Zerlegung gleich einer naiven Umsetzung (alles je Schritt neu gerechnet) und ihre Modularität gleich der von `networkx` berechneten Modularität derselben Zerlegung (und höchstens 0,03 neben der gierigen Zerlegung von `networkx`, größte Abweichung 0,012); k nächste Nachbarn gleich `scikit-learn`; geometrische Radiusformel gleich Monte-Carlo; Watts–Strogatz ohne Umlegen gleich `networkx`; Kantentausch gleichverteilt über alle **70** 2-regulären Graphen auf sechs Knoten, G(n, m) gleichverteilt über alle 120 Kantenmengen (Chi-Quadrat); die Prüfung hält bei Normalverteilung ihr Niveau (90 bis 99 % bestehen bei zu erwartenden 95 %). | `tests/test_oracle_modellwahl_netze.py` |
| Werden klare Familien erkannt? | Mit den Zufalls-Seeds 1 bis 8 (Erdős–Rényi) bzw. 1 bis 3 (Watts–Strogatz, geometrisch, Blockmodell): **jedes Netz wird richtig genannt, und das wahre Modell besteht die Prüfung jedes Mal**; der Klassifikator nennt für diese Familien nur sie selbst. Voreinstellung (Watts–Strogatz, Zufalls-Seed 35): 100 % der nächsten Nachbarn, nur Watts–Strogatz besteht (p = **0,77**, alle anderen 0,016). | `test_claims.py::test_the_clear_families_are_recognised_and_their_own_model_passes`, `::test_watts_strogatz_preset_is_the_clear_case` |
| Sind Barabási–Albert und Konfigurationsmodell zu trennen? | Nein, nicht mit diesen Größen: bei Zufalls-Seeds 1 bis 8 nennt der Klassifikator bei Barabási–Albert-Netzen **4 von 8** Mal die richtige Familie (sonst das Konfigurationsmodell), bei Konfigurationsnetzen **6 von 8**; das wahre Modell besteht **8 von 8** bzw. **7 von 8** Mal die Prüfung, und es besteht stets mindestens eines. Schnellstart (Zufalls-Seed 36): 53 % Barabási–Albert, 47 % Konfigurationsmodell, beide bestehen (p = **0,34** und **0,12**). | `test_claims.py::test_barabasi_albert_and_configuration_model_are_confused`, `::test_barabasi_albert_preset_cannot_be_told_from_the_configuration_model` |
| Wie oft stimmt der Klassifikator, und wie viel tragen die Gradgrößen? | Messreihe (100 Knoten, je Familie 20 frisch erzeugte Netze): mit allen sieben Größen **86,7 %** (104 von 120), mit nur den **zwei Gradgrößen 50,8 %** (61 von 120), **ohne die Gradgrößen 90,8 %** (109 von 120). Erdős–Rényi, Watts–Strogatz und geometrisch jedes Mal richtig; die Fehler sind fast nur Barabási–Albert gegen Konfigurationsmodell (**9** Konfigurationsnetze als Barabási–Albert, **6** umgekehrt) und ein Blockmodell als Erdős–Rényi. Die Gradgrößen helfen nicht, wenn die anderen fünf da sind; allein erkennen sie nur Watts–Strogatz (fast gleiche Grade) sicher. | `test_claims.py::test_confusion_series_shows_what_the_degree_sizes_alone_cannot_do` |
| Besteht das wahre Modell, und besteht ein falsches? | Messreihe (100 Knoten, je Familie 6 Netze, jedes gegen alle sechs Modelle, 40 gezogene Netze je Prüfung): das wahre Modell besteht in **97 %** der Fälle (nur Barabási–Albert in 5 von 6), ein falsches im Mittel in **11 %**. Die Ausnahmen haben einen Grund: **Erdős–Rényi-Netze bestehen gegen das Konfigurationsmodell immer** (ihre Grade sind mit der Gradfolge verträglich), **gegen das Blockmodell in 83 %** (die angepasste Zerlegung fängt Zufallsnetze auf), Barabási–Albert gegen das Konfigurationsmodell in 83 %, umgekehrt in 33 %. Alle anderen Paare höchstens 17 %. | `test_claims.py::test_fit_series_keeps_its_size_and_rejects_most_wrong_models`, `::test_fit_series_shows_which_wrong_models_still_pass` |
| Was antwortet der Klassifikator bei einem Netz aus keiner Familie? | Holme–Kim (Barabási–Albert mit Dreiecksschluss, Transitivität **0,19**, größter Grad das **8,4**-fache des mittleren): bei Zufalls-Seeds 1 bis 8 nennt der Klassifikator **nie** die Wahrheit (er hat sie nicht), nur Barabási–Albert oder Konfigurationsmodell, und **bei 8 von 8 Netzen besteht kein Modell** die Prüfung. Schnellstart (Zufalls-Seed 39): Konfigurationsmodell 60 %, bester p-Wert 0,016. | `test_claims.py::test_a_holme_kim_network_is_never_recognised_and_never_passes`, `::test_holme_kim_preset_matches_no_model_and_still_gets_a_confident_answer` |
| Und echte Netze? | Karateclub (34 Knoten): Klassifikator **Barabási–Albert 53 %**, aber **kein** Modell besteht (bester p-Wert 0,016, Erdős–Rényi); Les Misérables (77 Knoten): **Barabási–Albert 60 %**, kein Modell besteht (bester p-Wert 0,033). Florentiner Familien (15 Knoten): Erdős–Rényi 40 %, aber **vier Modelle bestehen** (Erdős–Rényi, Konfigurationsmodell, Barabási–Albert, Blockmodell; bester p-Wert 0,75): bei 15 Knoten ist nichts zu verwerfen. Die Transitivität des Karateclubs ist 0,256, die Assortativität −0,476. | `test_claims.py::test_real_networks_series`, `::test_real_network_presets_get_an_answer_but_pass_no_model`, `::test_the_observed_statistics_of_the_real_networks` |

## Ehrliche Grenzen

- **Bestanden heißt nicht erzeugt.** Das Blockmodell wird an die Zerlegung des Netzes angepasst und kann viel auffangen (Zufallsnetze zu 83 %); auch beim Florentiner Netz bestehen vier Modelle, weil bei 15 Knoten kein Test Kraft hat. Die Prüfung kann Modelle verwerfen, nicht bestätigen.
- **Eingesetzte Parameter, dieselben Daten.** Die Parameter kommen aus dem Netz, das geprüft wird; das macht die Prüfung eher nachsichtig. Die gemessene Quote des wahren Modells (97 % bei zu erwartenden 95 %) beruht auf 36 Netzen und sagt dazu wenig.
- **Grobe p-Werte.** Bei 60 gezogenen Netzen ist der kleinste p-Wert 0,016; „p < 0,001“ bräuchte mehr als 1 000 Netze je Modell. Der Gesamt-p-Wert hängt am gewählten Abstand (Mahalanobis, zur Diagonale hin geschrumpft) und an der Wahl der sieben Größen.
- **Die Antwort des Klassifikators hängt von den Trainingsnetzen ab.** Barabási–Albert gegen Konfigurationsmodell ist ein Münzwurf: derselbe Netztyp wechselte beim Runden des Trainingsgrades von 0,01 auf 0,1 zwischen den beiden Antworten. Der Klassifikator kennt nur die Parameterbereiche, in denen trainiert wurde.
- **Nur sechs Familien, sieben Größen, ungerichtete Netze.** Kein Modell mit Gemeinschaften plus Hubs, keine gerichteten oder gewichteten Netze. Das Holme–Kim-Netz zeigt genau diese Lücke; ein anderes Netz außerhalb der Liste kann sich anders verhalten (zum Beispiel vom Blockmodell aufgefangen werden: ein Ring mit Hubs wurde in einer Vorprüfung als Blockmodell bestanden).
- **Beispielnetze und ihre Herkunft.** Die drei Netze stammen aus `networkx` (Zachary 1977; Knuth, *The Stanford GraphBase*, 1993; Padgett und Ansell zu den Florentiner Familien, aus dem Gedächtnis); die Kantenlisten sind kleine Faktenlisten ohne weitere Angaben. Die Quellenangaben sind nicht im Original geprüft.
- **Literatur** (Gelman, Meng und Stern 1996; Clauset, Newman und Moore 2004; Barrat und Weigt 2000; Holme und Kim 2002) aus dem Gedächtnis, nicht im Original geprüft.

## Befunde und Korrekturen gegenüber dem Plan

- **Port:** 9170 statt 9163 (bis 9169 sind vergeben).
- **H1 geändert:** die Vormessung V9 fand die Größen „fast eindeutig“ (100 % bei n = 400 und 98 % bei n = 150) mit einem Random-Forest-Klassifikator auf festen Parametern. Hier, mit zufälligen Parametern von Watts–Strogatz und Blockmodell und k nächsten Nachbarn, sind es **86,7 %** bei 100 Knoten; die Verwechslung Barabási–Albert gegen Konfigurationsmodell, in V9 nur mit den Graden allein, bleibt auch mit allen sieben Größen. Die Gradgrößen allein erreichen 50,8 %.
- **H2 bestätigt:** ein Klassifikator liefert für reale Netze immer eine Antwort (V9: Karateclub Barabási–Albert 0,44, Les Misérables 0,61; hier 0,53 und 0,60); neu ist, dass die Prüfung bei beiden **kein** Modell besteht.
- **Posterior-Predictive im Plan:** gebaut ist eine Prüfung mit eingesetzten Parametern (parametrischer Bootstrap), nicht die bayessche Fassung.
- **Toronto-Ausschnitt** (V9b): nicht übernommen, die Straßenkarte steckt in `raeumliche-netze-demo` (Daten von OpenStreetMap, ODbL); ersatzweise das Holme–Kim-Netz als Netz außerhalb der Liste, dessen Herkunft bekannt ist.
- **Random-Forest-Klassifikator der Vormessung** durch k nächste Nachbarn ersetzt, damit `scikit-learn` nicht zur Laufzeit gebraucht wird; `scikit-learn` steht nur in `requirements-dev.txt` als Gegenprobe.
- **Gradschlüssel der Trainingsnetze:** auf 0,1 gerundet (in einer ersten Fassung auf 0,01); das spart je Netz die Trainingszeit, verschiebt aber die 50:50-Entscheidung zwischen Barabási–Albert und Konfigurationsmodell, siehe Grenzen.
- **Mischnetz verworfen:** ein Ring mit Hubs wurde vom angepassten Blockmodell als passend akzeptiert (in 6 von 6 Zufalls-Seeds), taugte also nicht als Beispiel für ein Netz ohne passendes Modell; das Holme–Kim-Netz besteht bei 8 von 8 Zufalls-Seeds keine Prüfung.

## Verifikation

- **Kern:** jede Einheit an einer Mini-Instanz von Hand (Weg 0–1–2–3, Dreieck mit Anhängsel, zwei getrennte Dreiecke, zwei durch eine Kante verbundene Dreiecke): Transitivität 3/5, Assortativität −1/2, mittlere Weglänge 5/3, Modularität 1/6 und 5/14, die Wahrscheinlichkeit des Abstands im Einheitsquadrat, Blockwahrscheinlichkeiten, k nächste Nachbarn mit Gleichstand, p-Werte und Gesamt-p-Wert an kleinen Wolken; Randfälle (kleinste und größte Reglerwerte für jedes erzeugte Netz, Netze ohne Kanten, ein Baum als dünnstes Netz, Wolken mit weniger als drei Netzen).
- **Orakel:** `networkx`, naive gierige Zerlegung, `scikit-learn`, Monte-Carlo, exakte Aufzählung aller Graphen einer Gradfolge, Chi-Quadrat; die Größe der Prüfung gegen eine bekannte Normalverteilung.
- **Verdikte:** reine Funktionen, jeder Zweig und jede Zahl im Text an handgesetzten Werten (`tests/test_verdicts.py`).
- **Presets, Layout, Oberfläche:** wie in allen Demos (Presets vollständig und im Regelbereich und auf dem Raster der Regler, keines auf einem Reglerrand, kein Preset entsteht durch Änderung eines einzigen Reglers, Schnellstart oben, aktives Preset hervorgehoben, AppTest jeder Button und jeder Regler an den Rändern).
- **Mutationsprüfung** (`tools/mutate_demo.py`): 297 Stellen im Kern `mwn_algorithm.py`, **278 gefangen** (73 davon über wortgleiche Zeilen in Nachbar-Demos), 13 als äquivalent festgehalten, 6 Zeitüberschreitungen (das Werkzeug zählt sie nicht). Der erste Lauf ließ 26 Mutanten überleben und deckte echte Lücken auf: Netze mit weniger Knoten als der Start (Barabási–Albert und Holme–Kim), ein Paar genau auf dem Radius des geometrischen Graphen (jetzt mit einem Ersatz-Zufallsgenerator mit festen Punkten geprüft), die Zusatzkanten des Rings an der Grenze des Abstands (n = 6 und n = 7), den gebrochenen Wert von m, ein Blockmodell mit Einzelknoten (Dichte 0 statt 0 / 0), die p-Werte je Größe bei konstanter Spalte und bei zwei Wolkennetzen sowie doppelte Ziele beim Dreiecksschluss.
  Die 13 äquivalenten Stellen: fünfmal `<` statt `<=` bei einer Ziehung mit `rng.random()` (gleich nur mit Wahrscheinlichkeit null), einmal in der Bisektion des Radius, einmal an einem Gleichstand zweier Gleitkommaabstände, viermal in Wächtern, die denselben Wert liefern (zweimal `frac > 0` beim Ring, `>=` statt `>` beim Schätzer der Umlegewahrscheinlichkeit am Rand, `r < 1` statt `r < 2` bei einem einzelnen Wolkennetz, das ohnehin p = 1 ergibt) und je einmal die Abfragen `len(cols) == 0` und `len(cloud) < 2`, deren Folgeberechnung für leere Spalten dasselbe liefert.
  Die 6 Zeitüberschreitungen sind Endlosschleifen: die Ziehschleife von G(n, m) (Obergrenze der Kantenzahl und Zeile mit dem Einfügen der Kante, dieselben Zeilen wie in `netzmotive-demo` und `raeumliche-netze-demo`) und die Auswahl der Ziele bei Barabási–Albert (`<=` statt `<`); in einer Kopie von Hand bestätigt, dass die Tests dann hängen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/netze.json` | Kantenlisten der drei echten Netze (ohne Gewichte) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx und scikit-learn (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` rechnet die dünnen Matrizen und die Wege) |
| `mwn_algorithm.py` | Kern: Erzeuger, sieben Größen, Klassifikator, Anpassung und Prüfung |
| `mwn_constants.py` | Regler-Grenzen, feste Pläne, Farben, Presets |
| `mwn_evaluation.py` | Analyse, Messreihen, Verdikte, Tabellen |
| `mwn_layout.py` | Fester Kopf (Stylesheet) |
| `mwn_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `mwn_scenario.py` | Erzeugte und echte Netze, Trainingsmengen |
| `mwn_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `messen.py` | Einmalige Messung der Zahlen in dieser README (nicht Teil der App) |
| `tests/` | Mini-Netze, Kern, Orakel, Verdikte, Randfälle, Presets, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

- **Bayessche Modellwahl** (Posterior, Evidenz, Minimum Description Length): würde die Familien über ihre Wahrscheinlichkeit für das ganze Netz vergleichen; das Blockmodell mit Evidenz steht in `stochastic-block-demo` nur ansatzweise. Hier bleibt es bei sieben Zusammenfassungsgrößen.
- **Weitere Familien** (Kronecker, gerichtete oder zeitliche Modelle, Chung–Lu): würden die Liste verlängern, aber nichts an der Grundfrage ändern, dass jede Liste endlich ist.
- **Mehr echte Netze und das Straßennetz von Toronto:** die Daten stehen in `netzmotive-demo` und `raeumliche-netze-demo`; ein größeres echtes Netz würde die Rechenzeit der Prüfung sprengen.
- **Automatische Wahl der Zahl gezogener Netze:** der Regler bleibt dem Nutzer überlassen; die Demo warnt nicht, wenn 30 Netze zu wenig sind.

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
