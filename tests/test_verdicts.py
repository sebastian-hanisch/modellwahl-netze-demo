"""Verdikte (reine Funktionen): Art und jede Zahl im Text an handgesetzten Werten, jeder Zweig."""

import numpy as np

import mwn_constants as C
import mwn_evaluation as ev


def probs(**kw):
    p = np.zeros(6)
    for key, v in kw.items():
        p[C.MODEL_ORDER.index(key)] = v
    return p


def test_verdict_choice_when_nothing_passes():
    kind, text = ev.verdict_choice(probs(ba=0.6, config=0.4), ())
    assert kind == "warning" and "**Barabási–Albert** (60 % der nächsten Nachbarn)" in text and "**kein** Modell der Liste besteht die Prüfung" in text and "erzwungen" in text


def test_verdict_choice_when_only_the_top_model_passes():
    kind, text = ev.verdict_choice(probs(ws=1.0), ("ws",))
    assert kind == "success" and "**Watts–Strogatz** (100 % der nächsten Nachbarn)" in text and "nur dieses Modell besteht" in text


def test_verdict_choice_when_the_top_model_and_others_pass():
    kind, text = ev.verdict_choice(probs(ba=0.73, config=0.27), ("config", "ba"))
    assert kind == "info" and "**Barabási–Albert** (73 % der nächsten Nachbarn)" in text and "auch Konfigurationsmodell zu" in text
    assert "die Prüfung trennt diese Modelle hier nicht" in text and "sieben Größen trennen" not in text        # die Aussage gilt der Prüfung, nicht den Größen
    kind, text = ev.verdict_choice(probs(er=0.5, ba=0.5), ("er", "config", "sbm"))                 # Gleichstand: der zuerst genannte
    assert kind == "info" and "**Erdős–Rényi** (50 %" in text and "Konfigurationsmodell und Blockmodell zu" in text


def test_verdict_choice_when_the_top_model_fails_but_others_pass():
    kind, text = ev.verdict_choice(probs(ba=0.53, config=0.47), ("config",))
    assert kind == "warning" and "**Barabási–Albert** (53 %" in text and "besteht die Prüfung nicht; bestanden haben Konfigurationsmodell" in text and "widersprechen" in text


def test_truth_sentence_cases():
    assert ev.truth_sentence("geo", "geo") == "Erzeugt wurde das Netz aus: **geometrisch**."
    assert "Holme und Kim" in ev.truth_sentence(None, "hk") and "**keiner**" in ev.truth_sentence(None, "hk")
    assert "echten Netzes ist unbekannt" in ev.truth_sentence(None, "karate")


def conf(diag, off=None):
    m = np.diag(diag).astype(int)
    for (i, j), v in (off or {}).items():
        m[i, j] = v
    return m


def test_verdict_confusion_by_hand():
    series = {"per_class": 10, "sets": {
        "alle sieben Größen": {"conf": conf([10, 8, 7, 10, 10, 10], {(1, 2): 2, (2, 1): 3}), "accuracy": 0.9167},
        "nur die zwei Gradgrößen": {"conf": conf([5] * 6), "accuracy": 0.5},
        "ohne die Gradgrößen": {"conf": conf([10] * 6), "accuracy": 1.0},
    }}
    kind, text = ev.verdict_confusion(series)
    assert kind == "info" and "**92 %** der 60 frisch erzeugten Netze" in text and "nur den beiden Gradgrößen 50 %" in text and "ohne die Gradgrößen 100 %" in text
    assert "Am häufigsten verwechselt wird Barabási–Albert mit Konfigurationsmodell (3 von 10 Netzen)" in text
    series["sets"]["alle sieben Größen"]["conf"] = conf([10] * 6)
    kind, text = ev.verdict_confusion(series)
    assert kind == "success" and text.endswith("Es gibt keine Verwechslung.")


def test_verdict_fit_by_hand():
    rate = np.diag([1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
    rate[0, 1] = 1.0
    rate[2, 1] = 0.5
    kind, text = ev.verdict_fit({"rate": rate, "per_class": 6, "sims": 40})
    assert kind == "info" and "besteht die Prüfung in **92 %** der Fälle" in text and "ein falsches im Mittel in 5 %" in text          # Diagonale 5,5 / 6; außerhalb 1,5 / 30
    assert "bei Erdős–Rényi-Netzen: Konfigurationsmodell in 100 % der Fälle (6 Netze je Familie, 40 Simulationen je Prüfung)" in text


def test_verdict_real_by_hand():
    rows = [
        {"name": "Karateclub", "top": "ba", "p_top": 0.6, "passes": (), "best_p": 0.016},
        {"name": "Florenz", "top": "er", "p_top": 0.53, "passes": ("er", "config"), "best_p": 0.75},
    ]
    kind, text = ev.verdict_real(rows)
    assert kind == "warning" and text.startswith("Der Klassifikator antwortet in allen 2 Fällen; bei 1 davon besteht kein Modell die Prüfung.")
    assert "Karateclub: Klassifikator Barabási–Albert (60 %), es besteht keines (bester p-Wert 0,016)" in text and "Florenz: Klassifikator Erdős–Rényi (53 %), es bestehen Erdős–Rényi und Konfigurationsmodell" in text
    kind, text = ev.verdict_real(rows[1:])
    assert kind == "info" and "bei 0 davon" in text
