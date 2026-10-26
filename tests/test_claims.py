"""Befunde der README als Bänder: jede Zahl, die dort steht, ist hier an einem festen Plan geprüft (Seeds und Pläne wie in messen.py)."""

import numpy as np
import pytest

import mwn_algorithm as A
import mwn_constants as C
import mwn_evaluation as ev

IDX = {k: i for i, k in enumerate(C.MODEL_ORDER)}


@pytest.fixture(scope="module")
def confusion():
    return ev.series_confusion()


@pytest.fixture(scope="module")
def fit():
    return ev.series_fit()


@pytest.fixture(scope="module")
def real():
    return {r["key"]: r for r in ev.series_real()}


@pytest.fixture(scope="module")
def recognition():
    return ev.recognition()


def preset(name):
    return ev.analyse(ev.Settings(**C.PRESETS[name]))


def test_watts_strogatz_preset_is_the_clear_case():
    a = preset("Watts–Strogatz, sauber erkannt")
    assert a.top == "ws" and a.probs.max() == 1.0 and a.passes == ("ws",) and a.truth == "ws"
    assert a.p_all[IDX["ws"]] == pytest.approx(0.77, abs=0.05) and ev.verdict_choice(a.probs, a.passes)[0] == "success"


def test_barabasi_albert_preset_cannot_be_told_from_the_configuration_model():
    a = preset("Barabási–Albert: kaum von Konfiguration zu trennen")
    assert a.top == "ba" and a.probs[IDX["ba"]] == pytest.approx(8 / 15) and a.probs[IDX["config"]] == pytest.approx(7 / 15)
    assert a.passes == ("config", "ba") and ev.verdict_choice(a.probs, a.passes)[0] == "info"
    assert a.p_all[IDX["ba"]] == pytest.approx(0.344, abs=0.05) and a.p_all[IDX["config"]] == pytest.approx(0.115, abs=0.05)


def test_real_network_presets_get_an_answer_but_pass_no_model():
    les = preset("Les Misérables: nach Zwang BA?")
    assert les.n == 77 and les.top == "ba" and les.probs.max() == pytest.approx(0.6) and les.passes == () and les.p_all.min() == pytest.approx(1 / 61)
    karate = preset("Karateclub: nach Zwang BA?")
    assert karate.n == 34 and karate.top == "ba" and karate.probs.max() == pytest.approx(8 / 15) and karate.passes == ()
    assert ev.verdict_choice(karate.probs, karate.passes)[0] == "warning"


def test_holme_kim_preset_matches_no_model_and_still_gets_a_confident_answer():
    a = preset("Holme–Kim-Netz: passt kein Modell")
    assert a.truth is None and a.top == "config" and a.probs.max() == pytest.approx(0.6) and a.passes == ()
    assert a.observed[2] == pytest.approx(0.188, abs=0.01) and a.observed[0] == pytest.approx(8.37, abs=0.2)      # viele Dreiecke und ein großer Hub zugleich


def test_the_clear_families_are_recognised_and_their_own_model_passes(recognition):
    for key in ("er", "ws", "geo", "sbm"):
        r = recognition[key]
        assert r["right"] == r["runs"] and r["own"] == r["runs"] and r["none"] == 0 and r["named"] == (key,), key


def test_barabasi_albert_and_configuration_model_are_confused(recognition):
    ba, cfg = recognition["ba"], recognition["config"]
    assert ba["runs"] == cfg["runs"] == 8 and ba["named"] == cfg["named"] == ("ba", "config")
    assert 3 <= ba["right"] <= 6 and 4 <= cfg["right"] <= 7                         # kaum besser als Raten
    assert ba["own"] == 8 and cfg["own"] >= 6 and ba["none"] == cfg["none"] == 0     # das wahre Modell besteht (fast) immer


def test_a_holme_kim_network_is_never_recognised_and_never_passes(recognition):
    r = recognition["hk"]
    assert r["right"] == 0 and r["own"] == 0 and r["none"] == r["runs"] == 8 and set(r["named"]) <= {"ba", "config"}


def test_confusion_series_shows_what_the_degree_sizes_alone_cannot_do(confusion):
    s = confusion["sets"]
    allc, deg, nodeg = s["alle sieben Größen"], s["nur die zwei Gradgrößen"], s["ohne die Gradgrößen"]
    assert allc["accuracy"] == pytest.approx(0.867, abs=0.04) and deg["accuracy"] == pytest.approx(0.508, abs=0.06) and nodeg["accuracy"] == pytest.approx(0.908, abs=0.04)
    assert nodeg["accuracy"] >= allc["accuracy"] - 0.02                              # die Gradgrößen helfen nicht, wenn die anderen fünf da sind
    conf = allc["conf"]
    assert [conf[i, i] for i in (IDX["er"], IDX["ws"], IDX["geo"])] == [20, 20, 20] and conf[IDX["sbm"], IDX["sbm"]] >= 18 and conf[IDX["sbm"], IDX["er"]] <= 2
    assert conf[IDX["config"], IDX["ba"]] >= 5 and conf[IDX["ba"], IDX["config"]] >= 3 and conf[IDX["config"], IDX["config"]] + conf[IDX["ba"], IDX["ba"]] <= 31
    assert all(deg["conf"][IDX[k], IDX[k]] < 15 for k in ("er", "config", "ba", "geo", "sbm")) and deg["conf"][IDX["ws"], IDX["ws"]] == 20          # Watts–Strogatz hat fast gleiche Grade, das ist sofort sichtbar


def test_fit_series_keeps_its_size_and_rejects_most_wrong_models(fit):
    rate = fit["rate"]
    assert fit["per_class"] == 6 and fit["sims"] == 40 and fit["n"] == 100
    own = np.diag(rate)
    assert own.mean() == pytest.approx(0.97, abs=0.05) and own.min() >= 0.66 and rate[IDX["ba"], IDX["ba"]] <= 1.0
    open_pairs = {(IDX["er"], IDX["config"]), (IDX["er"], IDX["sbm"]), (IDX["ba"], IDX["config"]), (IDX["config"], IDX["ba"])}
    for i in range(6):
        for j in range(6):
            if i != j and (i, j) not in open_pairs:
                assert rate[i, j] <= 1 / 6 + 1e-9, (C.MODEL_ORDER[i], C.MODEL_ORDER[j])             # höchstens ein Netz von sechs


def test_fit_series_shows_which_wrong_models_still_pass(fit):
    rate = fit["rate"]
    assert rate[IDX["er"], IDX["config"]] == 1.0                                   # Erdős–Rényi ist fast ein Spezialfall des Konfigurationsmodells
    assert rate[IDX["er"], IDX["sbm"]] >= 0.5                                      # das angepasste Blockmodell ist flexibel genug, um ein Zufallsnetz aufzufangen
    assert rate[IDX["ba"], IDX["config"]] == pytest.approx(0.83, abs=0.17) and rate[IDX["config"], IDX["ba"]] == pytest.approx(0.33, abs=0.17)
    off = (rate.sum() - np.trace(rate)) / 30
    assert off == pytest.approx(0.11, abs=0.05)


def test_real_networks_series(real):
    assert real["karate"]["passes"] == () and real["lesmis"]["passes"] == () and real["hk"]["passes"] == ()
    assert real["karate"]["best_p"] < C.ALPHA and real["lesmis"]["best_p"] < C.ALPHA and real["hk"]["best_p"] < C.ALPHA
    assert real["florenz"]["n"] == 15 and set(real["florenz"]["passes"]) >= {"er", "config", "ba", "sbm"} and real["florenz"]["top"] == "er"
    assert [(real[k]["top"], round(real[k]["p_top"], 3)) for k in ("karate", "lesmis", "florenz", "hk")] == [("ba", 0.533), ("ba", 0.6), ("er", 0.4), ("config", 0.6)]
    assert real["lesmis"]["best_p"] == pytest.approx(2 / 61) and real["florenz"]["best"] == "sbm" and real["florenz"]["best_p"] == pytest.approx(0.754, abs=0.01)


def test_the_observed_statistics_of_the_real_networks():
    karate = ev.analyse(ev.Settings(network="karate", sims=30, seed=1)).observed
    assert karate[2] == pytest.approx(0.2557, abs=0.001) and karate[3] == pytest.approx(-0.4756, abs=0.001) and karate[4] == pytest.approx(2.4082, abs=0.001)
    les = ev.analyse(ev.Settings(network="lesmis", sims=30, seed=1)).observed
    assert les[2] == pytest.approx(0.4989, abs=0.001) and les[0] == pytest.approx(36 / (508 / 77), abs=0.01)


def test_goodness_of_fit_check_on_a_known_cloud_matches_the_closed_form():
    """Streuung null in einer Spalte und Gleichstände: der Gesamt-p-Wert bleibt in (0, 1] und der kleinste Wert ist 1 / (R + 1)."""
    cloud = np.random.default_rng(1).normal(size=(60, 3))
    _, _, p = A.goodness_of_fit([40.0, 40.0, 40.0], cloud)
    assert p == pytest.approx(1 / 61)
