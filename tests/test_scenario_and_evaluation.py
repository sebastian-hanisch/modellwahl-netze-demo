"""Szenario und Auswertung: Netze der Familien und die echten Netze, Trainingsmenge, Determinismus und Konsistenz der Analyse, Zeilen der Tabellen, Aufbau der Messreihen."""

import numpy as np
import pytest

import mwn_algorithm as A
import mwn_constants as C
import mwn_evaluation as ev
from mwn_scenario import family_member, hk_member, make_network, training_member, training_set


def test_family_member_by_hand():
    rng = np.random.default_rng(1)
    n, e = family_member("er", 100, 3.5, rng, 0.1, 15.0)
    assert n == 100 and len(e) == 175                                              # round(3,5 · 100 / 2)
    n, e = family_member("ws", 10, 4.0, rng, 0.0, 15.0)
    assert len(e) == 20 and set(A.degrees(10, e)) == {4}
    n, e = family_member("sbm", 30, 3.0, rng, 0.1, 1.0)                            # Verhältnis 1: kein Blockeffekt, ein gewöhnliches Zufallsnetz mit Grad 3
    assert n == 30 and 10 <= len(e) <= 80
    n, e = family_member("config", 200, 3.5, rng, 0.1, 15.0)
    assert A.degrees(200, e).max() > 3 * 3.5                                       # die Grade eines Barabási–Albert-Netzes bleiben (Hubs)
    with pytest.raises(ValueError):
        family_member("x", 10, 3.0, rng, 0.1, 15.0)


def test_training_member_and_make_network_are_deterministic():
    assert training_member("ws", 60, 3.5, np.random.default_rng(1)) == training_member("ws", 60, 3.5, np.random.default_rng(1))
    assert make_network("er", 80, 3.0, 5) == make_network("er", 80, 3.0, 5)
    assert make_network("er", 80, 3.0, 5)[1] != make_network("er", 80, 3.0, 6)[1]
    n, e, truth = make_network("sbm", 80, 3.0, 5)
    assert n == 80 and truth == "sbm"


def test_make_network_real_networks_ignore_the_controls():
    assert [(make_network(key, 400, 8.0, 3)[0], len(make_network(key, 400, 8.0, 3)[1]), make_network(key, 400, 8.0, 3)[2]) for key in C.REAL] == [(34, 78, None), (77, 254, None), (15, 20, None)]
    assert make_network("karate", 50, 2.0, 1) == make_network("karate", 400, 8.0, 99)


def test_make_network_truth_labels_and_errors():
    for key in C.MODEL_ORDER:
        assert make_network(key, 60, 3.5, 1)[2] == key
    n, e, truth = make_network("hk", 200, 3.5, 1)
    assert truth is None and n == 200 and 2 * len(e) / n == pytest.approx(3.5, abs=0.3)
    with pytest.raises(ValueError):
        make_network("x", 60, 3.5, 1)


def test_hk_member_is_deterministic_and_has_hubs_and_triangles():
    e = hk_member(300, 3.5, np.random.default_rng(1))
    assert e == hk_member(300, 3.5, np.random.default_rng(1))
    assert A.degrees(300, e).max() > 4 * 3.5 and A.transitivity(300, e) > 0.08


def test_training_set_by_hand():
    x, y = training_set(60, 350)
    assert x.shape == (6 * C.TRAIN_PER_CLASS, 7) and np.bincount(y).tolist() == [C.TRAIN_PER_CLASS] * 6
    x2, y2 = training_set(60, 350)
    assert x2 is x                                                                  # zwischengespeichert, gleiches Objekt
    assert not np.array_equal(training_set(60, 400)[0], x)                          # anderer mittlerer Grad: andere Trainingsnetze
    assert x[y == 3][:, 2].mean() > x[y == 0][:, 2].mean() + 0.1                    # Watts–Strogatz hat viel mehr Dreiecke als Erdős–Rényi


def test_standardized_by_hand():
    assert ev.standardized(5.0, 3.0, 2.0) == 1.0
    assert ev.standardized(3.0, 3.0, 0.0) == 0.0
    assert ev.standardized(4.0, 3.0, 0.0) == 99.0 and ev.standardized(2.0, 3.0, 0.0) == -99.0
    assert ev.standardized(1000.0, 0.0, 1.0) == 99.0 and ev.standardized(-1000.0, 0.0, 1.0) == -99.0
    assert ev.standardized(3.0, 3.0, 1e-12) == 0.0


def test_names_and_formatting():
    assert ev.names(()) == "" and ev.names(("er",)) == "Erdős–Rényi"
    assert ev.names(("er", "ba")) == "Erdős–Rényi und Barabási–Albert"
    assert ev.names(("er", "config", "ba")) == "Erdős–Rényi, Konfigurationsmodell und Barabási–Albert"
    assert ev.num(3.14159, 2) == "3,14" and ev.pct(0.456) == "46 %" and ev.pct(0.0456, 1) == "4,6 %"


def test_analyse_is_deterministic_and_consistent():
    s = ev.Settings(network="config", n=60, k=3.0, sims=30, seed=3)
    a, b = ev.analyse(s), ev.analyse(s)
    assert a.probs.tolist() == b.probs.tolist() and a.p_all.tolist() == b.p_all.tolist() and a.observed.tolist() == b.observed.tolist()
    n, e, truth = make_network("config", 60, 3.0, 3)
    assert a.n == 60 and a.edges == len(e) and a.truth == "config" and a.mean_degree == pytest.approx(2 * len(e) / 60)
    assert a.observed.tolist() == A.statistics(60, e).tolist()
    assert a.probs.sum() == pytest.approx(1.0) and all(abs(p * C.KNN_K - round(p * C.KNN_K)) < 1e-9 for p in a.probs)
    assert all(1 / 31 - 1e-12 <= p <= 1.0 for p in a.p_all)                          # der kleinste mögliche p-Wert bei 30 Netzen
    assert a.passes == tuple(k for k, p in zip(C.MODEL_ORDER, a.p_all) if p >= C.ALPHA)
    assert a.top == C.MODEL_ORDER[int(np.argmax(a.probs))]
    assert a.p_each.shape == a.z.shape == a.cloud_mean.shape == (6, 7) and a.dist.shape == a.p_all.shape == (6,)


def test_the_configuration_model_keeps_the_degree_statistics_fixed():
    a = ev.analyse(ev.Settings(network="ba", n=60, k=3.0, sims=30, seed=3))
    ci = C.MODEL_ORDER.index("config")
    assert a.p_each[ci][:2].tolist() == [1.0, 1.0] and a.z[ci][:2].tolist() == [0.0, 0.0]          # Hub und Streuung des Grades sind festgehalten
    assert a.cloud_mean[ci][:2].tolist() == pytest.approx(a.observed[:2].tolist())


def test_analyse_real_network():
    a = ev.analyse(ev.Settings(network="karate", sims=30, seed=1))
    assert a.n == 34 and a.edges == 78 and a.truth is None and a.mean_degree == pytest.approx(156 / 34)
    assert a.observed[4] == pytest.approx(2.408, abs=0.001) and a.observed[6] == 1.0              # mittlere Weglänge und Anteil der größten Komponente des Karateclubs


def test_fit_and_stats_rows():
    a = ev.analyse(ev.Settings(network="er", n=60, k=3.0, sims=30, seed=3))
    rows = ev.fit_rows(a)
    assert [r["Modell"] for r in rows] == [C.MODEL_LABELS[k] for k in C.MODEL_ORDER]
    for r, p, prob in zip(rows, a.p_all, a.probs):
        assert r["Prüfung"] == ("besteht" if p >= C.ALPHA else "verworfen") and r["Gesamt-p-Wert"] == ev.num(float(p), 3) and r["Anteil Klassifikator"] == ev.pct(float(prob))
        assert "(z = " in r["auffälligste Größe"] and "." not in r["Gesamt-p-Wert"]
    srows = ev.stats_rows(a)
    assert [r["Größe"] for r in srows] == [C.STAT_LABELS[s] for s in A.STAT_NAMES] and srows[2]["beobachtet"] == ev.num(float(a.observed[2]), 3)


def test_off_diagonal_max_by_hand():
    assert ev.off_diagonal_max([[5, 1], [3, 9]]) == (1, 0, 3.0)
    assert ev.off_diagonal_max([[0, 2, 2], [0, 0, 0], [0, 0, 0]]) == (0, 1, 2.0)           # bei Gleichstand der erste in Zeilenfolge
    assert ev.off_diagonal_max([[9, 0], [0, 9]]) == (0, 1, 0.0)


def test_series_structure_with_small_plans():
    s = ev.series_confusion(n=60, k=3.0, per_class=2)
    assert list(s["sets"]) == ["alle sieben Größen", "nur die zwei Gradgrößen", "ohne die Gradgrößen"] and s["per_class"] == 2 and s["n"] == 60
    for r in s["sets"].values():
        assert r["conf"].shape == (6, 6) and r["conf"].sum() == 12 and r["conf"].sum(axis=1).tolist() == [2] * 6 and r["accuracy"] == pytest.approx(np.trace(r["conf"]) / 12)
    f = ev.series_fit(n=40, k=3.0, per_class=2, sims=30)
    assert f["rate"].shape == (6, 6) and set(np.unique(f["rate"])) <= {0.0, 0.5, 1.0} and f["per_class"] == 2 and f["sims"] == 30
    rows = ev.series_real(keys=("florenz",))
    assert rows[0]["key"] == "florenz" and rows[0]["n"] == 15 and rows[0]["name"] == C.NETWORKS["florenz"] and rows[0]["best"] in C.MODEL_ORDER


def test_the_fixed_plans_and_defaults_are_the_documented_ones():
    assert (C.SERIES_SEED, C.SERIES_N, C.SERIES_K, C.SERIES_PER_CLASS_CONFUSION, C.SERIES_PER_CLASS_FIT, C.SERIES_SIMS) == (100000, 100, 3.5, 20, 6, 40)
    assert (C.TRAIN_PER_CLASS, C.KNN_K, C.ALPHA) == (60, 15, 0.05) and C.WS_P_RANGE == (0.02, 0.30) and C.SBM_RATIO_RANGE == (6.0, 40.0)
    s = ev.Settings()
    assert (s.network, s.n, s.k, s.sims, s.seed) == ("ws", 150, 3.5, 60, 35)
    assert C.MODEL_ORDER == A.MODEL_KINDS and list(C.STAT_LABELS) == list(A.STAT_NAMES) == list(C.STAT_SHORT)
