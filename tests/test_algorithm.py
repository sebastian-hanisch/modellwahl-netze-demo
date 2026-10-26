"""Kern, Einheit für Einheit an Mini-Netzen von Hand gerechnet (Fixtures in conftest.py)."""

import numpy as np
import pytest

import mwn_algorithm as A


def test_normalize_adjacency_degrees(path4):
    n, e = path4
    assert A.normalize([(1, 0), (2, 1), (1, 2), (3, 3), (2, 3)]) == [(0, 1), (1, 2), (2, 3)]
    assert A.adjacency(n, e).toarray().tolist() == [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    assert A.degrees(n, e).tolist() == [1, 2, 2, 1]


def test_transitivity_by_hand(paw, path4, barbell):
    assert A.transitivity(*paw) == pytest.approx(3 / 5)
    assert A.transitivity(*path4) == 0.0
    assert A.transitivity(4, [(a, b) for a in range(4) for b in range(a)]) == pytest.approx(1.0)       # K4
    assert A.transitivity(5, []) == 0.0
    n, e = barbell                                                                                       # Grade 2, 2, 3, 3, 2, 2: Dreiergruppen 1 + 1 + 3 + 3 + 1 + 1 = 10, Dreiecke 2
    assert A.transitivity(n, e) == pytest.approx(6 / 10)


def test_assortativity_by_hand(path4):
    assert A.assortativity(*path4) == pytest.approx(-0.5)
    star = [(0, i) for i in range(1, 5)]
    assert A.assortativity(5, star) == pytest.approx(-1.0)
    cycle = [(i, (i + 1) % 6) for i in range(6)]
    assert A.assortativity(6, cycle) == 0.0                         # alle Grade gleich: nicht definiert, als 0 vereinbart
    assert A.assortativity(3, []) == 0.0


def test_giant_component_and_mean_path_length(two_triangles, path4):
    n, e = two_triangles
    mask, size = A.giant_component(n, e)
    assert size == 3 and mask.tolist() == [True, True, True, False, False, False, False]      # Gleichstand: die Komponente mit dem kleinsten Knoten
    assert A.mean_path_length(n, e) == pytest.approx(1.0)
    assert A.mean_path_length(*path4) == pytest.approx(5 / 3)
    assert A.mean_path_length(3, []) == 0.0                         # nur Einzelknoten
    assert A.giant_component(0, [])[1] == 0


def test_greedy_modularity_by_hand(path4, barbell):
    q, labels = A.greedy_modularity(*path4)
    assert q == pytest.approx(1 / 6) and labels.tolist() == [0, 0, 1, 1]
    q, labels = A.greedy_modularity(*barbell)                       # Zerlegung in die zwei Dreiecke: Q = 2 · (3/7 − (7/14)²) = 5/14
    assert q == pytest.approx(5 / 14) and labels.tolist() == [0, 0, 0, 1, 1, 1]
    q, labels = A.greedy_modularity(4, [])
    assert q == 0.0 and labels.tolist() == [0, 1, 2, 3]


def test_statistics_by_hand(path4):
    s = A.statistics(*path4)
    assert s.shape == (7,) and list(A.STAT_NAMES) == ["hub", "fano", "trans", "assort", "plen", "mod", "giant"]
    assert s.tolist() == pytest.approx([2 / 1.5, 0.25 / 1.5, 0.0, -0.5, 5 / 3, 1 / 6, 1.0])
    e = A.statistics(5, [])
    assert e.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]


def test_block_sizes_and_sbm_in_out():
    assert A.block_sizes(10, 3) == [4, 3, 3] and A.block_sizes(9, 3) == [3, 3, 3]
    p_in, p_out = A.sbm_in_out(6, 2, 2.0, 4.0)                       # Blöcke 3 + 3: 6 Paare innen, 9 außen; Kantenzahl k n / 2 = 6
    assert p_out == pytest.approx(6 / 33) and p_in == pytest.approx(24 / 33)
    assert 6 * p_in + 9 * p_out == pytest.approx(6.0)
    assert A.sbm_in_out(6, 2, 10.0, 4.0)[0] == 1.0                   # Wahrscheinlichkeit höchstens 1


def test_sbm_edges_with_extreme_probabilities():
    labels = [0, 0, 0, 1, 1, 1]
    rng = np.random.default_rng(1)
    e = A.sbm_edges(labels, [[1.0, 0.0], [0.0, 1.0]], rng)
    assert sorted(e) == [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    assert A.sbm_edges(labels, [[0.0, 0.0], [0.0, 0.0]], rng) == []
    assert len(A.sbm_edges(labels, [[1.0, 1.0], [1.0, 1.0]], rng)) == 15


def test_geometric_formulas():
    assert A.geo_pair_probability(1.0) == pytest.approx(np.pi - 8 / 3 + 0.5)                  # 0,9749
    assert A.geo_pair_probability(0.0) == 0.0 and A.geo_pair_probability(2.0) == A.geo_pair_probability(1.0)
    r = A.geo_radius(200, 4.0)
    assert A.geo_pair_probability(r) == pytest.approx(4.0 / 199)
    assert A.geo_radius(10, 100.0) == pytest.approx(1.0, abs=1e-6)                           # Grad über dem Möglichen: Radius höchstens 1


def test_geo_edges_threshold_and_count():
    rng = np.random.default_rng(3)
    assert A.geo_edges(30, 0.0, rng) == []
    assert len(A.geo_edges(30, 2.0, np.random.default_rng(3))) == 30 * 29 // 2               # Radius über der Diagonalen: vollständiger Graph


def test_generators_by_hand():
    rng = np.random.default_rng(1)
    e = A.gnm_edges(100, 200, rng)
    assert len(e) == 200 and len(set(e)) == 200 and all(u < v for u, v in e)
    assert len(A.gnm_edges(5, 100, rng)) == 10                      # höchstens alle Paare
    ba = A.ba_edges(50, 2.0, rng)
    assert len(ba) == 3 + 47 * 2 and len(set(ba)) == len(ba)         # Start: Dreieck, danach 2 Kanten je Knoten
    ba15 = A.ba_edges(2000, 1.5, np.random.default_rng(2))
    assert 2 * len(ba15) / 2000 == pytest.approx(3.0, abs=0.1)       # mittlerer Grad 2m
    tree = A.ba_edges(30, 1.0, rng)
    assert len(tree) == 1 + 28 and len(A.ba_edges(30, 0.4, rng)) == 29      # weniger als eine Kante je Knoten: ein Baum
    ring = A.ws_edges(10, 4.0, 0.0, rng)
    assert len(ring) == 20 and set(A.degrees(10, ring)) == {4}
    half = A.ws_edges(1000, 3.0, 0.0, np.random.default_rng(4))
    assert len(half) == pytest.approx(1500, abs=60)                  # Grad 3: floor(1,5) = 1, dazu jede zweite Kante des nächsten Abstands mit Wahrscheinlichkeit 0,5
    rewired = A.ws_edges(200, 4.0, 1.0, np.random.default_rng(5))
    assert len(rewired) == 400 and len(set(rewired)) == 400


def test_holme_kim_by_hand():
    e = A.holme_kim_edges(40, 2.0, 1.0, np.random.default_rng(1))
    assert len(e) == 3 + 37 * 2 and len(set(e)) == len(e)
    assert A.transitivity(40, e) > 0.2                                # Dreiecksschluss mit Wahrscheinlichkeit 1: viele Dreiecke
    plain = A.holme_kim_edges(400, 2.0, 0.0, np.random.default_rng(1))
    triad = A.holme_kim_edges(400, 2.0, 1.0, np.random.default_rng(1))
    assert A.transitivity(400, triad) > 3 * A.transitivity(400, plain)


def test_edge_switch_keeps_degrees_and_simplicity(barbell):
    n, e = barbell
    rng = np.random.default_rng(1)
    for _ in range(30):
        h = A.edge_switch(e, 50, rng)
        assert A.degrees(n, h).tolist() == A.degrees(n, e).tolist() and len(set(h)) == len(h) == len(e) and all(u < v for u, v in h)
    assert A.edge_switch([(0, 1)], 10, rng) == [(0, 1)]


def test_standardize_and_knn_by_hand():
    x = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [10.0, 5.0], [11.0, 5.0], [12.0, 5.0]])     # zweite Spalte ohne Streuung
    mu, sd = A.standardize(x)
    assert mu.tolist() == [6.0, 5.0] and sd[1] == 1.0 and sd[0] == pytest.approx(np.std([0, 1, 2, 10, 11, 12]))
    y = [0, 0, 0, 1, 1, 1]
    assert A.knn_proba(x, y, [1.5, 5.0], 2, 3).tolist() == [1.0, 0.0]
    assert A.knn_proba(x, y, [10.5, 5.0], 2, 3).tolist() == [0.0, 1.0]
    p = A.knn_proba(x, y, [6.0, 5.0], 2, 4)                                                       # Mitte: je zwei nächste auf jeder Seite
    assert p.tolist() == [0.5, 0.5] and p.sum() == 1.0
    tie = A.knn_proba(np.array([[-1.0], [1.0]]), [0, 1], [0.0], 2, 1)                             # gleicher Abstand: der kleinere Index gewinnt
    assert tie.tolist() == [1.0, 0.0]
    assert A.knn_proba(x, y, [1.5, 5.0], 2, 100).tolist() == [0.5, 0.5]                          # k größer als die Trainingsmenge: alle zählen


def test_ws_rewiring_estimate_by_hand():
    assert A.ws_rewiring_estimate(0.5, 4.0) == 0.0                   # C0 = 3 · 1 / (2 · 3) = 0,5
    assert A.ws_rewiring_estimate(0.9, 4.0) == 0.0                   # über C0: nicht umgelegt
    assert A.ws_rewiring_estimate(0.0, 4.0) == 1.0
    assert A.ws_rewiring_estimate(0.5 / 8, 4.0) == pytest.approx(0.5)
    assert A.ws_rewiring_estimate(0.1, 2.0) == 1.0                   # K = 1: ein Ring ohne Dreiecke
    assert A.ws_rewiring_estimate(0.3, 6.0) == pytest.approx(1 - (0.3 / 0.6) ** (1 / 3))        # K = 3: C0 = 3 · 2 / (2 · 5) = 0,6


def test_empirical_p_by_hand():
    assert A.empirical_p(5.0, [1, 2, 3, 4]) == pytest.approx(0.4)    # (1 + 0) / 5 · 2
    assert A.empirical_p(2.5, [1, 2, 3, 4]) == 1.0
    assert A.empirical_p(2.0, [2, 2, 2, 2]) == 1.0
    assert A.empirical_p(-1.0, [1, 2, 3, 4]) == pytest.approx(0.4)


def test_goodness_of_fit_by_hand():
    rng = np.random.default_rng(1)
    cloud = np.column_stack([rng.normal(0, 1, 50), rng.normal(5, 2, 50), np.full(50, 3.0)])
    p_each, dist, p_all = A.goodness_of_fit([0.0, 5.0, 3.0], cloud)
    assert p_each[2] == 1.0 and p_each[0] > 0.3 and p_each[1] > 0.3 and p_all > 0.3 and dist < 1.5
    p_each, dist, p_all = A.goodness_of_fit([10.0, 5.0, 3.0], cloud)
    assert p_each[0] == pytest.approx(2 / 51) and p_all == pytest.approx(1 / 51) and dist > 5          # weit außerhalb: der kleinste mögliche Wert
    _, dist, p_all = A.goodness_of_fit([0.0, 5.0, 4.0], cloud)                                         # konstante Größe weicht ab: zählt im Abstand nicht
    assert p_all > 0.3
    p_each, dist, p_all = A.goodness_of_fit([1.0, 1.0], np.ones((10, 2)))                              # lauter konstante Größen
    assert p_each.tolist() == [1.0, 1.0] and dist == 0.0 and p_all == 1.0


def test_fit_model_by_hand(barbell):
    n, e = barbell
    assert A.fit_model("er", n, e) == {"kind": "er", "n": 6, "m": 7}
    assert A.fit_model("ba", n, e)["m"] == pytest.approx(7 / 6 * 1.0)                                   # k = 14 / 6, m = k / 2 = 7 / 6
    ws = A.fit_model("ws", n, e)
    assert ws["k"] == pytest.approx(14 / 6) and 0.0 <= ws["p"] <= 1.0
    assert A.fit_model("geo", n, e)["radius"] == pytest.approx(A.geo_radius(6, 14 / 6))
    cfg = A.fit_model("config", n, e)
    assert cfg["edges"] == e and cfg["proposals"] == 70
    sbm = A.fit_model("sbm", n, e)
    assert sbm["labels"].tolist() == [0, 0, 0, 1, 1, 1]
    assert np.allclose(sbm["probs"], [[1.0, 1 / 9], [1 / 9, 1.0]])                      # innen alle 3 Paare, außen 1 von 9
    with pytest.raises(ValueError):
        A.fit_model("x", n, e)


def test_sample_fitted_by_hand(barbell):
    n, e = barbell
    rng = np.random.default_rng(1)
    assert len(A.sample_fitted(A.fit_model("er", n, e), rng)) == 7
    h = A.sample_fitted(A.fit_model("config", n, e), rng)
    assert A.degrees(n, h).tolist() == A.degrees(n, e).tolist()
    sbm = {"kind": "sbm", "n": 6, "labels": np.array([0, 0, 0, 1, 1, 1]), "probs": np.array([[1.0, 0.0], [0.0, 1.0]])}
    assert A.sample_fitted(sbm, rng) == [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    for kind in ("ba", "ws", "geo"):
        out = A.sample_fitted(A.fit_model(kind, 40, A.gnm_edges(40, 80, rng)), rng)
        assert all(u < v < 40 for u, v in out)
    with pytest.raises(ValueError):
        A.sample_fitted({"kind": "x", "n": 3}, rng)


def test_varying_columns_by_hand():
    assert A._varying_columns(np.array([[1.0, 2.0]])).tolist() == []                                 # eine Zeile: keine Streuung schätzbar
    assert A._varying_columns(np.array([[1.0, 2.0], [3.0, 2.0]])).tolist() == [0]                    # zwei Zeilen genügen; die konstante Spalte fällt weg
    assert A._varying_columns(np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 2.0]])).tolist() == [0, 1]


def test_goodness_of_fit_with_one_varying_column_and_with_three_cloud_nets():
    rng = np.random.default_rng(2)
    cloud = np.column_stack([rng.normal(size=40), np.full(40, 7.0)])
    p_each, dist, p_all = A.goodness_of_fit([30.0, 7.0], cloud)                                          # genau eine Größe streut: sie entscheidet allein
    assert p_each.tolist() == pytest.approx([2 / 41, 1.0]) and p_all == pytest.approx(1 / 41) and dist > 5
    _, dist3, p3 = A.goodness_of_fit([10.0, 10.0], np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]))      # drei Wolkennetze: schon ein Abstand, ein weit entfernter Punkt wird erkannt
    assert dist3 > 3 and p3 == pytest.approx(1 / 4)


class FixedPoints:
    """Ersatz für den Zufallsgenerator: liefert feste Punkte."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=float)

    def random(self, shape):
        return self.pts


def test_geo_edges_include_a_pair_exactly_at_the_radius():
    pts = [[0.0, 0.0], [0.3, 0.4]]                                      # Abstand 0,5 (in Gleitkommazahlen 0,5000000000000001)
    assert A.geo_edges(2, 0.5, FixedPoints(pts)) == [(0, 1)]
    assert A.geo_edges(2, 0.49, FixedPoints(pts)) == []


def test_holme_kim_with_fractional_m_and_tiny_n():
    e = A.holme_kim_edges(1500, 2.5, 0.5, np.random.default_rng(3))
    assert 2 * len(e) / 1500 == pytest.approx(5.0, abs=0.15)             # mittlerer Grad 2m
    tiny = A.holme_kim_edges(2, 2.0, 0.5, np.random.default_rng(3))
    assert len(tiny) == 3 + 2 and max(max(p) for p in tiny) == 3         # n wird auf m + 2 Knoten angehoben: Dreieck plus ein Knoten mit zwei Kanten


def test_ws_chord_edges_at_the_boundary_of_the_ring():
    assert len(A.ws_edges(7, 5.998, 0.0, np.random.default_rng(1))) == 21      # n = 7: der Abstand 3 ist noch ein eigener Abstand, vollständiger Graph
    assert len(A.ws_edges(6, 5.998, 0.0, np.random.default_rng(1))) == 12      # n = 6: Abstand 3 wäre doppelt gezählt, es gibt keine Zusatzkanten
    assert len(A.ws_edges(40, 4.0, 0.0, np.random.default_rng(1))) == 80       # ohne Bruchteil keine Zusatzkanten


def test_ba_with_tiny_n_is_raised_to_m_plus_two_nodes():
    e = A.ba_edges(2, 2.0, np.random.default_rng(3))
    assert len(e) == 3 + 2 and max(max(p) for p in e) == 3             # Dreieck plus ein Knoten mit zwei Kanten


def test_sbm_fit_with_a_singleton_group_has_no_nan():
    sbm = A.fit_model("sbm", 3, [(0, 1)])
    assert sbm["labels"].tolist() == [0, 0, 1] and sbm["probs"].tolist() == [[1.0, 0.0], [0.0, 0.0]]      # der Einzelknoten hat kein Paar: Dichte 0, nicht 0 / 0


def test_holme_kim_with_three_edges_and_full_triad_probability_has_no_duplicate_targets():
    e = A.holme_kim_edges(60, 3.0, 1.0, np.random.default_rng(1))
    assert len(e) == 6 + 56 * 3                                       # jedes neue Ziel ist ein anderer Knoten: sonst gingen Kanten als Doppelte verloren


def test_goodness_of_fit_per_column_p_values_with_tiny_and_constant_columns():
    cloud = np.array([[0.0, 3.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0]])
    p_each, _, _ = A.goodness_of_fit([9.0, 4.0], cloud)                # konstante Spalte: p = 1, obwohl der beobachtete Wert abweicht
    assert p_each.tolist() == pytest.approx([2 / 5, 1.0])
    p_each, dist, p_all = A.goodness_of_fit([5.0, 5.0], np.array([[0.0, 0.0], [1.0, 2.0]]))      # zwei Wolkennetze: p-Werte je Größe, aber kein Abstand
    assert p_each.tolist() == pytest.approx([2 / 3, 2 / 3]) and dist == 0.0 and p_all == 1.0
    p_each, _, _ = A.goodness_of_fit([5.0, 5.0], np.array([[0.0, 0.0]]))                         # ein Wolkennetz: nichts schätzbar
    assert p_each.tolist() == [1.0, 1.0]
