"""Orakel-Tests mit unabhängigem Rechenweg (ORACLE-PLAYBOOK): `networkx` für Transitivität, Assortativität, Weglänge, Komponenten, Modularität und Watts–Strogatz,
eine naive Umsetzung der gierigen Modularitätsmaximierung (alles in jedem Schritt neu gerechnet), `scikit-learn` für die k nächsten Nachbarn,
Monte-Carlo für die Formel des geometrischen Radius, exakte Aufzählung aller Graphen mit gegebener Gradfolge für den Kantentausch, Chi-Quadrat für G(n, m)."""

import itertools

import numpy as np
import pytest
from scipy import stats

import mwn_algorithm as A
import mwn_constants as C
from mwn_scenario import family_member, hk_member

nx = pytest.importorskip("networkx")


def graph(n, edges):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return g


def sample_networks(count=3, n=120, k=3.5):
    """Je Familie `count` Netze mit festen Zufalls-Seeds."""
    out = []
    for ci, kind in enumerate(C.MODEL_ORDER):
        for j in range(count):
            rng = np.random.default_rng([11, ci, j])
            out.append((kind,) + family_member(kind, n, k, rng, 0.1, 15.0))
    return out


def test_transitivity_assortativity_and_degrees_equal_networkx():
    for kind, n, e in sample_networks():
        g = graph(n, e)
        assert A.transitivity(n, e) == pytest.approx(nx.transitivity(g), abs=1e-12), kind
        r = nx.degree_assortativity_coefficient(g)
        assert A.assortativity(n, e) == pytest.approx(0.0 if r != r else r, abs=1e-9), kind
        assert A.degrees(n, e).tolist() == [d for _, d in sorted(g.degree())], kind


def test_giant_component_and_mean_path_length_equal_networkx():
    for kind, n, e in sample_networks():
        g = graph(n, e)
        big = max(nx.connected_components(g), key=len)
        mask, size = A.giant_component(n, e)
        assert size == len(big) and set(np.where(mask)[0].tolist()) == set(big), kind
        assert A.mean_path_length(n, e) == pytest.approx(nx.average_shortest_path_length(g.subgraph(big)), abs=1e-9), kind


def naive_greedy_modularity(n, edges):
    """Dieselbe gierige Zerlegung, in jedem Schritt vollständig neu gerechnet: Gewinn jedes verbundenen Gruppenpaares aus den Kantenzahlen."""
    m = len(edges)
    groups = {i: {i} for i in range(n)}
    deg = {i: 0 for i in range(n)}
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    def modularity():
        total = 0.0
        for g in groups.values():
            inside = sum(1 for u, v in edges if u in g and v in g)
            d = sum(deg[x] for x in g)
            total += inside / m - (d / (2 * m)) ** 2
        return total

    while True:
        best, pair = 1e-12, None
        for a, b in itertools.combinations(sorted(groups), 2):
            between = sum(1 for u, v in edges if (u in groups[a] and v in groups[b]) or (u in groups[b] and v in groups[a]))
            if between == 0:
                continue
            da, db = sum(deg[x] for x in groups[a]), sum(deg[x] for x in groups[b])
            gain = 2.0 * (between / (2 * m) - (da / (2 * m)) * (db / (2 * m)))
            if gain > best + 1e-12:
                best, pair = gain, (a, b)
        if pair is None:
            break
        a, b = pair
        groups[a] |= groups.pop(b)
    labels = np.empty(n, dtype=int)
    for new, key in enumerate(sorted(groups, key=lambda k: min(groups[k]))):
        for x in groups[key]:
            labels[x] = new
    return modularity(), labels


def test_greedy_modularity_equals_the_naive_recomputation():
    for kind, n, e in sample_networks(count=2, n=40, k=3.0):
        q, labels = A.greedy_modularity(n, e)
        q_naive, labels_naive = naive_greedy_modularity(n, e)
        assert q == pytest.approx(q_naive, abs=1e-9), kind
        assert labels.tolist() == labels_naive.tolist(), kind


def test_greedy_modularity_is_the_modularity_of_its_partition_and_close_to_networkx():
    for kind, n, e in sample_networks():
        q, labels = A.greedy_modularity(n, e)
        g = graph(n, e)
        groups = [set(np.where(labels == c)[0].tolist()) for c in range(int(labels.max()) + 1)]
        assert q == pytest.approx(nx.community.modularity(g, groups), abs=1e-9), kind
        q_nx = nx.community.modularity(g, nx.community.greedy_modularity_communities(g))
        assert abs(q - q_nx) < 0.03, kind                                  # gleiches Verfahren, die Reihenfolge bei Gleichstand kann sich unterscheiden


def test_knn_equals_scikit_learn():
    sk = pytest.importorskip("sklearn.neighbors")
    pre = pytest.importorskip("sklearn.preprocessing")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(180, 7)) * np.arange(1, 8) + np.arange(7)
    y = np.repeat(np.arange(6), 30)
    scaler = pre.StandardScaler().fit(x)
    clf = sk.KNeighborsClassifier(n_neighbors=15).fit(scaler.transform(x), y)
    for _ in range(25):
        q = rng.normal(size=7) * np.arange(1, 8) + np.arange(7)
        assert A.knn_proba(x, y, q, 6, 15).tolist() == pytest.approx(clf.predict_proba(scaler.transform([q]))[0].tolist())


def test_geometric_pair_probability_equals_monte_carlo():
    rng = np.random.default_rng(2)
    pts = rng.random((400000, 2, 2))
    d = np.sqrt(((pts[:, 0] - pts[:, 1]) ** 2).sum(axis=1))
    for r in (0.1, 0.3, 0.7, 1.0):
        assert A.geo_pair_probability(r) == pytest.approx(np.mean(d <= r), abs=0.004)


def test_geometric_graph_has_the_requested_mean_degree():
    n, k = 200, 4.0
    r = A.geo_radius(n, k)
    degs = [2 * len(A.geo_edges(n, r, np.random.default_rng(s))) / n for s in range(30)]
    assert np.mean(degs) == pytest.approx(k, abs=0.15)                     # der Randeffekt ist in der Formel enthalten


def test_watts_strogatz_without_rewiring_equals_networkx_and_with_rewiring_keeps_the_edge_count():
    for n, k in ((20, 4), (30, 6), (15, 2)):
        ours = set(A.ws_edges(n, float(k), 0.0, np.random.default_rng(1)))
        theirs = {(min(u, v), max(u, v)) for u, v in nx.watts_strogatz_graph(n, k, 0.0).edges()}
        assert ours == theirs
    assert len(A.ws_edges(100, 4.0, 0.5, np.random.default_rng(1))) == 200


def test_ws_rewiring_estimate_recovers_the_rewiring_probability():
    for p in (0.05, 0.1, 0.2, 0.4):
        est = [A.ws_rewiring_estimate(A.transitivity(400, A.ws_edges(400, 6.0, p, np.random.default_rng(s))), 6.0) for s in range(6)]
        assert np.mean(est) == pytest.approx(p, abs=0.04 + 0.1 * p)


def test_holme_kim_clustering_is_close_to_networkx_powerlaw_cluster():
    ours = np.mean([A.transitivity(300, A.holme_kim_edges(300, 2.0, 0.9, np.random.default_rng(s))) for s in range(12)])
    theirs = np.mean([nx.transitivity(nx.powerlaw_cluster_graph(300, 2, 0.9, seed=s)) for s in range(12)])
    assert abs(ours - theirs) < 0.06 and ours > 0.1 and theirs > 0.1
    plain = np.mean([nx.transitivity(nx.barabasi_albert_graph(300, 2, seed=s)) for s in range(12)])
    assert ours > 2 * plain


def test_geometric_and_er_and_ba_have_the_known_mean_degree():
    n, k = 300, 3.5
    for kind in ("er", "ba", "config", "ws", "sbm"):
        degs = [2 * len(family_member(kind, n, k, np.random.default_rng(s), 0.1, 15.0)[1]) / n for s in range(8)]
        assert np.mean(degs) == pytest.approx(k, abs=0.15), kind


def all_graphs_with_degrees(n, degrees):
    """Alle einfachen Graphen mit der gegebenen Gradfolge (Aufzählung aller Kantenmengen der richtigen Größe; nur für sehr kleine n)."""
    pairs = list(itertools.combinations(range(n), 2))
    m = sum(degrees) // 2
    out = []
    for es in itertools.combinations(pairs, m):
        d = [0] * n
        for u, v in es:
            d[u] += 1
            d[v] += 1
        if d == list(degrees):
            out.append(es)
    return out


def test_the_lazy_edge_switch_chain_is_uniform_over_all_graphs_with_the_degree_sequence():
    n, degrees = 6, [2] * 6
    graphs = all_graphs_with_degrees(n, degrees)
    assert len(graphs) == 70
    start = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]            # ein 6-Zyklus
    rng = np.random.default_rng(3)
    counts = {g: 0 for g in graphs}
    for _ in range(7000):
        counts[tuple(A.edge_switch(start, 120, rng))] += 1
    assert all(v > 0 for v in counts.values())
    assert stats.chisquare(list(counts.values())).pvalue > 0.001


def test_gnm_is_uniform_over_all_edge_sets():
    n, m, draws = 5, 3, 6000                                           # C(10, 3) = 120 mögliche Kantenmengen
    rng = np.random.default_rng(5)
    counts = {}
    for _ in range(draws):
        e = tuple(A.gnm_edges(n, m, rng))
        counts[e] = counts.get(e, 0) + 1
    assert len(counts) == 120 and stats.chisquare(list(counts.values())).pvalue > 0.001


def test_sbm_edge_probabilities_equal_the_matrix_by_chi_square():
    labels = [0, 0, 1, 1]
    probs = np.array([[0.8, 0.2], [0.2, 0.5]])
    rng = np.random.default_rng(6)
    counts = {}
    for _ in range(4000):
        for e in A.sbm_edges(labels, probs, rng):
            counts[e] = counts.get(e, 0) + 1
    expected = {(0, 1): 0.8, (2, 3): 0.5, (0, 2): 0.2, (0, 3): 0.2, (1, 2): 0.2, (1, 3): 0.2}
    for e, p in expected.items():
        assert counts[e] / 4000 == pytest.approx(p, abs=0.03), e


def test_goodness_of_fit_keeps_its_size_and_has_power():
    """Kalibrierung gegen eine Normalverteilung: ein frischer Zug aus derselben Verteilung besteht mit etwa 95 %, ein um 2,5 Streuungen verschobener fast nie."""
    rng = np.random.default_rng(8)
    passes = shifted = 0
    trials = 300
    for _ in range(trials):
        cloud = rng.normal(size=(100, 5))
        _, _, p_same = A.goodness_of_fit(rng.normal(size=5), cloud)
        _, _, p_far = A.goodness_of_fit(rng.normal(size=5) + 2.5, cloud)
        passes += p_same >= 0.05
        shifted += p_far >= 0.05
    assert 0.90 <= passes / trials <= 0.99
    assert shifted / trials < 0.05


def test_holme_kim_member_is_a_network_with_the_requested_degree():
    n, e = 300, hk_member(300, 3.5, np.random.default_rng(1))
    assert 2 * len(e) / n == pytest.approx(3.5, abs=0.2)
