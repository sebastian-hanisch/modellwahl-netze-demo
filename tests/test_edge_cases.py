"""Randfälle: kleinste und größte Reglerwerte für jedes erzeugte Netz, sehr dünne und sehr dichte Netze, Netze ohne Kanten, Einzelknoten."""

import numpy as np
import pytest

import mwn_algorithm as A
import mwn_constants as C
import mwn_evaluation as ev


def check(a):
    assert np.isfinite(a.observed).all() and np.isfinite(a.z).all() and np.isfinite(a.dist).all()
    assert a.probs.sum() == pytest.approx(1.0) and ((a.p_all > 0) & (a.p_all <= 1)).all() and ((a.p_each > 0) & (a.p_each <= 1)).all()
    assert ev.verdict_choice(a.probs, a.passes)[0] in ("success", "info", "warning") and len(ev.fit_rows(a)) == 6


def test_every_generated_network_at_the_smallest_settings():
    for key in C.GENERATED:
        check(ev.analyse(ev.Settings(network=key, n=C.N_MIN, k=C.K_MIN, sims=C.SIMS_MIN, seed=1)))


def test_dense_small_networks_run():
    for key in ("er", "ba", "sbm", "hk"):
        check(ev.analyse(ev.Settings(network=key, n=C.N_MIN, k=C.K_MAX, sims=C.SIMS_MIN, seed=2)))


def test_the_largest_settings_run():
    check(ev.analyse(ev.Settings(network="geo", n=C.N_MAX, k=C.K_MAX, sims=C.SIMS_MIN, seed=3)))


def test_every_real_network_runs_at_the_smallest_number_of_simulations():
    for key in C.REAL:
        check(ev.analyse(ev.Settings(network=key, sims=C.SIMS_MIN, seed=1)))


def test_statistics_of_degenerate_networks():
    assert A.statistics(1, []).tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    s = A.statistics(2, [(0, 1)])
    assert s.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])                  # eine Kante: beide Grade 1, Weglänge 1, nichts zu zerlegen
    assert np.isfinite(A.statistics(30, A.gnm_edges(30, 435, np.random.default_rng(1)))).all()                      # vollständiger Graph


def test_fitting_a_sparse_tree_like_network_does_not_fail():
    rng = np.random.default_rng(1)
    tree = A.ba_edges(60, 1.0, rng)                                                           # mittlerer Grad unter 2: Barabási–Albert-Anpassung mit m < 1
    for kind in C.MODEL_ORDER:
        fit = A.fit_model(kind, 60, tree)
        out = A.sample_fitted(fit, rng)
        assert all(0 <= u < v < 60 for u, v in out), kind


def test_the_smallest_real_network_cannot_reject_much():
    a = ev.analyse(ev.Settings(network="florenz", sims=C.SIMS_MIN, seed=5))
    assert a.n == 15 and len(a.passes) >= 3


def test_goodness_of_fit_with_tiny_clouds_raises_no_objection():
    for rows in (1, 2):
        p_each, dist, p_all = A.goodness_of_fit([0.5, 0.5], np.array([[0.0, 0.0], [1.0, 2.0]])[:rows])
        assert dist == 0.0 and p_all == 1.0 and p_each.tolist() == [1.0, 1.0]
    p_each, dist, p_all = A.goodness_of_fit([0.5, 0.5], np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]))      # drei Wolkennetze: der kleinste Fall mit Abstand
    assert np.isfinite(dist) and 0 < p_all <= 1
