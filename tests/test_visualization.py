"""Abbildungen: Spuren je Figur, Werte, Farben, Achsen gesperrt."""

import numpy as np

import mwn_constants as C
import mwn_evaluation as ev
import mwn_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def analysis(network="er", **kw):
    return ev.analyse(ev.Settings(network=network, n=60, k=3.0, sims=30, seed=3, **kw))


def test_probability_figure_highlights_the_top_and_marks_the_truth():
    a = analysis("er")
    fig = V.build_probs(a)
    assert _locked(fig) and list(fig.data[0].x) == V.MODEL_NAMES and list(fig.data[0].y) == [float(p) for p in a.probs]
    top = int(np.argmax(a.probs))
    assert fig.data[0].marker.color[top] == C.COLOR_TOP and sum(c == C.COLOR_TOP for c in fig.data[0].marker.color) == 1
    star = fig.data[1]
    assert star.name == "erzeugt aus" and list(star.x) == ["Erdős–Rényi"] and star.y[0] == float(a.probs[0]) + 0.1
    assert len(V.build_probs(analysis("karate")).data) == 1                      # echtes Netz: kein Stern


def test_check_figure_colours_by_threshold_and_draws_the_line():
    a = analysis("er")
    fig = V.build_check(a)
    assert _locked(fig) and list(fig.data[0].y) == [float(p) for p in a.p_all]
    assert list(fig.data[0].marker.color) == [C.COLOR_PASS if p >= C.ALPHA else C.COLOR_FAIL for p in a.p_all]
    assert fig.layout.shapes[0].y0 == C.ALPHA == fig.layout.shapes[0].y1


def test_heatmap_clips_to_the_colour_range_and_labels_every_cell():
    a = analysis("hk")
    fig = V.build_heatmap(a)
    z = np.array(fig.data[0].z)
    assert _locked(fig) and z.shape == (6, 7) and z.max() <= V.Z_CLIP and z.min() >= -V.Z_CLIP
    assert list(fig.data[0].x) == [C.STAT_SHORT[s] for s in ["hub", "fano", "trans", "assort", "plen", "mod", "giant"]] and list(fig.data[0].y) == V.MODEL_NAMES
    assert np.array(fig.data[0].text).shape == (6, 7)
    assert all(t.startswith(("+", "-", "−")) or t == "∞" for row in fig.data[0].text for t in row)


def test_heatmap_shows_infinity_for_huge_z():
    a = analysis("er")
    a = ev.Analysis(a.settings, a.n, a.edges, a.mean_degree, a.truth, a.observed, a.probs, a.p_each, np.full((6, 7), 99.0), a.cloud_mean, a.dist, a.p_all)
    fig = V.build_heatmap(a)
    assert fig.data[0].text[0][0] == "∞" and np.array(fig.data[0].z).max() == V.Z_CLIP


def test_confusion_and_pass_matrix_figures():
    conf = np.arange(36).reshape(6, 6)
    fig = V.build_confusion(conf)
    assert _locked(fig) and np.array(fig.data[0].z).tolist() == conf.tolist() and list(fig.data[0].y) == V.MODEL_NAMES
    rate = np.eye(6) * 0.5
    fig = V.build_pass_matrix(rate)
    assert _locked(fig) and np.array(fig.data[0].z).tolist() == rate.tolist() and fig.data[0].text[0][0] == "50 %" and fig.data[0].zmax == 1.0
