"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keiner auf einem Reglerrand und keiner, der durch eine einzelne Regleränderung vom Standard entsteht."""

import mwn_constants as C
import mwn_evaluation as ev
import mwn_presets as P

FIRST = "Watts–Strogatz, sauber erkannt"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["network"] in C.NETWORKS and C.N_MIN <= p["n"] <= C.N_MAX and C.K_MIN <= p["k"] <= C.K_MAX, name
        assert C.SIMS_MIN <= p["sims"] <= C.SIMS_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[FIRST]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_preset_sits_on_a_slider_bound():
    bounds = {"n": (C.N_MIN, C.N_MAX), "k": (C.K_MIN, C.K_MAX), "sims": (C.SIMS_MIN, C.SIMS_MAX)}
    for name, preset in C.PRESETS.items():
        for key, (lo, hi) in bounds.items():
            assert preset[key] not in (lo, hi), (name, key)


def test_no_single_change_of_the_default_produces_another_preset():
    default = C.PRESETS[FIRST]
    for name, p in C.PRESETS.items():
        if name != FIRST:
            assert sum(1 for k in p if p[k] != default[k]) >= 2, name


def test_presets_are_pairwise_different_and_cover_the_findings():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)
    assert {p["network"] for p in C.PRESETS.values()} == {"ws", "ba", "lesmis", "karate", "hk"}


def test_slider_steps_reach_the_presets():
    for name, p in C.PRESETS.items():
        for value, lo, step in ((p["n"], C.N_MIN, 50), (p["k"], C.K_MIN, 0.5), (p["sims"], C.SIMS_MIN, 10)):
            k = (value - lo) / step
            assert abs(k - round(k)) < 1e-9, (name, value)
