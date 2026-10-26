"""Szenario: erzeugte Testnetze (Modellfamilien mit typischen Parametern, Mischnetz), Trainingsnetze mit zufälligen Parametern und die echten Beispielnetze."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import mwn_algorithm as A
import mwn_constants as C

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def _data():
    return json.loads((DATA / "netze.json").read_text(encoding="utf-8"))


def family_member(kind: str, n: int, k: float, rng: np.random.Generator, ws_p: float, sbm_ratio: float):
    """Ein Netz der Familie `kind` mit n Knoten und mittlerem Grad k (die Kantenzahl stimmt bis auf Rundung und Zufall).
    Rückgabe: (Knotenzahl, Kanten). Beim Konfigurationsmodell werden die Grade eines Barabási–Albert-Netzes zufällig neu verdrahtet."""
    if kind == "er":
        return n, A.gnm_edges(n, int(round(k * n / 2.0)), rng)
    if kind == "ba":
        return n, A.ba_edges(n, k / 2.0, rng)
    if kind == "config":
        edges = A.ba_edges(n, k / 2.0, rng)
        return n, A.edge_switch(edges, 10 * len(edges), rng)
    if kind == "ws":
        return n, A.ws_edges(n, k, ws_p, rng)
    if kind == "geo":
        return n, A.geo_edges(n, A.geo_radius(n, k), rng)
    if kind == "sbm":
        labels = np.repeat(np.arange(C.SBM_BLOCKS), A.block_sizes(n, C.SBM_BLOCKS))
        p_in, p_out = A.sbm_in_out(n, C.SBM_BLOCKS, k, sbm_ratio)
        probs = np.full((C.SBM_BLOCKS, C.SBM_BLOCKS), p_out) + np.eye(C.SBM_BLOCKS) * (p_in - p_out)
        return n, A.normalize(A.sbm_edges(labels, probs, rng))
    raise ValueError(f"unbekannte Modellfamilie: {kind}")


def training_member(kind: str, n: int, k: float, rng: np.random.Generator):
    """Trainingsnetz: Parameter von Watts–Strogatz und Blockmodell zufällig aus den Bereichen der Konstanten."""
    ws_p = float(rng.uniform(*C.WS_P_RANGE))
    ratio = float(rng.uniform(*C.SBM_RATIO_RANGE))
    return family_member(kind, n, k, rng, ws_p, ratio)[1]


def hk_member(n: int, k: float, rng: np.random.Generator):
    """Holme–Kim-Netz mit mittlerem Grad k: Hubs wie bei Barabási–Albert, dazu Dreiecksschluss (Wahrscheinlichkeit HK_P_TRIAD). Kein Modell der Liste."""
    return A.holme_kim_edges(n, k / 2.0, C.HK_P_TRIAD, rng)


def make_network(key: str, n: int, k: float, seed: int):
    """(Knotenzahl, Kanten, Modellfamilie oder None) des gewählten Netzes. Erzeugte Netze hängen nur von (key, n, k, seed) ab; die echten Netze sind fest (n und k werden ignoriert)."""
    if key in C.REAL:
        d = _data()[key]
        return int(d["n"]), [tuple(e) for e in d["edges"]], None
    rng = np.random.default_rng(seed)
    if key == "hk":
        return n, hk_member(n, k, rng), None
    if key in C.MODEL_ORDER:
        n_eff, edges = family_member(key, n, k, rng, C.WS_P_TYPICAL, C.SBM_RATIO_TYPICAL)
        return n_eff, edges, key
    raise ValueError(f"unbekanntes Netz: {key}")


@lru_cache(maxsize=24)
def training_set(n: int, k_key: int):
    """Trainingsmenge für Netze mit n Knoten und mittlerem Grad k_key / 100: (Größen, Klassennummer je Netz). Deterministisch aus (n, k_key)."""
    k = k_key / 100.0
    rng = np.random.default_rng([C.TRAIN_SEED, n, k_key])
    xs, ys = [], []
    for ci, kind in enumerate(C.MODEL_ORDER):
        for _ in range(C.TRAIN_PER_CLASS):
            xs.append(A.statistics(n, training_member(kind, n, k, rng)))
            ys.append(ci)
    return np.array(xs), np.array(ys)
