"""Auswertung: Analyse eines Netzes (Klassifikator und Anpassungsprüfung), Messreihen und Verdikte als reine Funktionen."""

from dataclasses import dataclass

import numpy as np

import mwn_algorithm as A
import mwn_constants as C
from mwn_scenario import family_member, make_network, training_set


def num(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    """Anteil als Prozentzahl mit Dezimalkomma, zum Beispiel 0,44 → '44 %'."""
    return f"{num(100.0 * x, digits)} %"


def k_key(k: float) -> int:
    """Schlüssel der Trainingsmenge: mittlerer Grad auf 0,1 gerundet, in Hundertsteln (Netze mit fast gleichem Grad teilen sich die Trainingsnetze)."""
    return 10 * int(round(10.0 * k))


@dataclass(frozen=True)
class Settings:
    network: str = C.DEFAULT_NETWORK
    n: int = C.DEFAULT_N
    k: float = C.DEFAULT_K
    sims: int = C.DEFAULT_SIMS
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    n: int
    edges: int
    mean_degree: float
    truth: str | None            # Modellfamilie, aus der das Netz erzeugt wurde; None bei Mischnetz und echten Netzen
    observed: np.ndarray         # sieben Größen
    probs: np.ndarray            # Anteil der k nächsten Trainingsnetze je Modellfamilie (Reihenfolge MODEL_ORDER)
    p_each: np.ndarray           # (6, 7): p-Wert je Modell und Größe
    z: np.ndarray                # (6, 7): (beobachtet − Mittel der Wolke) / Streuung der Wolke, auf ±99 begrenzt
    cloud_mean: np.ndarray       # (6, 7)
    dist: np.ndarray             # (6,): Gesamtabstand je Modell
    p_all: np.ndarray            # (6,): Gesamt-p-Wert je Modell

    @property
    def top(self) -> str:
        """Modellfamilie mit dem größten Anteil unter den nächsten Nachbarn (bei Gleichstand die zuerst genannte)."""
        return C.MODEL_ORDER[int(np.argmax(self.probs))]

    @property
    def passes(self) -> tuple:
        """Modelle, die die Prüfung bestehen (Gesamt-p-Wert mindestens ALPHA), in der Reihenfolge MODEL_ORDER."""
        return tuple(kind for kind, p in zip(C.MODEL_ORDER, self.p_all) if p >= C.ALPHA)


def standardized(observed: float, mean: float, sd: float) -> float:
    """(beobachtet − Mittel) / Streuung, auf ±99 begrenzt; ohne Streuung 0 bei Gleichheit, sonst ±99."""
    if sd <= 1e-9:
        return 0.0 if abs(observed - mean) <= 1e-9 else float(np.sign(observed - mean)) * 99.0
    return float(max(-99.0, min(99.0, (observed - mean) / sd)))


def analyse(settings: Settings) -> Analysis:
    n, edges, truth = make_network(settings.network, settings.n, settings.k, settings.seed)
    m = len(edges)
    k_obs = 2.0 * m / n
    obs = A.statistics(n, edges)
    x, y = training_set(n, k_key(k_obs))
    probs = A.knn_proba(x, y, obs, len(C.MODEL_ORDER), C.KNN_K)
    p_each = np.zeros((6, 7))
    z = np.zeros((6, 7))
    mean = np.zeros((6, 7))
    dist = np.zeros(6)
    p_all = np.zeros(6)
    for mi, kind in enumerate(C.MODEL_ORDER):
        fit = A.fit_model(kind, n, edges)
        rng = np.random.default_rng([settings.seed, 2, mi])
        cloud = np.array([A.statistics(n, A.sample_fitted(fit, rng)) for _ in range(settings.sims)])
        pe, d, pa = A.goodness_of_fit(obs, cloud)
        p_each[mi], dist[mi], p_all[mi] = pe, d, pa
        mean[mi] = cloud.mean(axis=0)
        sd = cloud.std(axis=0, ddof=1)
        z[mi] = [standardized(obs[c], mean[mi][c], sd[c]) for c in range(7)]
    return Analysis(settings, n, m, k_obs, truth, obs, probs, p_each, z, mean, dist, p_all)


# --- Verdikte ------------------------------------------------------------------------------------------------------------------------------


def names(kinds) -> str:
    """Modellnamen als Aufzählung mit Komma und „und“."""
    labels = [C.MODEL_LABELS[k] for k in kinds]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " und " + labels[-1]


def verdict_choice(probs, passes) -> tuple[str, str]:
    """Verdikt der Kernfrage als (Art, Text), rein: `probs` Anteile je Modell in der Reihenfolge MODEL_ORDER, `passes` die Modelle, die die Prüfung bestehen."""
    probs = np.asarray(probs, dtype=float)
    top = C.MODEL_ORDER[int(np.argmax(probs))]
    ptop = float(probs.max())
    head = f"Der Klassifikator nennt **{C.MODEL_LABELS[top]}** ({pct(ptop)} der nächsten Nachbarn)"
    if not passes:
        return "warning", (
            f"{head}, aber **kein** Modell der Liste besteht die Prüfung: die Antwort ist erzwungen. Das Netz kann aus keiner der sechs Familien stammen, "
            "oder die angepassten Parameter treffen es nicht."
        )
    if top in passes and len(passes) == 1:
        return "success", f"{head}, und nur dieses Modell besteht die Prüfung: Klassifikator und Prüfung stimmen überein."
    if top in passes:
        others = [p for p in passes if p != top]
        return "info", f"{head}. Die Prüfung lässt auch {names(others)} zu: die sieben Größen trennen diese Modelle hier nicht."
    return "warning", f"{head}, aber dieses Modell besteht die Prüfung nicht; bestanden haben {names(passes)}. Klassifikator und Prüfung widersprechen sich."


def truth_sentence(truth: str | None, key: str) -> str:
    """Satz zur bekannten Herkunft des Netzes."""
    if truth is not None:
        return f"Erzeugt wurde das Netz aus: **{C.MODEL_LABELS[truth]}**."
    if key == "hk":
        return "Erzeugt wurde das Netz als Barabási–Albert-Netz mit Dreiecksschluss (Holme und Kim), also aus **keiner** der sechs Familien."
    return "Die wahre Herkunft eines echten Netzes ist unbekannt; zu erwarten ist, dass keines der Modelle genau passt."


def stats_rows(a: Analysis) -> list[dict]:
    return [{"Größe": C.STAT_LABELS[name], "beobachtet": num(float(a.observed[c]), 3)} for c, name in enumerate(A.STAT_NAMES)]


def fit_rows(a: Analysis) -> list[dict]:
    """Eine Zeile je Modell: Klassifikator-Anteil, Gesamt-p-Wert, Ergebnis der Prüfung und die auffälligste Größe (größter Betrag von z)."""
    rows = []
    for mi, kind in enumerate(C.MODEL_ORDER):
        worst = int(np.argmax(np.abs(a.z[mi])))
        rows.append({
            "Modell": C.MODEL_LABELS[kind],
            "Anteil Klassifikator": pct(float(a.probs[mi])),
            "Gesamt-p-Wert": num(float(a.p_all[mi]), 3),
            "Prüfung": "besteht" if a.p_all[mi] >= C.ALPHA else "verworfen",
            "auffälligste Größe": f"{C.STAT_LABELS[A.STAT_NAMES[worst]]} (z = {num(float(a.z[mi][worst]), 1)})",
        })
    return rows


# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------

FEATURE_SETS = {"alle sieben Größen": tuple(range(7)), "nur die zwei Gradgrößen": C.DEGREE_STATS, "ohne die Gradgrößen": C.NON_DEGREE_STATS}


def series_confusion(n: int = C.SERIES_N, k: float = C.SERIES_K, per_class: int = C.SERIES_PER_CLASS_CONFUSION) -> dict:
    """Verwechslungsmatrizen des Klassifikators auf frisch erzeugten Netzen mit typischen Parametern, je Größenauswahl. Zeilen: erzeugt aus, Spalten: erkannt als."""
    x, y = training_set(n, k_key(k))
    stats = {}
    for ci, kind in enumerate(C.MODEL_ORDER):
        for j in range(per_class):
            rng = np.random.default_rng([C.SERIES_SEED, ci, j])
            n_eff, edges = family_member(kind, n, k, rng, C.WS_P_TYPICAL, C.SBM_RATIO_TYPICAL)
            stats[(ci, j)] = A.statistics(n_eff, edges)
    out = {}
    for name, cols in FEATURE_SETS.items():
        cols = list(cols)
        conf = np.zeros((6, 6), dtype=int)
        for (ci, _), s in stats.items():
            pr = A.knn_proba(x[:, cols], y, s[cols], 6, C.KNN_K)
            conf[ci, int(np.argmax(pr))] += 1
        out[name] = {"conf": conf, "accuracy": float(np.trace(conf) / conf.sum())}
    return {"sets": out, "per_class": per_class, "n": n, "k": k}


def series_fit(n: int = C.SERIES_N, k: float = C.SERIES_K, per_class: int = C.SERIES_PER_CLASS_FIT, sims: int = C.SERIES_SIMS) -> dict:
    """Anteil der Netze je Familie, die die Prüfung gegen jedes der sechs Modelle bestehen. Zeilen: erzeugt aus, Spalten: geprüftes Modell."""
    passed = np.zeros((6, 6))
    for ci, kind in enumerate(C.MODEL_ORDER):
        for j in range(per_class):
            rng = np.random.default_rng([C.SERIES_SEED, 1, ci, j])
            n_eff, edges = family_member(kind, n, k, rng, C.WS_P_TYPICAL, C.SBM_RATIO_TYPICAL)
            obs = A.statistics(n_eff, edges)
            for mi, mkind in enumerate(C.MODEL_ORDER):
                fit = A.fit_model(mkind, n_eff, edges)
                rs = np.random.default_rng([C.SERIES_SEED, 3, ci, j, mi])
                cloud = np.array([A.statistics(n_eff, A.sample_fitted(fit, rs)) for _ in range(sims)])
                _, _, pa = A.goodness_of_fit(obs, cloud)
                passed[ci, mi] += pa >= C.ALPHA
    return {"rate": passed / per_class, "per_class": per_class, "sims": sims, "n": n, "k": k}


def series_real(keys=("karate", "lesmis", "florenz", "hk")) -> list[dict]:
    """Erzwungene Antwort des Klassifikators und Ergebnis der Prüfung für die echten Netze und das Holme–Kim-Netz (feste Zufalls-Seeds)."""
    rows = []
    for key in keys:
        a = analyse(Settings(network=key, seed=C.SERIES_SEED))
        best = int(np.argmax(a.p_all))
        rows.append({
            "key": key, "name": C.NETWORKS[key], "n": a.n, "k": a.mean_degree, "top": a.top, "p_top": float(a.probs.max()),
            "passes": a.passes, "best": C.MODEL_ORDER[best], "best_p": float(a.p_all[best]),
        })
    return rows


def recognition(plan=None) -> dict:
    """Für jedes Netz des Plans (Name → Zahl der Zufalls-Seeds 1 bis n) mit den Voreinstellungen: wie oft nennt der Klassifikator die wahre Familie, wie oft besteht sie die Prüfung,
    wie oft besteht kein Modell, und welche Familien nennt der Klassifikator überhaupt."""
    plan = C.RECOGNITION_PLAN if plan is None else plan
    out = {}
    for key, count in plan.items():
        right = own = none = 0
        named = set()
        for seed in range(1, count + 1):
            a = analyse(Settings(network=key, seed=seed))
            right += a.top == key
            own += key in a.passes
            none += not a.passes
            named.add(a.top)
        out[key] = {"runs": count, "right": right, "own": own, "none": none, "named": tuple(sorted(named))}
    return out


def off_diagonal_max(matrix):
    """(Zeile, Spalte, Wert) des größten Eintrags außerhalb der Diagonale; bei Gleichstand der erste in Zeilenfolge."""
    m = np.array(matrix, dtype=float)
    np.fill_diagonal(m, -np.inf)
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    return int(i), int(j), float(m[i, j])


def verdict_confusion(series: dict) -> tuple[str, str]:
    """Verdikt der Verwechslungsreihe als (Art, Text), rein."""
    sets = series["sets"]
    total = 6 * series["per_class"]
    allc = sets["alle sieben Größen"]
    text = (
        f"Mit allen sieben Größen erkennt der Klassifikator **{pct(allc['accuracy'])}** der {total} frisch erzeugten Netze, "
        f"mit nur den beiden Gradgrößen {pct(sets['nur die zwei Gradgrößen']['accuracy'])}, ohne die Gradgrößen {pct(sets['ohne die Gradgrößen']['accuracy'])}."
    )
    i, j, count = off_diagonal_max(allc["conf"])
    if count <= 0:
        return "success", text + " Es gibt keine Verwechslung."
    return "info", text + f" Am häufigsten verwechselt wird {C.MODEL_LABELS[C.MODEL_ORDER[i]]} mit {C.MODEL_LABELS[C.MODEL_ORDER[j]]} ({int(count)} von {series['per_class']} Netzen)."


def verdict_fit(series: dict) -> tuple[str, str]:
    """Verdikt der Prüfungsreihe als (Art, Text), rein: Anteil, mit dem das wahre Modell besteht, und der größte Anteil eines falschen Modells."""
    rate = np.asarray(series["rate"], dtype=float)
    own = float(np.mean(np.diag(rate)))
    i, j, worst = off_diagonal_max(rate)
    off = float((rate.sum() - np.trace(rate)) / 30.0)
    text = (
        f"Das wahre Modell besteht die Prüfung in **{pct(own)}** der Fälle (zu erwarten sind etwa 95 %), ein falsches im Mittel in {pct(off)}. "
        f"Am häufigsten besteht ein falsches Modell bei {C.MODEL_LABELS[C.MODEL_ORDER[i]]}-Netzen: {C.MODEL_LABELS[C.MODEL_ORDER[j]]} in {pct(worst)} der Fälle "
        f"({series['per_class']} Netze je Familie, {series['sims']} Simulationen je Prüfung)."
    )
    return "info", text


def verdict_real(rows: list[dict]) -> tuple[str, str]:
    """Verdikt der Reihe zu den echten Netzen als (Art, Text), rein."""
    none = [r for r in rows if not r["passes"]]
    parts = []
    for r in rows:
        if r["passes"]:
            parts.append(f"{r['name']}: Klassifikator {C.MODEL_LABELS[r['top']]} ({pct(r['p_top'])}), es bestehen {names(r['passes'])}")
        else:
            parts.append(f"{r['name']}: Klassifikator {C.MODEL_LABELS[r['top']]} ({pct(r['p_top'])}), es besteht keines (bester p-Wert {num(r['best_p'], 3)})")
    head = f"Der Klassifikator antwortet in allen {len(rows)} Fällen; bei {len(none)} davon besteht kein Modell die Prüfung. "
    return ("warning" if none else "info"), head + "; ".join(parts) + "."
