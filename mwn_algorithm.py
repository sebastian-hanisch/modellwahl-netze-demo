"""Kern: Netzgeneratoren, sieben Zusammenfassungsgrößen, k-nächste-Nachbarn-Klassifikator und Anpassungsprüfung (Posterior-Predictive-Stil).

Sieben Größen je Netz: größter Grad / mittlerer Grad, Varianz / Mittel des Grades, Transitivität, Grad-Assortativität, mittlere Weglänge in der größten Komponente,
Modularität der gierigen Zerlegung (Clauset, Newman, Moore) und Anteil der größten Komponente.
Sechs Modellfamilien: Erdős–Rényi G(n, m), Konfigurationsmodell (Kantentausch mit gegebenen Graden), Barabási–Albert, Watts–Strogatz, geometrischer Zufallsgraph im Einheitsquadrat, Blockmodell.
Anpassungsprüfung: ein Modell wird an das beobachtete Netz angepasst, daraus werden Netze gezogen, und das beobachtete Netz wird gegen diese Wolke gelegt."""

import math

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

STAT_NAMES = ("hub", "fano", "trans", "assort", "plen", "mod", "giant")
MODEL_KINDS = ("er", "config", "ba", "ws", "geo", "sbm")
RADIUS_SLACK = 1e-12


# --- Grundlagen ----------------------------------------------------------------------------------------------------------------------------


def normalize(edges):
    """Kanten als sortierte Paare (u < v), ohne Schleifen und Doppelte, aufsteigend sortiert."""
    return sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if int(u) != int(v)})


def adjacency(n: int, edges):
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (a + a.T).tocsr()


def degrees(n: int, edges):
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    return np.bincount(e.ravel(), minlength=n)


# --- Generatoren ---------------------------------------------------------------------------------------------------------------------------


def gnm_edges(n: int, m: int, rng: np.random.Generator):
    """m verschiedene Kanten, gleichverteilt unter allen Paaren (Erdős–Rényi G(n, m))."""
    m = min(m, n * (n - 1) // 2)
    out = set()
    while len(out) < m:
        need = m - len(out)
        a = rng.integers(0, n, size=2 * need + 8)
        b = rng.integers(0, n, size=2 * need + 8)
        for u, v in zip(a, b):
            if u != v:
                out.add((min(int(u), int(v)), max(int(u), int(v))))
                if len(out) == m:
                    break
    return sorted(out)


def edge_switch(edges, proposals: int, rng: np.random.Generator):
    """Träge Kantentauschkette mit `proposals` Vorschlägen (jeder Vorschlag zählt); die Gradfolge bleibt erhalten."""
    edges = [tuple(e) for e in edges]
    m = len(edges)
    if m < 2:
        return normalize(edges)
    present = {(min(u, v), max(u, v)) for u, v in edges}
    idx = rng.integers(0, m, size=(proposals, 2))
    flip = rng.integers(0, 2, size=(proposals, 2))
    for k in range(proposals):
        i, j = int(idx[k, 0]), int(idx[k, 1])
        if i == j:
            continue
        a, b = edges[i] if flip[k, 0] else edges[i][::-1]
        c, d = edges[j] if flip[k, 1] else edges[j][::-1]
        if a == d or c == b or a == c or b == d:
            continue
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if e1 in present or e2 in present:
            continue
        present.discard((min(a, b), max(a, b)))
        present.discard((min(c, d), max(c, d)))
        present.add(e1)
        present.add(e2)
        edges[i], edges[j] = e1, e2
    return normalize(edges)


def ba_edges(n: int, m: float, rng: np.random.Generator):
    """Barabási–Albert-Netz: jeder neue Knoten wählt floor(m) oder floor(m) + 1 (mindestens eine) verschiedene Enden aus der Liste der Kantenenden (so, dass der Mittelwert m ist);
    gestartet wird mit einem vollständigen Graphen auf floor(m) + 1 Knoten, damit der mittlere Grad bei etwa 2m liegt."""
    lo = max(1, int(math.floor(m)))      # weniger als eine Kante je neuen Knoten gibt es nicht (ein Baum ist das Dünnste)
    frac = max(0.0, m - lo)
    start = lo + 1
    n = max(n, start + 1)
    ends: list[int] = []
    edges = []
    for u in range(start):
        for v in range(u):
            ends += [u, v]
            edges.append((v, u))
    for t in range(start, n):
        want = lo + (1 if rng.random() < frac else 0)
        want = max(1, min(want, t))
        chosen: list[int] = []
        while len(chosen) < want:
            v = ends[int(rng.integers(0, len(ends)))]
            if v not in chosen:
                chosen.append(v)
        for v in chosen:
            ends += [v, t]
            edges.append((v, t))
    return normalize(edges)


def holme_kim_edges(n: int, m: float, p_triad: float, rng: np.random.Generator):
    """Barabási–Albert-Netz mit Dreiecksschluss (Holme und Kim): das erste Ziel eines neuen Knotens wird proportional zum Grad gezogen, jedes weitere
    mit Wahrscheinlichkeit p_triad unter den Nachbarn des letzten Ziels (schließt ein Dreieck), sonst wieder proportional zum Grad. Mittlere Kantenzahl je neuen Knoten m wie bei ba_edges."""
    lo = max(1, int(math.floor(m)))
    frac = max(0.0, m - lo)
    start = lo + 1
    n = max(n, start + 1)
    adj: list[set] = [set() for _ in range(n)]
    ends: list[int] = []
    edges = []
    for u in range(start):
        for v in range(u):
            ends += [u, v]
            edges.append((v, u))
            adj[u].add(v)
            adj[v].add(u)
    for t in range(start, n):
        want = min(lo + (1 if rng.random() < frac else 0), t)
        chosen: list[int] = [ends[int(rng.integers(0, len(ends)))]]
        while len(chosen) < want:
            if rng.random() < p_triad:
                cand = [w for w in adj[chosen[-1]] if w != t and w not in chosen]
                if cand:
                    chosen.append(cand[int(rng.integers(0, len(cand)))])
                    continue
            v = ends[int(rng.integers(0, len(ends)))]
            if v not in chosen:
                chosen.append(v)
        for v in chosen:
            ends += [v, t]
            edges.append((v, t))
            adj[v].add(t)
            adj[t].add(v)
    return normalize(edges)


def ws_edges(n: int, k: float, p: float, rng: np.random.Generator):
    """Watts–Strogatz-Netz: Ring, jeder Knoten mit den floor(k/2) nächsten Nachbarn auf jeder Seite, dazu jede Kante des nächstweiteren Abstands mit der Wahrscheinlichkeit k/2 − floor(k/2)
    (so liegt der mittlere Grad bei k); jede Ringkante wird mit Wahrscheinlichkeit p an ein anderes Ziel umgelegt (ohne Schleifen und Doppelte)."""
    half = int(math.floor(k / 2.0))
    frac = k / 2.0 - half
    edges = {(min(i, (i + j) % n), max(i, (i + j) % n)) for i in range(n) for j in range(1, half + 1)}
    if frac > 0 and half + 1 <= (n - 1) // 2:
        for i in range(n):
            if rng.random() < frac:
                j = (i + half + 1) % n
                edges.add((min(i, j), max(i, j)))
    for u, v in sorted(edges):
        if rng.random() < p:
            for _ in range(100):
                w = int(rng.integers(0, n))
                e = (min(u, w), max(u, w))
                if w != u and e not in edges:
                    edges.discard((u, v))
                    edges.add(e)
                    break
    return sorted(edges)


def geo_pair_probability(r: float) -> float:
    """Wahrscheinlichkeit, dass zwei gleichverteilte Punkte im Einheitsquadrat höchstens den Abstand r haben (r ≤ 1): π r² − 8 r³ / 3 + r⁴ / 2."""
    r = min(max(r, 0.0), 1.0)
    return math.pi * r * r - 8.0 * r ** 3 / 3.0 + r ** 4 / 2.0


def geo_radius(n: int, k: float) -> float:
    """Radius, bei dem der geometrische Zufallsgraph im Einheitsquadrat im Mittel den Grad k hat (Randeffekt eingerechnet)."""
    target = min(k / max(n - 1, 1), geo_pair_probability(1.0))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if geo_pair_probability(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def geo_edges(n: int, radius: float, rng: np.random.Generator):
    """Geometrischer Zufallsgraph: n gleichverteilte Punkte im Einheitsquadrat, Kante bei Abstand höchstens `radius`."""
    pos = rng.random((n, 2))
    pairs = cKDTree(pos).query_pairs(radius * (1.0 + RADIUS_SLACK), output_type="ndarray")
    return normalize([tuple(p) for p in pairs])


def block_sizes(n: int, blocks: int):
    """n Knoten möglichst gleichmäßig auf `blocks` Blöcke."""
    base, extra = divmod(n, blocks)
    return [base + (1 if b < extra else 0) for b in range(blocks)]


def sbm_edges(labels, probs, rng: np.random.Generator):
    """Blockmodell: Knoten i und j sind mit Wahrscheinlichkeit probs[label_i][label_j] verbunden. `labels`: Block je Knoten."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    n = len(labels)
    iu, ju = np.triu_indices(n, k=1)
    p = probs[labels[iu], labels[ju]]
    keep = rng.random(len(p)) < p
    return [(int(a), int(b)) for a, b in zip(iu[keep], ju[keep])]


def sbm_in_out(n: int, blocks: int, k: float, ratio: float):
    """Wahrscheinlichkeiten (p_in, p_out) im gleich großen Blockmodell mit mittlerem Grad k und p_in / p_out = ratio."""
    sizes = block_sizes(n, blocks)
    pairs_in = sum(s * (s - 1) / 2 for s in sizes)
    pairs_out = n * (n - 1) / 2 - pairs_in
    p_out = (k * n / 2.0) / (ratio * pairs_in + pairs_out)
    return min(1.0, ratio * p_out), p_out


# --- Zusammenfassungsgrößen ------------------------------------------------------------------------------------------------------------------


def transitivity(n: int, edges) -> float:
    """3 · Dreiecke / verbundene Dreiergruppen (globaler Clusterkoeffizient); 0 ohne Dreiergruppen."""
    if not len(edges):
        return 0.0
    a = adjacency(n, edges)
    d = np.asarray(a.sum(axis=1)).ravel()
    triples = float((d * (d - 1) / 2.0).sum())
    if triples == 0:
        return 0.0
    triangles = float(a.multiply(a @ a).sum()) / 6.0
    return 3.0 * triangles / triples


def assortativity(n: int, edges) -> float:
    """Pearson-Korrelation der Grade an den beiden Enden einer Kante (Newman); 0, wenn alle Kantenenden denselben Grad haben (nicht definiert)."""
    if not len(edges):
        return 0.0
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    d = degrees(n, edges).astype(float)
    j, k = d[e[:, 0]], d[e[:, 1]]
    m = len(e)
    s1 = np.sum(j * k) / m
    s2 = np.sum(j + k) / (2.0 * m)
    s3 = np.sum(j * j + k * k) / (2.0 * m)
    denom = s3 - s2 * s2
    if denom <= 1e-12:
        return 0.0
    return float((s1 - s2 * s2) / denom)


def giant_component(n: int, edges):
    """(Knotenmaske der größten Komponente, Zahl ihrer Knoten); bei Gleichstand die Komponente mit dem kleinsten Knoten."""
    if n == 0:
        return np.zeros(0, dtype=bool), 0
    a = adjacency(n, edges) if len(edges) else sparse.csr_matrix((n, n))
    _, labels = csgraph.connected_components(a, directed=False)
    counts = np.bincount(labels)
    big = int(np.argmax(counts))
    return labels == big, int(counts[big])


def mean_path_length(n: int, edges) -> float:
    """Mittlere kürzeste Weglänge über alle Knotenpaare der größten Komponente (0 bei weniger als zwei Knoten)."""
    mask, size = giant_component(n, edges)
    if size < 2:
        return 0.0
    a = adjacency(n, edges)[mask][:, mask]
    dist = csgraph.shortest_path(a, method="D", directed=False, unweighted=True)
    return float(dist.sum() / (size * (size - 1)))


def greedy_modularity(n: int, edges):
    """Gierige Modularitätsmaximierung (Clauset, Newman, Moore): immer das verbundene Gruppenpaar mit dem größten Gewinn vereinen, solange er positiv ist.
    Rückgabe: (Modularität, Gruppennummer je Knoten, fortlaufend ab 0)."""
    m = len(edges)
    if m == 0:
        return 0.0, np.arange(n)
    e = np.zeros((n, n))
    for u, v in edges:
        e[u, v] += 1.0 / (2 * m)
        e[v, u] += 1.0 / (2 * m)
    a = e.sum(axis=1)
    labels = np.arange(n)
    active = np.ones(n, dtype=bool)
    q = float(np.trace(e) - np.sum(a * a))
    dq = 2.0 * (e - np.outer(a, a))                       # Gewinn beim Vereinen von i und j; nur verbundene Paare zählen
    dq[e <= 0] = -np.inf
    np.fill_diagonal(dq, -np.inf)
    while True:
        flat = int(np.argmax(dq))
        i, j = divmod(flat, n)
        gain = dq[i, j]
        if not gain > 1e-12:
            break
        e[i, :] += e[j, :]
        e[:, i] += e[:, j]
        e[j, :] = 0.0
        e[:, j] = 0.0
        a[i] += a[j]
        a[j] = 0.0
        active[j] = False
        labels[labels == j] = i
        q += float(gain)
        row = 2.0 * (e[i, :] - a[i] * a)                  # nur Zeile und Spalte der vereinten Gruppe ändern sich
        row[(e[i, :] <= 0) | ~active] = -np.inf
        row[i] = -np.inf
        dq[i, :] = row
        dq[:, i] = row
        dq[j, :] = -np.inf
        dq[:, j] = -np.inf
    _, compact = np.unique(labels, return_inverse=True)
    return q, compact


def statistics(n: int, edges) -> np.ndarray:
    """Die sieben Zusammenfassungsgrößen in der Reihenfolge STAT_NAMES."""
    m = len(edges)
    if m == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / max(n, 1)])
    d = degrees(n, edges).astype(float)
    mean = d.mean()
    q, _ = greedy_modularity(n, edges)
    _, size = giant_component(n, edges)
    return np.array([d.max() / mean, d.var() / mean, transitivity(n, edges), assortativity(n, edges), mean_path_length(n, edges), q, size / n])


# --- Klassifikator -------------------------------------------------------------------------------------------------------------------------


def standardize(train_x):
    """(Mittel, Streuung) je Spalte; Streuung null wird zu eins (die Spalte trägt dann nichts bei)."""
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def knn_proba(train_x, train_y, x, classes: int, k: int):
    """Anteil der k nächsten Trainingsnetze (euklidisch, Größen standardisiert) je Klasse; bei gleichem Abstand gewinnt der kleinere Index."""
    mu, sd = standardize(train_x)
    z = (train_x - mu) / sd
    q = (np.asarray(x, dtype=float) - mu) / sd
    dist = np.sqrt(((z - q) ** 2).sum(axis=1))
    order = np.argsort(dist, kind="stable")[: min(k, len(dist))]
    votes = np.bincount(np.asarray(train_y)[order], minlength=classes).astype(float)
    return votes / votes.sum()


# --- Anpassung und Prüfung -----------------------------------------------------------------------------------------------------------------


def ws_rewiring_estimate(clustering: float, k: float) -> float:
    """Schätzer der Umlegewahrscheinlichkeit aus der Transitivität: C(p) ≈ C(0) (1 − p)³ mit C(0) = 3 (K − 1) / (2 (2 K − 1)), K = k / 2 (Barrat und Weigt, genähert); in [0, 1] begrenzt."""
    half = k / 2.0
    if half <= 1.0:
        return 1.0
    c0 = 3.0 * (half - 1.0) / (2.0 * (2.0 * half - 1.0))
    if clustering >= c0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - (clustering / c0) ** (1.0 / 3.0))))


def fit_model(kind: str, n: int, edges):
    """Parameter des Modells, angepasst an das beobachtete Netz (Funktion der Kanten allein, ohne Zufall)."""
    m = len(edges)
    k = 2.0 * m / n
    if kind == "er":
        return {"kind": kind, "n": n, "m": m}
    if kind == "config":
        return {"kind": kind, "n": n, "edges": list(edges), "proposals": 10 * m}
    if kind == "ba":
        return {"kind": kind, "n": n, "m": k / 2.0}
    if kind == "ws":
        return {"kind": kind, "n": n, "k": k, "p": ws_rewiring_estimate(transitivity(n, edges), k)}
    if kind == "geo":
        return {"kind": kind, "n": n, "radius": geo_radius(n, k)}
    if kind == "sbm":
        _, labels = greedy_modularity(n, edges)
        blocks = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=blocks)
        links = np.zeros((blocks, blocks))
        for u, v in edges:
            a, b = labels[u], labels[v]
            links[a, b] += 1
            if a != b:
                links[b, a] += 1
        probs = np.zeros((blocks, blocks))
        for a in range(blocks):
            for b in range(blocks):
                pairs = counts[a] * (counts[a] - 1) / 2.0 if a == b else counts[a] * counts[b]
                probs[a, b] = links[a, b] / pairs if pairs > 0 else 0.0
        return {"kind": kind, "n": n, "labels": labels, "probs": probs}
    raise ValueError(f"unbekanntes Modell: {kind}")


def sample_fitted(fit, rng: np.random.Generator):
    """Ein Netz aus dem angepassten Modell (Kantenliste)."""
    kind, n = fit["kind"], fit["n"]
    if kind == "er":
        return gnm_edges(n, fit["m"], rng)
    if kind == "config":
        return edge_switch(fit["edges"], fit["proposals"], rng)
    if kind == "ba":
        return ba_edges(n, fit["m"], rng)
    if kind == "ws":
        return ws_edges(n, fit["k"], fit["p"], rng)
    if kind == "geo":
        return geo_edges(n, fit["radius"], rng)
    if kind == "sbm":
        return normalize(sbm_edges(fit["labels"], fit["probs"], rng))
    raise ValueError(f"unbekanntes Modell: {kind}")


def empirical_p(observed: float, sample) -> float:
    """Zweiseitiger empirischer p-Wert 2 · min{(1 + Zahl ≥ beobachtet) / (N + 1), (1 + Zahl ≤ beobachtet) / (N + 1)}, höchstens 1; nie null."""
    s = np.asarray(sample, dtype=float)
    hi = (1 + np.sum(s >= observed)) / (len(s) + 1)
    lo = (1 + np.sum(s <= observed)) / (len(s) + 1)
    return float(min(1.0, 2.0 * min(hi, lo)))


SHRINK = 0.1
RIDGE = 1e-6


def _varying_columns(cloud):
    """Spalten der Wolke mit Streuung; konstante Größen sind ausgenommen."""
    if len(cloud) < 2:
        return np.zeros(0, dtype=int)
    return np.where(cloud.std(axis=0, ddof=1) > 1e-9)[0]


def _distance(x, others):
    """Mahalanobis-Abstand von x zum Mittel der Zeilen `others` (Kovarianz zur Diagonale hin geschrumpft, dazu ein kleiner Zuschlag gegen Entartung)."""
    cov = np.cov(others, rowvar=False).reshape(others.shape[1], others.shape[1])
    cov = (1.0 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + RIDGE * np.eye(len(cov))
    d = x - others.mean(axis=0)
    return float(np.sqrt(max(d @ np.linalg.solve(cov, d), 0.0)))


def goodness_of_fit(observed, cloud):
    """Beobachtete Größen gegen die Wolke simulierter Netze (Zeilen = Netze, Spalten = Größen).
    Rückgabe: (p-Wert je Größe, Gesamtabstand, Gesamt-p-Wert). Größen, die in der Wolke konstant sind, bekommen p = 1 und zählen im Abstand nicht;
    die übrigen werden mit Mittel und Streuung der ganzen Wolke standardisiert.
    Der Gesamt-p-Wert: (1 + Zahl der Wolkennetze, deren Abstand zu den übrigen Wolkennetzen mindestens so groß ist wie der des beobachteten Netzes zur Wolke) / (Zahl der Wolkennetze + 1)."""
    cloud = np.asarray(cloud, dtype=float)
    observed = np.asarray(observed, dtype=float)
    r = len(cloud)
    p_each = np.array([1.0 if r < 2 or cloud[:, c].std(ddof=1) <= 1e-9 else empirical_p(observed[c], cloud[:, c]) for c in range(cloud.shape[1])])
    cols = _varying_columns(cloud)
    if len(cols) == 0 or r < 3:                      # ohne Streuung oder mit weniger als drei Wolkennetzen lässt sich kein Abstand schätzen: kein Einwand
        return p_each, 0.0, 1.0
    mu, sd = cloud[:, cols].mean(axis=0), cloud[:, cols].std(axis=0, ddof=1)
    z = (cloud[:, cols] - mu) / sd
    zo = (observed[cols] - mu) / sd
    dist = _distance(zo, z)
    loo = np.array([_distance(z[j], np.delete(z, j, axis=0)) for j in range(r)])
    return p_each, dist, float((1 + np.sum(loo >= dist)) / (r + 1))
