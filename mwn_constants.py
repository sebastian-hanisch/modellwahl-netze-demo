"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Pläne der Messreihen und Presets."""

N_MIN, N_MAX, DEFAULT_N = 50, 400, 150
K_MIN, K_MAX, DEFAULT_K = 2.0, 8.0, 3.5          # mittlerer Grad der erzeugten Netze
SIMS_MIN, SIMS_MAX, DEFAULT_SIMS = 30, 120, 60   # Netze je angepasstem Modell in der Prüfung
SEED_MAX, DEFAULT_SEED = 999999, 35

NETWORKS = {
    "er": "Erdős–Rényi",
    "config": "Konfigurationsmodell (Grade eines Barabási–Albert-Netzes)",
    "ba": "Barabási–Albert",
    "ws": "Watts–Strogatz",
    "geo": "geometrischer Zufallsgraph",
    "sbm": "Blockmodell (3 Blöcke)",
    "hk": "Holme–Kim: Hubs mit Dreiecksschluss (kein Modell der Liste)",
    "karate": "Zachary-Karateclub",
    "lesmis": "Les Misérables (gemeinsame Auftritte)",
    "florenz": "Florentiner Familien (Heiraten)",
}
MODEL_LABELS = {
    "er": "Erdős–Rényi",
    "config": "Konfigurationsmodell",
    "ba": "Barabási–Albert",
    "ws": "Watts–Strogatz",
    "geo": "geometrisch",
    "sbm": "Blockmodell",
}
MODEL_ORDER = ("er", "config", "ba", "ws", "geo", "sbm")        # gleiche Reihenfolge wie mwn_algorithm.MODEL_KINDS
GENERATED = MODEL_ORDER + ("hk",)
REAL = ("karate", "lesmis", "florenz")
DEFAULT_NETWORK = "ws"

STAT_LABELS = {
    "hub": "größter Grad / mittlerer Grad",
    "fano": "Varianz / Mittel des Grades",
    "trans": "Transitivität",
    "assort": "Grad-Assortativität",
    "plen": "mittlere Weglänge",
    "mod": "Modularität (gierig)",
    "giant": "Anteil größte Komponente",
}
STAT_SHORT = {"hub": "Hub", "fano": "Grad-Streuung", "trans": "Transitivität", "assort": "Assortativität", "plen": "Weglänge", "mod": "Modularität", "giant": "Komponente"}
DEGREE_STATS = (0, 1)                 # Spalten, die nur die Gradverteilung beschreiben
NON_DEGREE_STATS = (2, 3, 4, 5, 6)

# Klassifikator und Prüfung
TRAIN_PER_CLASS = 60                  # Trainingsnetze je Modellfamilie (Parameter zufällig aus den Bereichen unten)
TRAIN_SEED = 77000
KNN_K = 15
ALPHA = 0.05                          # ein Modell besteht die Prüfung bei p-Wert >= ALPHA
WS_P_RANGE = (0.02, 0.30)             # Umlegewahrscheinlichkeit der Trainingsnetze; Testnetze: WS_P_TYPICAL
WS_P_TYPICAL = 0.10
SBM_BLOCKS = 3
SBM_RATIO_RANGE = (6.0, 40.0)         # Verhältnis p_in / p_out der Trainingsnetze; Testnetze: SBM_RATIO_TYPICAL
SBM_RATIO_TYPICAL = 15.0
HK_P_TRIAD = 0.9                      # Wahrscheinlichkeit des Dreiecksschlusses im Holme–Kim-Netz

# Messreihen (feste Pläne, unabhängig von den Reglern)
SERIES_SEED = 100000
SERIES_N, SERIES_K = 100, 3.5
SERIES_PER_CLASS_CONFUSION = 20
SERIES_PER_CLASS_FIT = 6
SERIES_SIMS = 40
RECOGNITION_PLAN = {"ba": 8, "config": 8, "hk": 8, "er": 8, "ws": 3, "geo": 3, "sbm": 3}   # erzeugte Netze mit den Zufalls-Seeds 1 bis n (n = 150, mittlerer Grad 3,5, 60 Netze je Modell)

COLOR_TOP = "#2F6B65"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_PASS = "#2F6B65"
COLOR_FAIL = "#b5483a"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Watts–Strogatz, sauber erkannt": {"network": "ws", "n": DEFAULT_N, "k": DEFAULT_K, "sims": DEFAULT_SIMS, "seed": DEFAULT_SEED},
    "Barabási–Albert: kaum von Konfiguration zu trennen": {"network": "ba", "n": DEFAULT_N, "k": DEFAULT_K, "sims": DEFAULT_SIMS, "seed": 36},
    "Les Misérables: nach Zwang BA?": {"network": "lesmis", "n": DEFAULT_N, "k": DEFAULT_K, "sims": DEFAULT_SIMS, "seed": 37},
    "Karateclub: nach Zwang BA?": {"network": "karate", "n": DEFAULT_N, "k": DEFAULT_K, "sims": DEFAULT_SIMS, "seed": 38},
    "Holme–Kim-Netz: passt kein Modell": {"network": "hk", "n": DEFAULT_N, "k": DEFAULT_K, "sims": DEFAULT_SIMS, "seed": 39},
}
PRESET_HELP = {
    "Watts–Strogatz, sauber erkannt": "Ein Watts–Strogatz-Netz mit 150 Knoten: der Klassifikator nennt es zu 100 %, und nur dieses Modell besteht die Prüfung. Der klare Fall.",
    "Barabási–Albert: kaum von Konfiguration zu trennen": "Ein Barabási–Albert-Netz: der Klassifikator nennt Barabási–Albert (53 %) und das Konfigurationsmodell (47 %), und beide Modelle bestehen die Prüfung. Gleiche Gradverteilung, die sieben Größen trennen sie nicht.",
    "Les Misérables: nach Zwang BA?": "Ein echtes Netz mit 77 Knoten: der Klassifikator nennt Barabási–Albert (60 %), doch kein Modell der Liste besteht die Prüfung.",
    "Karateclub: nach Zwang BA?": "Das Netz mit 34 Mitgliedern: der Klassifikator nennt Barabási–Albert (53 %), doch kein Modell der Liste besteht die Prüfung.",
    "Holme–Kim-Netz: passt kein Modell": "Ein Barabási–Albert-Netz mit Dreiecksschluss: Hubs und viele Dreiecke zugleich. Keines der sechs Modelle erzeugt beides; der Klassifikator nennt trotzdem das Konfigurationsmodell (60 %), und kein Modell besteht die Prüfung.",
}
