"""Einmalige Messung der README-Zahlen (nicht Teil der App): feste Pläne, Ausgabe auf stdout."""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import mwn_algorithm as A
import mwn_constants as C
import mwn_evaluation as ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()
print("PRESETS")
for name, p in C.PRESETS.items():
    t = time.time()
    a = ev.analyse(ev.Settings(**p))
    print(name, "n", a.n, "k", round(a.mean_degree, 2), "Klassifikator", a.top, [round(float(x), 2) for x in a.probs], "p_all", [round(float(x), 3) for x in a.p_all],
          "besteht", a.passes, "Größen", [round(float(x), 3) for x in a.observed], round(time.time() - t, 1), "s", ev.verdict_choice(a.probs, a.passes)[0])

print("SEEDS (erzeugte Netze, n = 150, k = 3,5, 60 Netze je Modell)")
for key, r in ev.recognition().items():
    print(key, "Seeds", r["runs"], "Klassifikator richtig", r["right"], "wahres Modell besteht", r["own"], "kein Modell besteht", r["none"], "genannt", r["named"], flush=True)

print("SERIE VERWECHSLUNG")
s = ev.series_confusion()
for name, r in s["sets"].items():
    print(name, "Trefferquote", round(r["accuracy"], 3))
    print(r["conf"])

print("SERIE PRÜFUNG")
f = ev.series_fit()
print(np.round(f["rate"], 2), "Netze je Familie", f["per_class"], "Simulationen", f["sims"])

print("SERIE ECHTE NETZE")
for r in ev.series_real():
    print(r)
print("sec", round(time.time() - t0, 1))
