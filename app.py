"""Welches Modell erzeugt dieses Netz? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 9 der Linie „Netzmodelle“ (Zusammenfluss): ein Klassifikator über sieben Zusammenfassungsgrößen muss immer eine von sechs Modellfamilien nennen; eine Anpassungsprüfung
fragt dagegen, ob irgendeines der Modelle das Netz erzeugt haben kann. Der Kern liegt in mwn_algorithm.py.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import mwn_algorithm as A
import mwn_constants as C
import mwn_evaluation as ev
from mwn_evaluation import Settings, analyse, fit_rows, num, pct, stats_rows, truth_sentence, verdict_choice, verdict_confusion, verdict_fit, verdict_real
from mwn_layout import dunkles_design, kopf_css
from mwn_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mwn_visualization import build_check, build_confusion, build_heatmap, build_pass_matrix, build_probs

st.set_page_config(page_title="Welches Modell erzeugt dieses Netz? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _series_confusion():
    return ev.series_confusion()


@st.cache_data(show_spinner=False)
def _series_fit():
    return ev.series_fit()


@st.cache_data(show_spinner=False)
def _series_real():
    return ev.series_real()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mwn_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Welches Modell erzeugt dieses Netz?")
st.markdown(
    """
Vorgabe: ein Netz, gesucht: die Zufallsfamilie, aus der es stammen könnte. Die Demo beantwortet das auf **zwei Wegen** und stellt sie nebeneinander.
Ein **Klassifikator** über sieben Zusammenfassungsgrößen (Grad, Dreiecke, Weglänge, Gemeinschaften …) **muss immer antworten**, auch wenn das Netz in keine der sechs Familien gehört.
Eine **Anpassungsprüfung** passt dagegen jedes Modell an das Netz an, zieht Netze daraus und fragt, ob das beobachtete Netz darin unauffällig liegt; sie darf „keines davon“ sagen.
Neuntes Stück der Linie „Netzmodelle“ (Zusammenfluss); die Familien stammen aus `zufallsgraph-demo`, `konfigurationsmodell-demo`, `wachstumsmodelle-demo`, `raeumliche-netze-demo` und dem Blockmodell aus `stochastic-block-demo`.

Weiter unten: 🔬 **Messreihen** (Verwechslung, Prüfung, echte Netze), 🚧 **Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So läuft es", expanded=True):
    st.markdown(
        """
1. **Ein Netz wählen.** Eine der sechs Familien (dann ist die Herkunft bekannt), ein Holme–Kim-Netz (gehört zu **keiner** der sechs) oder eines von drei echten Netzen.
2. **Sieben Größen messen.** Größter Grad, Streuung des Grades, Transitivität, Assortativität, mittlere Weglänge, Modularität und Anteil der größten Komponente.
3. **Klassifizieren.** Aus je 60 Trainingsnetzen jeder Familie (gleiche Knotenzahl und gleicher mittlerer Grad, zufällige Parameter) werden die 15 ähnlichsten gesucht; ihr Familienanteil ist die Antwort.
4. **Prüfen.** Jedes Modell wird an das Netz angepasst, daraus werden Netze gezogen, und der Abstand des beobachteten Netzes zu dieser Wolke wird mit den Abständen der Wolkennetze untereinander verglichen. Bestanden heißt: Gesamt-p-Wert mindestens 0,05.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    network = st.selectbox("Netz", list(C.NETWORKS), format_func=C.NETWORKS.get, key="network_select",
                           help="Eine der sechs Familien, ein Holme–Kim-Netz (keine der sechs) oder ein echtes Netz. Bei echten Netzen gelten ihre eigene Knotenzahl und ihr eigener Grad, nicht die Regler.")
    n = st.slider("Knoten", *bounds("n_slider"), key="n_slider", step=50, help="Knotenzahl des erzeugten Netzes (bei echten Netzen ohne Wirkung).")
    k = st.slider("Mittlerer Grad", *bounds("k_slider"), key="k_slider", step=0.5, help="Mittlerer Grad des erzeugten Netzes (bei echten Netzen ohne Wirkung); die Trainingsnetze haben denselben Grad.")
    sims = st.slider("Netze je Modell in der Prüfung", *bounds("sims_slider"), key="sims_slider", step=10,
                     help="Aus so vielen gezogenen Netzen besteht die Wolke je Modell; der kleinste mögliche p-Wert ist 1 / (Zahl + 1).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Netz ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"network_select": network, "n_slider": int(n), "k_slider": round(float(k), 1), "sims_slider": int(sims), "seed_input": int(seed)})

settings = Settings(network=network, n=int(n), k=round(float(k), 1), sims=int(sims), seed=int(seed))
if network in C.REAL:
    st.info("Echtes Netz: Knotenzahl und Grad kommen aus den Daten, die Regler „Knoten“ und „Mittlerer Grad“ gelten hier nicht. Der Zufalls-Seed steuert nur die Trainingsnetze der Prüfung.")
with st.spinner("Messe das Netz, trainiere den Klassifikator und prüfe die sechs Modelle …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)
r1c1.metric("Knoten", f"{a.n}", delta=f"{a.edges} Kanten, mittlerer Grad {num(a.mean_degree, 1)}", delta_color="off")
r1c2.metric("Herkunft", C.MODEL_LABELS[a.truth] if a.truth else "keine der sechs", delta="bekannt (erzeugt)" if a.truth else ("Holme–Kim" if network == "hk" else "echtes Netz, unbekannt"), delta_color="off")
r2c1.metric("Klassifikator", C.MODEL_LABELS[a.top], delta=f"{pct(float(a.probs.max()))} der nächsten Nachbarn", delta_color="off")
r2c2.metric("Prüfung bestanden", f"{len(a.passes)} von 6", delta=ev.names(a.passes) if a.passes else "kein Modell", delta_color="off")
c1, c2 = st.columns(2)
c1.plotly_chart(build_probs(a), width="stretch", key="main_probs")
c2.plotly_chart(build_check(a), width="stretch", key="main_check")
st.caption("Links die Antwort des Klassifikators (sie summiert sich immer zu 100 %); rechts, ob das jeweilige angepasste Modell das Netz erzeugt haben kann (grün: p-Wert mindestens 0,05).")

st.markdown("#### Wo passt ein Modell nicht?")
st.plotly_chart(build_heatmap(a), width="stretch", key="main_heatmap")
st.caption("z = (beobachtet − Mittel der Wolke) / Streuung der Wolke je Größe. Rot: das Netz hat mehr, als das Modell im Mittel erzeugt, blau: weniger. Betrag unter 2 ist unauffällig; eine Zeile ohne Streuung in einer Größe (zum Beispiel die Grade beim Konfigurationsmodell, die festgehalten werden) zeigt 0.")
with st.expander("Zahlen im Einzelnen"):
    st.table(fit_rows(a))
    st.table(stats_rows(a))

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Welches Modell war es?")
kind, text = verdict_choice(a.probs, a.passes)
getattr(st, kind)(text)
st.caption(truth_sentence(a.truth, network))

st.markdown("---")

# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Messreihen: wie oft stimmt was?")
st.caption(
    f"Feste Programme, unabhängig von den Reglern: Netze mit {C.SERIES_N} Knoten und mittlerem Grad {num(C.SERIES_K, 1)}, Parameter der erzeugten Netze typisch (Watts–Strogatz p = {num(C.WS_P_TYPICAL, 2)}, Blockmodell p_in / p_out = {num(C.SBM_RATIO_TYPICAL, 0)}). "
    "Jede Reihe rechnet auf Klick (die Prüfungsreihe dauert etwa eine halbe Minute)."
)
tab1, tab2, tab3 = st.tabs(["Verwechslung", "Prüfung", "Echte Netze"])
with tab1:
    st.markdown(f"**Wie oft nennt der Klassifikator die richtige Familie?** Je Familie {C.SERIES_PER_CLASS_CONFUSION} frisch erzeugte Netze; derselbe Klassifikator mit allen sieben Größen, mit nur den zwei Gradgrößen und ohne sie.")
    if st.button("Reihe rechnen", key="confusion_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe …"):
            _series_confusion()
        st.session_state["confusion_done"] = time.perf_counter() - t0
    if "confusion_done" in st.session_state:
        s = _series_confusion()
        cc1, cc2 = st.columns(2)
        cc1.markdown("**Alle sieben Größen**")
        cc1.plotly_chart(build_confusion(s["sets"]["alle sieben Größen"]["conf"]), width="stretch", key="confusion_all")
        cc2.markdown("**Nur die zwei Gradgrößen**")
        cc2.plotly_chart(build_confusion(s["sets"]["nur die zwei Gradgrößen"]["conf"]), width="stretch", key="confusion_deg")
        st.table([{"Größen": name, "Trefferquote": pct(r["accuracy"])} for name, r in s["sets"].items()])
        kind, text = verdict_confusion(s)
        getattr(st, kind)(text)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['confusion_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")
with tab2:
    st.markdown(f"**Wie oft besteht das wahre Modell die Prüfung, wie oft ein falsches?** Je Familie {C.SERIES_PER_CLASS_FIT} Netze, jedes gegen alle sechs angepassten Modelle mit {C.SERIES_SIMS} gezogenen Netzen je Prüfung.")
    if st.button("Reihe rechnen", key="fit_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe (etwa eine halbe Minute) …"):
            _series_fit()
        st.session_state["fit_done"] = time.perf_counter() - t0
    if "fit_done" in st.session_state:
        s = _series_fit()
        st.plotly_chart(build_pass_matrix(s["rate"]), width="stretch", key="fit_matrix")
        kind, text = verdict_fit(s)
        getattr(st, kind)(text)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['fit_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")
with tab3:
    st.markdown("**Was sagen Klassifikator und Prüfung zu den echten Netzen und zum Holme–Kim-Netz?** Alle vier mit festem Zufalls-Seed.")
    if st.button("Reihe rechnen", key="real_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe …"):
            _series_real()
        st.session_state["real_done"] = time.perf_counter() - t0
    if "real_done" in st.session_state:
        rows = _series_real()
        st.table([{
            "Netz": r["name"], "Knoten": r["n"], "Grad": num(r["k"], 1), "Klassifikator": f"{C.MODEL_LABELS[r['top']]} ({pct(r['p_top'])})",
            "besteht die Prüfung": ev.names(r["passes"]) if r["passes"] else "kein Modell", "bester p-Wert": num(r["best_p"], 3),
        } for r in rows])
        kind, text = verdict_real(rows)
        getattr(st, kind)(text)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['real_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Netz stammt aus einer der sechs Familien** | Der Klassifikator antwortet trotzdem und nennt die ähnlichste; nur die Prüfung kann „keines davon“ sagen. | diese Demo |
| **Die sieben Größen genügen** | Netze, die in allen sieben Größen übereinstimmen, aber anders gebaut sind, sind nicht zu trennen (Barabási–Albert gegen Konfigurationsmodell mit denselben Graden). | diese Demo |
| **Ein flexibles Modell passt fast immer** | Das Blockmodell wird an die gefundene Zerlegung des Netzes angepasst und kann deshalb viel absorbieren; „Prüfung bestanden“ heißt nicht „so ist das Netz entstanden“. | `stochastic-block-demo` |
| **Der p-Wert der Prüfung ist genau** | Die Parameter kommen aus denselben Daten wie das Netz, das geprüft wird: die Prüfung ist eher zu nachsichtig, und bei wenigen gezogenen Netzen sind die p-Werte grob (kleinster Wert 1 / (Zahl + 1)). | diese Demo |
| **Kleine Netze sind prüfbar** | Bei 15 Knoten besteht fast jedes Modell: nichts zu unterscheiden heißt nicht, dass alle stimmen. | diese Demo |
| **Zufällige Parameter der Trainingsnetze decken das Netz ab** | Liegt der Parameter außerhalb der Bereiche (Watts–Strogatz p von 0,02 bis 0,3, Blockmodell Verhältnis 6 bis 40), kann der Klassifikator danebenliegen. | diese Demo |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Ein Netz mit $n$ Knoten und $m$ Kanten, mittlerer Grad $k = 2m / n$.

| Symbol | Bedeutung |
|---|---|
| $s \in \mathbb{R}^7$ | die sieben Zusammenfassungsgrößen des beobachteten Netzes |
| $\hat\theta_M$ | die an das Netz angepassten Parameter des Modells $M$ |
| $s^{(1)}, \dots, s^{(R)}$ | Größen von $R$ aus $M(\hat\theta_M)$ gezogenen Netzen (die Wolke) |
| $D(x; S)$ | Mahalanobis-Abstand von $x$ zum Mittel der Menge $S$ (Kovarianz zur Diagonale hin geschrumpft) |

**Klassifikator.** Mit den standardisierten Größen $z$ der Trainingsnetze ist die Antwort für die Klasse $c$ der Anteil
$$\hat P(c) = \frac{1}{15} \, \#\{\text{unter den 15 nächsten Trainingsnetzen ist Klasse } c\}.$$

**Anpassungsprüfung.** Die Parameter: Erdős–Rényi $m$; Konfigurationsmodell die Gradfolge (Kantentausch); Barabási–Albert $m_{\mathrm{BA}} = k/2$; Watts–Strogatz $k$ und $p$ aus der Transitivität $C$ mit
$$\hat p = 1 - \left( \frac{C}{C_0} \right)^{1/3}, \qquad C_0 = \frac{3 (K - 1)}{2 (2K - 1)}, \quad K = \frac{k}{2};$$
geometrisch der Radius $r$ mit $n \, \Pr[\text{Abstand} \le r] = k$ und $\Pr[\text{Abstand} \le r] = \pi r^2 - \tfrac{8}{3} r^3 + \tfrac{1}{2} r^4$ im Einheitsquadrat; Blockmodell die Zerlegung der gierigen Modularitätsmaximierung und die Dichten zwischen den Gruppen.
Der Gesamt-p-Wert ist
$$p = \frac{1 + \#\{ j : D(s^{(j)}; S \setminus s^{(j)}) \ge D(s; S) \}}{R + 1}.$$
Je Größe gilt der zweiseitige empirische p-Wert $2 \min\{(1 + \#\{\ge\}) / (R + 1),\ (1 + \#\{\le\}) / (R + 1)\}$.

**Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Gelman, Meng und Stern, *Posterior predictive assessment of model fitness via realized discrepancies*, Statistica Sinica 6 (1996); Clauset, Newman und Moore, *Finding community structure in very large networks*, Physical Review E 70 (2004);
Barrat und Weigt, *On the properties of small-world network models*, European Physical Journal B 13 (2000); Holme und Kim, *Growing scale-free networks with tunable clustering*, Physical Review E 65 (2002).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
