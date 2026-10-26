"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import mwn_algorithm as A
import mwn_constants as C

MODEL_NAMES = [C.MODEL_LABELS[k] for k in C.MODEL_ORDER]
Z_CLIP = 6.0


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_probs(a):
    """Anteil der 15 nächsten Trainingsnetze je Modellfamilie; die gewählte Familie ist hervorgehoben, bei erzeugten Netzen steht ein Stern über der wahren Familie."""
    probs = [float(p) for p in a.probs]
    top = int(np.argmax(probs))
    colors = [C.COLOR_TOP if i == top else C.COLOR_OTHER for i in range(6)]
    fig = go.Figure(go.Bar(x=MODEL_NAMES, y=probs, marker_color=colors, text=[f"{p * 100:.0f} %" for p in probs], textposition="outside", name="Anteil der nächsten Nachbarn",
                           hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    if a.truth is not None:
        ti = C.MODEL_ORDER.index(a.truth)
        fig.add_trace(go.Scatter(x=[MODEL_NAMES[ti]], y=[probs[ti] + 0.1], mode="markers", marker=dict(symbol="star", size=14, color="#e0a800"), name="erzeugt aus", hoverinfo="skip"))
    fig.update_yaxes(title_text="Anteil der 15 nächsten Netze", range=[0, 1.25])
    return _base(fig)


def build_check(a):
    """Gesamt-p-Wert der Anpassungsprüfung je Modell; Balken unter der Schwelle ALPHA sind rot (verworfen), die übrigen grün (bestanden)."""
    p = [float(x) for x in a.p_all]
    colors = [C.COLOR_PASS if x >= C.ALPHA else C.COLOR_FAIL for x in p]
    fig = go.Figure(go.Bar(x=MODEL_NAMES, y=p, marker_color=colors, text=[f"{x:.3f}".replace(".", ",") for x in p], textposition="outside", name="Gesamt-p-Wert", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=C.ALPHA, line_dash="dash", line_color="#444", annotation_text=f"Schwelle {str(C.ALPHA).replace('.', ',')}", annotation_position="top left")
    fig.update_yaxes(title_text="Gesamt-p-Wert", range=[0, 1.15])
    return _base(fig)


def build_heatmap(a):
    """Wo ein Modell nicht passt: standardisierter Abstand z = (beobachtet − Mittel der Wolke) / Streuung je Modell und Größe (Farbskala auf ±6 begrenzt)."""
    z = np.clip(a.z, -Z_CLIP, Z_CLIP)
    labels = [[f"{v:+.1f}".replace(".", ",") if abs(v) < 99 else "∞" for v in row] for row in a.z]
    fig = go.Figure(go.Heatmap(z=z, x=[C.STAT_SHORT[s] for s in A.STAT_NAMES], y=MODEL_NAMES, zmin=-Z_CLIP, zmax=Z_CLIP, zmid=0.0, colorscale="RdBu_r", text=labels, texttemplate="%{text}",
                               colorbar=dict(title="z"), hovertemplate="%{y}, %{x}: z = %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=320)


def build_confusion(conf):
    """Verwechslungsmatrix: Zeile = erzeugt aus, Spalte = erkannt als; in den Zellen die Zahl der Netze."""
    conf = np.asarray(conf)
    fig = go.Figure(go.Heatmap(z=conf, x=MODEL_NAMES, y=MODEL_NAMES, colorscale="Greens", text=conf, texttemplate="%{text}", showscale=False, hovertemplate="erzeugt aus %{y}, erkannt als %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="erzeugt aus")
    fig.update_xaxes(title_text="erkannt als")
    return _base(fig, height=360)


def build_pass_matrix(rate):
    """Anteil der Netze je Familie (Zeile), die die Prüfung gegen das Modell der Spalte bestehen."""
    rate = np.asarray(rate, dtype=float)
    fig = go.Figure(go.Heatmap(z=rate, x=MODEL_NAMES, y=MODEL_NAMES, zmin=0.0, zmax=1.0, colorscale="Greens", text=[[f"{v * 100:.0f} %" for v in row] for row in rate], texttemplate="%{text}",
                               showscale=False, hovertemplate="erzeugt aus %{y}, geprüft gegen %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="erzeugt aus")
    fig.update_xaxes(title_text="geprüftes Modell")
    return _base(fig, height=360)
