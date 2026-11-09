"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import rsf_algorithm as A
import rsf_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _de(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def build_cell(a):
    """Kostenkurve der Zelle über der Menge; Mittel, Quantil und beste Menge (Strecke) markiert, die Nachfragen als Punkte am unteren Rand."""
    top = max(a.demands) + 1.0
    qs = np.linspace(0.0, top, 241)
    costs = A.cell_cost(qs, a.demands, a.b, a.h)
    fig = go.Figure()
    fig.add_vrect(x0=a.optimum.q_low, x1=a.optimum.q_high, fillcolor=C.COLOR_OPT, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=qs, y=costs, mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="erwartete Kosten",
                             hovertemplate="Menge %{x:.1f}: Kosten %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[a.mean], y=[a.mean_cost], mode="markers", marker=dict(color=C.COLOR_MEAN, size=13, symbol="diamond"),
                             name=f"Mittel {_de(a.mean, 2) if abs(a.mean - round(a.mean)) > 1e-9 else int(round(a.mean))}",
                             hovertemplate="Mittel: Kosten %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[a.quantile], y=[float(A.cell_cost(a.quantile, a.demands, a.b, a.h))], mode="markers",
                             marker=dict(color=C.COLOR_QUANTILE, size=13, symbol="circle"), name=f"Quantil τ = {_de(a.tau, 2)}: {a.quantile}",
                             hovertemplate="Quantil: Kosten %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(a.demands), y=[0.0] * len(a.demands), mode="markers", marker=dict(color=C.COLOR_OTHER, size=9),
                             name="Nachfragen der Zelle", hovertemplate="Nachfrage %{x}<extra></extra>"))
    fig.update_xaxes(title_text="Bestellmenge q", range=[-0.2, top + 0.2])
    fig.update_yaxes(title_text="erwartete Kosten", rangemode="tozero")
    return _base(fig)


def build_kern(rows):
    """Mehrkosten gegen die wahre Quantilregel je tau: Balken = Median, Fehlerbalken = Minimum bis Maximum über die festen Seeds."""
    cats = [f"τ = {_de(r['tau'], 2)}" for r in rows]
    fig = go.Figure()
    for key, name, color in (("mittel", "Mittel bestellen", C.COLOR_MEAN), ("quantil", "KQ plus Normalquantil", C.COLOR_QUANTILE)):
        med = [r[key][0] for r in rows]
        fig.add_trace(go.Bar(
            x=cats, y=med, name=name, marker_color=color,
            error_y=dict(type="data", symmetric=False, array=[r[key][2] - r[key][0] for r in rows], arrayminus=[r[key][0] - r[key][1] for r in rows]),
            text=[_de(v) + " %" for v in med], textposition="outside",
            hovertemplate="%{x}: Median %{y:.1f} %<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Mehrkosten gegen die wahre Quantilregel (%)", rangemode="tozero")
    return _base(fig, 400)


def build_instance(view):
    """Eine Instanz: Nachfrage der Trainingsfälle über dem wahren bedingten Mittelwert, dazu Mittel (rot) und wahres Quantil (grün)."""
    lo, hi = float(view.mu.min()), float(view.mu.max())
    xs = np.array([lo, hi])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.mu, y=view.D, mode="markers", marker=dict(color=C.COLOR_OTHER, size=5), name="Trainingsfälle",
                             hovertemplate="Mittelwert %{x:.1f}, Nachfrage %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color=C.COLOR_MEAN, width=3), name="Mittel (Prognose ohne Fehler)"))
    fig.add_trace(go.Scatter(x=xs, y=A.true_quantile(xs, view.sigma, view.tau), mode="lines", line=dict(color=C.COLOR_QUANTILE, width=3),
                             name=f"wahres Quantil τ = {_de(view.tau, 2)}"))
    fig.update_xaxes(title_text="wahrer bedingter Mittelwert μ(x)")
    fig.update_yaxes(title_text="Nachfrage")
    return _base(fig)


def build_shift(tau: float, s: float):
    """Zwei Felder über der Verschiebung s: Quadratfehler (Minimum bei 0) und Kosten (Minimum bei z); die gewählte Verschiebung ist markiert."""
    grid = np.linspace(C.S_MIN, C.S_MAX, 251)
    mse, cost = A.shift_curves(tau, grid)
    z = A.z_tau(tau)
    mse_s, cost_s = A.shift_curves(tau, [s, 0.0, z])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Quadratfehler der Prognose", "erwartete Kosten der Menge"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=grid, y=mse, mode="lines", line=dict(color=C.COLOR_MEAN, width=3), name="Quadratfehler",
                             hovertemplate="s = %{x:.2f}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=grid, y=cost, mode="lines", line=dict(color=C.COLOR_QUANTILE, width=3), name="Kosten",
                             hovertemplate="s = %{x:.2f}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[s], y=[mse_s[0]], mode="markers", marker=dict(color="black", size=11), name="gewähltes s", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[s], y=[cost_s[0]], mode="markers", marker=dict(color="black", size=11), name="gewähltes s", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0.0], y=[mse_s[1]], mode="markers", marker=dict(color=C.COLOR_MEAN, size=12, symbol="diamond"),
                             name="s = 0 (Mittel)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.0], y=[cost_s[1]], mode="markers", marker=dict(color=C.COLOR_MEAN, size=12, symbol="diamond"),
                             name="s = 0 (Mittel)", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[z], y=[cost_s[2]], mode="markers", marker=dict(color=C.COLOR_QUANTILE, size=12, symbol="circle"),
                             name=f"s = z = {_de(z, 2)} (Quantil)"), row=1, col=2)
    fig.update_xaxes(title_text="Verschiebung s (in Reststreuungen)")
    fig.update_yaxes(title_text="Quadratfehler (σ² = 1)", row=1, col=1)
    fig.update_yaxes(title_text="Kosten (σ = 1, b + h = 1)", row=1, col=2)
    return _base(fig, 380)


def build_learning(rows):
    """Mehrkosten über der Zahl der Trainingsfälle n: Median je Regel mit Band (Minimum bis Maximum über die festen Seeds)."""
    xs = [str(r["n"]) for r in rows]
    fig = go.Figure()
    for key, name, color, fill in (("mittel", "Mittel bestellen", C.COLOR_MEAN, "rgba(192,80,77,0.15)"),
                                   ("quantil", "KQ plus Normalquantil", C.COLOR_QUANTILE, "rgba(47,107,101,0.15)")):
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r[key][2] for r in rows] + [r[key][1] for r in rows][::-1], fill="toself", fillcolor=fill,
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=[r[key][0] for r in rows], mode="lines+markers", line=dict(color=color, width=3), name=name,
                                 hovertemplate="n = %{x}: Median %{y:.1f} %<extra>" + name + "</extra>"))
    fig.update_xaxes(title_text="Trainingsfälle n", type="category")
    fig.update_yaxes(title_text="Mehrkosten gegen die wahre Quantilregel (%)", rangemode="tozero")
    return _base(fig)
