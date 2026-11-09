"""Kennzahlen, Verdikt-Logik und Sweeps. Verdikte sind reine Funktionen (Art, Text); die App ruft `getattr(st, Art)(Text)`."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import rsf_algorithm as A
import rsf_constants as C


@dataclass(frozen=True)
class Settings:
    b: int = C.DEFAULT_B
    h: int = C.DEFAULT_H
    cell: str = C.DEFAULT_CELL
    p: int = C.DEFAULT_P
    n: int = C.DEFAULT_N
    s: float = C.DEFAULT_S
    seed: int = C.DEFAULT_SEED

    @property
    def tau(self) -> float:
        return A.tau_of(self.b, self.h)


# --- Zahlenformat (deutsches Dezimalkomma) ----------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 1) -> str:
    """Zahl mit Dezimalkomma und nd Nachkommastellen; ein negatives Vorzeichen als „−“ (Minuszeichen)."""
    text = f"{x:.{nd}f}".replace(".", ",")
    if text.startswith("-") and float(text.replace(",", ".")) != 0.0:
        return "−" + text[1:]
    return text.lstrip("-")


def qty(x: float) -> str:
    """Menge: ganze Zahl ohne Nachkommastelle, sonst zwei Stellen."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else de(x, 2)


# --- Experiment 1: Zelle ----------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CellAnalysis:
    demands: tuple
    b: float
    h: float
    tau: float
    mean: float
    mean_cost: float
    optimum: A.CellOptimum
    quantile: float
    excess: float          # Prozent: Mehrkosten der Menge „Mittel“ gegen die beste Menge


def analyse_cell(s: Settings) -> CellAnalysis:
    demands = C.CELLS[s.cell][1]
    mean = A.cell_mean(demands)
    mean_cost = float(A.cell_cost(mean, demands, s.b, s.h))
    opt = A.cell_optimum(demands, s.b, s.h)
    return CellAnalysis(demands, s.b, s.h, s.tau, mean, mean_cost, opt, A.cell_quantile(demands, s.b, s.h), A.excess_percent(mean_cost, opt.cost))


def verdict_cell(a: CellAnalysis):
    opt = a.optimum
    where = f"({qty(opt.q_low)})" if opt.q_high - opt.q_low < 1e-9 else f"(jede Menge von {qty(opt.q_low)} bis {qty(opt.q_high)} ist gleich gut)"
    if a.excess <= 1e-9:
        return "success", f"Das Mittel ({qty(a.mean)}) ist hier selbst eine beste Menge {where}: Kosten {de(a.mean_cost, 2)}, nichts geht billiger."
    kind = "warning" if a.excess >= C.EXCESS_WARN else "info"
    lead = "Das Mittel" if kind == "warning" else "Nur das Mittel"
    return kind, (f"{lead} ({qty(a.mean)}) zu bestellen kostet {de(a.mean_cost, 2)} statt {de(opt.cost, 2)}: "
                  f"**{de(a.excess)} % mehr** als die beste Menge {where}.")


# --- Experiment 2: perfektes Modell, teure Entscheidung ----------------------------------------------------------------------------------------------

def tau_list(tau: float) -> tuple:
    """Die drei Referenzverhältnisse der Vormessung plus das eingestellte, aufsteigend und ohne Doppelte."""
    return tuple(sorted(set(round(t, 9) for t in (*C.TAU_REFERENCE, tau))))


def sweep_rows(p: int, n: int, taus) -> list:
    """Zeilen {tau, mittel, quantil, mittel_exakt}: je (Median, Minimum, Maximum) über die festen Seeds."""
    res = A.sweep(p, n, taus)
    return [{"tau": t, **{rule: A.band(res[t][rule]) for rule in ("mittel", "quantil", "mittel_exakt")}} for t in taus]


def row_for(rows: list, tau: float) -> dict:
    return min(rows, key=lambda r: abs(r["tau"] - tau))


def verdict_kernfrage(row: dict, p: int, n: int):
    """Mehrkosten des Mittels gegen die Quantilregel mit Normalquantil; ab 5 Punkten Unterschied eine Warnung."""
    m, r = row["mittel"], row["quantil"]
    scope = f"Welt linear, p = {p}, n = {n}, Median über {len(C.SWEEP_SEEDS)} feste Seeds"
    text = (f"Bei τ = {de(row['tau'], 2)} kostet das Mittel als Bestellmenge **{de(m[0])} %** mehr als die wahre Quantilregel (Band {de(m[1])} bis {de(m[2])} %), "
            f"KQ plus Normalquantil **{de(r[0])} %** (Band {de(r[1])} bis {de(r[2])} %); {scope}.")
    if m[0] - r[0] >= 5.0:
        return "warning", text
    return "success", text + f" Der Unterschied beträgt {de(m[0] - r[0])} Punkte: bei diesen Kosten fällt das Mittel kaum ins Gewicht."


def vormessung(world: str, p: int, n: int, tau: float):
    """Eingefrorene Zeile der Vormessung (eo_v2_news.py, numpy-Zufallsstrom, nicht live): {mittel, kq_gauss} je (Median, Minimum, Maximum) oder None."""
    data = json.loads((Path(__file__).resolve().parent / "vormessung_eo_v2.json").read_text(encoding="utf-8"))["zeilen"]
    return data.get(f"{world}|{p}|{n}|{tau}")


@dataclass(frozen=True)
class InstanceView:
    mu: np.ndarray       # wahrer bedingter Mittelwert der Trainingsfälle
    D: np.ndarray        # Trainingsnachfragen
    tau: float
    z: float
    sigma: float
    excess_mean: float   # Mehrkosten in %, Prognose als Menge
    excess_quantile: float
    excess_exact_mean: float


def analyse_instance(s: Settings) -> InstanceView:
    inst = A.fit_instance(s.p, s.n, s.seed)
    Xt = A.draw_test_points(s.p, s.seed)
    ex = A.rule_excess(inst, (s.tau,), Xt)[s.tau]
    return InstanceView(inst.world.mu(inst.X), inst.D, s.tau, A.z_tau(s.tau), inst.world.sigma, ex["mittel"], ex["quantil"], ex["mittel_exakt"])


# --- Experiment 3: Verschiebung der Prognose ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ShiftAnalysis:
    tau: float
    z: float                 # Verschiebung mit den kleinsten Kosten
    s: float                 # gewählte Verschiebung
    mse_at_s: float          # Quadratfehler in Prozent der Reststreuungs-Varianz: 100 * (1 + s^2); der kleinste (s = 0) ist 100
    cost_at_s: float         # Kosten bei s, in Einheiten der Reststreuung
    cost_at_zero: float
    cost_at_opt: float
    excess_zero: float       # Prozent: Kosten bei s = 0 gegen s = z
    excess_at_s: float       # Prozent: Kosten bei s gegen s = z
    mse_increase_at_opt: float   # Prozent: Quadratfehler bei s = z gegen s = 0, 100 * z^2


def analyse_shift(tau: float, s: float) -> ShiftAnalysis:
    z = A.z_tau(tau)
    _, c = A.shift_curves(tau, [s, 0.0, z])
    return ShiftAnalysis(tau, z, s, 100.0 * (1.0 + s * s), float(c[0]), float(c[1]), float(c[2]),
                         A.excess_percent(float(c[1]), float(c[2])), A.excess_percent(float(c[0]), float(c[2])), 100.0 * z * z)


def verdict_shift(a: ShiftAnalysis):
    if abs(a.z) < 1e-9:
        return "info", (f"Bei τ = {de(a.tau, 2)} ist z = 0: Quadratfehler und Kosten haben dasselbe Minimum bei s = 0. "
                        "Bei symmetrischen Kosten und symmetrischer Streuung fallen beste Prognose und beste Menge zusammen.")
    return "warning", (f"Der Quadratfehler ist bei s = 0 am kleinsten, die Kosten bei s = z = {de(a.z, 2)}. Die Prognose mit dem kleinsten Quadratfehler kostet "
                       f"**{de(a.excess_zero)} % mehr** als die beste Menge; die beste Menge hat einen um {de(a.mse_increase_at_opt)} % größeren Quadratfehler "
                       "(Normalverteilung ohne Stutzung, s in Einheiten der Reststreuung).")


# --- Experiment 4: Lernkurve ------------------------------------------------------------------------------------------------------------------------

def learning_rows(p: int, tau: float) -> list:
    """Je n aus C.N_CHOICES: {n, mittel, quantil} mit (Median, Minimum, Maximum) über die festen Seeds."""
    rows = []
    for n in C.N_CHOICES:
        r = row_for(sweep_rows(p, n, (round(tau, 9),)), tau)
        rows.append({"n": n, "mittel": r["mittel"], "quantil": r["quantil"]})
    return rows


def _verb(delta: float) -> str:
    return "sinken" if delta >= 0 else "steigen"


def verdict_learning(rows: list, tau: float, p: int):
    first, last = rows[0], rows[-1]
    ms = " / ".join(de(r["mittel"][0]) for r in rows)
    qs = " / ".join(de(r["quantil"][0]) for r in rows)
    ns = " / ".join(str(r["n"]) for r in rows)
    d_m = first["mittel"][0] - last["mittel"][0]
    d_q = first["quantil"][0] - last["quantil"][0]
    text = (f"Mehrkosten im Median bei n = {ns} (τ = {de(tau, 2)}, p = {p}): Mittel bestellen {ms} %, KQ plus Normalquantil {qs} %. "
            f"Von n = {first['n']} auf n = {last['n']} {_verb(d_q)} die Mehrkosten der Quantilregel um {de(abs(d_q))} Punkte, "
            f"die des Mittels {_verb(d_m)} um {de(abs(d_m))} Punkte.")
    floor = A.mean_order_excess_closed_form(tau)
    text += (f" Selbst das exakte Mittel (unendlich viele Daten, Normalverteilung ohne Stutzung) kostet {de(floor)} % mehr: "
             f"exp(z²/2) − 1 mit z = {de(A.z_tau(tau), 2)}.")
    if last["mittel"][0] >= C.EXCESS_WARN:
        return "warning", text
    return "info", text
