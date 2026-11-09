"""Kern: Newsvendor-Kosten, Regret, Mittel- und Quantilregel, Kleinste Quadrate, Simulation der Welt (nur numpy und scipy.special).

Newsvendor: Fehlmenge b je Einheit, Überschuss h je Einheit, Kosten(q, D) = b * max(D - q, 0) + h * max(q - D, 0), kritisches Verhältnis
tau = b / (b + h). Regret einer Menge q = erwartete Kosten(q) minus erwartete Kosten der besten Menge bei bekannter bedingter Verteilung.

Zwei Rechenwege stehen hier nebeneinander:
* Zelle (Experiment 1): feste Nachfragen, Aufzählung der Kosten an den Knickstellen, exakte Bruchrechnung für das Quantil;
* Welt (Experimente 2 bis 4): Normalverteilung, erwartete Kosten in geschlossener Form (ohne Monte-Carlo-Rauschen der Nachfrage), gemittelt über feste Testpunkte x.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.special import ndtr, ndtri

import rsf_constants as C
import rsf_scenario as S

SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def pdf(d):
    """Dichte der Standardnormalverteilung."""
    d = np.asarray(d, dtype=float)
    return np.exp(-0.5 * d * d) / SQRT_2PI


def tau_of(b: float, h: float) -> float:
    """Kritisches Verhältnis b / (b + h)."""
    if b <= 0 or h <= 0:
        raise ValueError("Fehlmenge b und Überschuss h müssen positiv sein")
    return b / (b + h)


def excess_percent(cost: float, best: float) -> float:
    """Mehrkosten in % gegen die beste Menge: 100 * (cost - best) / best; bei best = 0 und cost = 0 gleich 0."""
    if best <= 0.0:
        if cost <= 0.0:
            return 0.0
        raise ValueError("Beste Kosten sind 0, die Kosten der Menge nicht: Mehrkosten nicht definiert")
    return 100.0 * (cost - best) / best


# --- Zelle: feste Nachfragen ----------------------------------------------------------------------------------------------------------------------

def cell_cost(q, demands, b: float, h: float):
    """Erwartete Kosten der Menge q (Skalar oder Feld) bei gleich wahrscheinlichen Nachfragen."""
    d = np.asarray(demands, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    over = np.maximum(q_arr[..., None] - d, 0.0)
    short = np.maximum(d - q_arr[..., None], 0.0)
    return (b * short + h * over).mean(axis=-1)


def cell_mean(demands) -> float:
    return float(np.mean(np.asarray(demands, dtype=float)))


def cell_quantile(demands, b: float, h: float):
    """Kleinste Nachfrage d mit F(d) >= tau, F die Verteilungsfunktion der Zelle; exakt in Brüchen (kein Gleitkommavergleich am Gleichstand)."""
    tau_b, tau_h = Fraction(b), Fraction(h)
    ds = sorted(demands)
    n = len(ds)
    for d in sorted(set(ds)):
        count = sum(1 for x in ds if x <= d)
        if Fraction(count, n) * (tau_b + tau_h) >= tau_b:
            return d
    return ds[-1]


@dataclass(frozen=True)
class CellOptimum:
    q_low: float       # kleinste optimale Menge (Gleichstandsregel Z4)
    q_high: float      # größte optimale Menge: jede Menge dazwischen ist gleich gut
    cost: float        # Kosten der optimalen Menge


def cell_optimum(demands, b: float, h: float, tol: float = C.TIE_TOL) -> CellOptimum:
    """Optimale Menge durch Aufzählung: die Kosten sind stückweise linear mit Knicken an den Nachfragen, das Minimum liegt dort."""
    points = np.array(sorted(set(demands)), dtype=float)
    costs = cell_cost(points, demands, b, h)
    best = float(costs.min())
    ok = points[costs <= best + tol]
    return CellOptimum(float(ok.min()), float(ok.max()), best)


# --- Welt: Normalverteilung ------------------------------------------------------------------------------------------------------------------------

def cost_normal(q, mu, sigma, b: float, h: float):
    """Erwartete Kosten der Menge q bei Y ~ N(mu, sigma^2) ohne Stutzung: sigma * ((b + h) * phi(d) + d * ((b + h) * Phi(d) - b)), d = (q - mu) / sigma."""
    q, mu, sigma = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float))
    d = (q - mu) / sigma
    short = sigma * (pdf(d) - d * ndtr(-d))        # E (Y - q)+
    over = sigma * (pdf(d) + d * ndtr(d))          # E (q - Y)+
    return b * short + h * over


def cost_truncated(q, mu, sigma, b: float, h: float):
    """Erwartete Kosten der Menge q >= 0 bei D = max(0, Y), Y ~ N(mu, sigma^2).

    (D - q)+ = (Y - q)+ für q >= 0; (q - D)+ = (q - Y)+ - (0 - Y)+ (für Y < 0 ist es q, sonst (q - Y)+). Daraus die geschlossene Form."""
    q = np.asarray(q, dtype=float)
    if np.any(q < 0):
        raise ValueError("Bestellmengen sind nicht negativ")
    q, mu, sigma = np.broadcast_arrays(q, np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float))
    d = (q - mu) / sigma
    d0 = -mu / sigma
    short = sigma * (pdf(d) - d * ndtr(-d))
    over = sigma * (pdf(d) + d * ndtr(d)) - sigma * (pdf(d0) + d0 * ndtr(d0))
    return b * short + h * over


def true_quantile(mu, sigma, tau: float):
    """Wahres tau-Quantil von D = max(0, Y): max(0, mu + sigma * z_tau) (Stutzung ist monoton)."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau liegt zwischen 0 und 1")
    return np.maximum(0.0, np.asarray(mu, dtype=float) + np.asarray(sigma, dtype=float) * float(ndtri(tau)))


def z_tau(tau: float) -> float:
    return float(ndtri(tau))


def mean_order_excess_closed_form(tau: float) -> float:
    """Mehrkosten in % für das exakte Mittel als Bestellmenge bei Normalverteilung ohne Stutzung: exp(z^2 / 2) - 1, z = z_tau.

    Herleitung: cost(mu) = sigma (b + h) phi(0), cost(mu + z sigma) = sigma (b + h) phi(z), denn (b + h) Phi(z) - b = 0."""
    return 100.0 * (float(np.exp(0.5 * z_tau(tau) ** 2)) - 1.0)


# --- Kleinste Quadrate -----------------------------------------------------------------------------------------------------------------------------

def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kleinste Quadrate mit Achsenabschnitt: Koeffizienten (c0, c1..cp)."""
    Xa = np.hstack([np.ones((len(X), 1)), X])
    return np.linalg.lstsq(Xa, y, rcond=None)[0]


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + np.asarray(X) @ coef[1:]


def sigma_hat(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    """Gepoolte Reststreuung mit n - p - 1 Freiheitsgraden; für n <= p + 1 nicht definiert."""
    n, p = X.shape
    if n <= p + 1:
        raise ValueError("Reststreuung braucht n > p + 1")
    res = y - predict(coef, X)
    return float(np.sqrt(np.sum(res**2) / (n - p - 1)))


# --- Simulation einer Instanz ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FittedInstance:
    world: S.World
    X: np.ndarray        # Trainingsmerkmale
    D: np.ndarray        # Trainingsnachfragen
    coef: np.ndarray
    sigma_hat: float


def fit_instance(p: int, n: int, seed: int) -> FittedInstance:
    """Welt, Trainingsdaten (n Fälle) und angepasstes lineares Modell für (p, n, seed)."""
    world = S.make_world(p, seed)
    X = S.draw_features(n, p, seed, tag=2)
    D = S.draw_demand(world, X, seed)
    coef = fit_ols(X, D)
    return FittedInstance(world, X, D, coef, sigma_hat(X, D, coef))


def draw_test_points(p: int, seed: int, n_test: int = C.N_TEST) -> np.ndarray:
    return S.draw_features(n_test, p, seed, tag=4)


def _mean_cost(q, mu, sigma, b, h) -> float:
    return float(np.mean(cost_truncated(np.maximum(q, 0.0), mu, sigma, b, h)))


def rule_excess(inst: FittedInstance, taus, Xt: np.ndarray) -> dict:
    """Mehrkosten in % gegen die wahre Quantilregel, je tau und Regel: `mittel` (Prognose als Menge), `quantil` (Prognose plus Normalquantil der
    gepoolten Reststreuung), `mittel_exakt` (wahres Mittel als Menge, ohne Schätzfehler). Erwartete Kosten exakt über die Testpunkte gemittelt."""
    mu = inst.world.mu(Xt)
    sig = inst.world.sigma
    y_hat = predict(inst.coef, Xt)
    out = {}
    for tau in taus:
        b, h = tau, 1.0 - tau                      # Mehrkosten in % hängen nur an tau (Skalierung der Kosten kürzt sich)
        best = _mean_cost(true_quantile(mu, sig, tau), mu, sig, b, h)
        out[tau] = {
            "mittel": excess_percent(_mean_cost(y_hat, mu, sig, b, h), best),
            "quantil": excess_percent(_mean_cost(y_hat + z_tau(tau) * inst.sigma_hat, mu, sig, b, h), best),
            "mittel_exakt": excess_percent(_mean_cost(mu, mu, sig, b, h), best),
        }
    return out


def simulate_seed(p: int, n: int, seed: int, taus, n_test: int = C.N_TEST) -> dict:
    """Eine Instanz von der Welt bis zu den Mehrkosten; `taus` ein Tupel kritischer Verhältnisse."""
    return rule_excess(fit_instance(p, n, seed), taus, draw_test_points(p, seed, n_test))


def sweep(p: int, n: int, taus, seeds=C.SWEEP_SEEDS, n_test: int = C.N_TEST) -> dict:
    """Mehrkosten je tau und Regel über feste Seeds: {tau: {regel: Feld über die Seeds}}."""
    taus = tuple(taus)
    rows = [simulate_seed(p, n, s, taus, n_test) for s in seeds]
    return {tau: {rule: np.array([r[tau][rule] for r in rows]) for rule in ("mittel", "quantil", "mittel_exakt")} for tau in taus}


def band(values) -> tuple:
    """(Median, Minimum, Maximum)."""
    v = np.asarray(values, dtype=float)
    return float(np.median(v)), float(v.min()), float(v.max())


# --- Experiment 3: Verschiebung der Prognose -------------------------------------------------------------------------------------------------------

def shift_curves(tau: float, s_grid, sigma: float = 1.0):
    """Prognose = wahres Mittel + s * sigma, Verteilung N(mu, sigma^2) ohne Stutzung. Liefert (Quadratfehler(s), Kosten(s)) mit
    b = tau, h = 1 - tau. Quadratfehler gegen die Nachfrage: sigma^2 (1 + s^2) (Rest plus Verschiebung zum Quadrat)."""
    s = np.asarray(s_grid, dtype=float)
    mse = sigma**2 * (1.0 + s**2)
    cost = cost_normal(s * sigma, 0.0, sigma, tau, 1.0 - tau)
    return mse, cost
