"""Orakel-Test regret-statt-fehler-demo: Kernroutinen gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12).

Orakel je Routine:
* Kosten der Zelle und beste Menge: exakte Bruchrechnung (`fractions`) und Aufzählung über ein feines Mengenraster, nicht die Knickstellen-Formel der App;
* erwartete Kosten der Normalverteilung (mit und ohne Stutzung): numerische Aufzählung über 200 000 gleich wahrscheinliche Nachfragen (Quantile der Normalverteilung auf der
  Wahrscheinlichkeitsachse, Mittelpunktregel), nicht die geschlossene Form mit Dichte und Verteilungsfunktion;
* wahres bedingtes Quantil: Monte Carlo mit 400 000 Ziehungen aus dem SplitMix64-Strom (Toleranz 1 %) und Aufzählung der Kostenfunktion über ein Mengenraster (mit Stutzung bei 0);
* Kleinste Quadrate: Normalgleichungen `solve(X'X, X'y)` statt `lstsq`; Reststreuung als Summe der Quadrate in reinem Python;
* SplitMix64: skalare Referenzimplementierung in reinem Python und der veröffentlichte erste Ausgabewert für den Zustand 0 (0xE220A8397B1DCDAF);
* Eigenschaften: Mehrkosten >= 0, die wahre Quantilregel ist je Testpunkt die billigste Menge (Regret 0), Skalierung der Kosten, Verschiebung der Nachfrage, Verschiebung s der Prognose (Quadratfehler = sigma^2 (1 + s^2), Kosten konvex).
"""

from fractions import Fraction

import numpy as np
import pytest
from scipy.special import ndtri

import rsf_algorithm as A
import rsf_rng as R

N_QUAD = 200_000
_U = (np.arange(N_QUAD) + 0.5) / N_QUAD
_Z = ndtri(_U)


def quad_cost(q, mu, sigma, b, h, truncated):
    """Erwartete Kosten per Aufzählung: Mittelwert über 200 000 gleich wahrscheinliche Nachfragen mu + sigma * z_i (gestutzt bei 0, falls verlangt)."""
    y = mu + sigma * _Z
    d = np.maximum(0.0, y) if truncated else y
    return float(np.mean(b * np.maximum(d - q, 0.0) + h * np.maximum(q - d, 0.0)))


# --- Zelle ----------------------------------------------------------------------------------------------------------------------------------------

def _cell_cost_fractions(q: Fraction, demands, b, h) -> Fraction:
    return sum(Fraction(b) * max(Fraction(d) - q, 0) + Fraction(h) * max(q - Fraction(d), 0) for d in demands) / len(demands)


def _random_cell(seed):
    rng = R.SplitMix64(seed)
    size = 1 + int(rng.u64(1)[0] % np.uint64(8))
    demands = tuple(int(v % np.uint64(20)) for v in rng.u64(size))
    b, h = 1 + int(rng.u64(1)[0] % np.uint64(19)), 1 + int(rng.u64(1)[0] % np.uint64(19))
    return demands, b, h


def test_zellkosten_gegen_bruchrechnung_an_60_zellen():
    for seed in range(60):
        demands, b, h = _random_cell(seed)
        for q4 in range(-4, 4 * 22):                      # Mengen in Vierteln, auch außerhalb des Nachfragebereichs
            q = Fraction(q4, 4)
            assert float(A.cell_cost(float(q), demands, b, h)) == pytest.approx(float(_cell_cost_fractions(q, demands, b, h))), (seed, q)


def test_beste_menge_gegen_aufzaehlung_ueber_ein_raster_an_60_zellen():
    for seed in range(60):
        demands, b, h = _random_cell(seed)
        grid = [Fraction(q4, 4) for q4 in range(-4, 4 * 22)]
        costs = {q: _cell_cost_fractions(q, demands, b, h) for q in grid}
        best = min(costs.values())
        opt = A.cell_optimum(demands, b, h)
        assert opt.cost == pytest.approx(float(best)), seed
        best_set = [q for q in grid if costs[q] == best]                 # alle optimalen Rastermengen (Nachfragen sind ganzzahlig, die Strecke liegt im Raster)
        assert float(min(best_set)) == opt.q_low and float(max(best_set)) == opt.q_high, (seed, demands, b, h)
        assert A.cell_quantile(demands, b, h) == opt.q_low, seed          # das Quantil ist die kleinste optimale Menge (Newsvendor-Satz, Gleichstand Z4)


def test_quantil_der_zelle_gegen_bruchdefinition():
    for seed in range(60):
        demands, b, h = _random_cell(seed)
        tau = Fraction(b, b + h)
        f = lambda x: Fraction(sum(1 for d in demands if d <= x), len(demands))   # noqa: E731
        expected = min(d for d in demands if f(d) >= tau)
        assert A.cell_quantile(demands, b, h) == expected, (seed, demands, b, h)


# --- Normalverteilung --------------------------------------------------------------------------------------------------------------------------

CASES = [(60.0, 8.0, 4, 1), (60.0, 8.0, 19, 1), (60.0, 8.0, 1, 1), (2.0, 8.0, 4, 1), (3.0, 8.0, 1, 19), (0.0, 5.0, 2, 3)]


@pytest.mark.parametrize("mu, sigma, b, h", CASES)
def test_erwartete_kosten_ohne_stutzung_gegen_aufzaehlung(mu, sigma, b, h):
    for q in (mu - 12.0, mu - 3.0, mu, mu + 2.5, mu + 11.0):
        assert float(A.cost_normal(q, mu, sigma, b, h)) == pytest.approx(quad_cost(q, mu, sigma, b, h, False), rel=2e-4, abs=2e-4)


@pytest.mark.parametrize("mu, sigma, b, h", CASES)
def test_erwartete_kosten_mit_stutzung_gegen_aufzaehlung(mu, sigma, b, h):
    for q in (0.0, 1.0, max(mu - 3.0, 0.0), mu + 2.5, mu + 11.0):
        assert float(A.cost_truncated(q, mu, sigma, b, h)) == pytest.approx(quad_cost(q, mu, sigma, b, h, True), rel=2e-4, abs=2e-4)


@pytest.mark.parametrize("mu, sigma, tau", [(60.0, 8.0, 0.5), (60.0, 8.0, 0.8), (60.0, 8.0, 0.95), (2.0, 8.0, 0.8), (2.0, 8.0, 0.4)])
def test_wahres_quantil_gegen_monte_carlo_400000_ziehungen(mu, sigma, tau):
    y = np.maximum(0.0, mu + sigma * R.stream(99, 7).normal(400_000))
    mc = float(np.quantile(y, tau))
    assert float(A.true_quantile(mu, sigma, tau)) == pytest.approx(mc, rel=0.01)


def test_wahres_quantil_bei_stutzung_gegen_aufzaehlung_der_kostenfunktion():
    """mu = 2, sigma = 8: die kostenbeste Menge auf einem 0,05er Raster (Kosten per Aufzählung, gestutzt bei 0) ist das gestutzte Quantil max(0, mu + sigma z)."""
    grid = np.arange(0.0, 40.0, 0.05)
    for tau in (0.05, 0.1, 0.3, 0.5, 0.8, 0.95):
        costs = np.array([quad_cost(q, 2.0, 8.0, tau, 1 - tau, True) for q in grid[::4]])          # Raster 0,2: schnell genug für 6 Niveaus
        q_enum = grid[::4][int(np.argmin(costs))]
        assert float(A.true_quantile(2.0, 8.0, tau)) == pytest.approx(q_enum, abs=0.2 + 1e-9), tau


# --- Kleinste Quadrate ---------------------------------------------------------------------------------------------------------------------------

def test_kleinste_quadrate_gegen_normalgleichungen_und_reststreuung_in_reinem_python():
    rng = R.SplitMix64(5)
    for p, n in ((1, 6), (3, 40), (10, 60)):
        X = rng.normal(n * p).reshape(n, p)
        y = 3.0 + X @ np.linspace(1.0, -1.0, p) + rng.normal(n)
        coef = A.fit_ols(X, y)
        Xa = np.hstack([np.ones((n, 1)), X])
        assert coef == pytest.approx(np.linalg.solve(Xa.T @ Xa, Xa.T @ y), abs=1e-8)
        sse = 0.0
        for i in range(n):
            fit = coef[0] + sum(coef[1 + j] * X[i, j] for j in range(p))
            sse += (y[i] - fit) ** 2
        assert A.sigma_hat(X, y, coef) == pytest.approx((sse / (n - p - 1)) ** 0.5)


# --- Zufallsstrom ---------------------------------------------------------------------------------------------------------------------------------

def _splitmix_scalar(state: int, count: int):
    mask = (1 << 64) - 1
    out = []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        out.append(z ^ (z >> 31))
    return out


def test_splitmix64_gegen_skalare_referenz_und_veroeffentlichten_wert():
    assert int(R.SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF
    for seed in (0, 1, 12345, 2**63 + 7):
        assert [int(v) for v in R.SplitMix64(seed).u64(25)] == _splitmix_scalar(seed, 25)
    u = R.SplitMix64(3).uniform(100_000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.005
    z = R.SplitMix64(3).normal(100_000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


# --- Eigenschaften -------------------------------------------------------------------------------------------------------------------------------

def test_mehrkosten_aller_regeln_sind_nie_negativ():
    for seed in range(6):
        for p in (3, 10):
            inst = A.fit_instance(p, 100, seed)
            Xt = A.draw_test_points(p, seed, 3000)
            ex = A.rule_excess(inst, (0.5, 0.8, 0.95), Xt)
            for tau, row in ex.items():
                assert min(row.values()) >= -1e-9, (seed, p, tau, row)


def test_die_wahre_quantilregel_ist_je_testpunkt_die_billigste_menge():
    mu = np.linspace(5.0, 90.0, 40)
    for tau in (0.2, 0.8, 0.95):
        best = A.true_quantile(mu, 8.0, tau)
        c_best = A.cost_truncated(best, mu, 8.0, tau, 1 - tau)
        for delta in (-3.0, -0.5, 0.5, 3.0):
            q = np.maximum(best + delta, 0.0)
            assert np.all(A.cost_truncated(q, mu, 8.0, tau, 1 - tau) >= c_best - 1e-9)


def test_skalieren_der_kosten_skaliert_die_kosten_und_laesst_die_entscheidung_gleich():
    demands = (1, 3, 3, 9, 14)
    for lam in (0.5, 7.0):
        a, b_ = A.cell_optimum(demands, 3, 2), A.cell_optimum(demands, 3 * lam, 2 * lam)
        assert (a.q_low, a.q_high) == (b_.q_low, b_.q_high) and b_.cost == pytest.approx(lam * a.cost)
        assert A.cell_quantile(demands, 3, 2) == A.cell_quantile(demands, 3 * lam, 2 * lam)
        assert float(A.cost_normal(65.0, 60.0, 8.0, 3 * lam, 2 * lam)) == pytest.approx(lam * float(A.cost_normal(65.0, 60.0, 8.0, 3, 2)))


def test_verschieben_der_nachfrage_verschiebt_die_menge_und_laesst_die_kosten_gleich():
    demands = (1, 3, 3, 9, 14)
    for c in (-1, 5, 100):
        shifted = tuple(d + c for d in demands)
        a, b_ = A.cell_optimum(demands, 4, 1), A.cell_optimum(shifted, 4, 1)
        assert (b_.q_low, b_.q_high) == (a.q_low + c, a.q_high + c) and b_.cost == pytest.approx(a.cost)
        assert A.cell_quantile(shifted, 4, 1) == A.cell_quantile(demands, 4, 1) + c


def test_verschiebung_der_prognose_quadratfehler_und_konvexe_kosten_gegen_aufzaehlung():
    s_grid = np.linspace(-2.0, 3.0, 26)
    for tau in (0.5, 0.8, 0.95):
        mse, cost = A.shift_curves(tau, s_grid, sigma=2.0)
        for s, m, c in zip(s_grid, mse, cost):
            y = 2.0 * _Z                                                           # mu = 0, sigma = 2
            assert m == pytest.approx(float(np.mean((y - 2.0 * s) ** 2)), rel=1e-3)
            assert c == pytest.approx(quad_cost(2.0 * s, 0.0, 2.0, tau, 1 - tau, False), rel=2e-4, abs=2e-4)
        assert np.all(np.diff(cost, 2) >= -1e-12)                                    # konvex
        assert s_grid[int(np.argmin(cost))] == s_grid[int(np.argmin(np.abs(s_grid - A.z_tau(tau))))]            # Minimum am Rasterpunkt nahe z
        assert s_grid[int(np.argmin(mse))] == pytest.approx(0.0, abs=1e-9)          # der Quadratfehler ist bei s = 0 am kleinsten


def test_mehrkosten_des_exakten_mittels_gegen_aufzaehlung():
    """exp(z^2 / 2) - 1 gegen das Verhältnis der aufgezählten Kosten bei q = mu und q = mu + z sigma."""
    for tau in (0.5, 0.8, 0.9, 0.95, 0.99):
        z = float(ndtri(tau))
        at_mean = quad_cost(0.0, 0.0, 1.0, tau, 1 - tau, False)
        at_opt = quad_cost(z, 0.0, 1.0, tau, 1 - tau, False)
        assert 100 * (at_mean / at_opt - 1) == pytest.approx(A.mean_order_excess_closed_form(tau), abs=0.15)
