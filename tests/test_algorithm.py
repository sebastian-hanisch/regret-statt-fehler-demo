"""Kern gegen Handrechnung (Handbeispiel, Zelle mit Ausreißer, kleine Regression) und einfache Bibliotheksgegenprobe."""

from fractions import Fraction

import numpy as np
import pytest

import rsf_algorithm as A
import rsf_constants as C
import rsf_evaluation as ev
import rsf_rng as R
import rsf_scenario as S


def test_handbeispiel_kosten_von_hand(hand):
    demands, b, h = hand
    assert A.cell_mean(demands) == 6.0
    assert A.cell_cost(6.0, demands, b, h) == pytest.approx(6.0)     # 30 / 5
    assert A.cell_cost(8.0, demands, b, h) == pytest.approx(4.0)     # 20 / 5
    assert A.cell_cost(10.0, demands, b, h) == pytest.approx(4.0)    # 20 / 5


def test_handbeispiel_beste_menge_ist_eine_strecke_und_die_kleinste_wird_genannt(hand):
    demands, b, h = hand
    opt = A.cell_optimum(demands, b, h)
    assert (opt.q_low, opt.q_high) == (8.0, 10.0)
    assert opt.cost == pytest.approx(4.0)
    assert A.cell_quantile(demands, b, h) == 8          # F(8) = 4/5 = tau genau: die kleinste Menge mit F >= tau (Entscheidung Z4)


def test_handbeispiel_mehrkosten_der_mitte_sind_50_prozent(hand):
    demands, b, h = hand
    assert A.excess_percent(A.cell_cost(6.0, demands, b, h), A.cell_optimum(demands, b, h).cost) == pytest.approx(50.0)
    a = ev.analyse_cell(ev.Settings(b=4, h=1, cell="hand"))
    assert a.excess == pytest.approx(50.0) and a.quantile == 8 and a.tau == pytest.approx(0.8)


def test_zelle_mit_ausreisser_von_hand():
    """Nachfragen 3, 4, 4, 5, 6, 24, b = h = 1: Mittel 23/3; Kosten dort 49/9 (Summe der Abstände 98/3, durch 6); beste Menge 4 mit Kosten 4 (Summe 24, durch 6)."""
    demands = C.CELLS["schief"][1]
    mean = Fraction(sum(demands), len(demands))
    assert mean == Fraction(23, 3)
    assert A.cell_cost(float(mean), demands, 1, 1) == pytest.approx(49 / 9)
    opt = A.cell_optimum(demands, 1, 1)
    assert (opt.q_low, opt.q_high, opt.cost) == (4.0, 5.0, pytest.approx(4.0))   # F(4) = 3/6 = tau: jede Menge von 4 bis 5
    assert ev.analyse_cell(ev.Settings(b=1, h=1, cell="schief")).excess == pytest.approx(100 * 13 / 36)   # (49/9) / 4 - 1


def test_kritisches_verhaeltnis():
    assert A.tau_of(4, 1) == pytest.approx(0.8) and A.tau_of(19, 1) == pytest.approx(0.95) and A.tau_of(1, 1) == 0.5
    with pytest.raises(ValueError):
        A.tau_of(0, 1)


def test_kleinste_quadrate_und_reststreuung_von_hand():
    """x = 0, 1, 2, 3; y = 1, 3, 2, 4: Steigung Sxy / Sxx = 4 / 5 = 0,8, Achsenabschnitt 2,5 - 0,8 * 1,5 = 1,3; Reste -0,3, 0,9, -0,9, 0,3;
    Summe der Quadrate 1,8; n - p - 1 = 2 Freiheitsgrade: sigma^2 = 0,9."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    coef = A.fit_ols(X, y)
    assert coef == pytest.approx([1.3, 0.8])
    assert A.predict(coef, X) == pytest.approx([1.3, 2.1, 2.9, 3.7])
    assert A.sigma_hat(X, y, coef) == pytest.approx(np.sqrt(0.9))


def test_wahres_quantil_der_welt_von_hand():
    assert A.true_quantile(60.0, 8.0, 0.5) == pytest.approx(60.0)
    assert A.true_quantile(60.0, 8.0, 0.8) == pytest.approx(60.0 + 8.0 * 0.8416212335729143)   # z_0,8 aus der Tabelle der Normalverteilung
    assert A.true_quantile(2.0, 8.0, 0.1) == 0.0                                                 # Stutzung: 2 - 8 * 1,2816 < 0


def test_kosten_der_normalverteilung_am_mittelwert_von_hand():
    """Bei q = mu: E (Y - mu)+ = E (mu - Y)+ = sigma * phi(0), also Kosten (b + h) * sigma / sqrt(2 pi)."""
    assert A.cost_normal(60.0, 60.0, 8.0, 4, 1) == pytest.approx(5 * 8.0 / np.sqrt(2 * np.pi))


def test_wahre_menge_hat_kleinste_kosten_und_das_mittel_kostet_exp_z2_halbe_mehr():
    mu, sigma = 60.0, 8.0
    for tau in (0.5, 0.8, 0.95):
        b, h = tau, 1 - tau
        best = float(A.cost_normal(A.true_quantile(mu, sigma, tau), mu, sigma, b, h))
        at_mean = float(A.cost_normal(mu, mu, sigma, b, h))
        assert at_mean / best - 1 == pytest.approx(np.exp(0.5 * A.z_tau(tau) ** 2) - 1)
        assert A.mean_order_excess_closed_form(tau) == pytest.approx(100 * (at_mean / best - 1))


def test_zufallsstrom_ist_reproduzierbar_und_die_welt_normiert():
    assert np.array_equal(R.stream(5, 2).u64(10), R.stream(5, 2).u64(10))
    assert not np.array_equal(R.stream(5, 2).u64(10), R.stream(5, 3).u64(10))
    w = S.make_world(10, 3)
    assert np.linalg.norm(w.beta) == pytest.approx(1.0)
    assert S.make_world(10, 3) == w and S.make_world(10, 4) != w


def test_simulation_ist_deterministisch():
    a = A.simulate_seed(3, 100, 4, (0.8,), n_test=2000)
    assert a == A.simulate_seed(3, 100, 4, (0.8,), n_test=2000)
