"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1 und n = 2, Gleichstand, tau nahe 0 und 1, sigma gegen 0,
Stutzung bei 0, ein Merkmal ohne Wirkung, tau = 0,5."""

import numpy as np
import pytest

import rsf_algorithm as A
import rsf_scenario as S


def test_zelle_mit_einer_einzigen_nachfrage():
    """n = 1: die Nachfrage selbst ist Mittel, Quantil und beste Menge, die Kosten sind 0 und die Mehrkosten des Mittels ebenfalls (0 durch 0 ist 0)."""
    opt = A.cell_optimum((7,), 4, 1)
    assert (opt.q_low, opt.q_high, opt.cost) == (7.0, 7.0, 0.0)
    assert A.cell_quantile((7,), 4, 1) == 7
    assert A.excess_percent(A.cell_cost(7.0, (7,), 4, 1), opt.cost) == 0.0


def test_zelle_mit_genau_zwei_nachfragen():
    """n = 2, b = h: F(2) = 1/2 = tau, jede Menge von 2 bis 6 kostet 2; das Mittel 4 gehört dazu (Mehrkosten 0), bei b = 3, h = 1 (tau = 3/4) ist die Menge 6 allein optimal."""
    assert A.cell_quantile((2, 6), 1, 1) == 2
    opt = A.cell_optimum((2, 6), 1, 1)
    assert (opt.q_low, opt.q_high, opt.cost) == (2.0, 6.0, pytest.approx(2.0))
    opt = A.cell_optimum((2, 6), 3, 1)
    assert (opt.q_low, opt.q_high) == (6.0, 6.0) and opt.cost == pytest.approx(2.0)   # q = 6: D = 2 -> Überschuss 4 (Kosten 4), D = 6 -> 0; Mittel 2; bei q = 2 wären es 6


def test_gleichstand_toleranz_ist_inklusive_und_eng():
    """Nachfragen 0 und 10, b = 1: bei q = 0 kostet es 5, bei q = 10 kostet es 5 * h. Mit h = 1 + 1e-10 beträgt der Unterschied 5e-10 (gleich gut), mit h = 1 + 1e-6 sind es 5e-6."""
    d = (0.0, 10.0)
    assert A.cell_optimum(d, 1.0, 1.0 + 1e-10).q_high == 10.0
    assert A.cell_optimum(d, 1.0, 1.0 + 1e-6).q_high == 0.0


def test_kritisches_verhaeltnis_nahe_0_und_1(hand):
    demands, _, _ = hand
    assert A.cell_quantile(demands, 1, 19) == 2          # tau = 0,05: die kleinste Nachfrage
    assert A.cell_quantile(demands, 19, 1) == 10         # tau = 0,95: F(8) = 0,8 < 0,95, also die größte Nachfrage
    assert A.cell_quantile(demands, 1, 10**9) == 2 and A.cell_quantile(demands, 10**9, 1) == 10
    assert A.true_quantile(60.0, 8.0, 1e-9) == pytest.approx(60.0 - 8.0 * 6.0, abs=0.2)   # z_1e-9 = -5,998
    with pytest.raises(ValueError):
        A.true_quantile(60.0, 8.0, 1.0)
    with pytest.raises(ValueError):
        A.true_quantile(60.0, 8.0, 0.0)


def test_reststreuung_braucht_mehr_faelle_als_parameter():
    """n = 2 mit einem Merkmal (p = 1): n - p - 1 = 0 Freiheitsgrade, die Streuung ist nicht definiert; n = 3 geht."""
    X2, y2 = np.array([[0.0], [1.0]]), np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        A.sigma_hat(X2, y2, A.fit_ols(X2, y2))
    X3, y3 = np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 2.0])
    assert A.sigma_hat(X3, y3, A.fit_ols(X3, y3)) > 0.0


def test_streuung_gegen_null_bleibt_endlich_und_wird_zur_kostenfunktion_ohne_zufall():
    """sigma = 1e-9: die Kosten gehen in b * (mu - q)+ + h * (q - mu)+ über (hier q = mu + 1: h * 1)."""
    c = float(A.cost_normal(61.0, 60.0, 1e-9, 4, 1))
    assert np.isfinite(c) and c == pytest.approx(1.0, abs=1e-6)
    c = float(A.cost_truncated(59.0, 60.0, 1e-9, 4, 1))
    assert c == pytest.approx(4.0, abs=1e-6)


def test_stutzung_bei_null_ist_aktiv_wenn_mu_klein_ist():
    """mu = 2, sigma = 8: P(Y < 0) = Phi(-0,25) = 0,40; die Menge 0 kostet b * E max(Y, 0), kein Überschuss."""
    from scipy.special import ndtr

    mu, sigma = 2.0, 8.0
    ey_plus = sigma * (np.exp(-0.5 * (mu / sigma) ** 2) / np.sqrt(2 * np.pi) + (mu / sigma) * ndtr(mu / sigma))   # E max(Y, 0) = mu Phi(mu/s) + s phi(mu/s)
    assert float(A.cost_truncated(0.0, mu, sigma, 4, 1)) == pytest.approx(4 * ey_plus)
    assert float(A.cost_truncated(5.0, mu, sigma, 4, 1)) != pytest.approx(float(A.cost_normal(5.0, mu, sigma, 4, 1)), rel=1e-3)   # ohne Stutzung wäre es anders
    with pytest.raises(ValueError):
        A.cost_truncated(-1.0, mu, sigma, 4, 1)


def test_merkmal_ohne_wirkung_schaetzt_steigung_nahe_null_und_mittel_bleibt_teuer():
    """Welt ohne Merkmalswirkung (Steigung 0): D hängt nicht von x ab. Die geschätzte Steigung ist klein gegen die Streuung (8 / sqrt(1000) = 0,25),
    und das Mittel zu bestellen kostet weiter etwa exp(z^2 / 2) - 1 mehr, die Quantilregel wenig."""
    w = S.World(p=1, beta=(1.0,), slope=0.0)
    X = S.draw_features(1000, 1, 11, tag=2)
    D = S.draw_demand(w, X, 11)
    inst = A.FittedInstance(w, X, D, A.fit_ols(X, D), 0.0)
    assert abs(inst.coef[1]) < 1.0
    inst = A.FittedInstance(w, X, D, inst.coef, A.sigma_hat(X, D, inst.coef))
    ex = A.rule_excess(inst, (0.8,), A.draw_test_points(1, 11, 4000))[0.8]
    assert ex["mittel"] == pytest.approx(A.mean_order_excess_closed_form(0.8), abs=5.0)
    assert ex["quantil"] < 3.0


def test_bei_tau_halb_sind_mittel_und_quantilregel_gleich():
    """z_0,5 = 0: die Quantilregel ist die Mittelregel; das exakte Mittel hat Mehrkosten 0."""
    ex = A.simulate_seed(3, 100, 2, (0.5,), n_test=2000)[0.5]
    assert ex["mittel"] == ex["quantil"] and ex["mittel_exakt"] == pytest.approx(0.0, abs=1e-9)
    assert A.mean_order_excess_closed_form(0.5) == 0.0


def test_mehrkosten_bei_null_besten_kosten():
    assert A.excess_percent(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        A.excess_percent(1.0, 0.0)


def test_zelle_ohne_nachfragen_und_negative_nachfrage_sind_ungueltig():
    with pytest.raises(ValueError):
        S.Cell(())
    with pytest.raises(ValueError):
        S.Cell((1, -2))
