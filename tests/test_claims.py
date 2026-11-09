"""Claims: jede Zahl der README steht hier als Rechnung, Band oder Zählung (DEMO-PLAYBOOK §11a). Die Tabelle „Zahl -> Test“ steht in der README.

Gemessen wird mit den Funktionen der App (`rsf_algorithm.sweep`), Welt linear, 30 feste Seeds (C.SWEEP_SEEDS), 20 000 Testpunkte je Seed, erwartete Kosten je Testpunkt
exakt. Alle Zahlen sind deterministisch (ganzzahliger Zufallsstrom, Kleinste Quadrate ohne Training); die Toleranz 0,06 deckt die Rundung auf eine Nachkommastelle
und Unterschiede der Linearen Algebra zwischen Plattformen ab. Werte der Vormessung stammen aus `vormessung_eo_v2.json` (eingefroren aus `out_eo_v2_news.txt`)."""

import json
from pathlib import Path

import numpy as np
import pytest

import rsf_algorithm as A
import rsf_constants as C
import rsf_evaluation as ev

TAUS = (0.5, 0.8, 0.95)
TOL = 0.06
VORMESSUNG = json.loads((Path(__file__).resolve().parent.parent / "vormessung_eo_v2.json").read_text(encoding="utf-8"))["zeilen"]

# (p, n, tau): {Regel: (Median, Minimum, Maximum)}; Mehrkosten in % gegen die wahre Quantilregel, 30 feste Seeds (README, Tabelle „Messung dieser Demo“)
MESSUNG = {
    (3, 30, 0.5): {"mittel": (4.5, 0.5, 25.4), "quantil": (4.5, 0.5, 25.4)},
    (3, 30, 0.8): {"mittel": (50.2, 18.0, 104.2), "quantil": (6.4, 0.5, 28.6)},
    (3, 30, 0.95): {"mittel": (312.9, 155.0, 490.6), "quantil": (10.0, 1.1, 35.6)},
    (3, 100, 0.5): {"mittel": (2.2, 0.8, 7.8), "quantil": (2.2, 0.8, 7.8)},
    (3, 100, 0.8): {"mittel": (47.3, 29.8, 84.1), "quantil": (2.3, 0.7, 10.1)},
    (3, 100, 0.95): {"mittel": (301.6, 232.2, 441.2), "quantil": (3.0, 0.8, 13.4)},
    (3, 1000, 0.5): {"mittel": (0.1, 0.0, 0.5), "quantil": (0.1, 0.0, 0.5)},
    (3, 1000, 0.8): {"mittel": (42.5, 36.7, 47.5), "quantil": (0.1, 0.0, 0.6)},
    (3, 1000, 0.95): {"mittel": (286.7, 262.7, 306.6), "quantil": (0.2, 0.0, 0.8)},
    (10, 30, 0.5): {"mittel": (21.2, 4.3, 66.4), "quantil": (21.2, 4.3, 66.4)},
    (10, 30, 0.8): {"mittel": (78.9, 36.1, 133.9), "quantil": (29.0, 6.1, 65.6)},
    (10, 30, 0.95): {"mittel": (390.8, 233.2, 559.4), "quantil": (38.2, 8.4, 100.7)},
    (10, 100, 0.5): {"mittel": (4.8, 2.8, 9.3), "quantil": (4.8, 2.8, 9.3)},
    (10, 100, 0.8): {"mittel": (49.7, 32.5, 82.2), "quantil": (5.1, 3.1, 11.6)},
    (10, 100, 0.95): {"mittel": (307.7, 237.7, 431.9), "quantil": (5.5, 3.4, 17.1)},
    (10, 1000, 0.5): {"mittel": (0.5, 0.3, 1.3), "quantil": (0.5, 0.3, 1.3)},
    (10, 1000, 0.8): {"mittel": (42.9, 36.8, 48.2), "quantil": (0.5, 0.3, 1.3)},
    (10, 1000, 0.95): {"mittel": (287.5, 263.0, 308.0), "quantil": (0.6, 0.3, 1.3)},
}


@pytest.fixture(scope="module")
def grid():
    """Mehrkosten je (p, n, tau) und Regel über die 30 festen Seeds."""
    out = {}
    for p in (3, 10):
        for n in (30, 100, 1000):
            res = A.sweep(p, n, TAUS)
            for t in TAUS:
                out[(p, n, t)] = {rule: A.band(v) for rule, v in res[t].items()}
    return out


# --- Handbeispiel und Zelle ------------------------------------------------------------------------------------------------------------------------

def test_handbeispiel_zahlen_der_readme():
    a = ev.analyse_cell(ev.Settings(b=4, h=1, cell="hand"))
    assert (a.mean, a.mean_cost, a.optimum.cost, a.quantile) == (6.0, pytest.approx(6.0), pytest.approx(4.0), 8)
    assert (a.optimum.q_low, a.optimum.q_high) == (8.0, 10.0)
    assert a.excess == pytest.approx(50.0)


def test_zelle_bei_tau_095_und_tau_halb_zahlen_der_readme():
    """Handbeispiel mit b = 19, h = 1: Mittel 6 kostet 120 / 5 = 24, die beste Menge 10 kostet 20 / 5 = 4: 500 %. Mit b = h = 1: Mittel ist beste Menge, Kosten 2,4."""
    a = ev.analyse_cell(ev.Settings(b=19, h=1, cell="hand"))
    assert a.mean_cost == pytest.approx(24.0) and a.optimum.cost == pytest.approx(4.0) and (a.optimum.q_low, a.optimum.q_high) == (10.0, 10.0)
    assert a.excess == pytest.approx(500.0)
    a = ev.analyse_cell(ev.Settings(b=1, h=1, cell="hand"))
    assert a.excess == 0.0 and a.mean_cost == pytest.approx(2.4)


def test_schiefe_zelle_zahlen_der_readme():
    """Nachfragen 3, 4, 4, 5, 6, 24; b = h = 1: Mittel 7,67 kostet 5,44, die beste Menge (4 bis 5) kostet 4,00: 36,1 % mehr (49/9 gegen 4)."""
    a = ev.analyse_cell(ev.Settings(b=1, h=1, cell="schief"))
    assert a.mean == pytest.approx(23 / 3) and a.mean_cost == pytest.approx(49 / 9) and a.optimum.cost == pytest.approx(4.0)
    assert (a.optimum.q_low, a.optimum.q_high) == (4.0, 5.0)
    assert round(a.excess, 1) == 36.1


# --- Geschlossene Form -----------------------------------------------------------------------------------------------------------------------------

def test_mehrkosten_des_exakten_mittels_exp_z2_halbe():
    """0 / 42,5 / 286,8 % bei tau = 0,5 / 0,8 / 0,95; hängt bei festem tau nicht an b und h, nicht an mu und sigma (Normalverteilung ohne Stutzung)."""
    assert [round(A.mean_order_excess_closed_form(t), 1) for t in TAUS] == [0.0, 42.5, 286.8]
    for mu, sigma, b, h in ((60.0, 8.0, 4, 1), (500.0, 3.0, 8, 2), (100.0, 40.0, 0.4, 0.1)):
        best = float(A.cost_normal(A.true_quantile(mu, sigma, b / (b + h)), mu, sigma, b, h))
        assert 100 * (float(A.cost_normal(mu, mu, sigma, b, h)) / best - 1) == pytest.approx(A.mean_order_excess_closed_form(0.8), abs=1e-9)


def test_exaktes_mittel_in_der_welt_liegt_in_jedem_seed_auf_der_formel(grid):
    """Das wahre Mittel als Menge (ohne Schätzfehler): in allen 18 Zellen Median, Minimum und Maximum gleich der geschlossenen Form (Stutzung bei 0 ändert weniger als 0,01 Punkte)."""
    for (p, n, t), r in grid.items():
        for value in r["mittel_exakt"]:
            assert value == pytest.approx(A.mean_order_excess_closed_form(t), abs=0.01), (p, n, t)


# --- Tabelle der Messung dieser Demo ---------------------------------------------------------------------------------------------------------------

def test_tabelle_der_messung_dieser_demo(grid):
    for key, expected in MESSUNG.items():
        for rule, values in expected.items():
            assert grid[key][rule] == pytest.approx(values, abs=TOL), (key, rule)


def test_vormessung_und_nachbau_liegen_in_jeder_der_36_zellen_im_band_der_anderen(grid):
    """18 Zellen (p, n, tau) mal 2 Regeln: der Median des Nachbaus liegt im Minimum-Maximum-Band der Vormessung und umgekehrt: 36 von 36 in beide Richtungen."""
    mine_in, theirs_in = 0, 0
    for (p, n, t), r in grid.items():
        v = VORMESSUNG[f"linear|{p}|{n}|{t}"]
        for mine, theirs in ((r["mittel"], v["mittel"]), (r["quantil"], v["kq_gauss"])):
            mine_in += theirs[1] <= mine[0] <= theirs[2]
            theirs_in += mine[1] <= theirs[0] <= mine[2]
    assert (mine_in, theirs_in) == (36, 36)


def test_vormessung_zahlen_der_readme():
    v = VORMESSUNG
    assert v["linear|3|1000|0.8"]["mittel"] == [44.2, 34.8, 48.8] and v["linear|3|1000|0.95"]["mittel"] == [292.9, 256.9, 312.5]
    assert v["linear|3|1000|0.5"]["mittel"][0] == 0.2 and v["linear|3|1000|0.8"]["kq_gauss"][0] == 0.2
    assert v["streuung-linear|3|1000|0.95"]["kq_gauss"][0] == 33.7
    assert v["linear|10|30|0.95"]["kq_gauss"][0] == 38.8 and v["linear|10|30|0.95"]["mittel"][0] == 408.1


# --- Zählungen und Aussagen ------------------------------------------------------------------------------------------------------------------------

def test_quantilregel_bei_n_1000_liegt_im_median_in_allen_sechs_zellen_unter_einem_prozent(grid):
    zellen = [(p, 1000, t) for p in (3, 10) for t in TAUS]
    assert len(zellen) == 6
    assert all(grid[z]["quantil"][0] < 1.0 for z in zellen)
    assert max(grid[z]["quantil"][0] for z in zellen) == pytest.approx(0.6, abs=TOL)       # größter Median: p = 10, tau = 0,95


def test_bei_tau_halb_sind_beide_regeln_in_allen_sechs_zellen_gleich(grid):
    assert all(grid[(p, n, 0.5)]["mittel"] == grid[(p, n, 0.5)]["quantil"] for p in (3, 10) for n in C.N_CHOICES)


def test_mittel_kostet_in_allen_zwoelf_zellen_bei_schiefen_kosten_mindestens_42_punkte_mehr_als_die_quantilregel(grid):
    zellen = [(p, n, t) for p in (3, 10) for n in C.N_CHOICES for t in (0.8, 0.95)]
    assert len(zellen) == 12
    diffs = [grid[z]["mittel"][0] - grid[z]["quantil"][0] for z in zellen]
    assert min(diffs) >= 42.0 and round(min(diffs), 1) == 42.3                              # kleinster Abstand: p = 10, n = 1000, tau = 0,8


def test_mittel_liegt_bei_n_1000_hoechstens_0_8_punkte_ueber_der_untergrenze(grid):
    for p in (3, 10):
        for t in (0.8, 0.95):
            gap = grid[(p, 1000, t)]["mittel"][0] - A.mean_order_excess_closed_form(t)
            assert -0.1 <= gap <= 0.8, (p, t, gap)


def test_mittel_bei_kleinem_n_liegt_ueber_der_untergrenze(grid):
    """n = 30 und n = 100, tau = 0,8 und 0,95, p = 3 und 10: der Median des Mittels liegt in allen 8 Zellen über der Untergrenze (der Schätzfehler kommt hinzu)."""
    zellen = [(p, n, t) for p in (3, 10) for n in (30, 100) for t in (0.8, 0.95)]
    assert all(grid[z]["mittel"][0] > A.mean_order_excess_closed_form(z[2]) for z in zellen)


def test_wenig_daten_viele_merkmale_p10_n30_tau095(grid):
    assert grid[(10, 30, 0.95)]["quantil"][0] > 30.0 and grid[(10, 30, 0.95)]["mittel"][0] > grid[(10, 30, 0.95)]["quantil"][0]
    assert [round(grid[(10, 30, t)]["quantil"][0], 1) for t in TAUS] == [21.2, 29.0, 38.2]


def test_quantilregel_wird_mit_mehr_daten_in_allen_sechs_reihen_besser(grid):
    """Median der Quantilregel fällt in den 6 Reihen (p, tau) mit n = 30, 100, 1000 jeweils streng."""
    for p in (3, 10):
        for t in TAUS:
            m = [grid[(p, n, t)]["quantil"][0] for n in C.N_CHOICES]
            assert m[0] > m[1] > m[2], (p, t, m)


def test_preset_hilfetext_nennt_die_mehrkosten_des_mittels_bei_tau_halb(grid):
    assert round(grid[(3, 1000, 0.5)]["mittel"][0], 1) == 0.1 and round(grid[(10, 1000, 0.5)]["mittel"][0], 1) == 0.5
    assert "0,1 % (p = 3) bzw. 0,5 % (p = 10)" in C.PRESET_HELP["Symmetrische Kosten (τ = 0,5)"]


def test_lernkurve_zahlen_der_readme():
    rows = ev.learning_rows(3, 0.8)
    assert [r["n"] for r in rows] == [30, 100, 1000]
    assert [round(r["mittel"][0], 1) for r in rows] == [50.2, 47.3, 42.5] and [round(r["quantil"][0], 1) for r in rows] == [6.4, 2.3, 0.1]
    rows = ev.learning_rows(3, 0.95)
    assert [round(r["mittel"][0], 1) for r in rows] == [312.9, 301.6, 286.7] and [round(r["quantil"][0], 1) for r in rows] == [10.0, 3.0, 0.2]


# --- Experiment 3 ------------------------------------------------------------------------------------------------------------------------------------

def test_verschiebung_zahlen_der_readme():
    """z = 0,84 / 1,64 bei tau = 0,8 / 0,95; Kosten des Mittels 42,5 / 286,8 % über dem Minimum; der Quadratfehler der kostenbesten Prognose ist um 70,8 / 270,5 % größer."""
    a8, a95 = ev.analyse_shift(0.8, 0.0), ev.analyse_shift(0.95, 0.0)
    assert (round(a8.z, 2), round(a95.z, 2)) == (0.84, 1.64)
    assert (round(a8.excess_zero, 1), round(a95.excess_zero, 1)) == (42.5, 286.8)
    assert (round(a8.mse_increase_at_opt, 1), round(a95.mse_increase_at_opt, 1)) == (70.8, 270.6)
    assert ev.analyse_shift(0.5, 0.0).excess_zero == 0.0


def test_verschiebung_hat_ihr_minimum_bei_z_und_der_quadratfehler_bei_null_in_der_ganzen_schar():
    for tau in np.linspace(0.05, 0.95, 19):
        grid_s = np.linspace(C.S_MIN, C.S_MAX, 5001)
        mse, cost = A.shift_curves(float(tau), grid_s)
        assert abs(grid_s[int(np.argmin(cost))] - A.z_tau(float(tau))) <= 0.001 + 1e-12
        assert grid_s[int(np.argmin(mse))] == pytest.approx(0.0, abs=1e-12)


def test_abgeschnittene_bestellmengen_sind_extrem_selten():
    """README: 20 von 14,4 Millionen Testmengen unter 0 (6 Zellen (p, n) mal 30 Seeds mal 20 000 Testpunkte mal 4 Mengen: Prognose und drei Quantilregeln), kleinste etwa -13,3."""
    negativ, gesamt, kleinste = 0, 0, 0.0
    for p in (3, 10):
        for n in C.N_CHOICES:
            for seed in C.SWEEP_SEEDS:
                inst = A.fit_instance(p, n, seed)
                y_hat = A.predict(inst.coef, A.draw_test_points(p, seed))
                for q in (y_hat, *(y_hat + A.z_tau(t) * inst.sigma_hat for t in TAUS)):
                    negativ += int((q < 0).sum())
                    gesamt += q.size
                    kleinste = min(kleinste, float(q.min()))
    assert gesamt == 14_400_000 and 15 <= negativ <= 25 and kleinste == pytest.approx(-13.3, abs=0.2)


def test_preset_hilfetexte_nennen_gemessene_zahlen(grid):
    assert "36,1 %" in C.PRESET_HELP["Schiefe Zelle, symmetrische Kosten"] and "7,67" in C.PRESET_HELP["Schiefe Zelle, symmetrische Kosten"]
    assert round(grid[(10, 30, 0.95)]["quantil"][0], 1) == 38.2 and "38,2 %" in C.PRESET_HELP["Wenig Daten, viele Merkmale"]
