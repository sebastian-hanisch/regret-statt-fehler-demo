"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier. Verdikte sind reine Funktionen (Art, Text)."""

import pytest

import rsf_algorithm as A
import rsf_constants as C
import rsf_evaluation as ev


def _cell(excess, mean=6.0, mean_cost=6.0, opt=(8.0, 10.0, 4.0)):
    return ev.CellAnalysis((2, 4, 6, 8, 10), 4, 1, 0.8, mean, mean_cost, A.CellOptimum(*opt), 8, excess)


def test_zahlenformat_deutsch():
    assert ev.de(3.14159, 2) == "3,14" and ev.de(-3.14159, 2) == "−3,14" and ev.de(-0.04) == "0,0" and ev.de(12.0, 0) == "12"
    assert ev.qty(6.0) == "6" and ev.qty(23 / 3) == "7,67"


def test_verdikt_zelle_handbeispiel_nennt_kosten_mehrkosten_und_die_strecke():
    kind, text = ev.verdict_cell(ev.analyse_cell(ev.Settings(b=4, h=1, cell="hand")))
    assert kind == "warning"
    assert "Das Mittel (6) zu bestellen kostet 6,00 statt 4,00" in text and "**50,0 % mehr**" in text
    assert "jede Menge von 8 bis 10 ist gleich gut" in text


def test_verdikt_zelle_schwelle_zehn_prozent_ist_inklusive():
    assert C.EXCESS_WARN == 10.0
    kind, text = ev.verdict_cell(_cell(10.0, mean_cost=4.4))
    assert kind == "warning" and "**10,0 % mehr**" in text
    kind, text = ev.verdict_cell(_cell(9.9, mean_cost=4.396))
    assert kind == "info" and text.startswith("Nur das Mittel") and "**9,9 % mehr**" in text


def test_verdikt_zelle_eindeutige_beste_menge_ohne_strecke():
    kind, text = ev.verdict_cell(_cell(25.0, mean_cost=5.0, opt=(8.0, 8.0, 4.0)))
    assert kind == "warning" and "die beste Menge (8)" in text and "gleich gut" not in text


def test_verdikt_zelle_mittel_ist_optimal_bei_null_mehrkosten():
    kind, text = ev.verdict_cell(ev.analyse_cell(ev.Settings(b=1, h=1, cell="hand")))   # F(6) = 3/5 >= 1/2, F(4) = 2/5 < 1/2: Menge 6 ist die einzige beste
    assert kind == "success" and "Das Mittel (6) ist hier selbst eine beste Menge (6)" in text and "Kosten 2,40" in text   # (4 + 2 + 0 + 2 + 4) / 5


def test_verdikt_kernfrage_warnung_ab_fuenf_punkten_unterschied():
    row = {"tau": 0.8, "mittel": (42.5, 36.7, 47.5), "quantil": (0.1, 0.0, 0.6)}
    kind, text = ev.verdict_kernfrage(row, 3, 1000)
    assert kind == "warning"
    for part in ("τ = 0,80", "**42,5 %**", "(Band 36,7 bis 47,5 %)", "**0,1 %**", "(Band 0,0 bis 0,6 %)", "p = 3, n = 1000", "30 feste Seeds"):
        assert part in text
    assert ev.verdict_kernfrage({"tau": 0.8, "mittel": (7.0, 1.0, 9.0), "quantil": (2.0, 0.0, 3.0)}, 10, 30)[0] == "warning"      # genau 5,0 Punkte
    kind, text = ev.verdict_kernfrage({"tau": 0.5, "mittel": (6.9, 1.0, 9.0), "quantil": (2.0, 0.0, 3.0)}, 10, 30)               # 4,9 Punkte
    assert kind == "success" and "Der Unterschied beträgt 4,9 Punkte" in text and "p = 10, n = 30" in text


def test_verdikt_verschiebung_beim_mittel_und_bei_tau_halb():
    a = ev.analyse_shift(0.8, 0.0)
    assert a.z == pytest.approx(0.8416212335729143) and a.excess_zero == pytest.approx(42.5, abs=0.05) and a.excess_at_s == pytest.approx(a.excess_zero)
    assert a.mse_at_s == 100.0 and a.mse_increase_at_opt == pytest.approx(100 * a.z**2) and a.cost_at_opt < a.cost_at_zero
    kind, text = ev.verdict_shift(a)
    assert kind == "warning" and "s = z = 0,84" in text and "**42,5 % mehr**" in text and "um 70,8 % größeren Quadratfehler" in text
    b = ev.analyse_shift(0.8, 0.84162123357)
    assert b.excess_at_s == pytest.approx(0.0, abs=1e-6) and b.mse_at_s == pytest.approx(100 * (1 + b.z**2))
    kind, text = ev.verdict_shift(ev.analyse_shift(0.5, 1.5))
    assert kind == "info" and "τ = 0,50 ist z = 0" in text


def test_verdikt_verschiebung_bei_tau_unter_halb_hat_negatives_z():
    a = ev.analyse_shift(0.2, 0.0)
    assert a.z == pytest.approx(-0.8416212335729143) and a.excess_zero == pytest.approx(42.5, abs=0.05)
    assert "s = z = −0,84" in ev.verdict_shift(a)[1]


ROWS = [{"n": 30, "mittel": (50.0, 20.0, 100.0), "quantil": (6.0, 1.0, 30.0)},
        {"n": 100, "mittel": (47.0, 30.0, 80.0), "quantil": (2.0, 0.5, 10.0)},
        {"n": 1000, "mittel": (42.5, 36.0, 48.0), "quantil": (0.1, 0.0, 0.6)}]


def test_verdikt_lernkurve_nennt_zahlen_aenderungen_und_die_untergrenze():
    kind, text = ev.verdict_learning(ROWS, 0.8, 3)
    assert kind == "warning"
    for part in ("n = 30 / 100 / 1000", "τ = 0,80, p = 3", "Mittel bestellen 50,0 / 47,0 / 42,5 %", "KQ plus Normalquantil 6,0 / 2,0 / 0,1 %",
                 "sinken die Mehrkosten der Quantilregel um 5,9 Punkte", "die des Mittels sinken um 7,5 Punkte",
                 "kostet 42,5 % mehr: exp(z²/2) − 1 mit z = 0,84"):
        assert part in text, part


def test_verdikt_lernkurve_steigende_mehrkosten_und_schwelle():
    rows = [{**r} for r in ROWS]
    rows[0] = {"n": 30, "mittel": (9.0, 1.0, 20.0), "quantil": (1.0, 0.0, 3.0)}
    rows[2] = {"n": 1000, "mittel": (9.9, 5.0, 12.0), "quantil": (2.5, 0.0, 3.0)}
    kind, text = ev.verdict_learning(rows, 0.5, 10)
    assert kind == "info" and "steigen die Mehrkosten der Quantilregel um 1,5 Punkte" in text and "die des Mittels steigen um 0,9 Punkte" in text
    rows[2] = {"n": 1000, "mittel": (10.0, 5.0, 12.0), "quantil": (0.0, 0.0, 3.0)}
    assert ev.verdict_learning(rows, 0.5, 10)[0] == "warning"
