"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a Punkt 7)."""

import pytest

import rsf_constants as C
import rsf_evaluation as ev
import rsf_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        assert C.B_MIN <= preset["b"] <= C.B_MAX and C.H_MIN <= preset["h"] <= C.H_MAX, name
        assert preset["cell"] in C.CELLS and preset["p"] in C.P_CHOICES and preset["n"] in C.N_CHOICES, name
        assert C.S_MIN <= preset["s"] <= C.S_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Handbeispiel (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_defaults_are_valid():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for spec in P.SETTING_SPECS.values():
        if spec.choices is not None:
            assert spec.default in spec.choices
        else:
            assert spec.lo <= spec.default <= spec.hi


def test_preset_names_tell_the_truth_about_tau():
    """Die Namen nennen τ: das Verhältnis aus b und h muss stimmen."""
    assert ev.Settings(**C.PRESETS["Symmetrische Kosten (τ = 0,5)"]).tau == 0.5
    assert ev.Settings(**C.PRESETS["Sehr schiefe Kosten (τ = 0,95)"]).tau == pytest.approx(0.95)
    assert ev.Settings(**C.PRESETS["Wenig Daten, viele Merkmale"]).tau == pytest.approx(0.95)
    assert ev.Settings(**C.PRESETS["Handbeispiel (Voreinstellung)"]).tau == pytest.approx(0.8)


@pytest.mark.parametrize("name, cell_kind, cell_text, shift_kind", [
    ("Handbeispiel (Voreinstellung)", "warning", "**50,0 % mehr**", "warning"),
    ("Symmetrische Kosten (τ = 0,5)", "success", "ist hier selbst eine beste Menge", "info"),
    ("Sehr schiefe Kosten (τ = 0,95)", "warning", "**500,0 % mehr**", "warning"),            # Mittel 6: Kosten 24; beste Menge 10: Kosten 4 (von Hand)
    ("Schiefe Zelle, symmetrische Kosten", "warning", "**36,1 % mehr**", "info"),             # 49/9 gegen 4, siehe test_algorithm
    ("Wenig Daten, viele Merkmale", "warning", "**500,0 % mehr**", "warning"),
])
def test_jedes_preset_hat_die_erwartete_verdikt_zeile(name, cell_kind, cell_text, shift_kind):
    s = ev.Settings(**C.PRESETS[name])
    kind, text = ev.verdict_cell(ev.analyse_cell(s))
    assert kind == cell_kind and cell_text in text, (name, text)
    assert ev.verdict_shift(ev.analyse_shift(s.tau, s.s))[0] == shift_kind


def test_preset_wenig_daten_zeigt_im_kernverdikt_eine_warnung():
    s = ev.Settings(**C.PRESETS["Wenig Daten, viele Merkmale"])
    rows = ev.sweep_rows(s.p, s.n, ev.tau_list(s.tau))
    kind, text = ev.verdict_kernfrage(ev.row_for(rows, s.tau), s.p, s.n)
    assert kind == "warning" and "p = 10, n = 30" in text


def test_sweep_uses_the_fixed_seeds_and_a_consistent_band():
    rows = ev.sweep_rows(3, 100, ev.tau_list(0.8))
    assert [r["tau"] for r in rows] == [0.5, 0.8, 0.95]          # 0,8 ist schon eines der Referenzverhältnisse: keine Doppelten
    for r in rows:
        for rule in ("mittel", "quantil", "mittel_exakt"):
            med, lo, hi = r[rule]
            assert lo <= med <= hi
    assert len(C.SWEEP_SEEDS) == 30 and len(set(C.SWEEP_SEEDS)) == 30


def test_tau_list_nimmt_ein_eigenes_verhaeltnis_auf():
    assert ev.tau_list(0.25) == (0.25, 0.5, 0.8, 0.95)
    assert ev.row_for(ev.sweep_rows(3, 30, ev.tau_list(0.25)), 0.25)["tau"] == 0.25
