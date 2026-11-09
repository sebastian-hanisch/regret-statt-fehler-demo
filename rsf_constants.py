"""Konstanten: Regler-Grenzen, feste Annahmen der Welt, Farben und Presets."""

# --- Kosten: Fehlmenge b und Überschuss h je Einheit, kritisches Verhältnis tau = b / (b + h) ---------------------------------------------------
B_MIN, B_MAX, DEFAULT_B = 1, 19, 4
H_MIN, H_MAX, DEFAULT_H = 1, 19, 1

# --- Welt der Experimente 2 bis 4 (Vormessung eo_v2_news.py, Welt „linear“) -----------------------------------------------------------------------
MU0, MU_SLOPE, SIGMA = 60.0, 10.0, 8.0       # D = max(0, 60 + 10 * beta'x + 8 * z), beta normiert, z ~ N(0, 1)
P_CHOICES = (3, 10)
N_CHOICES = (30, 100, 1000)
DEFAULT_P, DEFAULT_N = 3, 1000
TAU_REFERENCE = (0.5, 0.8, 0.95)             # die drei Verhältnisse der Vormessung: (b, h) = (1, 1), (4, 1), (19, 1)
SWEEP_SEEDS = tuple(range(30))               # feste Seeds für alle Sweeps: nie ein einzelner Seed als Befund
N_TEST = 20000                               # Testfälle je Seed (Merkmalspunkte; die Kosten je Punkt werden exakt gerechnet)
SEED_MAX, DEFAULT_SEED = 999999, 7

# --- Experiment 3: Verschiebung der Prognose in Einheiten der Reststreuung ------------------------------------------------------------------------
S_MIN, S_MAX, DEFAULT_S = -2.0, 3.0, 0.0

# --- Experiment 1: Zellen mit festen Nachfragen (gleich wahrscheinlich) ---------------------------------------------------------------------------
CELLS = {
    "hand": ("Handbeispiel: 2, 4, 6, 8, 10", (2, 4, 6, 8, 10)),
    "schief": ("Schiefe Zelle: 3, 4, 4, 5, 6, 24", (3, 4, 4, 5, 6, 24)),
}
DEFAULT_CELL = "hand"
EXCESS_WARN = 10.0                           # Prozent Mehrkosten, ab denen die App warnt (Grenze inklusive)
TIE_TOL = 1e-9                               # Toleranz für „gleich gute Mengen“ (Entscheidung Z4)

COLOR_MEAN = "#c0504d"
COLOR_QUANTILE = "#2F6B65"
COLOR_OPT = "rgba(47,107,101,0.18)"
COLOR_LINE = "#4c78a8"
COLOR_OTHER = "rgba(150,150,150,0.55)"

PRESETS = {
    "Handbeispiel (Voreinstellung)": {"b": 4, "h": 1, "cell": "hand", "p": 3, "n": 1000, "s": 0.0, "seed": DEFAULT_SEED},
    "Symmetrische Kosten (τ = 0,5)": {"b": 1, "h": 1, "cell": "hand", "p": 3, "n": 1000, "s": 0.0, "seed": DEFAULT_SEED},
    "Sehr schiefe Kosten (τ = 0,95)": {"b": 19, "h": 1, "cell": "hand", "p": 3, "n": 1000, "s": 0.0, "seed": DEFAULT_SEED},
    "Schiefe Zelle, symmetrische Kosten": {"b": 1, "h": 1, "cell": "schief", "p": 3, "n": 1000, "s": 0.0, "seed": DEFAULT_SEED},
    "Wenig Daten, viele Merkmale": {"b": 19, "h": 1, "cell": "hand", "p": 10, "n": 30, "s": 0.0, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Handbeispiel (Voreinstellung)": "Fehlmenge 4, Überschuss 1 (τ = 0,8); Zelle mit den Nachfragen 2, 4, 6, 8, 10; Welt mit 3 Merkmalen und 1000 Trainingsfällen.",
    "Symmetrische Kosten (τ = 0,5)": "Fehlmenge und Überschuss gleich teuer: Mittel und Median der Welt fallen zusammen; bei n = 1000 betragen die Mehrkosten des Mittels im Median 0,1 % (p = 3) bzw. 0,5 % (p = 10), siehe Experiment 2.",
    "Sehr schiefe Kosten (τ = 0,95)": "Fehlmenge 19, Überschuss 1: eine fehlende Einheit kostet 19-mal so viel wie eine überschüssige.",
    "Schiefe Zelle, symmetrische Kosten": "Zelle mit einem Ausreißer nach oben (Nachfragen 3, 4, 4, 5, 6, 24): selbst bei gleichen Kosten kostet das Mittel 7,67 um 36,1 % mehr als die beste Menge (4 bis 5).",
    "Wenig Daten, viele Merkmale": "30 Trainingsfälle, 10 Merkmale, τ = 0,95: auch KQ plus Normalquantil kostet im Median 38,2 % mehr als die wahre Quantilregel (Experiment 2).",
}
