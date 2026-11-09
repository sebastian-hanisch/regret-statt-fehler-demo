"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Handbeispiel: Nachfragen 2, 4, 6, 8, 10 gleich wahrscheinlich, Fehlmenge b = 4, Überschuss h = 1, tau = 0,8.

    Von Hand (je Nachfrage D die Kosten 4 * max(D - q, 0) + 1 * max(q - D, 0), Mittel der fünf):
      q = 6:  D = 2 -> 4, 4 -> 2, 6 -> 0, 8 -> 8, 10 -> 16: 30 / 5 = 6,00 (das Mittel)
      q = 8:  D = 2 -> 6, 4 -> 4, 6 -> 2, 8 -> 0, 10 -> 8: 20 / 5 = 4,00
      q = 10: D = 2 -> 8, 4 -> 6, 6 -> 4, 8 -> 2, 10 -> 0: 20 / 5 = 4,00 (Gleichstand: F(8) = 4/5 = tau, jede Menge von 8 bis 10 ist optimal)
    Mehrkosten des Mittels: (6 - 4) / 4 = 50 %."""
    return (2, 4, 6, 8, 10), 4, 1
