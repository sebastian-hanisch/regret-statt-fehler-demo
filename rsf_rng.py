"""Zufallsstrom: SplitMix64 (Steele, Lea, Flood 2014), ganzzahlig, plattform- und versionsunabhängig.

Herkunft: kopiert und gekürzt aus `entrollte-netze-demo/inverse_generator.py` (Klasse SplitMix64: u64, uniform, normal), nichts aus anderen Demos
importiert. Neu hinzugefügt: `stream(seed, tag)`, ein eigener Strom je (Seed, Verwendungszweck), damit Training, Test und Welt unabhängig bleiben.

Ausgabe k hängt nur von (Seed, k) ab. Nur numpy; Python-`hash()` und `numpy.random` kommen nirgends vor."""

import numpy as np

_GAMMA = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB

# Verwendungszwecke eines Seeds: je ein eigener Strom
TAG_WORLD, TAG_TRAIN_X, TAG_TRAIN_Z, TAG_TEST_X = 1, 2, 3, 4


class SplitMix64:
    """SplitMix64-Strom. `u64(size)` rückt den Zustand um `size` Schritte weiter."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilte Werte in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormalverteilte Werte (Box-Muller, nur der Kosinus-Zweig); verbraucht 2 * size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)


def stream(seed: int, tag: int) -> SplitMix64:
    """Eigener Strom für (Seed, Zweck): der Anfangszustand ist die erste Ausgabe eines SplitMix64 mit dem gemischten Schlüssel."""
    key = (int(seed) * 256 + int(tag)) & _MASK
    return SplitMix64(int(SplitMix64(key).u64(1)[0]))
