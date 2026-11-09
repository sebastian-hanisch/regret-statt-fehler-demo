"""Szenarien: die Zelle mit festen Nachfragen (Experiment 1) und die Welt mit Merkmalen (Experimente 2 bis 4)."""

from dataclasses import dataclass

import numpy as np

import rsf_constants as C
import rsf_rng as R


@dataclass(frozen=True)
class Cell:
    """Eine Merkmalszelle: feste Nachfragen, jede gleich wahrscheinlich."""

    demands: tuple

    def __post_init__(self):
        if len(self.demands) < 1:
            raise ValueError("Eine Zelle braucht mindestens eine Nachfrage")
        if min(self.demands) < 0:
            raise ValueError("Nachfragen sind nicht negativ")


@dataclass(frozen=True)
class World:
    """Welt: D = max(0, mu0 + slope * beta'x + sigma * z), beta aus U(-1, 1)^p, auf Länge 1 normiert (Vormessung eo_v2_news.py, Welt „linear“)."""

    p: int
    beta: tuple
    mu0: float = C.MU0
    slope: float = C.MU_SLOPE
    sigma: float = C.SIGMA

    def mu(self, X: np.ndarray) -> np.ndarray:
        """Bedingter Mittelwert des Rohwerts Y (vor der Stutzung bei 0)."""
        return self.mu0 + self.slope * (np.asarray(X) @ np.array(self.beta))


def make_world(p: int, seed: int) -> World:
    if p < 1:
        raise ValueError("p >= 1")
    beta = 2.0 * R.stream(seed, R.TAG_WORLD).uniform(p) - 1.0
    norm = float(np.linalg.norm(beta))
    if norm == 0.0:
        beta = np.ones(p)
        norm = float(np.linalg.norm(beta))
    return World(p=p, beta=tuple(float(v) for v in beta / norm))


def draw_features(n: int, p: int, seed: int, tag: int) -> np.ndarray:
    """n Merkmalszeilen, standardnormal, aus dem Strom (seed, tag)."""
    return R.stream(seed, tag).normal(n * p).reshape(n, p)


def draw_demand(world: World, X: np.ndarray, seed: int) -> np.ndarray:
    """Nachfrage D = max(0, mu(x) + sigma z) zu den Zeilen von X."""
    z = R.stream(seed, R.TAG_TRAIN_Z).normal(len(X))
    return np.maximum(0.0, world.mu(X) + world.sigma * z)
