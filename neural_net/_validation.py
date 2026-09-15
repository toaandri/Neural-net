"""Validation commune des tableaux utilisés par le moteur."""

import numpy as np


def matrix(value, name="X"):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or 0 in value.shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} doit être une matrice non vide et finie")
    return value


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} doit être un entier strictement positif")
    return int(value)
