"""Perceptron binaire, implémenté uniquement avec Python et NumPy.

Pour chaque observation x, le modèle calcule :

    z = w · x + b
    y = 1 si z >= 0, sinon 0

La règle d'apprentissage corrige les paramètres à chaque erreur :

    erreur = y - prediction
    w <- w + learning_rate * erreur * x
    b <- b + learning_rate * erreur
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """Classifieur linéaire pour des sorties binaires 0/1."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 100, random_state: int | None = 42):
        if learning_rate <= 0:
            raise ValueError("learning_rate doit être strictement positif")
        if epochs <= 0:
            raise ValueError("epochs doit être strictement positif")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.errors_: list[int] = []

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X doit être une matrice non vide de forme (n_samples, n_features)")

        if y is None:
            return X, None

        y = np.asarray(y, dtype=int)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y doit être un vecteur de même longueur que X")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y ne peut contenir que les classes 0 et 1")
        return X, y

    def _ensure_fitted(self, n_features: int) -> None:
        if self.weights is None:
            raise RuntimeError("Le perceptron doit être entraîné avec fit() avant predict()")
        if self.weights.shape[0] != n_features:
            raise ValueError("Le nombre de caractéristiques ne correspond pas à celui de fit()")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Entraîne le perceptron et retourne le modèle entraîné."""
        X, y = self._validate_inputs(X, y)
        assert y is not None

        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(0.0, 0.01, size=X.shape[1])
        self.bias = 0.0
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for features, target in zip(X, y):
                prediction = self._predict_one(features)
                update = self.learning_rate * (target - prediction)
                self.weights += update * features
                self.bias += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def _predict_one(self, features: np.ndarray) -> int:
        assert self.weights is not None
        return int(np.dot(features, self.weights) + self.bias >= 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne les classes prédites (0 ou 1)."""
        X, _ = self._validate_inputs(X)
        self._ensure_fitted(X.shape[1])
        return np.array([self._predict_one(row) for row in X], dtype=int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Retourne l'accuracy sur un ensemble de données."""
        X, y = self._validate_inputs(X, y)
        assert y is not None
        return float(np.mean(self.predict(X) == y))
