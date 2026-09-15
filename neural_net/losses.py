"""Coûts moyens et dérivées par rapport aux prédictions.

Les entropies croisées attendent des probabilités, pas des logits.
Le clipping rend le coût fini ; sa dérivée est nulle hors de l'intervalle
ouvert de clipping, conformément à la fonction effectivement calculée.
"""

import numpy as np

from ._validation import matrix


class _Loss:
    def _validate(self, predictions, targets):
        predictions = matrix(predictions, "predictions")
        targets = matrix(targets, "targets")
        if predictions.shape != targets.shape:
            raise ValueError("Prédictions et cibles doivent avoir la même forme")
        return predictions, targets


class MeanSquaredError(_Loss):
    def forward(self, predictions, targets):
        p, y = self._validate(predictions, targets)
        return float(np.mean((p - y) ** 2))

    def backward(self, predictions, targets):
        p, y = self._validate(predictions, targets)
        return 2 * (p - y) / p.size


class BinaryCrossEntropy(_Loss):
    epsilon = 1e-12

    def _validate(self, predictions, targets):
        p, y = super()._validate(predictions, targets)
        if np.any((p < 0) | (p > 1)) or np.any((y < 0) | (y > 1)):
            raise ValueError("Probabilités et cibles doivent appartenir à [0, 1]")
        return p, y

    def forward(self, predictions, targets):
        p, y = self._validate(predictions, targets)
        q = np.clip(p, self.epsilon, 1 - self.epsilon)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))

    def backward(self, predictions, targets):
        p, y = self._validate(predictions, targets)
        q = np.clip(p, self.epsilon, 1 - self.epsilon)
        active = (p > self.epsilon) & (p < 1 - self.epsilon)
        return ((q - y) / (q * (1 - q))) * active / p.size


class CategoricalCrossEntropy(BinaryCrossEntropy):
    def _validate(self, predictions, targets):
        p, y = super()._validate(predictions, targets)
        if p.shape[1] < 2 or not np.allclose(p.sum(axis=1), 1) or not np.allclose(y.sum(axis=1), 1):
            raise ValueError("Chaque ligne doit être une distribution sur au moins deux classes")
        return p, y

    def forward(self, predictions, targets):
        p, y = self._validate(predictions, targets)
        return float(-np.mean(np.sum(y * np.log(np.clip(p, self.epsilon, 1)), axis=1)))

    def backward(self, predictions, targets):
        p, y = self._validate(predictions, targets)
        return -y / np.clip(p, self.epsilon, 1) * (p > self.epsilon) / p.shape[0]
