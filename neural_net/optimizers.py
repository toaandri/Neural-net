"""Mises à jour des paramètres ; état indépendant pour chaque tableau."""

import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate doit être fini et strictement positif")
        self.learning_rate = float(learning_rate)
        self._state = {}
        self.iterations = 0

    def step(self, layers):
        pairs = [pair for layer in layers if hasattr(layer, "parameters_and_gradients")
                 for pair in layer.parameters_and_gradients()]
        # Valider tous les gradients avant de modifier le moindre paramètre.
        if len({id(p) for p, _ in pairs}) != len(pairs):
            raise ValueError("Un paramètre ne doit apparaître qu'une seule fois")
        for parameter, gradient in pairs:
            if parameter.shape != gradient.shape or not np.all(np.isfinite(gradient)):
                raise ValueError("Gradient incompatible ou non fini")
        if not pairs:
            raise ValueError("Aucun paramètre à mettre à jour")
        self.iterations += 1
        for parameter, gradient in pairs:
            key = id(parameter)
            if key not in self._state:
                # Garder une référence empêche la réutilisation d'un identifiant.
                self._state[key] = (parameter, np.zeros_like(parameter), np.zeros_like(parameter))
            _, first, second = self._state[key]
            parameter -= self.learning_rate * self._direction(gradient, first, second)

    def _direction(self, gradient, first, second):
        return gradient


def _decay(value, name):
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} doit appartenir à [0, 1[")
    return float(value)


def _epsilon(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError("epsilon doit être fini et strictement positif")
    return float(value)


class Momentum(SGD):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = _decay(momentum, "momentum")

    def _direction(self, gradient, first, second):
        first *= self.momentum
        first += gradient
        return first


class RMSProp(SGD):
    def __init__(self, learning_rate=0.01, decay=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay = _decay(decay, "decay")
        self.epsilon = _epsilon(epsilon)

    def _direction(self, gradient, first, second):
        second *= self.decay
        second += (1 - self.decay) * gradient ** 2
        return gradient / (np.sqrt(second) + self.epsilon)


class Adam(SGD):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = _decay(beta1, "beta1")
        self.beta2 = _decay(beta2, "beta2")
        self.epsilon = _epsilon(epsilon)

    def _direction(self, gradient, first, second):
        first *= self.beta1
        first += (1 - self.beta1) * gradient
        second *= self.beta2
        second += (1 - self.beta2) * gradient ** 2
        corrected_first = first / (1 - self.beta1 ** self.iterations)
        corrected_second = second / (1 - self.beta2 ** self.iterations)
        return corrected_first / (np.sqrt(corrected_second) + self.epsilon)
