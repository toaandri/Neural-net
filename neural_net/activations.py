"""Activations et produits du gradient amont par leur dérivée locale."""

import numpy as np

from ._validation import matrix


class _Activation:
    def __init__(self):
        self._output = None

    def forward(self, inputs):
        self._output = self._apply(matrix(inputs))
        return self._output.copy()

    def backward(self, gradient):
        if self._output is None:
            raise RuntimeError("forward() doit précéder backward()")
        gradient = matrix(gradient, "gradient")
        if gradient.shape != self._output.shape:
            raise ValueError("Forme du gradient incompatible avec l'activation")
        return self._backward(gradient)


class Sigmoid(_Activation):
    def _apply(self, x):
        # Deux branches évitent le débordement de exp pour les grands |x|.
        result = np.empty_like(x)
        positive = x >= 0
        result[positive] = 1 / (1 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        result[~positive] = exp_x / (1 + exp_x)
        return result

    def _backward(self, gradient):
        return gradient * self._output * (1 - self._output)


class ReLU(_Activation):
    def _apply(self, x):
        return np.maximum(0, x)

    def _backward(self, gradient):
        return gradient * (self._output > 0)


class Tanh(_Activation):
    def _apply(self, x):
        return np.tanh(x)

    def _backward(self, gradient):
        return gradient * (1 - self._output ** 2)


class Softmax(_Activation):
    def _apply(self, x):
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def _backward(self, gradient):
        # Produit vectoriel par la Jacobienne complète, sans la matérialiser.
        return self._output * (gradient - (gradient * self._output).sum(axis=1, keepdims=True))
