"""Couche affine : Z = XW + b ; dX = dZ Wᵀ, dW = Xᵀ dZ."""

import numpy as np

from ._validation import matrix, positive_integer


class Dense:
    def __init__(self, input_size, output_size, random_state=None):
        self.input_size = positive_integer(input_size, "input_size")
        self.output_size = positive_integer(output_size, "output_size")
        rng = np.random.default_rng(random_state)
        # Xavier : conserve l'ordre de grandeur des signaux entre les couches.
        scale = np.sqrt(2 / (self.input_size + self.output_size))
        self.weights = rng.normal(0, scale, (self.input_size, self.output_size))
        self.bias = np.zeros((1, self.output_size))
        self.dweights = None
        self.dbias = None
        self._inputs = None

    def forward(self, inputs):
        inputs = matrix(inputs)
        if inputs.shape[1] != self.input_size:
            raise ValueError("Nombre de caractéristiques incompatible avec Dense")
        self._inputs = inputs.copy()
        self.dweights = self.dbias = None
        return inputs @ self.weights + self.bias

    def backward(self, gradient):
        if self._inputs is None:
            raise RuntimeError("forward() doit précéder backward()")
        gradient = matrix(gradient, "gradient")
        if gradient.shape != (self._inputs.shape[0], self.output_size):
            raise ValueError("Forme du gradient incompatible avec la sortie")
        # La moyenne du batch est déjà prise dans la dérivée de la loss.
        self.dweights = self._inputs.T @ gradient
        self.dbias = gradient.sum(axis=0, keepdims=True)
        return gradient @ self.weights.T

    def parameters_and_gradients(self):
        if self.dweights is None or self.dbias is None:
            raise RuntimeError("backward() doit précéder la mise à jour")
        return [(self.weights, self.dweights), (self.bias, self.dbias)]
