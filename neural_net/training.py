"""Boucle explicite de phase 2, avant l'API Sequential de phase 3."""

import numpy as np

from ._validation import matrix, positive_integer


def forward(layers, inputs):
    output = inputs
    for layer in layers:
        output = layer.forward(output)
    return output


def backward(layers, gradient):
    for layer in reversed(layers):
        gradient = layer.backward(gradient)
    return gradient


def train(layers, inputs, targets, loss, optimizer, epochs=100):
    """Descente sur le batch complet ; historique mesuré avant chaque update."""
    epochs = positive_integer(epochs, "epochs")
    inputs, targets = matrix(inputs), matrix(targets, "targets")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("Les entrées et les cibles doivent avoir le même nombre de lignes")
    layers = list(layers)
    if not layers or len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("Fournir des instances de couches distinctes")
    history = []
    for _ in range(epochs):
        predictions = forward(layers, inputs)
        history.append(loss.forward(predictions, targets))
        backward(layers, loss.backward(predictions, targets))
        optimizer.step(layers)
    return history
