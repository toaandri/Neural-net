"""Expérience reproductible de phase 2 : même réseau, données et budget."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_net import Dense, Tanh, MeanSquaredError, SGD, Momentum, RMSProp, Adam, train, forward


def main():
    x = np.linspace(-1, 1, 80).reshape(-1, 1)
    y = 0.5 * x ** 2 + 0.8 * x - 0.2
    # Validation sur des points intercalés, jamais utilisés pour les updates.
    x_validation = (x[:-1] + x[1:]) / 2
    y_validation = 0.5 * x_validation ** 2 + 0.8 * x_validation - 0.2
    print("Optimiseur  MSE initiale  MSE finale  MSE validation  Epoch seuil  Max hausse")
    for kind in [SGD, Momentum, RMSProp, Adam]:
        layers = [Dense(1, 8, 0), Tanh(), Dense(8, 1, 1)]
        loss = MeanSquaredError()
        history = train(layers, x, y, loss, kind(learning_rate=0.01), epochs=1000)
        final = loss.forward(forward(layers, x), y)
        validation = loss.forward(forward(layers, x_validation), y_validation)
        first = next((i + 1 for i, value in enumerate(history) if value < 0.005), None)
        increase = max(0, float(np.max(np.diff(history))))
        print(f"{kind.__name__:10}  {history[0]:.6f}     {final:.6f}    {validation:.6f}        {str(first):>4}         {increase:.6f}")


if __name__ == "__main__":
    main()
