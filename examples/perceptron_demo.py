"""Démonstration de la V1 : apprentissage des portes logiques AND et OR."""

import sys
from pathlib import Path

import numpy as np

# Permet l'exécution directe depuis la racine : python examples/perceptron_demo.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_net import Perceptron


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
datasets = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
}

for name, y in datasets.items():
    model = Perceptron(learning_rate=0.1, epochs=20, random_state=42)
    model.fit(X, y)
    print(f"{name}: prédictions={model.predict(X).tolist()}, accuracy={model.score(X, y):.0%}")
    print(f"  erreurs par époque : {model.errors_}")
