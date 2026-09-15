# Neural Network From Scratch

Projet d'apprentissage : construire progressivement un moteur de réseau neuronal
avec Python et NumPy, sans PyTorch ni TensorFlow.

Les **phases 1 et 2 sont implémentées** : perceptron, couches denses,
activations, fonctions de coût, rétropropagation et optimiseurs.

Installation :

```bash
python -m pip install -r requirements.txt
```

## Partie 1 — V1 : perceptron

Le perceptron calcule `z = w · x + b`, puis applique un seuil : la sortie vaut 1
si `z >= 0`, sinon 0. En cas d'erreur, ses paramètres sont corrigés par la règle
du perceptron. Cette première version apprend les portes logiques AND et OR.

```python
from neural_net import Perceptron

model = Perceptron(learning_rate=0.1, epochs=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.score(X_test, y_test))
```

Lancer la démonstration :

```bash
python examples/perceptron_demo.py
```

Lancer les tests :

```bash
python -m pytest -q
```

Un perceptron simple ne peut apprendre que des problèmes linéairement séparables ;
XOR sera donc traité dans une étape ultérieure avec un réseau multicouche.

## Phase 2 — Entraînement par rétropropagation

```python
import numpy as np
from neural_net import Dense, Tanh, MeanSquaredError, Adam, train, forward

X = np.linspace(-1, 1, 80).reshape(-1, 1)
y = 0.5 * X**2 + 0.8 * X - 0.2
layers = [Dense(1, 8, random_state=0), Tanh(), Dense(8, 1, random_state=1)]
loss = MeanSquaredError()
history = train(layers, X, y, loss, Adam(learning_rate=0.01), epochs=1000)
predictions = forward(layers, X)
print(loss.forward(predictions, y))
```

`train` enchaîne forward, calcul du coût, backward en ordre inverse et mise
à jour des paramètres. L'historique contient le coût avant chaque update.
L'entraînement utilise le batch complet.

Composants disponibles :

- `Dense(input_size, output_size)` avec initialisation Xavier, biais et gradients.
- `Sigmoid`, `ReLU`, `Tanh`, `Softmax` avec `forward` et `backward`.
- `MeanSquaredError`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`.
- `SGD`, `Momentum`, `RMSProp`, `Adam`.

Les [formules et conventions](docs/mathematics.md) expliquent la règle de
chaîne, les dimensions, la normalisation et les limites numériques.
Les modules sont séparés par rôle dans `neural_net/` ; `tests/` vérifie les
gradients et l'apprentissage, et `examples/` fournit les démonstrations.

Comparer les optimiseurs :

```bash
python examples/optimizer_comparison.py
```

L'[expérience documentée](docs/optimizer_experiment.md) compare convergence,
stabilité et erreur finale à initialisation identique. Sur cette expérience,
Adam passe d'une MSE de 0.298905 à 0.000137 en 1 000 epochs.

## Roadmap

- Phase 1 terminée : V1 perceptron, V2 Dense, V3 activations, V4 coûts.
- Phase 2 terminée : V5 rétropropagation, V6 SGD / Momentum / RMSProp / Adam.
- Phase 3 à venir : API Sequential, XOR, MNIST, sauvegarde et chargement.
- Phase 4 à venir : régularisation et CNN.
- Phase 5 à venir : performances, benchmarks et comparaison avec PyTorch.

Ce moteur reste un outil d'apprentissage. Il ne propose pas encore d'API
`compile` / `fit` pour les réseaux multicouches, de mini-batchs ou de
sérialisation. La comparaison avec PyTorch est réservée à la dernière phase.
