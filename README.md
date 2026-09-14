# Neural Network From Scratch

Projet d'apprentissage : construire progressivement un moteur de réseau neuronal
avec Python et NumPy, sans PyTorch ni TensorFlow.

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
pytest -q
```

Un perceptron simple ne peut apprendre que des problèmes linéairement séparables ;
XOR sera donc traité dans une étape ultérieure avec un réseau multicouche.
