# Phases 1 et 2 : formules et conventions

Tous les tableaux ont la forme `(batch, caractéristiques)`. Une couche Dense
contient `W` de forme `(entrée, sortie)` et `b` de forme `(1, sortie)`.
Les composants utilisent Python et NumPy uniquement.

## Propagation et règle de chaîne

Pour `Z = XW + b`, le gradient amont `G = ∂L/∂Z` donne :

- `∂L/∂W = XᵀG` ;
- `∂L/∂b = somme(G, axe=batch)` ;
- `∂L/∂X = GWᵀ`.

Chaque `forward` conserve les valeurs nécessaires à son `backward`. On calcule
tous les gradients en parcourant les couches dans l'ordre inverse, puis on met
à jour les paramètres. Les couches sont des instances distinctes ; leurs caches
correspondent au dernier forward. Ne pas intercaler une autre prédiction entre
le forward d'entraînement et son backward.

## Activations

| Fonction | Valeur | Dérivée locale |
| --- | --- | --- |
| Sigmoid | `s = 1 / (1 + exp(-x))` | `s(1-s)` |
| ReLU | `max(0, x)` | `1` si `x > 0`, sinon `0` |
| Tanh | `tanh(x)` | `1 - tanh(x)²` |
| Softmax | `sᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)` | `Jᵢⱼ = sᵢ(δᵢⱼ - sⱼ)` |

Pour Softmax, `backward(g)` calcule `s * (g - somme(g*s))` par ligne.
Sa dérivée ne se réduit pas à `s(1-s)` : les classes dépendent les unes des autres.
La dérivée de ReLU en zéro est choisie égale à zéro.

## Coûts

`N` est le nombre d'observations, `D` le nombre de sorties.

| Coût | Usage | Formule |
| --- | --- | --- |
| MeanSquaredError | Régression | `Σ(p-y)² / (ND)` |
| BinaryCrossEntropy | Classification binaire ou multilabel, après Sigmoid | `-Σ[y log(p) + (1-y) log(1-p)] / (ND)` |
| CategoricalCrossEntropy | Classes exclusives, après Softmax | `-Σ y log(p) / N` |

Les cibles ont la même forme que les prédictions. Pour la classification
catégorielle, utiliser des distributions de classes (notamment one-hot).
Les dérivées sont respectivement `2(p-y)/(ND)`,
`(p-y)/(p(1-p)ND)` et `-y/(pN)`, dans l'intérieur du domaine non clippé.
La moyenne est appliquée dans la loss, jamais une seconde fois dans Dense.

Les logarithmes sont protégés par clipping à `epsilon = 1e-12`. Le gradient
est nul dans les régions clippées ; une probabilité exactement saturée peut
donc arrêter l'apprentissage. Cette version sépare activations et losses ;
elle n'implémente pas encore de coût fusionné à partir des logits.

## Optimiseurs

Avec le gradient `g`, le taux `η` et des états initialisés à zéro :

- SGD : `θ ← θ - ηg`.
- Momentum : `v ← μv + g`, puis `θ ← θ - ηv`.
- RMSProp : `v ← ρv + (1-ρ)g²`, puis `θ ← θ - ηg/(√v + ε)`.
- Adam : `m ← β₁m + (1-β₁)g`, `v ← β₂v + (1-β₂)g²` ;
  `m̂ = m/(1-β₁ᵗ)`, `v̂ = v/(1-β₂ᵗ)`, puis `θ ← θ - ηm̂/(√v̂ + ε)`.

Les opérations sont élément par élément. Un optimiseur conserve un état par
paramètre et compte une itération par appel à `step`, pour toutes les couches.
Créer un nouvel optimiseur pour chaque expérience indépendante.

## Vérification numérique

Pour chaque élément d'un tableau, comparer le gradient analytique à
`[L(θ + h) - L(θ - h)] / (2h)`, avec `h = 1e-6`.
Les tests vérifient entrées, poids et biais d'une couche, chaque activation,
chaque coût et des chaînes de plusieurs couches, dont Softmax avec entropie
croisée. Les points non dérivables et les limites de clipping sont évités
dans les différences finies et testés séparément pour leur stabilité.
