# Expérience : comparaison des optimiseurs

Hypothèse : accumuler les gradients ou adapter les pas peut accélérer la
convergence par rapport à SGD sur une régression non linéaire simple.

Lancer `python examples/optimizer_comparison.py`.

Configuration : 80 observations uniformes sur `[-1, 1]`, cible
`y = 0.5x² + 0.8x - 0.2`, réseau `Dense(1, 8) → Tanh → Dense(8, 1)`.
Les graines des deux couches sont 0 et 1 pour chaque optimiseur.
Budget : 1 000 epochs, batch complet, MSE, learning rate 0.01 commun.
Validation : 79 points intercalés, exclus des mises à jour.
Le seuil est une MSE d'entraînement inférieure à 0.005 ; son epoch est
mesurée avant la mise à jour. La hausse maximale mesure la plus grande
augmentation de loss entre deux epochs, limitée à zéro si elle décroît toujours.

Résultats obtenus :

| Optimiseur | MSE initiale | MSE finale | MSE validation | Epoch seuil | Hausse maximale |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGD | 0.298905 | 0.021328 | 0.020265 | Non atteint | 0.000000 |
| Momentum | 0.298905 | 0.000566 | 0.000490 | 629 | 0.001331 |
| RMSProp | 0.298905 | 0.000994 | 0.000994 | 102 | 0.000756 |
| Adam | 0.298905 | 0.000137 | 0.000125 | 156 | 0.000860 |

Dans cette configuration, RMSProp atteint le seuil en premier et Adam obtient
la plus faible erreur finale. SGD décroît sans hausse, mais reste plus lent.
Les autres méthodes présentent de petites oscillations. Les erreurs de
validation restent proches des erreurs d'entraînement.

Cette expérience illustre un comportement, sans classement universel : un
taux commun n'est pas nécessairement optimal pour chaque méthode et une seule
initialisation ne mesure pas la variabilité. La métrique retenue est la MSE,
car il s'agit d'une régression ; l'accuracy ne serait pas appropriée ici.
