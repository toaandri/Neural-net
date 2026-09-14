import numpy as np
import pytest

from neural_net import Perceptron


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


@pytest.mark.parametrize("targets", [[0, 0, 0, 1], [0, 1, 1, 1]])
def test_perceptron_apprend_portes_lineairement_separables(targets):
    model = Perceptron(learning_rate=0.1, epochs=100, random_state=0)
    model.fit(X, targets)
    assert np.array_equal(model.predict(X), targets)
    assert model.score(X, targets) == 1.0
    assert model.errors_[-1] == 0


def test_predict_avant_fit_leve_une_erreur():
    with pytest.raises(RuntimeError):
        Perceptron().predict([[0, 1]])


def test_rejette_des_labels_non_binaires():
    with pytest.raises(ValueError):
        Perceptron().fit(X, [0, 1, 2, 1])


def test_rejette_des_hyperparametres_invalides():
    with pytest.raises(ValueError):
        Perceptron(learning_rate=0)
    with pytest.raises(ValueError):
        Perceptron(epochs=0)
