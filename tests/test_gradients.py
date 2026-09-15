import numpy as np
import pytest

from neural_net import (Dense, Sigmoid, ReLU, Tanh, Softmax, MeanSquaredError,
                        BinaryCrossEntropy, CategoricalCrossEntropy, forward, backward)


def numerical_gradient(function, values, epsilon=1e-6):
    result = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + epsilon
        upper = function()
        values[index] = original - epsilon
        lower = function()
        values[index] = original
        result[index] = (upper - lower) / (2 * epsilon)
    return result


def test_dense_forward_and_gradients():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 2))
    gradient = rng.normal(size=(3, 4))
    layer = Dense(2, 4, random_state=8)
    np.testing.assert_allclose(layer.forward(x), x @ layer.weights + layer.bias)
    dx = layer.backward(gradient)
    dw, db = layer.dweights.copy(), layer.dbias.copy()
    objective = lambda: np.sum(layer.forward(x) * gradient)
    for values, analytical in [(x, dx), (layer.weights, dw), (layer.bias, db)]:
        np.testing.assert_allclose(numerical_gradient(objective, values), analytical, atol=1e-8)


@pytest.mark.parametrize("activation", [Sigmoid, ReLU, Tanh, Softmax])
def test_activation_gradients(activation):
    x = np.array([[-1.2, 0.3, 2.1], [0.8, -0.6, 1.4]])
    gradient = np.array([[0.2, -0.7, 0.9], [1.1, 0.4, -0.3]])
    layer = activation()
    layer.forward(x)
    analytical = layer.backward(gradient)
    numerical = numerical_gradient(lambda: np.sum(layer.forward(x) * gradient), x)
    np.testing.assert_allclose(analytical, numerical, atol=1e-8)


@pytest.mark.parametrize("loss,p,y", [
    (MeanSquaredError(), [[0.3, -0.5], [0.7, 1.2]], [[1, 0], [0, 1]]),
    (BinaryCrossEntropy(), [[0.3, 0.5], [0.7, 0.9]], [[1, 0], [0, 1]]),
    (CategoricalCrossEntropy(), [[0.3, 0.7], [0.8, 0.2]], [[1, 0], [0, 1]]),
])
def test_loss_gradients(loss, p, y):
    p, y = np.array(p, dtype=float), np.array(y, dtype=float)
    analytical = loss.backward(p, y)
    numerical = numerical_gradient(lambda: loss.forward(p, y), p)
    np.testing.assert_allclose(analytical, numerical, atol=1e-8)


@pytest.mark.parametrize("activation,loss,targets", [
    (Tanh, MeanSquaredError, [[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]),
    (Sigmoid, BinaryCrossEntropy, [[0, 1], [1, 0], [1, 1]]),
    (Softmax, CategoricalCrossEntropy, [[0, 1], [1, 0], [0, 1]]),
])
def test_chain_rule_through_multiple_layers(activation, loss, targets):
    layers = [Dense(2, 3, 1), Tanh(), Dense(3, 2, 2), activation()]
    x = np.array([[0.2, -0.3], [0.6, 0.4], [-0.2, 0.7]])
    y = np.array(targets, dtype=float)
    cost = loss()
    output = forward(layers, x)
    dx = backward(layers, cost.backward(output, y))
    pairs = [(x, dx)] + [(p, g.copy()) for layer in layers if isinstance(layer, Dense)
                            for p, g in layer.parameters_and_gradients()]
    for values, analytical in pairs:
        numerical = numerical_gradient(lambda: cost.forward(forward(layers, x), y), values)
        np.testing.assert_allclose(analytical, numerical, atol=1e-8)


def test_extreme_inputs_and_clipping():
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        np.testing.assert_allclose(Sigmoid().forward([[-1000, 0, 1000]]), [[0, 0.5, 1]])
        output = Softmax().forward([[10000, 10001, 9999], [-10000, -10000, -10000]])
        np.testing.assert_allclose(output.sum(axis=1), 1)
        for loss in [BinaryCrossEntropy(), CategoricalCrossEntropy()]:
            assert np.isfinite(loss.forward([[0, 1]], [[1, 0]]))
            assert np.all(np.isfinite(loss.backward([[0, 1]], [[1, 0]])))
    np.testing.assert_array_equal(ReLU().forward([[-1, 0, 2]]), [[0, 0, 2]])


def test_invalid_shapes_and_call_order():
    for layer in [Dense(2, 1), Sigmoid(), ReLU(), Tanh(), Softmax()]:
        with pytest.raises(RuntimeError):
            layer.backward([[1]])
        with pytest.raises(ValueError):
            layer.forward([1, 2])
        layer.forward([[1, 2]])
        with pytest.raises(ValueError):
            layer.backward([[1, 2, 3]])
    for size in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            Dense(size, 2)
    with pytest.raises(ValueError):
        Dense(2, 1).forward([[1, float("nan")]])
    with pytest.raises(ValueError):
        BinaryCrossEntropy().forward([[1.2]], [[1]])
    with pytest.raises(ValueError):
        CategoricalCrossEntropy().forward([[0.2, 0.2]], [[1, 0]])
    with pytest.raises(ValueError):
        MeanSquaredError().forward([[1, 2]], [[1]])
