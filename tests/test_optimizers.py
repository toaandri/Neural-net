import numpy as np
import pytest

from neural_net import Dense, SGD, Momentum, RMSProp, Adam, MeanSquaredError, train, forward


@pytest.mark.parametrize("kind", [SGD, Momentum, RMSProp, Adam])
def test_optimizer_matches_two_manual_updates(kind):
    layer = Dense(2, 1, 0)
    layer.weights[:] = 1
    layer.bias[:] = 2
    optimizer = kind(learning_rate=0.1)
    expected_w, expected_b = layer.weights.copy(), layer.bias.copy()
    moments = {"w": [np.zeros_like(expected_w), np.zeros_like(expected_w)],
               "b": [np.zeros_like(expected_b), np.zeros_like(expected_b)]}
    for t, (gw, gb) in enumerate([(np.array([[2.], [-3.]]), np.array([[0.5]])),
                                 (np.array([[-1.], [4.]]), np.array([[-2.]]))], 1):
        layer.dweights, layer.dbias = gw, gb
        for name, expected, g in [("w", expected_w, gw), ("b", expected_b, gb)]:
            m, v = moments[name]
            if kind is SGD:
                direction = g
            elif kind is Momentum:
                m[:] = 0.9 * m + g
                direction = m
            elif kind is RMSProp:
                v[:] = 0.9 * v + 0.1 * g ** 2
                direction = g / (np.sqrt(v) + 1e-8)
            else:
                m[:] = 0.9 * m + 0.1 * g
                v[:] = 0.999 * v + 0.001 * g ** 2
                direction = (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            expected -= 0.1 * direction
        optimizer.step([layer])
        np.testing.assert_allclose(layer.weights, expected_w)
        np.testing.assert_allclose(layer.bias, expected_b)
    assert optimizer.iterations == 2


@pytest.mark.parametrize("kind", [SGD, Momentum, RMSProp, Adam])
def test_training_learns_regression(kind):
    x = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = 2 * x + 0.5
    layers = [Dense(1, 1, 3)]
    loss = MeanSquaredError()
    history = train(layers, x, y, loss, kind(learning_rate=0.03), epochs=400)
    final = loss.forward(forward(layers, x), y)
    assert final < history[0] * 0.01
    assert final < 0.002


@pytest.mark.parametrize("kind", [SGD, Momentum, RMSProp, Adam])
@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_learning_rate(kind, rate):
    with pytest.raises(ValueError):
        kind(learning_rate=rate)


def test_optimizer_validates_before_update():
    a, b = Dense(1, 1, 0), Dense(1, 1, 1)
    for layer in [a, b]:
        layer.forward([[1]])
        layer.backward([[1]])
    original = a.weights.copy()
    b.dweights[:] = np.nan
    optimizer = Adam()
    with pytest.raises(ValueError):
        optimizer.step([a, b])
    np.testing.assert_array_equal(a.weights, original)
    assert optimizer.iterations == 0
    with pytest.raises(RuntimeError):
        optimizer.step([Dense(1, 1)])
    with pytest.raises(ValueError):
        train([a], [[1]], [[1]], MeanSquaredError(), SGD(), epochs=1.5)


@pytest.mark.parametrize("factory", [lambda: Momentum(momentum=1), lambda: RMSProp(decay=-0.1),
                                     lambda: Adam(beta1=float("nan")), lambda: Adam(beta2=1),
                                     lambda: RMSProp(epsilon=0), lambda: Adam(epsilon=float("inf"))])
def test_invalid_optimizer_configuration(factory):
    with pytest.raises(ValueError):
        factory()
