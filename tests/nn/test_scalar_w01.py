"""Public tests for assignment 01: Neuron, Layer and MLP in nn/scalar.py."""

import math
import random

import pytest

from vrovaris_ai_lib.core.scalar import Value
from vrovaris_ai_lib.nn.scalar import MLP, Layer, Neuron

# The four examples from lecture C
XS = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
YS = [1.0, -1.0, -1.0, 1.0]


def set_params(params, values):
    assert len(params) == len(values)
    for p, v in zip(params, values):
        p.data = v


def test_neuron_parameter_count_and_types():
    n = Neuron(3)
    params = n.parameters()
    assert len(params) == 4
    assert all(isinstance(p, Value) for p in params)


def test_neuron_parameter_order_is_weights_then_bias():
    n = Neuron(2, nonlin=False)
    set_params(n.parameters(), [2.0, 3.0, 1.0])
    out = n([1.0, 10.0])
    assert out.data == pytest.approx(2.0 * 1.0 + 3.0 * 10.0 + 1.0)


def test_neuron_nonlin_applies_tanh():
    n = Neuron(2, nonlin=True)
    set_params(n.parameters(), [0.5, -0.25, 0.1])
    out = n([1.0, 2.0])
    assert out.data == pytest.approx(math.tanh(0.5 - 0.5 + 0.1))


def test_initial_weights_are_between_minus_one_and_one():
    random.seed(0)
    net = MLP(3, [4, 4, 1])
    assert all(-1.0 <= p.data <= 1.0 for p in net.parameters())


def test_same_seed_gives_same_weights():
    random.seed(123)
    first = [p.data for p in MLP(3, [4, 4, 1]).parameters()]
    random.seed(123)
    second = [p.data for p in MLP(3, [4, 4, 1]).parameters()]
    assert first == second


def test_layer_returns_list_or_single_value():
    many = Layer(3, 4)([1.0, 2.0, 3.0])
    assert isinstance(many, list)
    assert len(many) == 4
    one = Layer(3, 1)([1.0, 2.0, 3.0])
    assert isinstance(one, Value)


def test_layer_parameter_count():
    assert len(Layer(3, 4).parameters()) == 4 * (3 + 1)


def test_mlp_parameter_count():
    # lecture C: MLP(3, [4, 4, 1]) has 41 parameters
    assert len(MLP(3, [4, 4, 1]).parameters()) == 41
    assert len(MLP(4, [8, 1]).parameters()) == 49


# MLP(1, [2, 1]) has 7 parameters in this order:
# hidden neuron 1 (w, b), hidden neuron 2 (w, b), output neuron (w1, w2, b)
SMALL_NET_PARAMS = [2.0, 0.0, -1.0, 0.5, 3.0, 1.0, 0.5]


def test_mlp_last_layer_is_linear():
    net = MLP(1, [2, 1])
    set_params(net.parameters(), SMALL_NET_PARAMS)
    out = net([1.0])
    assert isinstance(out, Value)
    h1, h2 = math.tanh(2.0), math.tanh(-0.5)
    assert out.data == pytest.approx(3.0 * h1 + 1.0 * h2 + 0.5)


def test_mlp_gradients_by_hand():
    net = MLP(1, [2, 1])
    set_params(net.parameters(), SMALL_NET_PARAMS)
    out = net([1.0])
    out.backward()
    w_h1, b_h1, w_h2, b_h2, w_o1, w_o2, b_o = net.parameters()
    h1, h2 = math.tanh(2.0), math.tanh(-0.5)
    assert b_o.grad == pytest.approx(1.0)
    assert w_o1.grad == pytest.approx(h1)
    assert w_o2.grad == pytest.approx(h2)
    assert w_h1.grad == pytest.approx(3.0 * (1 - h1**2) * 1.0)
    assert b_h1.grad == pytest.approx(3.0 * (1 - h1**2))
    assert w_h2.grad == pytest.approx(1.0 * (1 - h2**2) * 1.0)
    assert b_h2.grad == pytest.approx(1.0 * (1 - h2**2))


def test_zero_grad():
    random.seed(1)
    net = MLP(3, [4, 4, 1])
    loss = sum((net(x) - y) ** 2 for x, y in zip(XS, YS))
    loss.backward()
    assert any(p.grad != 0 for p in net.parameters())
    net.zero_grad()
    assert all(p.grad == 0 for p in net.parameters())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_training_loss_goes_down(seed):
    random.seed(seed)
    net = MLP(3, [4, 4, 1])
    lr = 0.05
    losses = []
    for _ in range(200):
        loss = sum((net(x) - y) ** 2 for x, y in zip(XS, YS))
        net.zero_grad()
        loss.backward()
        for p in net.parameters():
            p.data -= lr * p.grad
        losses.append(loss.data)
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.05
