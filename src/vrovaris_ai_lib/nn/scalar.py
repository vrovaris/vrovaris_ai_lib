"""Neuron, Layer and MLP built on Value."""

import random
from typing import Any

from ..core.scalar import Value


class Neuron:
    """One neuron: tanh(w1*x1 + ... + wn*xn + b), or the plain sum if nonlin is False.

    The n_in weights and the bias start as Values drawn from
    random.uniform(-1, 1), using Python's `random` module, so
    random.seed() makes a net reproducible.
    """

    def __init__(self, n_in: int, nonlin: bool = True) -> None:
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.bias = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x: list[Any]) -> Value:
        """Return the neuron's output for a list of n_in numbers or Values."""

        activation = sum(
            (weight * input for weight, input in zip(self.weights, x)), self.bias
        )

        out = activation.tanh() if self.nonlin else activation

        return out

    def parameters(self) -> list[Value]:
        """Return [w1, ..., wn, b]: the weights in input order, then the bias."""

        params = [w for w in self.weights] + [self.bias]

        return params


class Layer:
    """n_out neurons that each take the same n_in inputs. `nonlin` goes to every neuron."""

    def __init__(self, n_in: int, n_out: int, nonlin: bool = True) -> None:
        self.neurons = [Neuron(n_in, nonlin) for _ in range(n_out)]

    def __call__(self, x: list[Any]) -> list[Value] | Value:
        """Return a list of n_out outputs, or a single Value when n_out is 1."""
        out = [n(x) for n in self.neurons]

        return out if len(out) > 1 else out[0]

    def parameters(self) -> list[Value]:
        """Return every neuron's parameters in one flat list, neuron by neuron."""

        params = []

        for n in self.neurons:
            params.extend(n.parameters())

        return params


class MLP:
    """A stack of layers. MLP(3, [4, 4, 1]) has 3 inputs, two layers of 4 and 1 output.

    Every layer uses tanh except the last, which is built with nonlin=False,
    so the output is the raw weighted sum.
    """

    def __init__(self, n_in: int, n_outs: list[int]) -> None:

        # Build a list with n_in + n_outs
        sizes = [n_in] + n_outs
        length = len(sizes)

        # Build each layer as n_in = sizes[i], n_out = sizes[i + 1] (output of previous layer is input of the next one)
        # Up to the penultimate layer
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=True) for i in range(length - 2)
        ]

        # Last layer should have nonlin = False
        last_layer = Layer(sizes[length - 2], sizes[length - 1], nonlin=False)
        self.layers.append(last_layer)

    def __call__(self, x: Any) -> list[Value] | Value:
        """Run x through every layer and return what the last layer returns."""

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self) -> list[Value]:
        """Return every layer's parameters in one flat list, first layer first."""
        params = []

        for l in self.layers:
            params.extend(l.parameters())
        return params

    def zero_grad(self) -> None:
        """Set `grad` to 0 on every parameter."""

        params = self.parameters()
        for param in params:
            param.grad = 0.0
