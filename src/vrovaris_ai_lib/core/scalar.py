"""Scalar autograd: a Value holds one number and records how it was computed."""

from __future__ import annotations

import math


class Value:
    """One number in a computation graph.

    `data` is the number. `grad` is the derivative of the output you called
    `backward()` on with respect to this value. It starts at 0.

    `_prev` holds the values this one was computed from, and `_op` names
    the operation, such as "+". Leaves, such as inputs and weights, use the
    defaults.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op

        self.grad = 0.0
        self._backward = lambda: None
        self.label = label

    # Primitive operations (need a backward definition)
    def __add__(self, other: Value | int | float) -> Value:
        """Return self + other. `other` may be a Value, an int or a float."""

        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | int | float) -> Value:
        """Return self * other. `other` may be a Value, an int or a float."""

        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: int | float) -> Value:
        """Return self ** other.

        `other` must be an int or a float. Raise TypeError for anything else,
        including a Value.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("Must be an int or float for power operation")

        out = Value(self.data**other, (self, Value(other)), "**")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def exp(self) -> Value:
        """Return e ** self."""

        val = math.exp(self.data)
        out = Value(val, (self,), "exp")

        def _backward():
            self.grad += out.grad * val

        out._backward = _backward

        return out

    # Backpropagation
    def backward(self) -> None:
        """Fill `grad` on every value in the graph that ends at self.

        Sets self.grad to 1, then runs each node's backward step in reverse
        topological order. Gradients are added with +=, so they accumulate
        until something resets them to 0.
        """

        self.grad = 1.0
        order = self._reverse_topo_sort()

        for node in order:
            node._backward()

    # Reverse topological order for backpropagation
    def _reverse_topo_sort(self) -> list[Value]:

        stack = [(self, False)]  # (node, children_done) pair
        visited = set()
        order = []

        while stack != []:
            curr, children_done = stack.pop()  #  (node, children_done)

            if children_done:
                order.append(curr)

            elif curr not in visited:
                visited.add(curr)
                stack.append((curr, True))

                for child in curr._prev:
                    stack.append((child, False))

        return order[::-1]

    # Activation functions
    def tanh(self) -> Value:
        """Return tanh(self). Must not overflow for inputs such as 500 or -500."""
        val = math.tanh(self.data)
        out = Value(val, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1.0 - val**2)

        out._backward = _backward

        return out

    def relu(self) -> Value:
        """Return max(0, self). The local derivative at exactly 0 is 0."""
        val = max(0, self.data)

        out = Value(val, (self,), "relu")

        def _backward():
            self.grad += out.grad * (1.0 if val > 0 else 0)

        out._backward = _backward

        return out

    # Derived operations to facilitate usage
    def __neg__(self) -> Value:
        """Return -self."""
        return self * -1

    def __sub__(self, other: Value | int | float) -> Value:
        """Return self - other."""
        return self + (-other)

    def __truediv__(self, other: Value | int | float) -> Value:
        """Return self / other."""
        return self * (other**-1)

    # Reverse operations
    def __radd__(self, other: Value | int | float) -> Value:
        """Return other + self, for expressions like `1 + v`."""
        return self + other

    def __rmul__(self, other: Value | int | float) -> Value:
        """Return other * self, for expressions like `2 * v`."""
        return self * other

    def __rsub__(self, other: Value | int | float) -> Value:
        """Return other - self, for expressions like `5 - v`."""
        return other + (-self)

    def __rtruediv__(self, other: Value | int | float) -> Value:
        """Return other / self, for expressions like `1 / v`."""
        return other * (self**-1)

    # Print representation
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
