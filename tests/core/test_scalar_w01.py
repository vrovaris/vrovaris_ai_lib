"""Public tests for assignment 01: Value in core/scalar.py."""

import math

import pytest

from vrovaris_ai_lib.core.scalar import Value


def numerical_grad(f, x, h=1e-6):
    """Centered difference estimate of df/dx, as in lecture B, idea 6."""
    return (f(x + h) - f(x - h)) / (2 * h)


def test_lecture_a_example():
    # L = (a*b + c) * f with the numbers from lecture A
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    e = a * b
    d = e + c
    L = d * f
    L.backward()
    assert L.data == pytest.approx(-8.0)
    assert L.grad == pytest.approx(1.0)
    assert d.grad == pytest.approx(-2.0)
    assert f.grad == pytest.approx(4.0)
    assert c.grad == pytest.approx(-2.0)
    assert e.grad == pytest.approx(-2.0)
    assert a.grad == pytest.approx(6.0)
    assert b.grad == pytest.approx(-4.0)


def test_lecture_b_neuron():
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432)
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()
    assert o.data == pytest.approx(0.7071, abs=1e-4)
    assert x1.grad == pytest.approx(-1.5, abs=1e-4)
    assert w1.grad == pytest.approx(1.0, abs=1e-4)
    assert x2.grad == pytest.approx(0.5, abs=1e-4)
    assert w2.grad == pytest.approx(0.0, abs=1e-4)
    assert b.grad == pytest.approx(0.5, abs=1e-4)


def test_value_used_twice_in_add():
    a = Value(3.0)
    b = a + a
    b.backward()
    assert b.data == pytest.approx(6.0)
    assert a.grad == pytest.approx(2.0)


def test_value_used_twice_in_mul():
    a = Value(3.0)
    b = a * a
    b.backward()
    assert b.data == pytest.approx(9.0)
    assert a.grad == pytest.approx(6.0)


def test_value_feeds_two_nodes():
    a = Value(-2.0)
    b = Value(3.0)
    d = a * b
    e = a + b
    f = d * e
    f.backward()
    # f = a*b*(a+b) = a^2*b + a*b^2, so df/da = 2ab + b^2 and df/db = a^2 + 2ab
    assert f.data == pytest.approx(-6.0)
    assert a.grad == pytest.approx(2 * -2 * 3 + 3**2)
    assert b.grad == pytest.approx((-2) ** 2 + 2 * -2 * 3)


def test_plain_numbers_on_either_side():
    a = Value(4.0)
    assert (a + 1).data == pytest.approx(5.0)
    assert (1 + a).data == pytest.approx(5.0)
    assert (a * 2).data == pytest.approx(8.0)
    assert (2 * a).data == pytest.approx(8.0)
    assert (a - 1).data == pytest.approx(3.0)
    assert (10 - a).data == pytest.approx(6.0)
    assert (a / 2).data == pytest.approx(2.0)
    assert (2 / a).data == pytest.approx(0.5)
    assert (-a).data == pytest.approx(-4.0)


def test_rsub_gradient_sign():
    a = Value(2.0)
    out = 5 - a
    out.backward()
    assert out.data == pytest.approx(3.0)
    assert a.grad == pytest.approx(-1.0)


def test_division_gradients():
    a, b = Value(3.0), Value(-4.0)
    out = a / b
    out.backward()
    assert out.data == pytest.approx(-0.75)
    assert a.grad == pytest.approx(1 / -4.0)
    assert b.grad == pytest.approx(-3.0 / 16.0)


def test_pow_gradient():
    x = Value(2.0)
    y = x**3
    y.backward()
    assert y.data == pytest.approx(8.0)
    assert x.grad == pytest.approx(12.0)


def test_exp_gradient():
    x = Value(0.7)
    y = x.exp()
    y.backward()
    assert y.data == pytest.approx(math.exp(0.7))
    assert x.grad == pytest.approx(math.exp(0.7))


def test_tanh_gradient_matches_nudge():
    x0 = 0.3
    x = Value(x0)
    y = x.tanh()
    y.backward()
    assert y.data == pytest.approx(math.tanh(x0))
    assert x.grad == pytest.approx(numerical_grad(math.tanh, x0), rel=1e-6)


def test_relu():
    for x0, want_data, want_grad in [(2.5, 2.5, 1.0), (-1.5, 0.0, 0.0), (0.0, 0.0, 0.0)]:
        x = Value(x0)
        y = x.relu()
        y.backward()
        assert y.data == pytest.approx(want_data)
        assert x.grad == pytest.approx(want_grad)


def test_tanh_from_exp_gives_same_gradient():
    # lecture B, 1:36:32: tanh rebuilt from exp, plus, minus and division
    n1 = Value(0.8814)
    o1 = n1.tanh()
    o1.backward()

    n2 = Value(0.8814)
    e = (2 * n2).exp()
    o2 = (e - 1) / (e + 1)
    o2.backward()

    assert o2.data == pytest.approx(o1.data)
    assert n2.grad == pytest.approx(n1.grad)


def test_grad_starts_at_zero_and_output_grad_is_one():
    a, b = Value(1.5), Value(2.0)
    assert a.grad == 0
    out = a * b
    out.backward()
    assert out.grad == pytest.approx(1.0)
