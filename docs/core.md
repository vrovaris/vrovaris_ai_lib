# core

The `core` sub-package holds the autograd engine. So far it has one module.

| Module | What it has |
|---|---|
| `core/scalar.py` | `Value`, a single number that tracks its gradient |

```python
from vrovaris_ai_lib.core.scalar import Value
```

## Value

A `Value` wraps one number, an int or a float. Every operation on Values returns a new Value that remembers the Values it came from and how to send gradients back to them. Calling `backward()` on the final Value fills in the gradient of every Value it was built from.

Public attributes:

- `data` is the number.
- `grad` is the derivative of the output you called `backward()` on with respect to this Value. It starts at 0.
- `label` is an optional name, used when drawing the graph.

### Example

```python
a, b = Value(3.0), Value(2.0)
c = a * b + a
c.backward()
```

After this, `a.grad` is 3.0 and `b.grad` is 3.0. `b.grad` is the partial derivative of `c` with respect to `b`, which is `a.data`. `a.grad` is `b.data + 1`, because `a` reaches `c` through two paths and `backward()` adds the gradient from each path.

### Operations

| Operation | Example | Notes |
|---|---|---|
| add, subtract, multiply, divide | `a + b`, `2 - a`, `a / 3`, `1 / a` | Either side can be a plain int or float |
| power | `a ** 2`, `a ** -1` | The exponent must be an int or float. A Value exponent raises `TypeError`, and so does `2 ** a` |
| negate | `-a` | |
| exp | `a.exp()` | Returns e to the power of `a` |
| tanh | `a.tanh()` | Uses `math.tanh`, so inputs like 500 and -500 do not overflow |
| relu | `a.relu()` | Returns `max(0, a)`. The derivative at exactly 0 is 0 |

### backward()

`backward()` sets the output's `grad` to 1. Then it visits every Value the output was built from, in reverse topological order, so each Value's gradient is complete before it passes gradient on to its own inputs. The order comes from an iterative depth-first search, so it works on graphs deeper than Python's recursion limit.

Every gradient is added with `+=`, and `backward()` never resets anything. Gradients pile up across calls. Calling `backward()` twice on `y = x * x` with `x = Value(2.0)` leaves `x.grad` at 8.0, not 4.0.

### Resetting gradients

Set `grad` back to 0 before each new backward pass. For a network, `MLP.zero_grad()` does this for every parameter (see [nn.md](nn.md)).
