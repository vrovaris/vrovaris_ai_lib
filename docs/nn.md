# nn

The `nn` sub-package builds neural networks out of `Value` (see [core.md](core.md)). So far it has one module.

| Module | What it has |
|---|---|
| `nn/scalar.py` | `Neuron`, `Layer` and `MLP`, built from Values |

```python
from vrovaris_ai_lib.nn.scalar import Neuron, Layer, MLP
```

Each class can be called on an input, and each has `parameters()`, which returns its weights and biases as one flat list of Values.

## Neuron(n_in, nonlin=True)

One neuron with `n_in` weights and a bias. Each starts as a Value drawn from `random.uniform(-1, 1)`, so `random.seed()` makes a network reproducible.

- `n(x)` takes a list of `n_in` numbers or Values and returns one Value, `tanh(w1*x1 + ... + wn*xn + b)`.
- With `nonlin=False` it skips tanh and returns the weighted sum itself.
- `parameters()` returns `[w1, ..., wn, b]`, the weights in input order and then the bias.

The neuron does not check the length of `x`. It pairs inputs with weights using `zip`, which stops at the shorter list, so a wrong length gives a wrong answer with no error.

## Layer(n_in, n_out, nonlin=True)

`n_out` neurons that each take the same `n_in` inputs. `nonlin` goes to every neuron.

- `layer(x)` returns a list of `n_out` Values, or a single Value when `n_out` is 1.
- `parameters()` returns every neuron's parameters in one list, neuron by neuron.

## MLP(n_in, n_outs)

A multi-layer perceptron, which is a stack of layers where each layer's output is the next layer's input. `n_outs` lists the size of each layer. `MLP(3, [4, 4, 1])` takes 3 inputs, has two layers of 4 neurons, and ends in 1 output.

- Every layer uses tanh except the last, which is built with `nonlin=False`. The last layer is linear, so the output is the raw weighted sum. Later loss functions, such as week 4's cross-entropy, expect raw outputs like this.
- `mlp(x)` takes one example, a list of `n_in` numbers, and returns what the last layer returns.
- `parameters()` returns every layer's parameters in one list, first layer first. `MLP(3, [4, 4, 1])` has 41.
- `zero_grad()` sets `grad` to 0 on every parameter.

## Training

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
mlp = MLP(3, [4, 4, 1])
lr = 0.05

for step in range(200):
    mlp.zero_grad()
    preds = [mlp(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))
    loss.backward()
    for p in mlp.parameters():
        p.data -= lr * p.grad
```

Each step runs four things in order:

1. `zero_grad()` clears the gradients from the last step. `backward()` adds with `+=`, so without this each step's gradients would include every earlier step's.
2. The forward pass runs each example through the network and adds up the squared errors into one loss Value. Any loss built from Value operations works.
3. `loss.backward()` fills `grad` on every parameter. It does not change any weights.
4. The update moves each parameter a small step against its gradient, which lowers the loss.
