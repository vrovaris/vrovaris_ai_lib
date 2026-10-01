# vrovaris_ai_lib

A machine learning library written from scratch in Python and numpy. It starts with scalar autograd and a tensor engine, then adds neural network modules, classical machine learning, data mining, a BPE tokenizer and a small GPT.

The library is under construction. Most modules are empty files until the week that builds them.

## Layout

| Sub-package | Contents |
|---|---|
| `core` | Scalar autograd, `Tensor`, gradient checking |
| `nn` | Modules, layers, losses, optimizers, data loading, checkpoints |
| `ml` | PCA, k-means, decision trees |
| `mine` | Frequent itemsets, MinHash, LSH |
| `text` | BPE tokenizers, corpus tools |
| `gpt` | The GPT model and its training loop |
| `analyze` | Embedding and attention analysis |

## Install

Python 3.12 or newer is required.

```
pip install -e ".[dev]"
```

## Run the tests

```
python -m pytest
```

## License

MIT. See `LICENSE`.
