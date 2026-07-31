# mojo-scikit-multilearn

A focused Mojo port of the compute-heavy parts of
[scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn).
It keeps the upstream `skmultilearn` import paths and estimator signatures,
uses scikit-learn estimators where problem transformation calls for one, and
moves dense nearest-neighbor search, neighbor-label aggregation, and label-set
decoding into a compiled Mojo shared library.

This is a covered-subset port, not a reimplementation of every module in the
upstream project.

## Coverage

| Upstream module | Covered API | Execution |
| --- | --- | --- |
| `skmultilearn.problem_transform` | `BinaryRelevance` | scikit-learn base classifiers; upstream-compatible sparse assembly |
| `skmultilearn.problem_transform` | `ClassifierChain` | scikit-learn base classifiers; upstream-compatible chain conditioning |
| `skmultilearn.problem_transform` | `LabelPowerset` | compatible combination mapping; Mojo inverse transform |
| `skmultilearn.adapt` | `BRkNNaClassifier` | Mojo dense kNN and label votes |
| `skmultilearn.adapt` | `BRkNNbClassifier` | Mojo dense kNN and label votes |
| `skmultilearn.adapt` | `MLkNN` | Mojo dense kNN and label counts; Bayesian fit and prediction |
| `skmultilearn.base` | `MLClassifierBase`, `ProblemTransformationBase` | Python compatibility layer |

Dense feature matrices use the Mojo path. Sparse feature matrices are accepted
by the kNN estimators and deliberately fall back to scikit-learn's sparse
nearest-neighbor implementation. Dense and sparse label matrices are accepted.

Not covered are `MLARAM`, `MLTSVM`, embedding methods, ensembles, label-space
clusterers, dataset download helpers, and the MEKA interface. Those either have
substantially different kernels or are integration and I/O code with little to
gain from a Mojo rewrite.

## Install

The repository is self-contained under Pixi:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles the single Mojo compilation unit to
`dist/libmojo-scikit-multilearn.so`. Imports also rebuild a missing or stale
library. Set `MOJO_SKMULTILEARN_LIB` to load an already-built library from a
different path.

## Usage

This example runs as written after `pixi run build`:

```python
import numpy as np
from skmultilearn.adapt import MLkNN

X = np.array([
    [0.0, 0.1],
    [0.2, 0.0],
    [0.9, 1.0],
    [1.0, 0.8],
])
y = np.array([
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
])

classifier = MLkNN(k=2, s=1.0).fit(X, y)
prediction = classifier.predict(np.array([[0.1, 0.1], [0.9, 0.9]]))
print(prediction.toarray())
```

Problem transformations accept ordinary scikit-learn estimators:

```python
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import ClassifierChain

classifier = ClassifierChain(
    classifier=LogisticRegression(max_iter=1000)
).fit(X, y)
prediction = classifier.predict(X)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(72 logical CPUs), Linux 6.8.0-136-generic. The environment used Mojo
1.0.0b3.dev2026072406, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0, and
scikit-multilearn 0.2.0. Each value is the best of three runs after warm-up.
The Pixi task holds `/tmp/mojo-bench.lock` for the complete run.

| case | Mojo port | upstream 0.2.0 | speedup |
| --- | ---: | ---: | ---: |
| BRkNNa.predict (8k train, 2k query, 24d, 12 labels) | 44.54 ms | 1031.29 ms | 23.15x |
| BRkNNb.predict (8k train, 2k query, 24d, 12 labels) | 173.00 ms | 1712.84 ms | 9.90x |
| MLkNN.fit (3k x 24, 12 labels, k=10) | 51.00 ms | 2945.44 ms | 57.76x |
| MLkNN.predict (8k train, 2k query, 24d, 12 labels) | 83.51 ms | 1766.57 ms | 21.15x |
| LabelPowerset.transform (100k x 40, 10% density) | 143.52 ms | 205.18 ms | 1.43x |
| LabelPowerset.inverse_transform (100k x 40) | 368.50 ms | 9799.10 ms | 26.59x |

`LabelPowerset.transform` is still ordered dictionary construction in Python
because first-seen class IDs and arbitrary label counts are part of the
upstream contract. It uses compact byte keys for up to 256 labels and tuple
keys above that, while constructing the public string mapping only for
first-seen combinations. Its measured difference comes from that Python key
representation, not a Mojo kernel. The inverse operation is a regular integer
gather.

No GPU path is provided. These kernels have low arithmetic intensity, so this
port keeps execution on the CPU and avoids device transfer and launch costs.

Upstream 0.2.0 passes `n_neighbors` positionally, which scikit-learn 1.9 no
longer accepts. Tests and benchmarks apply a constructor-only compatibility
shim to the installed upstream class before comparing algorithms. Classifier
Chain parity similarly requests CSC output from SciPy's `hstack`, avoiding a
new COO slicing failure in the old upstream code.

## How it works

The Python package allocates all inputs, outputs, and scratch arrays. Its
`ctypes` layer passes their addresses and dimensions to the C-ABI exports in
`src/capi.mojo`. Mojo reconstructs mutable pointers inside the exported
functions; it does not allocate or retain Python-owned memory.

Feature matrices cross the boundary as C-contiguous row-major `float64`.
Neighbor indices, dense labels, counts, class assignments, and decoded output
use row-major `int64`. Non-contiguous or differently typed dense inputs are
converted once at the boundary. Sparse kNN inputs stay sparse and use the
scikit-learn fallback instead of being expanded unexpectedly.

The nearest-neighbor kernel scans each training row, computes squared
Euclidean distance with native-width SIMD and a scalar remainder, and
maintains a sorted size-`k` candidate list. Large query batches are split into
a bounded set of independent row chunks above a workload threshold; each
worker calls Mojo with zero-copy NumPy slices, while smaller batches stay
serial. Self
queries exclude their matching training row inside the kernel, avoiding
temporary `k+1` outputs and Python row filtering. A second native-width SIMD
kernel accumulates every selected neighbor's labels in one pass. ML-kNN builds
its conditional tables from those integer counts and evaluates the normalized
Bayesian posterior in vectorized NumPy.

`MLkNN.predict_proba` returns that normalized posterior. This is the documented
algorithm; current SciPy sparse scalar arithmetic causes upstream 0.2.0's old
implementation to expose its unnormalized numerator. Parity tests compare the
upstream priors, conditional tables, and decisions, then check probabilities
directly against the published posterior formula.

## License

MIT
