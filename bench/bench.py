"""Benchmarks against the installed scikit-multilearn 0.2.0."""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PYTHON = os.path.join(ROOT, "python")
if LOCAL_PYTHON not in sys.path:
    sys.path.insert(0, LOCAL_PYTHON)

from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN
from skmultilearn.problem_transform import LabelPowerset


def load_upstream():
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "skmultilearn" or name.startswith("skmultilearn.")
    }
    saved_path = list(sys.path)
    for name in saved_modules:
        del sys.modules[name]
    sys.path = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or os.getcwd()) != os.path.abspath(LOCAL_PYTHON)
    ]
    try:
        from skmultilearn.adapt import (
            BRkNNaClassifier as UpstreamBRkNNa,
            BRkNNbClassifier as UpstreamBRkNNb,
            MLkNN as UpstreamMLkNN,
        )
        from skmultilearn.problem_transform import (
            LabelPowerset as UpstreamLabelPowerset,
        )
        import skmultilearn.adapt.brknn as upstream_brknn
        import skmultilearn.adapt.mlknn as upstream_mlknn

        class CompatibleNearestNeighbors(NearestNeighbors):
            def __init__(self, n_neighbors=5, **kwargs):
                super().__init__(n_neighbors=n_neighbors, **kwargs)

        upstream_brknn.NearestNeighbors = CompatibleNearestNeighbors
        upstream_mlknn.NearestNeighbors = CompatibleNearestNeighbors
        loaded = SimpleNamespace(
            BRkNNaClassifier=UpstreamBRkNNa,
            BRkNNbClassifier=UpstreamBRkNNb,
            MLkNN=UpstreamMLkNN,
            LabelPowerset=UpstreamLabelPowerset,
        )
    finally:
        for name in list(sys.modules):
            if name == "skmultilearn" or name.startswith("skmultilearn."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path = saved_path
    return loaded


UPSTREAM = load_upstream()


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def features(rows, columns, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(rows, columns)))


def labels(X, count, seed=0):
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(X.shape[1], count))
    return np.ascontiguousarray((X @ weights > 0.4).astype(np.int64))


CASES = []


def case(name):
    def register(builder):
        CASES.append((name, builder))
        return builder

    return register


@case("BRkNNa.predict (8k train, 2k query, 24d, 12 labels)")
def _():
    X = features(8_000, 24)
    y = labels(X, 12, seed=1)
    Q = features(2_000, 24, seed=2)
    ours = BRkNNaClassifier(k=10).fit(X, y)
    theirs = UPSTREAM.BRkNNaClassifier(k=10).fit(X, y)
    return lambda: ours.predict(Q), lambda: theirs.predict(Q)


@case("BRkNNb.predict (8k train, 2k query, 24d, 12 labels)")
def _():
    X = features(8_000, 24)
    y = labels(X, 12, seed=3)
    Q = features(2_000, 24, seed=4)
    ours = BRkNNbClassifier(k=10).fit(X, y)
    theirs = UPSTREAM.BRkNNbClassifier(k=10).fit(X, y)
    return lambda: ours.predict(Q), lambda: theirs.predict(Q)


@case("MLkNN.fit (3k x 24, 12 labels, k=10)")
def _():
    X = features(3_000, 24, seed=5)
    y = labels(X, 12, seed=6)
    return (
        lambda: MLkNN(k=10).fit(X, y),
        lambda: UPSTREAM.MLkNN(k=10).fit(X, y),
    )


@case("MLkNN.predict (8k train, 2k query, 24d, 12 labels)")
def _():
    X = features(8_000, 24, seed=7)
    y = labels(X, 12, seed=8)
    Q = features(2_000, 24, seed=9)
    ours = MLkNN(k=10).fit(X, y)
    theirs = UPSTREAM.MLkNN(k=10).fit(X, y)
    return lambda: ours.predict(Q), lambda: theirs.predict(Q)


@case("LabelPowerset.transform (100k x 40, 10% density)")
def _():
    rng = np.random.default_rng(10)
    y = sparse.lil_matrix(rng.random((100_000, 40)) < 0.1, dtype=np.int64)
    ours = LabelPowerset()
    theirs = UPSTREAM.LabelPowerset()
    return lambda: ours.transform(y), lambda: theirs.transform(y)


@case("LabelPowerset.inverse_transform (100k x 40)")
def _():
    rng = np.random.default_rng(11)
    combinations = (rng.random((128, 40)) < 0.15).astype(np.int64)
    assignments = rng.integers(0, len(combinations), size=100_000)
    ours = LabelPowerset()
    theirs = UPSTREAM.LabelPowerset()
    ours.transform(combinations)
    theirs.transform(combinations)
    return (
        lambda: ours.inverse_transform(assignments),
        lambda: theirs.inverse_transform(assignments),
    )


def machine_name():
    cpu = platform.processor()
    if cpu.lower() in {"", "x86_64", "amd64"} and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    return f"{cpu or 'unknown CPU'}; {platform.system()} {platform.release()}"


def main():
    print(f"Machine: {machine_name()}")
    mojo_version = subprocess.run(
        ["mojo", "--version"], capture_output=True, check=True, text=True
    ).stdout.strip()
    package_versions = ", ".join(
        f"{name} {importlib.metadata.version(name)}"
        for name in ("numpy", "scipy", "scikit-learn", "scikit-multilearn")
    )
    print(f"Environment: {mojo_version}; {package_versions}")
    print(f"Logical CPUs: {os.cpu_count() or 'unknown'}")
    print()
    print("| case | Mojo port | upstream 0.2.0 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, builder in CASES:
        ours, theirs = builder()
        ours()
        theirs()
        mojo_seconds = timeit(ours)
        upstream_seconds = timeit(theirs)
        speedup = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
