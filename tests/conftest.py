from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@pytest.fixture(scope="session")
def upstream():
    local_python = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "python")
    )
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
        if os.path.abspath(entry or os.getcwd()) != local_python
    ]

    try:
        from skmultilearn.adapt import (
            BRkNNaClassifier,
            BRkNNbClassifier,
            MLkNN,
        )
        from skmultilearn.problem_transform import (
            BinaryRelevance,
            ClassifierChain,
            LabelPowerset,
        )
        import skmultilearn.adapt.brknn as upstream_brknn
        import skmultilearn.adapt.mlknn as upstream_mlknn
        import skmultilearn.problem_transform.cc as upstream_cc

        class CompatibleNearestNeighbors(NearestNeighbors):
            def __init__(self, n_neighbors=5, **kwargs):
                super().__init__(n_neighbors=n_neighbors, **kwargs)

        upstream_brknn.NearestNeighbors = CompatibleNearestNeighbors
        upstream_mlknn.NearestNeighbors = CompatibleNearestNeighbors
        scipy_hstack = sparse.hstack
        upstream_cc.hstack = lambda blocks: scipy_hstack(blocks, format="csc")

        loaded = SimpleNamespace(
            BRkNNaClassifier=BRkNNaClassifier,
            BRkNNbClassifier=BRkNNbClassifier,
            MLkNN=MLkNN,
            BinaryRelevance=BinaryRelevance,
            ClassifierChain=ClassifierChain,
            LabelPowerset=LabelPowerset,
        )
    finally:
        for name in list(sys.modules):
            if name == "skmultilearn" or name.startswith("skmultilearn."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path = saved_path

    return loaded

