"""Shared formatting behavior for problem-transformation estimators."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .base import MLClassifierBase


class ProblemTransformationBase(MLClassifierBase):
    def __init__(self, classifier=None, require_dense=None):
        super().__init__()
        self.copyable_attrs = ["classifier", "require_dense"]
        self.classifier = classifier
        if require_dense is not None:
            if isinstance(require_dense, bool):
                self.require_dense = [require_dense, require_dense]
            else:
                assert (
                    len(require_dense) == 2
                    and isinstance(require_dense[0], bool)
                    and isinstance(require_dense[1], bool)
                )
                self.require_dense = require_dense
        elif isinstance(classifier, MLClassifierBase):
            self.require_dense = [False, False]
        else:
            self.require_dense = [True, True]

    def _ensure_multi_label_from_single_class(self, matrix, matrix_format="csr"):
        if sparse.issparse(matrix):
            result = matrix
        else:
            array = np.asarray(matrix)
            if array.ndim == 1:
                array = array.reshape((-1, 1))
            if array.ndim != 2:
                raise ValueError("Matrix dimensions too large (>2) or other value error")
            result = sparse.csr_matrix(array)
        return result.asformat(matrix_format)

