"""Base estimator compatible with scikit-multilearn's public helpers."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin


class MLClassifierBase(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.copyable_attrs = []

    def _generate_data_subset(self, y, subset, axis):
        if axis == 1:
            return y.tocsc()[:, subset]
        if axis == 0:
            return y.tocsr()[subset, :]
        return None

    def _ensure_input_format(self, X, sparse_format="csr", enforce_sparse=False):
        if sparse.issparse(X):
            if self.require_dense[0] and not enforce_sparse:
                return X.toarray()
            return X if sparse_format is None else X.asformat(sparse_format)
        if self.require_dense[0] and not enforce_sparse:
            return X
        return sparse.csr_matrix(X).asformat(sparse_format)

    def _ensure_output_format(
        self, matrix, sparse_format="csr", enforce_sparse=False
    ):
        if sparse.issparse(matrix):
            if self.require_dense[1] and not enforce_sparse:
                dense = matrix.toarray()
                return np.ravel(dense) if dense.shape[1] == 1 else dense
            return matrix if sparse_format is None else matrix.asformat(sparse_format)
        array = np.asarray(matrix)
        if self.require_dense[1] and not enforce_sparse:
            return np.ravel(array) if array.ndim > 1 else array
        if array.ndim == 1:
            array = array.reshape((-1, 1))
        return sparse.csr_matrix(array).asformat(sparse_format)

    def fit(self, X, y):
        raise NotImplementedError("MLClassifierBase::fit()")

    def predict(self, X):
        raise NotImplementedError("MLClassifierBase::predict()")

