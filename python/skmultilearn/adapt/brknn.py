"""Binary-relevance kNN classifiers with Mojo neighbor aggregation."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .._neighbors import MojoNearestNeighbors, neighbor_label_counts
from ..base import MLClassifierBase


class _BinaryRelevanceKNN(MLClassifierBase):
    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.copyable_attrs = ["k"]

    def fit(self, X, y):
        self.train_labelspace = sparse.csc_matrix(y)
        self._n_samples, self._n_labels = self.train_labelspace.shape
        if np.shape(X)[0] != self._n_samples:
            raise ValueError("X and y have inconsistent numbers of samples")
        self.knn_ = MojoNearestNeighbors(n_neighbors=self.k).fit(X)
        return self

    def predict(self, X):
        self.neighbors_ = self.knn_.kneighbors(
            X, self.k, return_distance=False
        )
        counts = neighbor_label_counts(self.neighbors_, self.train_labelspace)
        self.confidences_ = counts.astype(np.float64) / self.k
        return self._predict_variant(X)


class BRkNNaClassifier(_BinaryRelevanceKNN):
    """Assign each label selected by the rounded neighbor vote."""

    def _predict_variant(self, X):
        return sparse.csr_matrix(np.rint(self.confidences_), dtype="i8")


class BRkNNbClassifier(_BinaryRelevanceKNN):
    """Assign the top-m labels, where m is mean neighbor cardinality."""

    def _predict_variant(self, X):
        cardinalities = np.asarray(self.train_labelspace.sum(axis=1)).reshape(-1)
        sample_count = X.shape[0] if hasattr(X, "shape") else len(X)
        avg_labels = [
            int(np.average(cardinalities[neighbors]).round())
            for neighbors in self.neighbors_
        ]
        prediction = sparse.lil_matrix((sample_count, self._n_labels), dtype="i8")
        kth = min(avg_labels + [self._n_labels])
        if kth >= self._n_labels:
            top_labels = np.tile(
                np.arange(self._n_labels, dtype=np.int64), (sample_count, 1)
            ).tolist()
        else:
            top_labels = np.argpartition(
                self.confidences_, kth=kth, axis=1
            ).tolist()
        for row in range(sample_count):
            if avg_labels[row] == 0:
                continue
            for label in top_labels[row][-avg_labels[row] :]:
                prediction[row, label] += 1
        return prediction
