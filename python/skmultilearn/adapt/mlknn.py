"""ML-kNN with Bayesian decisions over Mojo-computed neighborhoods."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .._neighbors import MojoNearestNeighbors, neighbor_label_counts
from ..base import MLClassifierBase


class MLkNN(MLClassifierBase):
    def __init__(self, k=10, s=1.0, ignore_first_neighbours=0, n_jobs=None):
        super().__init__()
        self.k = k
        self.s = s
        self.ignore_first_neighbours = ignore_first_neighbours
        self.n_jobs = n_jobs
        self.knn_ = MojoNearestNeighbors(self.k, n_jobs=self.n_jobs)
        self.copyable_attrs = ["k", "s", "ignore_first_neighbours", "n_jobs"]

    def _compute_prior(self, y):
        positive = np.asarray(y.sum(axis=0)).reshape(-1)
        prior_prob_true = (self.s + positive) / (
            self.s * 2 + self._num_instances
        )
        return prior_prob_true, 1.0 - prior_prob_true

    def _neighbors(self, X):
        all_neighbors = self.knn_.kneighbors(
            X,
            self.k + self.ignore_first_neighbours,
            return_distance=False,
        )
        return all_neighbors[:, self.ignore_first_neighbours :]

    def _compute_cond(self, X, y):
        self.knn_.fit(X)
        neighbors = self._neighbors(X)
        dense_y = np.ascontiguousarray(y.toarray(), dtype=np.int64)
        deltas = neighbor_label_counts(neighbors, dense_y)

        true_counts = np.zeros((self._num_labels, self.k + 1), dtype=np.int64)
        false_counts = np.zeros_like(true_counts)
        for label in range(self._num_labels):
            assigned = dense_y[:, label] == 1
            np.add.at(true_counts[label], deltas[assigned, label], 1)
            np.add.at(false_counts[label], deltas[~assigned, label], 1)

        true_denominator = (
            self.s * (self.k + 1) + true_counts.sum(axis=1)
        )[:, None]
        false_denominator = (
            self.s * (self.k + 1) + false_counts.sum(axis=1)
        )[:, None]
        cond_prob_true = (self.s + true_counts) / true_denominator
        cond_prob_false = (self.s + false_counts) / false_denominator
        return sparse.lil_matrix(cond_prob_true), sparse.lil_matrix(cond_prob_false)

    def fit(self, X, y):
        self._label_cache = sparse.lil_matrix(y)
        self._num_instances, self._num_labels = self._label_cache.shape
        if np.shape(X)[0] != self._num_instances:
            raise ValueError("X and y have inconsistent numbers of samples")
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(
            self._label_cache
        )
        self._cond_prob_true, self._cond_prob_false = self._compute_cond(
            X, self._label_cache
        )
        return self

    def _posterior(self, X):
        neighbors = self._neighbors(X)
        deltas = neighbor_label_counts(neighbors, self._label_cache)
        labels = np.arange(self._num_labels)[None, :]
        cond_true = self._cond_prob_true.toarray()
        cond_false = self._cond_prob_false.toarray()
        p_true = self._prior_prob_true[None, :] * cond_true[labels, deltas]
        p_false = self._prior_prob_false[None, :] * cond_false[labels, deltas]
        return p_true, p_false

    def predict(self, X):
        p_true, p_false = self._posterior(X)
        return sparse.lil_matrix((p_true >= p_false).astype(np.int64))

    def predict_proba(self, X):
        p_true, p_false = self._posterior(X)
        return sparse.lil_matrix(p_true / (p_true + p_false))
