"""Dense Mojo nearest-neighbor path with a sparse sklearn fallback."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numbers
import os

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from ._lib import addr, lib

_KNN_PARALLEL_WORK = 2_000_000
_KNN_MAX_WORKERS = 16


def _dense_f64(matrix) -> np.ndarray:
    original = np.asarray(matrix)
    if original.dtype.kind not in "biuf":
        raise TypeError("feature matrices must contain real numeric values")
    if original.dtype.kind in "iu" and original.size:
        # Every integer in this interval is exactly representable as float64.
        limit = 1 << 53
        if np.any(original < -limit) or np.any(original > limit):
            raise ValueError("integer features are not exactly representable as float64")
    if original.dtype.kind == "f" and original.dtype.itemsize > 8:
        raise TypeError("feature dtype would be narrowed to float64")
    array = np.asarray(original, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional feature matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("feature matrices must not contain NaN or infinity")
    return np.ascontiguousarray(array)


def _dense_i64(matrix, name) -> np.ndarray:
    original = np.asarray(matrix)
    if original.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain integer values")
    if original.size:
        if original.dtype.kind == "f":
            if not np.all(np.isfinite(original)) or np.any(original != np.trunc(original)):
                raise ValueError(f"{name} must contain finite integer values")
            info = np.iinfo(np.int64)
            if np.any(original < info.min) or np.any(original > info.max):
                raise OverflowError(f"{name} values do not fit in int64")
        elif original.dtype.kind == "u" and np.any(original > np.iinfo(np.int64).max):
            raise OverflowError(f"{name} values do not fit in int64")
    return np.ascontiguousarray(original, dtype=np.int64)


class MojoNearestNeighbors:
    def __init__(self, n_neighbors=5, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(self, X):
        if sparse.issparse(X):
            self._fallback = NearestNeighbors(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
            ).fit(X)
            self._fit_X = X
        else:
            self._fallback = None
            self._fit_X = _dense_f64(X)
        self.n_samples_fit_, self.n_features_in_ = self._fit_X.shape
        return self

    def _worker_count(self, query_rows):
        if self.n_jobs == 1:
            return 1
        available = os.cpu_count() or 1
        if isinstance(self.n_jobs, numbers.Integral) and self.n_jobs > 1:
            available = min(available, int(self.n_jobs))
        return min(query_rows, available, _KNN_MAX_WORKERS)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if not hasattr(self, "_fit_X"):
            raise ValueError("nearest-neighbor index is not fitted")
        neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        if not isinstance(neighbors, numbers.Integral) or neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        neighbors = int(neighbors)
        query_is_train = X is None
        query = self._fit_X if query_is_train else X

        if self._fallback is not None or sparse.issparse(query):
            if self._fallback is None:
                self._fallback = NearestNeighbors(
                    n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
                ).fit(self._fit_X)
            return self._fallback.kneighbors(
                None if query_is_train else query,
                n_neighbors=neighbors,
                return_distance=return_distance,
            )

        query_array = _dense_f64(query)
        if query_array.shape[1] != self.n_features_in_:
            raise ValueError("query and training matrices have different feature counts")
        available = self.n_samples_fit_ - int(query_is_train)
        if neighbors > available:
            raise ValueError(
                f"Expected n_neighbors <= n_samples_fit, but "
                f"n_neighbors = {neighbors + int(query_is_train)}, "
                f"n_samples_fit = {self.n_samples_fit_}"
            )
        if len(query_array) == 0:
            indices = np.empty((0, neighbors), dtype=np.int64)
            distances = np.empty((0, neighbors), dtype=np.float64)
            return (distances, indices) if return_distance else indices

        indices = np.empty((len(query_array), neighbors), dtype=np.int64)
        squared = np.empty((len(query_array), neighbors), dtype=np.float64)
        function = lib().msml_knn_query

        def query_chunk(start, end):
            function(
                addr(self._fit_X),
                addr(query_array[start:end]),
                addr(indices[start:end]),
                addr(squared[start:end]),
                self.n_samples_fit_,
                self.n_features_in_,
                end - start,
                neighbors,
                int(query_is_train),
                start,
            )

        work = len(query_array) * self.n_samples_fit_ * self.n_features_in_
        workers = self._worker_count(len(query_array))
        if len(query_array) >= 8 and work >= _KNN_PARALLEL_WORK and workers > 1:
            bounds = [
                (
                    worker * len(query_array) // workers,
                    (worker + 1) * len(query_array) // workers,
                )
                for worker in range(workers)
            ]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(lambda bound: query_chunk(*bound), bounds))
        else:
            query_chunk(0, len(query_array))

        if return_distance:
            np.sqrt(squared, out=squared)
            return squared, indices
        return indices


def neighbor_label_counts(indices, labels) -> np.ndarray:
    neighbor_indices = _dense_i64(indices, "indices")
    dense_labels = (
        labels.toarray() if sparse.issparse(labels) else np.asarray(labels)
    )
    dense_labels = _dense_i64(dense_labels, "labels")
    if neighbor_indices.ndim != 2 or dense_labels.ndim != 2:
        raise ValueError("indices and labels must both be two-dimensional")
    if dense_labels.size and np.any((dense_labels != 0) & (dense_labels != 1)):
        raise ValueError("labels must be a binary indicator matrix")
    if neighbor_indices.size and (
        np.any(neighbor_indices < 0) or np.any(neighbor_indices >= len(dense_labels))
    ):
        raise IndexError("neighbor index is out of range")
    result = np.empty(
        (neighbor_indices.shape[0], dense_labels.shape[1]), dtype=np.int64
    )
    if result.size:
        lib().msml_neighbor_label_counts(
            addr(neighbor_indices),
            addr(dense_labels),
            addr(result),
            neighbor_indices.shape[0],
            neighbor_indices.shape[1],
            dense_labels.shape[1],
        )
    return result


def take_labelsets(combinations, assignments) -> np.ndarray:
    table = _dense_i64(combinations, "combinations")
    raw_classes = (
        assignments.toarray() if sparse.issparse(assignments) else assignments
    )
    classes = _dense_i64(np.asarray(raw_classes).reshape(-1), "assignments")
    if table.ndim != 2:
        raise ValueError("combinations must be two-dimensional")
    if np.any(classes < 0) or np.any(classes >= len(table)):
        raise IndexError("label combination assignment is out of range")
    result = np.empty((len(classes), table.shape[1]), dtype=np.int64)
    if result.size:
        lib().msml_take_labelsets(
            addr(table), addr(classes), addr(result), len(classes), table.shape[1]
        )
    return result
