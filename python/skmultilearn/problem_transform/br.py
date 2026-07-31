"""Binary Relevance problem transformation."""

from __future__ import annotations

import copy

import numpy as np
from scipy import sparse

from ..base import MLClassifierBase, ProblemTransformationBase


class BinaryRelevance(ProblemTransformationBase):
    def __init__(self, classifier=None, require_dense=None):
        super().__init__(classifier, require_dense)

    def _generate_partition(self, X, y):
        self.partition_ = list(range(y.shape[1]))
        self.model_count_ = y.shape[1]

    def fit(self, X, y):
        X = self._ensure_input_format(X, sparse_format="csr", enforce_sparse=True)
        y = self._ensure_output_format(y, sparse_format="csc", enforce_sparse=True)
        self.classifiers_ = []
        self._generate_partition(X, y)
        self._label_count = y.shape[1]
        for label in range(self.model_count_):
            classifier = copy.deepcopy(self.classifier)
            target = self._generate_data_subset(y, self.partition_[label], axis=1)
            if sparse.issparse(target) and target.shape[1] == 1:
                target = np.ravel(target.toarray())
            classifier.fit(
                self._ensure_input_format(X), self._ensure_output_format(target)
            )
            self.classifiers_.append(classifier)
        return self

    def predict(self, X):
        predictions = [
            self._ensure_multi_label_from_single_class(
                self.classifiers_[label].predict(self._ensure_input_format(X))
            )
            for label in range(self.model_count_)
        ]
        return sparse.hstack(predictions)

    def predict_proba(self, X):
        result = sparse.lil_matrix((X.shape[0], self._label_count), dtype="float")
        for label_assignment, classifier in zip(
            self.partition_, self.classifiers_
        ):
            if isinstance(self.classifier, MLClassifierBase):
                result[:, label_assignment] = classifier.predict_proba(X)
            else:
                probabilities = classifier.predict_proba(
                    self._ensure_input_format(X)
                )
                result[:, label_assignment] = (
                    self._ensure_multi_label_from_single_class(probabilities)[:, 1]
                )
        return result

