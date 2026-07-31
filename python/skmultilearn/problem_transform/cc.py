"""Classifier Chain problem transformation."""

from __future__ import annotations

import copy

from scipy import sparse
from sklearn.exceptions import NotFittedError

from ..base import ProblemTransformationBase


class ClassifierChain(ProblemTransformationBase):
    def __init__(self, classifier=None, require_dense=None, order=None):
        super().__init__(classifier, require_dense)
        self.order = order
        self.copyable_attrs = ["classifier", "require_dense", "order"]

    def fit(self, X, y, order=None):
        X_extended = self._ensure_input_format(
            X, sparse_format="csc", enforce_sparse=True
        )
        y = self._ensure_output_format(y, sparse_format="csc", enforce_sparse=True)
        self._label_count = y.shape[1]
        self.classifiers_ = [None for _ in range(self._label_count)]

        for label in self._order():
            classifier = copy.deepcopy(self.classifier)
            target = self._generate_data_subset(y, label, axis=1)
            self.classifiers_[label] = classifier.fit(
                self._ensure_input_format(X_extended),
                self._ensure_output_format(target),
            )
            X_extended = sparse.hstack([X_extended, target])
        return self

    def predict(self, X):
        X_extended = self._ensure_input_format(
            X, sparse_format="csc", enforce_sparse=True
        )
        for label in self._order():
            prediction = self.classifiers_[label].predict(
                self._ensure_input_format(X_extended)
            )
            prediction = self._ensure_multi_label_from_single_class(prediction)
            X_extended = sparse.hstack([X_extended, prediction]).tocsc()
        return X_extended[:, -self._label_count :]

    def predict_proba(self, X):
        X_extended = self._ensure_input_format(
            X, sparse_format="csc", enforce_sparse=True
        )
        results = []
        for label in self._order():
            prediction = self.classifiers_[label].predict(
                self._ensure_input_format(X_extended)
            )
            prediction = self._ensure_output_format(
                prediction, sparse_format="csc", enforce_sparse=True
            )
            probabilities = self.classifiers_[label].predict_proba(
                self._ensure_input_format(X_extended)
            )
            probabilities = self._ensure_output_format(
                probabilities, sparse_format="csc", enforce_sparse=True
            )[:, 1]
            X_extended = sparse.hstack([X_extended, prediction]).tocsc()
            results.append(probabilities)
        return sparse.hstack(results)

    def _order(self):
        if self.order is not None:
            return self.order
        try:
            return list(range(self._label_count))
        except AttributeError:
            raise NotFittedError("This Classifier Chain has not been fit yet")
