"""Label Powerset problem transformation."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .._neighbors import take_labelsets
from ..base import ProblemTransformationBase


class LabelPowerset(ProblemTransformationBase):
    def __init__(self, classifier=None, require_dense=None):
        super().__init__(classifier=classifier, require_dense=require_dense)
        self._clean()

    def _clean(self):
        self.unique_combinations_ = {}
        self.reverse_combinations_ = []
        self._label_count = None

    def fit(self, X, y):
        X = self._ensure_input_format(X, sparse_format="csr", enforce_sparse=True)
        self.classifier.fit(self._ensure_input_format(X), self.transform(y))
        return self

    def predict(self, X):
        assignments = self.classifier.predict(self._ensure_input_format(X))
        return self.inverse_transform(assignments)

    def predict_proba(self, X):
        powerset_probabilities = self.classifier.predict_proba(
            self._ensure_input_format(X)
        )
        result = sparse.lil_matrix((X.shape[0], self._label_count), dtype="float")
        for row, assignment in enumerate(powerset_probabilities):
            for combination_id, probability in enumerate(assignment):
                for label in self.reverse_combinations_[combination_id]:
                    result[row, label] += probability
        return result

    def transform(self, y):
        labels = self._ensure_output_format(
            y, sparse_format="lil", enforce_sparse=True
        )
        self._clean()
        self._label_count = labels.shape[1]
        train_vector = np.empty(labels.shape[0], dtype=np.int64)
        tuple_combinations = {}
        key_type = bytes if self._label_count <= 256 else tuple
        label_names = tuple(map(str, range(self._label_count)))
        get_label_name = label_names.__getitem__
        for row, labels_applied in enumerate(labels.rows):
            combination = key_type(labels_applied)
            combination_id = tuple_combinations.get(combination)
            if combination_id is None:
                combination_id = len(self.reverse_combinations_)
                tuple_combinations[combination] = combination_id
                label_string = ",".join(map(get_label_name, labels_applied))
                self.unique_combinations_[label_string] = combination_id
                self.reverse_combinations_.append(labels_applied)
            train_vector[row] = combination_id
        return train_vector

    def inverse_transform(self, y):
        table = np.zeros(
            (len(self.reverse_combinations_), self._label_count), dtype=np.int64
        )
        for combination_id, labels in enumerate(self.reverse_combinations_):
            table[combination_id, labels] = 1
        return sparse.lil_matrix(take_labelsets(table, y), dtype="i8")
