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
        train_vector = []
        append_class = train_vector.append
        combinations = self.unique_combinations_
        get_combination = combinations.get
        reverse_combinations = self.reverse_combinations_
        append_combination = reverse_combinations.append
        label_names = tuple(map(str, range(self._label_count)))
        get_label_name = label_names.__getitem__
        for labels_applied in labels.rows:
            combination = ",".join(map(get_label_name, labels_applied))
            combination_id = get_combination(combination)
            if combination_id is None:
                combination_id = len(reverse_combinations)
                combinations[combination] = combination_id
                append_combination(labels_applied)
            append_class(combination_id)
        return np.array(train_vector)

    def inverse_transform(self, y):
        table = np.zeros(
            (len(self.reverse_combinations_), self._label_count), dtype=np.int64
        )
        for combination_id, labels in enumerate(self.reverse_combinations_):
            table[combination_id, labels] = 1
        return sparse.lil_matrix(take_labelsets(table, y), dtype="i8")
