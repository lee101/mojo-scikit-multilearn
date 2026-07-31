"""Behavioral and numerical parity with scikit-multilearn 0.2.0."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from skmultilearn._neighbors import (
    MojoNearestNeighbors,
    neighbor_label_counts,
    take_labelsets,
)
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN
from skmultilearn.problem_transform import (
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset,
)


@pytest.fixture
def multilabel_data():
    rng = np.random.default_rng(42)
    X = np.ascontiguousarray(rng.normal(size=(180, 9)))
    weights = rng.normal(size=(9, 6))
    logits = X @ weights + rng.normal(scale=0.6, size=(180, 6))
    y = (logits > np.quantile(logits, 0.58, axis=0)).astype(np.int64)
    Q = np.ascontiguousarray(rng.normal(size=(31, 9)))
    return X, y, Q


def test_mojo_neighbors_match_sklearn(multilabel_data):
    X, _, Q = multilabel_data
    distances, indices = MojoNearestNeighbors(7).fit(X).kneighbors(Q)
    expected_distances, expected_indices = NearestNeighbors(
        n_neighbors=7
    ).fit(X).kneighbors(Q)
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(distances, expected_distances, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("rows", "features"), [(7, 3), (64, 11)])
def test_mojo_neighbors_simd_tail_and_parallel_threshold(rows, features):
    rng = np.random.default_rng(rows + features)
    X = np.ascontiguousarray(rng.normal(size=(rows, features)))
    query_rows = 5 if rows < 8 else 4000
    Q = np.ascontiguousarray(rng.normal(size=(query_rows, features)))
    distances, indices = MojoNearestNeighbors(3).fit(X).kneighbors(Q)
    expected_distances, expected_indices = NearestNeighbors(
        n_neighbors=3
    ).fit(X).kneighbors(Q)
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(distances, expected_distances, rtol=1e-12, atol=1e-12)


def test_mojo_neighbors_without_explicit_query(multilabel_data):
    X, _, _ = multilabel_data
    distances, indices = MojoNearestNeighbors(4).fit(X).kneighbors()
    expected_distances, expected_indices = NearestNeighbors(
        n_neighbors=4
    ).fit(X).kneighbors()
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(distances, expected_distances, atol=1e-12)


def test_neighbor_label_counts_matches_numpy(multilabel_data):
    X, y, Q = multilabel_data
    indices = MojoNearestNeighbors(5).fit(X).kneighbors(
        Q, return_distance=False
    )
    actual = neighbor_label_counts(indices, y)
    expected = y[indices].sum(axis=1)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("label_count", [3, 11])
def test_neighbor_label_counts_simd_tail(label_count):
    rng = np.random.default_rng(label_count)
    labels = rng.integers(0, 2, size=(23, label_count), dtype=np.int64)
    indices = rng.integers(0, len(labels), size=(9, 5), dtype=np.int64)
    assert np.array_equal(
        neighbor_label_counts(indices, labels), labels[indices].sum(axis=1)
    )


def test_neighbor_label_counts_sparse_labels(multilabel_data):
    X, y, Q = multilabel_data
    indices = MojoNearestNeighbors(3).fit(X).kneighbors(
        Q, return_distance=False
    )
    assert np.array_equal(
        neighbor_label_counts(indices, sparse.csc_matrix(y)),
        y[indices].sum(axis=1),
    )


@pytest.mark.parametrize(
    ("indices", "labels", "error"),
    [
        (np.array([[0, -1]]), np.ones((2, 1)), IndexError),
        (np.array([[0, 2]]), np.ones((2, 1)), IndexError),
        (np.array([[0.0, 0.5]]), np.ones((2, 1)), ValueError),
        (np.array([[0]]), np.array([[1.5]]), ValueError),
        (
            np.array([[0]], dtype=np.uint64),
            np.array([[2**63]], dtype=np.uint64),
            OverflowError,
        ),
        (np.array([[0]]), np.array([[2]]), ValueError),
    ],
)
def test_neighbor_label_counts_rejects_unsafe_inputs(indices, labels, error):
    with pytest.raises(error):
        neighbor_label_counts(indices, labels)


@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf]
)
def test_mojo_neighbors_rejects_non_finite_features(bad_value):
    X = np.array([[0.0, 1.0], [1.0, bad_value]])
    with pytest.raises(ValueError):
        MojoNearestNeighbors(1).fit(X)


def test_mojo_neighbors_rejects_lossy_feature_conversion():
    with pytest.raises(ValueError, match="exactly representable"):
        MojoNearestNeighbors(1).fit(np.array([[2**53 + 1]], dtype=np.int64))
    with pytest.raises(TypeError, match="narrowed"):
        MojoNearestNeighbors(1).fit(np.array([[0.1]], dtype=np.longdouble))


@pytest.mark.parametrize(
    ("ours_class", "upstream_name"),
    [
        (BRkNNaClassifier, "BRkNNaClassifier"),
        (BRkNNbClassifier, "BRkNNbClassifier"),
    ],
)
def test_brknn_prediction_parity(
    multilabel_data, upstream, ours_class, upstream_name
):
    X, y, Q = multilabel_data
    ours = ours_class(k=6).fit(X, y)
    theirs = getattr(upstream, upstream_name)(k=6).fit(X, y)
    assert np.array_equal(
        ours.predict(Q).toarray(), theirs.predict(Q).toarray()
    )
    assert np.array_equal(ours.neighbors_, theirs.neighbors_)
    assert np.allclose(ours.confidences_, theirs.confidences_)


def test_brknn_sparse_feature_fallback(multilabel_data, upstream):
    X, y, Q = multilabel_data
    X_sparse = sparse.csr_matrix(X)
    Q_sparse = sparse.csr_matrix(Q)
    ours = BRkNNaClassifier(k=5).fit(X_sparse, y)
    theirs = upstream.BRkNNaClassifier(k=5).fit(X_sparse, y)
    assert np.array_equal(
        ours.predict(Q_sparse).toarray(), theirs.predict(Q_sparse).toarray()
    )


def test_brknnb_zero_neighbor_cardinality_predicts_no_labels():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.zeros((3, 2), dtype=np.int64)
    prediction = BRkNNbClassifier(k=1).fit(X, y).predict([[0.1]])
    assert prediction.nnz == 0


def test_mlknn_fit_probability_tables(multilabel_data, upstream):
    X, y, _ = multilabel_data
    ours = MLkNN(k=5, s=0.7).fit(X, y)
    theirs = upstream.MLkNN(k=5, s=0.7).fit(X, y)
    assert np.allclose(ours._prior_prob_true, theirs._prior_prob_true)
    assert np.allclose(ours._prior_prob_false, theirs._prior_prob_false)
    assert np.allclose(
        ours._cond_prob_true.toarray(), theirs._cond_prob_true.toarray()
    )
    assert np.allclose(
        ours._cond_prob_false.toarray(), theirs._cond_prob_false.toarray()
    )


def test_mlknn_prediction_parity(multilabel_data, upstream):
    X, y, Q = multilabel_data
    ours = MLkNN(k=7).fit(X, y)
    theirs = upstream.MLkNN(k=7).fit(X, y)
    assert np.array_equal(
        ours.predict(Q).toarray(), theirs.predict(Q).toarray()
    )


def test_mlknn_ignore_first_neighbours_parity(multilabel_data, upstream):
    X, y, Q = multilabel_data
    ours = MLkNN(k=4, ignore_first_neighbours=2).fit(X, y)
    theirs = upstream.MLkNN(k=4, ignore_first_neighbours=2).fit(X, y)
    assert np.allclose(
        ours._cond_prob_true.toarray(), theirs._cond_prob_true.toarray()
    )
    assert np.array_equal(
        ours.predict(Q).toarray(), theirs.predict(Q).toarray()
    )


def test_mlknn_predict_proba_is_bayesian_posterior(multilabel_data):
    X, y, Q = multilabel_data
    model = MLkNN(k=5, s=1.0).fit(X, y)
    probabilities = model.predict_proba(Q).toarray()
    neighbors = model._neighbors(Q)
    deltas = y[neighbors].sum(axis=1)
    labels = np.arange(y.shape[1])[None, :]
    p_true = model._prior_prob_true * model._cond_prob_true.toarray()[
        labels, deltas
    ]
    p_false = model._prior_prob_false * model._cond_prob_false.toarray()[
        labels, deltas
    ]
    assert np.allclose(probabilities, p_true / (p_true + p_false))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_label_powerset_transform_parity(multilabel_data, upstream):
    _, y, _ = multilabel_data
    ours = LabelPowerset()
    theirs = upstream.LabelPowerset()
    ours_classes = ours.transform(y)
    their_classes = theirs.transform(y)
    assert np.array_equal(ours_classes, their_classes)
    assert ours.unique_combinations_ == theirs.unique_combinations_
    assert ours.reverse_combinations_ == theirs.reverse_combinations_
    assert np.array_equal(
        ours.inverse_transform(ours_classes).toarray(),
        theirs.inverse_transform(their_classes).toarray(),
    )


def test_label_powerset_transform_wide_label_space(upstream):
    y = np.zeros((5, 300), dtype=np.int64)
    y[0, [1, 299]] = 1
    y[1, [0, 256]] = 1
    y[2, [1, 299]] = 1
    y[4, 255] = 1
    ours = LabelPowerset()
    theirs = upstream.LabelPowerset()
    assert np.array_equal(ours.transform(y), theirs.transform(y))
    assert ours.unique_combinations_ == theirs.unique_combinations_
    assert ours.reverse_combinations_ == theirs.reverse_combinations_


def test_take_labelsets_kernel():
    combinations = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]])
    assignments = np.array([2, 0, 1, 2])
    assert np.array_equal(
        take_labelsets(combinations, assignments), combinations[assignments]
    )


def test_take_labelsets_rejects_unsafe_inputs():
    with pytest.raises(ValueError, match="two-dimensional"):
        take_labelsets(np.array([0, 1]), np.array([0]))
    with pytest.raises(IndexError):
        take_labelsets(np.eye(2, dtype=np.int64), np.array([-1]))
    with pytest.raises(ValueError, match="integer"):
        take_labelsets(np.eye(2, dtype=np.int64), np.array([0.5]))


def test_label_powerset_classifier_parity(multilabel_data, upstream):
    X, y, Q = multilabel_data
    base = LogisticRegression(max_iter=1000, random_state=0)
    ours = LabelPowerset(base).fit(X, y)
    theirs = upstream.LabelPowerset(
        LogisticRegression(max_iter=1000, random_state=0)
    ).fit(X, y)
    assert np.array_equal(
        ours.predict(Q).toarray(), theirs.predict(Q).toarray()
    )
    assert np.allclose(
        ours.predict_proba(Q).toarray(), theirs.predict_proba(Q).toarray()
    )


def test_binary_relevance_classifier_parity(multilabel_data, upstream):
    X, y, Q = multilabel_data
    base = LogisticRegression(max_iter=1000, random_state=0)
    ours = BinaryRelevance(base).fit(X, y)
    theirs = upstream.BinaryRelevance(
        LogisticRegression(max_iter=1000, random_state=0)
    ).fit(X, y)
    assert np.array_equal(
        ours.predict(Q).toarray(), theirs.predict(Q).toarray()
    )
    assert np.allclose(
        ours.predict_proba(Q).toarray(), theirs.predict_proba(Q).toarray()
    )


def test_classifier_chain_classifier_parity(multilabel_data, upstream):
    X, y, Q = multilabel_data
    base = LogisticRegression(max_iter=1000, random_state=0)
    ours = ClassifierChain(base, order=[2, 0, 1, 5, 4, 3]).fit(X, y)
    theirs = upstream.ClassifierChain(
        LogisticRegression(max_iter=1000, random_state=0),
        order=[2, 0, 1, 5, 4, 3],
    ).fit(X, y)
    assert np.array_equal(
        ours.predict(Q).toarray(), theirs.predict(Q).toarray()
    )
    assert np.allclose(
        ours.predict_proba(Q).toarray(), theirs.predict_proba(Q).toarray()
    )


def test_problem_transform_accepts_sparse_inputs(multilabel_data):
    X, y, Q = multilabel_data
    model = BinaryRelevance(
        LogisticRegression(max_iter=1000), require_dense=[False, True]
    ).fit(sparse.csr_matrix(X), sparse.csr_matrix(y))
    prediction = model.predict(sparse.csr_matrix(Q))
    assert sparse.issparse(prediction)
    assert prediction.shape == y[: len(Q)].shape


def test_estimator_parameter_api():
    estimator = MLkNN(k=3, s=0.5)
    assert estimator.get_params()["k"] == 3
    assert estimator.set_params(k=8) is estimator
    assert estimator.k == 8
