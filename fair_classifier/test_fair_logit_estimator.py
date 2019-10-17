# pylint: disable=missing-docstring
# pylint: disable=no-self-use
# pylint: disable=invalid-name

import numpy as np
from fair_logit_estimator import _get_fairness_constraint

class TestFairnessConstraint:
    def test_perfect_correlation(self):
        sensitive_x = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
        unsensitive_x = np.array([[x, 0] for x in sensitive_x])
        w = np.array([1, 0])
        constraint = _get_fairness_constraint(unsensitive_x, sensitive_x, 0)
        constraint_fn = constraint['fun']
        assert constraint_fn(w, unsensitive_x, sensitive_x, 0) == -1

    def test_perfect_correlation_scaled(self):
        sensitive_x = 100*np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
        unsensitive_x = np.array([[x, 0] for x in sensitive_x])
        w = np.array([1, 0])
        constraint = _get_fairness_constraint(unsensitive_x, sensitive_x, 0)
        constraint_fn = constraint['fun']
        assert np.isclose(constraint_fn(w, unsensitive_x, sensitive_x, 0), -1,
                          atol=1e-10, rtol=1e-10)

    def test_perfect_anticorrelation(self):
        sensitive_x = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
        unsensitive_x = np.array([[x, 0] for x in sensitive_x])
        w = np.array([-1, 0])
        constraint = _get_fairness_constraint(unsensitive_x, sensitive_x, 0)
        constraint_fn = constraint['fun']
        assert constraint_fn(w, unsensitive_x, sensitive_x, 0) == -1

    def test_zero_correlation(self):
        num_samples = 100000
        sensitive_x = np.array([np.random.random() for _ in range(num_samples)])
        unsensitive_x = np.array([[np.random.random(), np.random.random()]
                                  for _ in range(num_samples)])
        w = np.array([1, 1])
        constraint = _get_fairness_constraint(unsensitive_x, sensitive_x, 0)
        constraint_fn = constraint['fun']
        assert np.isclose(constraint_fn(w, unsensitive_x, sensitive_x, 0), 0, atol=0.01)

    def test_zero_correlation_different_scales(self):
        num_samples = 100000
        sensitive_x = np.array([np.random.random() for _ in range(num_samples)])
        unsensitive_x = np.array([[np.random.random(), np.random.random()]
                                  for _ in range(num_samples)])
        w = np.array([1, 1])
        constraint = _get_fairness_constraint(unsensitive_x, sensitive_x, 0)
        constraint_fn = constraint['fun']
        small_x_correlation = constraint_fn(w, unsensitive_x, sensitive_x, 0)
        large_x_correlation = constraint_fn(w, 100*unsensitive_x, 100*sensitive_x, 0)
        assert np.isclose(small_x_correlation, large_x_correlation,
                          atol=1e-10, rtol=1e-10)
