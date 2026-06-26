"""
Unit tests for the Decision Tree classifier.
These tests do not require an internet connection.
"""

import numpy as np
import pytest


VALID_CLASSES = {"GALAXY", "QSO", "STAR"}


def test_decision_tree_returns_one_prediction(test_predictor, single_example):
    """Predictor should return exactly one row for a single input."""
    result = test_predictor.predict(single_example)
    assert len(result) == 1, "The model did not return exactly one prediction."


def test_decision_tree_predicted_class_is_valid(test_predictor, single_example):
    """Predicted class must be one of GALAXY, QSO, or STAR."""
    result = test_predictor.predict(single_example)
    assert result.loc[0, "predicted_class"] in VALID_CLASSES, (
        f"Unexpected class: {result.loc[0, 'predicted_class']}"
    )


def test_decision_tree_confidence_is_finite(test_predictor, single_example):
    """Confidence score must be a finite number."""
    result = test_predictor.predict(single_example)
    assert np.isfinite(result.loc[0, "confidence_%"]), "Confidence is not finite."


def test_decision_tree_confidence_in_range(test_predictor, single_example):
    """Confidence score must be between 0 and 100."""
    result = test_predictor.predict(single_example)
    assert 0.0 <= result.loc[0, "confidence_%"] <= 100.0, (
        f"Confidence out of range: {result.loc[0, 'confidence_%']}"
    )
