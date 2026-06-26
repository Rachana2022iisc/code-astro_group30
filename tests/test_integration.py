"""
Integration tests: download SDSS data and run it through the Decision Tree.
Requires an internet connection plus astropy and astroquery.
"""

import pytest

VALID_CLASSES = {"GALAXY", "QSO", "STAR"}
REQUIRED_OUTPUT_COLUMNS = {"predicted_class", "confidence_%", "model_used"}


def test_integration_returns_predictions(test_predictor, downloaded_sdss_data):
    """Combined SDSS + model pipeline should return a non-empty result."""
    result = test_predictor.predict(downloaded_sdss_data)
    assert not result.empty, "The combined test returned no predictions."


def test_integration_has_required_columns(test_predictor, downloaded_sdss_data):
    """Result must contain predicted_class, confidence_%, and model_used columns."""
    result = test_predictor.predict(downloaded_sdss_data)
    missing = REQUIRED_OUTPUT_COLUMNS - set(result.columns)
    assert not missing, f"Combined result is missing columns: {sorted(missing)}"


def test_integration_classes_are_valid(test_predictor, downloaded_sdss_data):
    """All predicted classes must be GALAXY, QSO, or STAR."""
    result = test_predictor.predict(downloaded_sdss_data)
    assert result["predicted_class"].isin(VALID_CLASSES).all(), (
        f"Unexpected classes found: {set(result['predicted_class']) - VALID_CLASSES}"
    )


def test_integration_confidence_in_range(test_predictor, downloaded_sdss_data):
    """All confidence scores must be between 0 and 100."""
    result = test_predictor.predict(downloaded_sdss_data)
    assert result["confidence_%"].between(0.0, 100.0).all(), (
        "Some confidence scores are out of the 0–100 range."
    )
