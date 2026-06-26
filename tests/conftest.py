"""
Shared fixtures for StellaClass tests.
"""

import os
import pytest
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from StellaClass.predict import (
    DEFAULT_DEC,
    DEFAULT_RA,
    PHOTO_FEATURES,
    PredictingStellarClass,
    StellarClassPredictor,
    add_features,
)

# Coordinates — can be overridden via environment variables
TEST_RA = os.getenv("TEST_RA", DEFAULT_RA)
TEST_DEC = os.getenv("TEST_DEC", DEFAULT_DEC)
TEST_RADIUS_ARCSEC = float(os.getenv("TEST_RADIUS_ARCSEC", "2.0"))

# Minimal labelled dataset for unit tests
_TEST_TRAINING_DATA = pd.DataFrame(
    {
        "u": [22.2, 21.8, 22.5, 19.3, 19.0, 19.6, 17.4, 17.8, 17.1],
        "g": [20.8, 20.4, 21.0, 18.9, 18.6, 19.1, 16.2, 16.6, 15.9],
        "r": [19.8, 19.5, 20.0, 18.7, 18.4, 18.9, 15.7, 16.1, 15.4],
        "i": [19.3, 19.0, 19.5, 18.6, 18.3, 18.8, 15.5, 15.9, 15.2],
        "z": [19.0, 18.7, 19.2, 18.5, 18.2, 18.7, 15.4, 15.8, 15.1],
        "class": [
            "GALAXY", "GALAXY", "GALAXY",
            "QSO",    "QSO",    "QSO",
            "STAR",   "STAR",   "STAR",
        ],
    }
)


@pytest.fixture(scope="session")
def test_predictor():
    """A lightweight StellarClassPredictor fitted on test data."""
    features = add_features(_TEST_TRAINING_DATA)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(random_state=42, max_depth=4)),
        ]
    )
    pipeline.fit(features[PHOTO_FEATURES], _TEST_TRAINING_DATA["class"])

    predictor = StellarClassPredictor(random_state=42)
    predictor.photo_model = pipeline
    predictor.redshift_model = None
    return predictor


@pytest.fixture(scope="session")
def sdss_query():
    """A PredictingStellarClass instance pointed at the default test coordinate."""
    return PredictingStellarClass(
        ra=TEST_RA,
        dec=TEST_DEC,
        radius_arcsec=TEST_RADIUS_ARCSEC,
        try_redshift=False,
    )


@pytest.fixture(scope="session")
def downloaded_sdss_data(sdss_query):
    """Downloaded SDSS DataFrame (fetched once per session)."""
    return sdss_query.data_call()


@pytest.fixture
def single_example():
    """A one-row DataFrame representing a single astronomical object."""
    return pd.DataFrame(
        {"u": [19.2], "g": [18.8], "r": [18.6], "i": [18.5], "z": [18.4]}
    )
