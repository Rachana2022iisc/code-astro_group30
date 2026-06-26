"""
Tests for SDSS data downloading.
Requires an internet connection plus astropy and astroquery.
"""

import pandas as pd
import pytest

from StellaClass.predict import MAGNITUDE_COLUMNS


def test_sdss_returns_dataframe(downloaded_sdss_data):
    """SDSS result should be a pandas DataFrame."""
    assert isinstance(downloaded_sdss_data, pd.DataFrame), (
        "SDSS result is not a DataFrame."
    )


def test_sdss_not_empty(downloaded_sdss_data):
    """SDSS should return at least one object for the test coordinate."""
    assert not downloaded_sdss_data.empty, (
        "SDSS returned no objects for this coordinate."
    )


def test_sdss_has_magnitude_columns(downloaded_sdss_data):
    """Downloaded data must contain all required ugriz magnitude columns."""
    missing = [col for col in MAGNITUDE_COLUMNS if col not in downloaded_sdss_data.columns]
    assert not missing, f"Downloaded SDSS data are missing columns: {missing}"


def test_sdss_magnitudes_not_all_null(downloaded_sdss_data):
    """At least some ugriz values should be non-null."""
    assert downloaded_sdss_data[MAGNITUDE_COLUMNS].notna().any(axis=None), (
        "All downloaded ugriz values are missing."
    )
