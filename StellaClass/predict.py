#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run directly:
    python predict_sdss_stellar_class.py

On the first run:
    1. Put train.csv in the same directory as this Python file.
    2. The script trains the classifiers and saves stellar_class_model.joblib, or you can download this from zonedo using the link give in the documentation.
    3. It queries SDSS at the coordinates below (astro-query)
    4. It prints GALAXY/QSO/STAR confidence percentages.



You can either edit the USER SETTINGS section below or provide coordinates:
    python predict_sdss_stellar_class.py \
        --ra "10h05m54.678s" --dec "+31d22m27.476s" --radius 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from astroquery.sdss import SDSS
    import astropy.coordinates as coord
    import astropy.units as u
    ASTRONOMY_IMPORT_ERROR = None
except ImportError as error:
    SDSS = None
    coord = None
    u = None
    ASTRONOMY_IMPORT_ERROR = error

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_RA = "10h05m54.678s"
DEFAULT_DEC = "+31d22m27.476s"
DEFAULT_RADIUS_ARCSEC = 1.0

# train.csv should contain: u, g, r, i, z, class

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = SCRIPT_DIRECTORY / "train.csv"
DEFAULT_MODEL_FILE = SCRIPT_DIRECTORY / "stellar_class_model.joblib"
DEFAULT_OUTPUT_CSV = SCRIPT_DIRECTORY / "sdss_stellar_predictions.csv"

RETRAIN_MODEL = False

# If a spectrum exists, try to obtain its redshift and use the enhanced model.
TRY_SDSS_SPECTROSCOPIC_REDSHIFT = True

# Current astroquery SDSS default is DR17; this is written explicitly here.
SDSS_DATA_RELEASE = 17


#adding some more features

MAGNITUDE_COLUMNS = ["u", "g", "r", "i", "z"]
COLOUR_COLUMNS = [
    "u-g", "g-r", "r-i", "i-z",
    "u-r", "u-i", "u-z",
    "g-i", "g-z", "r-z",
]
PHOTO_FEATURES = MAGNITUDE_COLUMNS + COLOUR_COLUMNS
REDSHIFT_FEATURES = PHOTO_FEATURES + ["redshift", "log1p_redshift"]
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]

COLUMN_ALIASES = {
    "obj_id": "objid",
    "objectid": "objid",
    "object_id": "objid",
    "alpha": "ra",
    "delta": "dec",
    "specz": "redshift",
    "spec_z": "redshift",
    "zspec": "redshift",
    "z_spec": "redshift",
    "red_shift": "redshift",
}

CLASS_ALIASES = {
    "GAL": "GALAXY",
    "GALAXIES": "GALAXY",
    "QUASAR": "QSO",
    "QUASARS": "QSO",
    "QSOS": "QSO",
    "STARS": "STAR",
}


def canonical_name(name: Any) -> str:
    """ Convert a column name to the standard form used by this script

    Normalises spacing and casing, then maps known synonyms (for example
    'obj_id' or 'alpha') to the canonical column name used internally.

    Args:
        name (Any): Any. Column name to normalise. Will be converted to a string before processing.

    Returns:
        String. The lower-case, underscore-separated, alias-resolved column name.
    """
    clean = str(name).strip().lower().replace(" ", "_")
    return COLUMN_ALIASES.get(clean, clean)


def normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    """ Return a copy whose column names are lower-case and standardised

    Renames every column using canonical_name and raises an error if two
    columns end up sharing the same standardised name.

    Args:
        data (DataFrame): DataFrame. Table whose columns need to be standardised.

    Returns:
        DataFrame. Copy of data with renamed columns. Raises ValueError if renaming produces duplicate column names.
    """
    output = data.copy()
    output.columns = [canonical_name(column) for column in output.columns]

    if output.columns.duplicated().any():
        duplicates = output.columns[output.columns.duplicated()].tolist()
        raise ValueError(
            "Duplicate columns appeared after renaming: "
            f"{duplicates}. Please rename those columns."
        )
    return output


def to_dataframe(data: Any) -> pd.DataFrame:
    """ Convert a pandas table, Astropy Table, dictionary, or array to DataFrame

    Accepts several common input shapes and converts each into a DataFrame
    with u, g, r, i, z (and optionally redshift) columns so the rest of the
    pipeline can work with a single, consistent input type.

    Args:
        data (Any): Any. A pandas DataFrame/Series, an Astropy Table, a dictionary of scalar or array-like values, or a 1D/2D array-like ordered as u, g, r, i, z[, redshift].

    Returns:
        DataFrame. Standardised table representation of the input. Raises TypeError if data does not match any of the supported input shapes.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, pd.Series):
        return data.to_frame().T

    if hasattr(data, "to_pandas") and callable(data.to_pandas):
        return data.to_pandas()

    if isinstance(data, dict):
        if all(np.isscalar(value) or value is None for value in data.values()):
            return pd.DataFrame([data])
        return pd.DataFrame(data)

    array = np.asarray(data)
    if array.ndim == 1 and array.size in (5, 6):
        columns = MAGNITUDE_COLUMNS.copy()
        if array.size == 6:
            columns.append("redshift")
        return pd.DataFrame([array], columns=columns)

    if array.ndim == 2 and array.shape[1] in (5, 6):
        columns = MAGNITUDE_COLUMNS.copy()
        if array.shape[1] == 6:
            columns.append("redshift")
        return pd.DataFrame(array, columns=columns)

    raise TypeError(
        "Input must be a DataFrame, Astropy Table, dictionary, or a 5/6-column "
        "array ordered as u, g, r, i, z[, redshift]."
    )


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """ Clean the magnitudes and create SDSS colour features

    Standardises column names, coerces the ugriz magnitudes and redshift to
    numeric values, discards values outside plausible physical ranges, and
    derives the colour-index and log1p(redshift) features used by the model.

    Args:
        data (DataFrame): DataFrame. Table containing at least the u, g, r, i, z magnitude columns, optionally including a redshift column.

    Returns:
        DataFrame. Copy of data with cleaned magnitudes plus the added colour-index and log1p_redshift feature columns. Raises ValueError if one or more required magnitude columns are missing.
    """
    df = normalise_columns(data)

    missing = [column for column in MAGNITUDE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing photometric column(s): {missing}. "
            "Required columns are u, g, r, i, z."
        )

    for column in MAGNITUDE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[(df[column] < -10) | (df[column] > 50), column] = np.nan

    if "redshift" not in df.columns:
        df["redshift"] = np.nan
    else:
        df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
        df.loc[(df["redshift"] < 0) | (df["redshift"] > 20), "redshift"] = np.nan

    df["u-g"] = df["u"] - df["g"]
    df["g-r"] = df["g"] - df["r"]
    df["r-i"] = df["r"] - df["i"]
    df["i-z"] = df["i"] - df["z"]
    df["u-r"] = df["u"] - df["r"]
    df["u-i"] = df["u"] - df["i"]
    df["u-z"] = df["u"] - df["z"]
    df["g-i"] = df["g"] - df["i"]
    df["g-z"] = df["g"] - df["z"]
    df["r-z"] = df["r"] - df["z"]
    df["log1p_redshift"] = np.log1p(df["redshift"])

    return df


def normalise_target(values: pd.Series) -> pd.Series:
    """ Convert class labels to GALAXY, QSO, and STAR

    Supports a previously label-encoded target (0/1/2) as well as free-text
    labels, mapping known synonyms such as 'GAL' or 'QUASAR' onto the three
    canonical class names.

    Args:
        values (Series): Series. Class labels to normalise, either numeric codes (0=GALAXY, 1=QSO, 2=STAR) or string labels.

    Returns:
        Series. Class labels standardised to 'GALAXY', 'QSO', or 'STAR'.
    """
    # Support a previously LabelEncoded target: 0=GALAXY, 1=QSO, 2=STAR.
    numeric = pd.to_numeric(values, errors="coerce")
    finite_numeric = numeric.dropna()
    if len(finite_numeric) == len(values) and set(finite_numeric.unique()).issubset({0, 1, 2}):
        return numeric.astype(int).map({0: "GALAXY", 1: "QSO", 2: "STAR"})

    labels = values.astype(str).str.strip().str.upper()
    return labels.replace(CLASS_ALIASES)


def make_random_forest(random_state: int = 42) -> Pipeline:
    """ Create the classification pipeline used for both feature sets

    Builds a scikit-learn Pipeline that first imputes missing values with the
    median, then classifies with a tuned RandomForestClassifier.

    Args:
        random_state (int): Integer. Seed used to make the forest's training reproducible. Defaults to 42.

    Returns:
        Pipeline. Unfitted scikit-learn pipeline ready to be trained with .fit().
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=22,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


class StellarClassPredictor:
    """ Train, save, load, and apply the GALAXY/QSO/STAR classifiers

    Wraps two RandomForest pipelines: one trained on photometry and colours
    alone, and one trained on photometry, colours, and redshift, used
    whenever redshift is available for a given object.

    Attributes:
        random_state (int): Integer. Seed used for training and validation splits.
        photo_model (Pipeline): Pipeline. Fitted model using only photometric features. None until trained.
        redshift_model (Pipeline): Pipeline. Fitted model that also uses redshift. None until trained, or if redshift data was insufficient.
        training_summary (dict): Dictionary. Metrics and row counts recorded after the most recent fit_csv call.
    """

    def __init__(self, random_state: int = 42):
        """ Create an untrained predictor

        Args:
            random_state (int): Integer. Seed used for training and validation splits. Defaults to 42.

        Returns:
            None.
        """
        self.random_state = random_state
        self.photo_model: Optional[Pipeline] = None
        self.redshift_model: Optional[Pipeline] = None
        self.training_summary: dict[str, Any] = {}

    @staticmethod
    def _validation_report(
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        random_state: int,
    ) -> dict[str, float]:
        """ Evaluate a fresh model on a held-out validation sample

        Splits the data, trains a throwaway RandomForest pipeline on 80% of
        it, and prints an accuracy/F1 report on the remaining 20%. Skipped
        when there are too few rows or too few examples of any class.

        Args:
            X (DataFrame): DataFrame. Feature matrix to validate on.
            y (Series): Series. Class labels matching the rows of X.
            model_name (string): String. Label used in the printed validation report.
            random_state (int): Integer. Seed used for the train/validation split and the throwaway model.

        Returns:
            Dictionary. Contains 'accuracy' and 'macro_f1' as floats, or NaN for both if validation was skipped.
        """
        class_counts = y.value_counts()
        if len(y) < 20 or class_counts.min() < 3:
            print(f"\nSkipping validation report for {model_name}: too few rows.")
            return {"accuracy": np.nan, "macro_f1": np.nan}

        X_train, X_validate, y_train, y_validate = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=random_state,
            stratify=y,
        )

        validation_model = make_random_forest(random_state=random_state)
        validation_model.fit(X_train, y_train)
        prediction = validation_model.predict(X_validate)

        accuracy = accuracy_score(y_validate, prediction)
        macro_f1 = f1_score(y_validate, prediction, average="macro")

        print("\n" + "=" * 72)
        print(f"VALIDATION: {model_name}")
        print("=" * 72)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Macro F1 : {macro_f1:.4f}")
        print(
            classification_report(
                y_validate,
                prediction,
                labels=CLASS_NAMES,
                zero_division=0,
            )
        )

        return {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}

    def fit_csv(self, train_csv: str | Path, target: str = "class") -> "StellarClassPredictor":
        """ Train the classifiers from a labelled CSV file

        Reads and cleans the CSV, fits the photometry-only model on all
        valid rows, and additionally fits a photometry+redshift model
        whenever enough rows with finite redshift and all three classes
        are available.

        Args:
            train_csv (string or Path): String or Path. Path to a CSV file containing u, g, r, i, z, and a class column (optionally also a redshift column).
            target (string): String. Name of the column holding the class label. Defaults to 'class'.

        Returns:
            StellarClassPredictor. This instance, now fitted, to allow method chaining. Raises FileNotFoundError if train_csv does not exist, or ValueError if the target column is missing, no valid rows remain after cleaning, or not all three classes are present.
        """
        train_path = Path(train_csv).expanduser().resolve()
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {train_path}\n"
                "Put train.csv in the same directory as this script, or use "
                "--train /full/path/to/train.csv"
            )

        raw = pd.read_csv(train_path)
        raw = normalise_columns(raw)
        target = canonical_name(target)

        if target not in raw.columns:
            raise ValueError(
                f"Target column '{target}' was not found. "
                f"Available columns: {raw.columns.tolist()}"
            )

        data = add_features(raw)
        data[target] = normalise_target(data[target])
        data = data[data[target].isin(CLASS_NAMES)].copy()

        # Only use reliable rows to define the training distribution.
        finite_magnitudes = np.isfinite(data[MAGNITUDE_COLUMNS]).all(axis=1)
        plausible_magnitudes = (
            (data[MAGNITUDE_COLUMNS] >= 5) &
            (data[MAGNITUDE_COLUMNS] <= 40)
        ).all(axis=1)
        data = data[finite_magnitudes & plausible_magnitudes].copy()

        if data.empty:
            raise ValueError("No valid training rows remain after cleaning.")

        missing_classes = sorted(set(CLASS_NAMES) - set(data[target].unique()))
        if missing_classes:
            raise ValueError(
                "The training data must contain GALAXY, QSO, and STAR. "
                f"Missing classes: {missing_classes}"
            )

        y = data[target]

        photo_metrics = self._validation_report(
            data[PHOTO_FEATURES],
            y,
            model_name="u, g, r, i, z + colours",
            random_state=self.random_state,
        )
        self.photo_model = make_random_forest(self.random_state)
        self.photo_model.fit(data[PHOTO_FEATURES], y)

        redshift_rows = data[np.isfinite(data["redshift"])].copy()
        redshift_metrics = None

        if (
            not redshift_rows.empty
            and set(redshift_rows[target].unique()) == set(CLASS_NAMES)
        ):
            redshift_metrics = self._validation_report(
                redshift_rows[REDSHIFT_FEATURES],
                redshift_rows[target],
                model_name="u, g, r, i, z + colours + redshift",
                random_state=self.random_state,
            )
            self.redshift_model = make_random_forest(self.random_state)
            self.redshift_model.fit(
                redshift_rows[REDSHIFT_FEATURES], redshift_rows[target]
            )
        else:
            self.redshift_model = None
            print(
                "\nA redshift model was not trained because the training CSV "
                "does not have finite redshifts for all three classes."
            )

        self.training_summary = {
            "n_training_rows": int(len(data)),
            "class_counts": data[target].value_counts().to_dict(),
            "n_redshift_rows": int(len(redshift_rows)),
            "photo_validation": photo_metrics,
            "redshift_validation": redshift_metrics,
        }
        return self

    def save(self, model_file: str | Path) -> Path:
        """ Save fitted sklearn objects without pickling this custom class

        Bundles the photo model, redshift model, random state, and training
        summary into a dictionary and writes it with joblib, so the saved
        file does not depend on this class's source code to reload.

        Args:
            model_file (string or Path): String or Path. Destination path for the saved .joblib file. Parent directories are created if needed.

        Returns:
            Path. Resolved path of the saved model file. Raises RuntimeError if called before the predictor has been trained.
        """
        if self.photo_model is None:
            raise RuntimeError("The model must be trained before it can be saved.")

        path = Path(model_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format_version": 1,
            "random_state": self.random_state,
            "photo_model": self.photo_model,
            "redshift_model": self.redshift_model,
            "training_summary": self.training_summary,
        }
        joblib.dump(bundle, path)
        return path

    @classmethod
    def load(cls, model_file: str | Path) -> "StellarClassPredictor":
        """ Load a previously trained model bundle

        Args:
            model_file (string or Path): String or Path. Path to a .joblib file previously produced by save().

        Returns:
            StellarClassPredictor. New instance populated with the saved photo model, redshift model, random state, and training summary. Raises TypeError if the loaded file is not a model bundle produced by this script.
        """
        path = Path(model_file).expanduser().resolve()
        bundle = joblib.load(path)

        if not isinstance(bundle, dict) or "photo_model" not in bundle:
            raise TypeError(
                f"{path} is not a model bundle created by this script. "
                "Delete it and run with --retrain."
            )

        predictor = cls(random_state=int(bundle.get("random_state", 42)))
        predictor.photo_model = bundle["photo_model"]
        predictor.redshift_model = bundle.get("redshift_model")
        predictor.training_summary = bundle.get("training_summary", {})
        return predictor

    @staticmethod
    def _place_probabilities(
        model: Pipeline,
        X: pd.DataFrame,
        indices: pd.Index,
        probability_table: pd.DataFrame,
    ) -> None:
        """ Write a model's per-class probabilities into a shared probability table

        Args:
            model (Pipeline): Pipeline. Fitted classifier exposing predict_proba and classes_.
            X (DataFrame): DataFrame. Feature rows to score with model.
            indices (Index): Index. Row labels of probability_table that correspond to X.
            probability_table (DataFrame): DataFrame. Table of per-class probabilities, modified in place.

        Returns:
            None. probability_table is updated in place.
        """
        probabilities = model.predict_proba(X)
        model_classes = [str(value).upper() for value in model.classes_]

        for class_index, class_name in enumerate(model_classes):
            if class_name in probability_table.columns:
                probability_table.loc[indices, class_name] = probabilities[:, class_index]

    def predict(self, objects: Any) -> pd.DataFrame:
        """ Predict class and percentage confidence for one or more objects

        Cleans the input, chooses the redshift model for rows with a finite
        redshift (when available) and the photometry-only model otherwise,
        then attaches the predicted class, per-class confidence percentages,
        which model was used, and a warning about any imputed magnitudes.

        Args:
            objects (Any): Any. One or more objects to classify, in any form accepted by to_dataframe (DataFrame, Astropy Table, dictionary, or array).

        Returns:
            DataFrame. Copy of the input with added columns: predicted_class, confidence_%, P_GALAXY_%, P_QSO_%, P_STAR_%, model_used, and input_warning. Raises RuntimeError if called before training or if a row receives invalid probabilities, or ValueError if there are no rows to predict or a row has no usable ugriz magnitudes.
        """
        if self.photo_model is None:
            raise RuntimeError("No fitted model is available.")

        original = normalise_columns(to_dataframe(objects))
        data = add_features(original)

        if data.empty:
            raise ValueError("There are no rows to predict.")

        usable_count = np.isfinite(data[MAGNITUDE_COLUMNS]).sum(axis=1)
        if (usable_count == 0).any():
            bad_rows = data.index[usable_count == 0].tolist()
            raise ValueError(f"No usable ugriz magnitudes in rows: {bad_rows}")

        probabilities = pd.DataFrame(
            0.0,
            index=data.index,
            columns=CLASS_NAMES,
        )
        model_used = pd.Series(index=data.index, dtype="object")

        use_redshift = np.isfinite(data["redshift"]) & (self.redshift_model is not None)
        use_photo = ~use_redshift

        if use_photo.any():
            indices = data.index[use_photo]
            self._place_probabilities(
                self.photo_model,
                data.loc[indices, PHOTO_FEATURES],
                indices,
                probabilities,
            )
            model_used.loc[indices] = "photometry"

        if use_redshift.any():
            indices = data.index[use_redshift]
            self._place_probabilities(
                self.redshift_model,
                data.loc[indices, REDSHIFT_FEATURES],
                indices,
                probabilities,
            )
            model_used.loc[indices] = "photometry+redshift"

        row_sums = probabilities.sum(axis=1)
        if (row_sums <= 0).any():
            raise RuntimeError("The classifier returned invalid class probabilities.")
        probabilities = probabilities.div(row_sums, axis=0)

        result = original.copy()
        result["predicted_class"] = probabilities.idxmax(axis=1)
        result["confidence_%"] = (100 * probabilities.max(axis=1)).round(2)
        result["P_GALAXY_%"] = (100 * probabilities["GALAXY"]).round(2)
        result["P_QSO_%"] = (100 * probabilities["QSO"]).round(2)
        result["P_STAR_%"] = (100 * probabilities["STAR"]).round(2)
        result["model_used"] = model_used

        warning = pd.Series("", index=result.index, dtype="object")
        missing_magnitudes = 5 - usable_count
        warning.loc[missing_magnitudes > 0] = (
            missing_magnitudes.loc[missing_magnitudes > 0].astype(str)
            + " missing magnitude(s) imputed"
        )
        result["input_warning"] = warning

        return result


def require_astronomy_packages() -> None:
    """ Raise a readable error if Astropy/Astroquery are unavailable

    Returns:
        None. Returns silently if both packages imported successfully. Raises ImportError if astropy or astroquery could not be imported, with installation instructions.
    """
    if ASTRONOMY_IMPORT_ERROR is not None:
        raise ImportError(
            "The SDSS query requires astropy and astroquery. Install them with:\n"
            "pip install astropy astroquery"
        ) from ASTRONOMY_IMPORT_ERROR


def parse_sky_position(ra: str | float, dec: str | float) -> coord.SkyCoord:
    """ Accept sexagesimal strings or decimal-degree coordinates

    Tries, in order: numeric decimal degrees, sexagesimal strings such as
    '10h05m54.678s', space-separated sexagesimal strings, and finally
    plain numeric strings, returning as soon as one interpretation succeeds.

    Args:
        ra (string or float): String or float. Right ascension, e.g. '10h05m54.678s' or a decimal-degree value.
        dec (string or float): String or float. Declination, e.g. '+31d22m27.476s' or a decimal-degree value.

    Returns:
        SkyCoord. Astropy sky coordinate in the ICRS frame. Raises ValueError if none of the supported coordinate formats could be parsed.
    """
    # Numeric values mean decimal degrees.
    if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
        return coord.SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")

    ra_text = str(ra).strip()
    dec_text = str(dec).strip()

    try:
        # Works for strings such as 10h05m54.678s and +31d22m27.476s.
        return coord.SkyCoord(ra_text, dec_text, frame="icrs")
    except Exception:
        pass

    try:
        # Works for strings such as "10 05 54.678" and "+31 22 27.476".
        return coord.SkyCoord(
            ra_text,
            dec_text,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )
    except Exception:
        pass

    try:
        # Finally interpret plain numeric strings as decimal degrees.
        return coord.SkyCoord(
            ra=float(ra_text) * u.deg,
            dec=float(dec_text) * u.deg,
            frame="icrs",
        )
    except Exception as error:
        raise ValueError(
            "Could not understand the coordinates. Examples: "
            "RA='10h05m54.678s', DEC='+31d22m27.476s', or decimal degrees."
        ) from error


class PredictingStellarClass:
    """ Query an SDSS coordinate and immediately classify every match

    Wraps an SDSS cone search at a given position and radius, optionally
    attaching spectroscopic redshift, so the resulting table can be passed
    straight to a fitted StellarClassPredictor.

    Attributes:
        ra (string or float): String or float. Right ascension of the search position.
        dec (string or float): String or float. Declination of the search position.
        radius_arcsec (float): Float. Search radius in arcseconds.
        data_release (int): Integer. SDSS data release number to query.
        try_redshift (bool): Boolean. Whether to additionally try to fetch a spectroscopic redshift.
        position (SkyCoord): SkyCoord. Parsed sky coordinate used for the SDSS query.
    """

    def __init__(
        self,
        ra: str | float,
        dec: str | float,
        radius_arcsec: float = 1.0,
        data_release: int = SDSS_DATA_RELEASE,
        try_redshift: bool = True,
    ) -> None:
        """ Create a query for a single sky position

        Args:
            ra (string or float): String or float. Right ascension, sexagesimal or decimal degrees.
            dec (string or float): String or float. Declination, sexagesimal or decimal degrees.
            radius_arcsec (float): Float. SDSS cone-search radius in arcseconds. Defaults to 1.0.
            data_release (int): Integer. SDSS data release to query. Defaults to SDSS_DATA_RELEASE.
            try_redshift (bool): Boolean. Whether to also attempt fetching a spectroscopic redshift. Defaults to True.

        Returns:
            None. Raises ImportError if astropy/astroquery are not installed, or ValueError if ra/dec cannot be parsed into a sky position.
        """
        require_astronomy_packages()
        self.ra = ra
        self.dec = dec
        self.radius_arcsec = float(radius_arcsec)
        self.data_release = int(data_release)
        self.try_redshift = bool(try_redshift)
        self.position = parse_sky_position(ra, dec)

    def data_call(self) -> pd.DataFrame:
        """ Download objID, coordinates, and ugriz photometry from SDSS

        Runs an SDSS cone search around self.position, adds each match's
        angular separation from the requested position, sorts by that
        separation, and optionally attaches a spectroscopic redshift.

        Returns:
            DataFrame. Photometric matches with objid, ra, dec, u, g, r, i, z, separation_arcsec, and (if requested) redshift, sorted by separation. Raises LookupError if no SDSS photometric object is found within the search radius.
        """
        radius = self.radius_arcsec * u.arcsec

        xid = SDSS.query_region(
            self.position,
            radius=radius,
            photoobj_fields=["objid", "ra", "dec", "u", "g", "r", "i", "z"],
            spectro=False,
            data_release=self.data_release,
        )

        if xid is None or len(xid) == 0:
            raise LookupError(
                f"No SDSS photometric object was found within "
                f"{self.radius_arcsec:g} arcsec."
            )

        photometry = normalise_columns(xid.to_pandas())

        # Add distance of every returned object from the requested coordinate.
        object_positions = coord.SkyCoord(
            ra=pd.to_numeric(photometry["ra"], errors="coerce").to_numpy() * u.deg,
            dec=pd.to_numeric(photometry["dec"], errors="coerce").to_numpy() * u.deg,
            frame="icrs",
        )
        photometry["separation_arcsec"] = self.position.separation(
            object_positions
        ).arcsec
        photometry = photometry.sort_values("separation_arcsec").reset_index(drop=True)

        if self.try_redshift:
            photometry = self._attach_spectroscopic_redshift(photometry)

        return photometry

    def _attach_spectroscopic_redshift(self, photometry: pd.DataFrame) -> pd.DataFrame:
        """ Attach SDSS spectroscopic z as a separate column named redshift

        Queries SDSS for spectroscopic matches, joins them to the
        photometry table first by exact objid and then, for any rows still
        missing a redshift, by nearest sky position within the search radius.

        Args:
            photometry (DataFrame): DataFrame. Photometric table to attach a redshift column to. Must include 'objid', 'ra', and 'dec' for matching.

        Returns:
            DataFrame. Copy of photometry with a redshift column, populated with NaN for any rows with no spectroscopic match.
        """
        output = photometry.copy()
        output["redshift"] = np.nan

        try:
            spectral_match = SDSS.query_region(
                self.position,
                radius=self.radius_arcsec * u.arcsec,
                photoobj_fields=["objid", "ra", "dec"],
                specobj_fields=["z"],
                spectro=True,
                data_release=self.data_release,
            )
        except Exception as error:
            print(f"Warning: SDSS redshift query failed: {error}")
            return output

        if spectral_match is None or len(spectral_match) == 0:
            return output

        spectrum = normalise_columns(spectral_match.to_pandas())

        # The SpecObj redshift is normally returned as 'z'.
        redshift_column = None
        for candidate in ("redshift", "z", "specz", "spec_z"):
            if candidate in spectrum.columns:
                redshift_column = candidate
                break

        if redshift_column is None:
            print(
                "Warning: a spectroscopic match was found, but no redshift "
                f"column was returned. Columns: {spectrum.columns.tolist()}"
            )
            return output

        spectrum["redshift"] = pd.to_numeric(
            spectrum[redshift_column], errors="coerce"
        )
        spectrum = spectrum[np.isfinite(spectrum["redshift"])].copy()
        if spectrum.empty:
            return output

        # First try the exact SDSS object identifier.
        if "objid" in output.columns and "objid" in spectrum.columns:
            object_to_redshift = (
                spectrum.drop_duplicates(subset="objid")
                .set_index("objid")["redshift"]
            )
            output["redshift"] = output["objid"].map(object_to_redshift)

        # For any unmatched rows, use the closest spectroscopic position.
        still_missing = ~np.isfinite(pd.to_numeric(output["redshift"], errors="coerce"))
        required_coordinates = {"ra", "dec"}
        if still_missing.any() and required_coordinates.issubset(spectrum.columns):
            photo_positions = coord.SkyCoord(
                ra=output.loc[still_missing, "ra"].to_numpy(dtype=float) * u.deg,
                dec=output.loc[still_missing, "dec"].to_numpy(dtype=float) * u.deg,
                frame="icrs",
            )
            spec_positions = coord.SkyCoord(
                ra=spectrum["ra"].to_numpy(dtype=float) * u.deg,
                dec=spectrum["dec"].to_numpy(dtype=float) * u.deg,
                frame="icrs",
            )
            nearest_index, separation, _ = photo_positions.match_to_catalog_sky(
                spec_positions
            )

            missing_indices = output.index[still_missing]
            for row_index, spec_index, sep in zip(
                missing_indices, nearest_index, separation.arcsec
            ):
                if sep <= self.radius_arcsec:
                    output.loc[row_index, "redshift"] = spectrum.iloc[
                        int(spec_index)
                    ]["redshift"]

        return output

    def predict(self, predictor: StellarClassPredictor) -> pd.DataFrame:
        """ Download the SDSS data and apply the fitted model

        Args:
            predictor (StellarClassPredictor): StellarClassPredictor. Fitted predictor used to classify the downloaded SDSS objects.

        Returns:
            DataFrame. SDSS photometric matches with added prediction columns, as returned by StellarClassPredictor.predict.
        """
        sdss_data = self.data_call()
        return predictor.predict(sdss_data)


def load_or_train_predictor(
    train_csv: str | Path,
    model_file: str | Path,
    retrain: bool = False,
) -> StellarClassPredictor:
    """ Load a saved model, or train and save one when necessary

    Args:
        train_csv (string or Path): String or Path. Path to the labelled training CSV, used only if a new model needs to be trained.
        model_file (string or Path): String or Path. Path to the saved .joblib model file.
        retrain (bool): Boolean. If True, train a fresh model even when a saved one already exists at model_file. Defaults to False.

    Returns:
        StellarClassPredictor. A predictor that is either freshly loaded from disk or newly trained and saved.
    """
    model_path = Path(model_file).expanduser().resolve()

    if model_path.exists() and not retrain:
        print(f"Loading saved model: {model_path}")
        return StellarClassPredictor.load(model_path)

    print(f"Training from: {Path(train_csv).expanduser().resolve()}")
    predictor = StellarClassPredictor(random_state=42)
    predictor.fit_csv(train_csv)
    saved_path = predictor.save(model_path)
    print(f"Saved trained model: {saved_path}")
    return predictor



def predict_sdss_coordinates(
    ra: str | float,
    dec: str | float,
    radius_arcsec: float = 1.0,
    model_file: str | Path = DEFAULT_MODEL_FILE,
    try_redshift: bool = True,
    output_csv: str | Path | None = None,
    show_result: bool = True,
) -> pd.DataFrame:
    """
    Load an already-trained model and directly classify SDSS objects.

    This function never trains the model. The saved ``.joblib`` model must
    already exist. It is intended for use from a Jupyter notebook::

        from predict_sdss_stellar_class import predict_sdss_coordinates

        result = predict_sdss_coordinates(
            ra="10h05m54.678s",
            dec="+31d22m27.476s",
            radius_arcsec=1.0,
        )

    Args:
        ra (string or float): String or float. Right ascension of the search position, sexagesimal or decimal degrees.
        dec (string or float): String or float. Declination of the search position, sexagesimal or decimal degrees.
        radius_arcsec (float): Float. SDSS cone-search radius in arcseconds. Defaults to 1.0.
        model_file (string or Path): String or Path. Path to a previously saved .joblib model. Defaults to DEFAULT_MODEL_FILE.
        try_redshift (bool): Boolean. Whether to also attempt fetching a spectroscopic redshift. Defaults to True.
        output_csv (string or Path): String or Path. If given, the full prediction table is also written to this CSV path. Defaults to None.
        show_result (bool): Boolean. Whether to print a formatted prediction summary. Defaults to True.

    Returns:
        DataFrame. SDSS photometric matches with predicted_class, confidence_%, per-class probability, model_used, and input_warning columns. Raises FileNotFoundError if the model file does not exist, or LookupError if no SDSS photometric object is found within the search radius.
    """
    model_path = Path(model_file).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "Train and save the model once before using this prediction function."
        )

    predictor = StellarClassPredictor.load(model_path)
    sdss_predictor = PredictingStellarClass(
        ra=ra,
        dec=dec,
        radius_arcsec=radius_arcsec,
        data_release=SDSS_DATA_RELEASE,
        try_redshift=try_redshift,
    )
    result = sdss_predictor.predict(predictor)

    if show_result:
        print_prediction_summary(result)

    if output_csv is not None:
        output_path = Path(output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        if show_result:
            print(f"\nSaved full prediction table to: {output_path}")

    return result

def print_prediction_summary(result: pd.DataFrame) -> None:
    """ Print the most useful columns without truncating the probabilities

    Args:
        result (DataFrame): DataFrame. Prediction table as returned by StellarClassPredictor.predict, expected to include predicted_class and the P_*_% confidence columns.

    Returns:
        None. Output is printed to standard output.
    """
    preferred_columns = [
        "objid",
        "ra",
        "dec",
        "separation_arcsec",
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
        "predicted_class",
        "confidence_%",
        "P_GALAXY_%",
        "P_QSO_%",
        "P_STAR_%",
        "model_used",
    ]
    shown_columns = [column for column in preferred_columns if column in result.columns]

    print("\n" + "=" * 100)
    print("SDSS STELLAR-CLASS PREDICTION")
    print("=" * 100)
    print(result[shown_columns].to_string(index=False))

    print("\nInterpretation:")
    for row_number, row in result.reset_index(drop=True).iterrows():
        print(
            f"  Match {row_number + 1}: {row['predicted_class']} "
            f"with {row['confidence_%']:.2f}% model confidence "
            f"(GALAXY={row['P_GALAXY_%']:.2f}%, "
            f"QSO={row['P_QSO_%']:.2f}%, "
            f"STAR={row['P_STAR_%']:.2f}%)."
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """ Build the command-line argument parser for this script

    Returns:
        ArgumentParser. Parser configured with --ra, --dec, --radius, --train, --model, --output, --retrain, and --no-redshift options.
    """
    parser = argparse.ArgumentParser(
        description="Query SDSS and predict GALAXY, QSO, or STAR in one run."
    )
    parser.add_argument("--ra", default=DEFAULT_RA, help="RA in HMS or decimal degrees")
    parser.add_argument("--dec", default=DEFAULT_DEC, help="Dec in DMS or decimal degrees")
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS_ARCSEC,
        help="SDSS search radius in arcseconds",
    )
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_FILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train again even when a saved model exists",
    )
    parser.add_argument(
        "--no-redshift",
        action="store_true",
        help="Do not try the additional SDSS spectroscopic-redshift query",
    )
    return parser


def main() -> int:
    """ Run the end-to-end command-line workflow

    Loads or trains the model according to the CLI arguments, queries SDSS
    at the requested coordinates, prints the prediction summary, and writes
    the full results to a CSV file.

    Returns:
        Integer. Exit code 0 on success.
    """
    args = build_argument_parser().parse_args()

    predictor = load_or_train_predictor(
        train_csv=args.train,
        model_file=args.model,
        retrain=bool(args.retrain or RETRAIN_MODEL),
    )

    sdss_predictor = PredictingStellarClass(
        ra=args.ra,
        dec=args.dec,
        radius_arcsec=args.radius,
        data_release=SDSS_DATA_RELEASE,
        try_redshift=(
            TRY_SDSS_SPECTROSCOPIC_REDSHIFT and not args.no_redshift
        ),
    )

    print(
        f"\nQuerying SDSS DR{SDSS_DATA_RELEASE}: "
        f"RA={args.ra}, Dec={args.dec}, radius={args.radius:g} arcsec"
    )
    result = sdss_predictor.predict(predictor)
    print_prediction_summary(result)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved full prediction table to: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
    except Exception as error:
        print("\nERROR")
        print("-" * 72)
        print(error)
        print("\nRequired packages:")
        print("pip install numpy pandas scikit-learn joblib astropy astroquery")
        raise SystemExit(1)
