"""Helpers: CSV loading, validation, and task-type detection."""
import numpy as np
import pandas as pd


class DataError(Exception):
    """Raised for problems with the user's dataset (shown to the user)."""


def load_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataError(f"Could not read the CSV file: {e}")
    if df.empty or df.shape[1] < 2:
        raise DataError("The CSV must contain at least 2 columns and 1 row.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_task(y):
    """Return 'classification' or 'regression' from the target column."""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if pd.api.types.is_float_dtype(y) and not np.allclose(y, np.round(y)):
        return "regression"
    # Integer-valued numeric target: few distinct values -> classification
    return "classification" if y.nunique() <= 10 else "regression"