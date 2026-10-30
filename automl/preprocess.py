"""Data cleaning + a reusable preprocessing pipeline (no data leakage)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl.utils import DataError, detect_task


def prepare_data(df, target):
    """Validate the dataset and return (X, y, task, notes).
    Only structural cleaning happens here (nothing is learned from the data
    distribution), so it is safe to do before the train/test split."""
    notes = []
    if target not in df.columns:
        raise DataError(f"Target column '{target}' not found in the CSV.")

    df = df.dropna(subset=[target]).copy()
    if len(df) < 30:
        raise DataError("Need at least 30 rows with a non-missing target.")

    y = df[target]
    X = df.drop(columns=[target])
    task = detect_task(y)
    if y.nunique() < 2:
        raise DataError("Target column has only one unique value.")

    # Drop constant columns
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    # Drop ID-like text columns (almost every value unique)
    ids = [c for c in X.columns
           if not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 0.5 * len(X)]
    # Target leakage: features that are an exact copy of the target
    leak = [c for c in X.columns if X[c].astype(str).equals(y.astype(str))]
    for label, cols in [("constant", const), ("ID-like text", ids), ("identical to target (leakage)", leak)]:
        if cols:
            notes.append(f"Dropped {label} column(s): {', '.join(cols)}")
    X = X.drop(columns=set(const + ids + leak))
    if X.shape[1] == 0:
        raise DataError("No usable feature columns left after cleaning.")

    if task == "classification":
        counts = y.value_counts()
        if counts.min() < 2:
            raise DataError("Each class needs at least 2 rows. Remove rare classes.")
        if y.nunique() > 20:
            raise DataError("Target has more than 20 classes; choose a different target.")
    return X, y, task, notes


def build_preprocessor(X):
    """ColumnTransformer: impute + scale numbers, impute + one-hot categories."""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num_cols),
                              ("cat", categorical, cat_cols)])