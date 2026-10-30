"""End-to-end AutoML: compare models, pick the best, test it, save it."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score,
                             root_mean_squared_error)

from automl.preprocess import prepare_data
from automl.trainer import SELECTION_METRIC, compare_models, split_data
from automl.utils import DataError

MODEL_PATH = Path("models/best_model.joblib")


def _feature_schema(X):
    """Describe each feature so the prediction form can be built from it."""
    schema = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            schema.append({"name": col, "type": "numeric",
                           "default": round(float(X[col].median()), 4)})
        else:
            cats = sorted(X[col].dropna().astype(str).unique().tolist())
            schema.append({"name": col, "type": "categorical", "options": cats[:50]})
    return schema


def evaluate(pipe, X_test, y_test, task):
    """Final evaluation on the held-out test set (used once)."""
    pred = pipe.predict(X_test)
    if task == "classification":
        kw = {"average": "weighted", "zero_division": 0}
        return {"accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, **kw),
                "recall": recall_score(y_test, pred, **kw),
                "f1": f1_score(y_test, pred, **kw)}
    return {"mae": mean_absolute_error(y_test, pred),
            "rmse": root_mean_squared_error(y_test, pred),
            "r2": r2_score(y_test, pred)}


def run_automl(df, target, model_path=MODEL_PATH):
    """Full workflow. Returns a summary dict and saves the model bundle."""
    X, y, task, notes = prepare_data(df, target)
    X_train, X_test, y_train, y_test = split_data(X, y, task)

    # 1) Compare candidates by cross-validation on TRAIN data only
    cv_results, pipelines = compare_models(X_train, y_train, task)
    best_name = cv_results.loc[0, "model"]
    metric = SELECTION_METRIC[task][0]

    # 2) Fit the winner on the training set, evaluate once on the test set
    best = pipelines[best_name]
    best.fit(X_train, y_train)
    test_metrics = {k: round(float(v), 4) for k, v in evaluate(best, X_test, y_test, task).items()}

    # 3) Save everything needed for prediction
    bundle = {"pipeline": best, "task": task, "target": target,
              "best_model": best_name, "selection_metric": metric,
              "features": _feature_schema(X), "test_metrics": test_metrics,
              "cv_results": cv_results.to_dict("records"), "notes": notes,
              "n_rows": len(X), "n_train": len(X_train), "n_test": len(X_test),
              "classes": best.classes_.tolist() if task == "classification" else None}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    try:                                    # tracking must never break training
        from automl.tracking import log_experiment
        log_experiment(bundle, model_path)
    except Exception as e:
        print(f"MLflow tracking skipped: {e}")
    return bundle


def load_bundle(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise DataError("No trained model found. Train a model first.")
    return joblib.load(model_path)


def predict_one(bundle, values):
    """values: {feature_name: raw string/number}. Returns the prediction."""
    row = {}
    for f in bundle["features"]:
        raw = values.get(f["name"], "")
        raw = raw.strip() if isinstance(raw, str) else raw
        if raw in ("", None):
            row[f["name"]] = np.nan          # imputer handles missing values
        elif f["type"] == "numeric":
            try:
                row[f["name"]] = float(raw)
            except ValueError:
                raise DataError(f"'{f['name']}' must be a number (got '{raw}').")
        else:
            row[f["name"]] = str(raw)
    pred = bundle["pipeline"].predict(pd.DataFrame([row]))[0]
    return pred.item() if hasattr(pred, "item") else pred