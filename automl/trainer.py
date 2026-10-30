"""Train several models and compare them with cross-validation (train set only)."""
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from automl.preprocess import build_preprocessor

RANDOM_STATE = 42
progress_hook = lambda msg, done, total: None  # replaced by automl/live.py

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}
REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

SCORING = {
    "classification": {"accuracy": "accuracy", "precision": "precision_weighted",
                       "recall": "recall_weighted", "f1": "f1_weighted"},
    "regression": {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
                   "r2": "r2"},
}
# Metric used to pick the best model, and whether higher is better
SELECTION_METRIC = {"classification": ("f1", True), "regression": ("rmse", False)}


def split_data(X, y, task, test_size=0.2):
    """Hold out a test set. It is NOT used for model selection."""
    strat = y if task == "classification" else None
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat)


def compare_models(X_train, y_train, task, cv_folds=5):
    """Cross-validate every candidate on the training data.
    Returns (results_df sorted best-first, dict of unfitted pipelines)."""
    models = CLASSIFIERS if task == "classification" else REGRESSORS
    if task == "classification":
        cv_folds = max(2, min(cv_folds, int(y_train.value_counts().min())))
        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)

    pipelines, rows = {}, []
    for i, (name, model) in enumerate(models.items()):
        progress_hook(f"Training & cross-validating {name}...", i, len(models))
    
        # Preprocessing lives inside the pipeline -> re-fit on each CV training fold
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", model)])
        pipelines[name] = pipe
        try:
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING[task])
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
        row = {"model": name}
        for metric in SCORING[task]:
            val = scores[f"test_{metric}"].mean()
            row[metric] = round(-val if metric in ("mae", "rmse") else val, 4)
        rows.append(row)

    if not rows:
        raise RuntimeError("All candidate models failed to train.")
    metric, higher_better = SELECTION_METRIC[task]
    results = pd.DataFrame(rows).sort_values(metric, ascending=not higher_better).reset_index(drop=True)
    return results, pipelines