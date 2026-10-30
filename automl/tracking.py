"""MLflow experiment tracking (local SQLite database, no server needed)."""
import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "AutoML-Deploy"


def log_experiment(bundle, model_path):
    """One MLflow run per candidate model; the winner also gets test metrics + the model file."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    for row in bundle["cv_results"]:
        is_best = row["model"] == bundle["best_model"]
        with mlflow.start_run(run_name=row["model"] + (" (BEST)" if is_best else "")):
            mlflow.log_params({
                "model": row["model"], "task": bundle["task"], "target": bundle["target"],
                "selection_metric": bundle["selection_metric"],
                "n_train": bundle["n_train"], "n_test": bundle["n_test"]})
            mlflow.log_metrics({f"cv_{k}": v for k, v in row.items() if k != "model"})
            mlflow.set_tag("selected_best", str(is_best))
            if is_best:
                mlflow.log_metrics({f"test_{k}": v for k, v in bundle["test_metrics"].items()})
                mlflow.log_artifact(str(model_path))