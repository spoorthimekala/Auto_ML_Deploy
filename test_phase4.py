"""Phase 4 test: python test_phase4.py <csv> <target>"""
import sys
from automl.utils import load_csv, DataError
from automl.pipeline import run_automl, load_bundle, predict_one

if len(sys.argv) != 3:
    sys.exit("Usage: python test_phase4.py <csv_path> <target_column>")
try:
    b = run_automl(load_csv(sys.argv[1]), sys.argv[2])
except DataError as e:
    sys.exit(f"Data error: {e}")

print(f"Task: {b['task']} | Best model: {b['best_model']} (selected by CV {b['selection_metric']})")
print(f"Split: train={b['n_train']} test={b['n_test']}")
print("Test-set metrics:", b["test_metrics"])
print("Saved: models/best_model.joblib")

# Reload from disk and predict using default feature values
loaded = load_bundle()
sample = {f["name"]: f.get("default", f.get("options", [""])[0]) for f in loaded["features"]}
print("Reloaded model prediction on default inputs:", predict_one(loaded, sample))