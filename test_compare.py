"""Command-line test for Phase 3.  Usage: python test_compare.py <csv> <target>"""
import sys
from automl.utils import load_csv, DataError
from automl.preprocess import prepare_data
from automl.trainer import compare_models, split_data, SELECTION_METRIC

if len(sys.argv) != 3:
    sys.exit("Usage: python test_compare.py <csv_path> <target_column>")

try:
    df = load_csv(sys.argv[1])
    X, y, task, notes = prepare_data(df, sys.argv[2])
except DataError as e:
    sys.exit(f"Data error: {e}")

print(f"Task detected: {task} | rows={len(X)} | features={X.shape[1]}")
for n in notes:
    print("Note:", n)

X_train, X_test, y_train, y_test = split_data(X, y, task)
results, _ = compare_models(X_train, y_train, task)
print(f"\nCross-validation results (train set only, sorted by {SELECTION_METRIC[task][0]}):")
print(results.to_string(index=False))
print(f"\nBest model by CV: {results.loc[0, 'model']}")