"""Exports scikit-learn's built-in datasets as CSV files for testing."""
from pathlib import Path
from sklearn.datasets import load_breast_cancer, load_diabetes

out = Path("sample_data")
out.mkdir(exist_ok=True)

load_breast_cancer(as_frame=True).frame.to_csv(out / "breast_cancer.csv", index=False)
load_diabetes(as_frame=True).frame.to_csv(out / "diabetes.csv", index=False)
print("Created: sample_data/breast_cancer.csv (target column: 'target')")
print("Created: sample_data/diabetes.csv      (target column: 'target')")