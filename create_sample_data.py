
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

DATA_DIR = Path(__file__).parent / "sample_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Classification dataset
cancer = load_breast_cancer(as_frame=True)
cancer_df = cancer.frame
cancer_df.to_csv(DATA_DIR / "breast_cancer.csv", index=False)

# Regression dataset
diabetes = load_diabetes(as_frame=True)
diabetes_df = diabetes.frame
diabetes_df.to_csv(DATA_DIR / "diabetes.csv", index=False)

print("Sample datasets created successfully!")
print(f"Saved in: {DATA_DIR}")
print("1. breast_cancer.csv - Classification")
print("2. diabetes.csv - Regression")
