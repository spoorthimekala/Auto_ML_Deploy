# AutoML Deploy: Intelligent Model Selection and Deployment System

A web app that takes any CSV file, automatically cleans it, trains several machine learning models,
picks the best one using a leakage-free evaluation, and serves live predictions in the browser.

**Live demo:** https://YOUR-APP.onrender.com (free hosting; the first load may take about a minute)

## Features
- CSV upload with validation (type, size, structure)
- Automatic dataset inspection: rows, columns, missing values, detected task
- Automatic preprocessing: missing-value imputation, scaling, one-hot encoding
- Automatic classification vs. regression detection (user can pick any target column)
- Trains and compares 4 models per task using 5-fold cross-validation
- Best model chosen by a fixed metric, then tested once on held-out data
- Dashboard with test metrics, model-comparison bars and table
- Live predictions that update as you type, with class probabilities
- Download of the trained model (`.joblib`)
- MLflow experiment tracking (optional, local)
- Private session per visitor, so many people can use the deployed app at once

## Architecture
```
Browser (HTML/CSS/JS)
   |  upload CSV, choose target, live predict
   v
Flask app (app.py)  -- per-visitor session: uploads/<id>.csv, models/<id>.joblib
   |
   +-- automl/utils.py       CSV loading, task detection, DataError
   +-- automl/preprocess.py  validation, cleaning, ColumnTransformer
   +-- automl/trainer.py     train/test split, candidate models, cross-validation
   +-- automl/pipeline.py    select best -> test once -> save with Joblib -> predict
   +-- automl/live.py        background training thread, live progress, probabilities
   +-- automl/tracking.py    MLflow logging (skipped automatically if not installed)
```

### Evaluation procedure (no data leakage)
1. Split data 80% train / 20% test (stratified for classification).
2. Preprocessing lives **inside** a scikit-learn `Pipeline`, so it is fitted only on training folds.
3. Candidate models are compared with 5-fold cross-validation **on the training set only**.
4. The winner is chosen by **F1 (weighted)** for classification or **RMSE** for regression.
5. The winner is scored **once** on the untouched test set.
6. The saved `.joblib` file contains the full pipeline, so prediction uses identical preprocessing.

| Task | Models compared | Selection metric | Reported metrics |
|---|---|---|---|
| Classification | Logistic Regression, Random Forest, Gradient Boosting, KNN | F1 (weighted) | Accuracy, Precision, Recall, F1 |
| Regression | Linear Regression, Ridge, Random Forest, Gradient Boosting | RMSE (lower is better) | MAE, RMSE, R² |

### Assumptions
- The CSV has a header row, and one column is the target.
- A text target, or an integer target with 10 or fewer distinct values, means classification. Otherwise regression.
- Constant columns, ID-like text columns and columns identical to the target are dropped automatically.
- At least 30 rows; classification targets need 2 to 20 classes, with at least 2 rows each.

## Installation (Windows)
Requires Python 3.10 or newer.
```
git clone https://github.com/spoorthimekala/Auto_ML_Deploy.git
cd Auto_ML_Deploy
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```
python app.py
```
Open http://127.0.0.1:5000, then:
1. Upload a CSV (try `sample_data/breast_cancer.csv` or `sample_data/diabetes.csv`; target column: `target`).
2. Choose the target column and click **Train & Compare Models**.
3. View the dashboard, then open **Predict** and change values to see live predictions.

### Optional: MLflow tracking
```
pip install mlflow
python app.py                      # train a model; runs are logged to mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Open http://127.0.0.1:5001 and select the **AutoML-Deploy** experiment to compare runs.

### Run the tests
```
python test_workflow.py
```
Expected: `25/25 tests passed`.

## Deployment (Render, free plan)
| Setting | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app --workers 1 --threads 4 --timeout 120` |
| Environment | `PYTHON_VERSION=3.11.9`, `SECRET_KEY=<random text>` |

## Results on the sample datasets
| Dataset | Best model (by CV) | Test result |
|---|---|---|
| Breast Cancer (classification) | Logistic Regression | Accuracy and F1 about 0.98 |
| Diabetes (regression) | Ridge | RMSE about 53.8, R² about 0.45 |

## Screenshots (add yours in `docs/screenshots/`)
| Page | File to add |
|---|---|
| Upload page | `docs/screenshots/upload.png` |
| Dataset overview and target selection | `docs/screenshots/configure.png` |
| Live training progress | `docs/screenshots/training.png` |
| Dashboard | `docs/screenshots/dashboard.png` |
| Live prediction | `docs/screenshots/predict.png` |
| MLflow comparison | `docs/screenshots/mlflow.png` |

Then show them here with `![Dashboard](docs/screenshots/dashboard.png)`.

## Limitations and future work
- No hyperparameter tuning, and only 4 models per task
- Numeric columns stored as text are treated as categories
- Free hosting sleeps after inactivity, and files are not kept after a restart
- Future: hyperparameter search, feature importance chart, more models (XGBoost), user accounts

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Flask, Joblib, MLflow, HTML/CSS/JavaScript, Git, Gunicorn

## Author
Spoorthi Mekala ([@spoorthimekala](https://github.com/spoorthimekala))