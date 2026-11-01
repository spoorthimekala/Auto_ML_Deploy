"""Phase 7: automated end-to-end tests.  Run: python test_workflow.py"""
import io, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

from automl.pipeline import load_bundle, predict_one, run_automl
from automl.utils import DataError

TMP = Path(tempfile.mkdtemp())
try:  # keep test runs out of your real MLflow history
    import automl.tracking as _t
    _t.TRACKING_URI = f"sqlite:///{(TMP / 'test_mlflow.db').as_posix()}"
except ImportError:
    pass

results = []
def check(name, ok, info=""):
    results.append(ok)
    print(("PASS" if ok else "FAIL"), "-", name, info)

def expect_error(name, df, target, text):
    try:
        run_automl(df, target, TMP / "m.joblib")
        check(name, False, "no error raised")
    except DataError as e:
        check(name, text in str(e), f"-> {e}")
    except Exception as e:
        check(name, False, f"wrong exception: {e!r}")

rng = np.random.default_rng(1)
n = 300

# 1-2. Built-in datasets
for f, task in [("breast_cancer", "classification"), ("diabetes", "regression")]:
    b = run_automl(pd.read_csv(f"sample_data/{f}.csv"), "target", TMP / "m.joblib")
    check(f"{f}: task={task}", b["task"] == task, f"best={b['best_model']} test={b['test_metrics']}")

# 3. Messy classification: missing values, categories, ID, bool, constant, 3 text classes
df = pd.DataFrame({
    "customer_id": [f"C{i}" for i in range(n)],
    "age": rng.integers(18, 70, n).astype(float),
    "city": rng.choice(["Hyd", "Blr", "Chn"], n),
    "member": rng.choice([True, False], n),
    "const": 1,
    "income": rng.normal(50, 12, n)})
df.loc[:30, "age"] = np.nan
df.loc[20:50, "city"] = None
df["plan"] = pd.cut(df.income, [-99, 42, 58, 999], labels=["basic", "plus", "pro"]).astype(str)
b = run_automl(df, "plan", TMP / "m.joblib")
dropped = " ".join(b["notes"])
check("messy: 3-class text target", b["task"] == "classification" and len(b["classes"]) == 3)
check("messy: ID + constant dropped", "customer_id" in dropped and "const" in dropped, dropped)
loaded = load_bundle(TMP / "m.joblib")
check("messy: predict w/ unseen city + blanks", predict_one(loaded, {"age": "", "city": "NewCity", "member": "True", "income": "70"}) in b["classes"])
try:
    predict_one(loaded, {"income": "abc"}); check("predict: bad number rejected", False)
except DataError:
    check("predict: bad number rejected", True)

# 4. Regression with categoricals + missing target rows
df2 = pd.DataFrame({"area": rng.normal(100, 20, n), "type": rng.choice(["flat", "villa"], n)})
df2["price"] = df2.area * 3 + (df2.type == "villa") * 50 + rng.normal(0, 5, n)
df2.loc[:9, "price"] = np.nan
b = run_automl(df2, "price", TMP / "m.joblib")
check("regression w/ categorical + NaN target", b["task"] == "regression" and b["n_rows"] == n - 10, f"r2={b['test_metrics']['r2']}")

# 5. Error handling
expect_error("tiny dataset", df.head(10), "plan", "at least 30")
expect_error("missing target column", df, "nope", "not found")
expect_error("single-value target", df.assign(plan="x"), "plan", "only one unique")
expect_error("rare class (1 row)", pd.concat([df, df.head(1).assign(plan="rare")]), "plan", "at least 2 rows")
expect_error("only ID columns", pd.DataFrame({"id": [f"a{i}" for i in range(n)], "y": rng.integers(0, 2, n)}), "y", "No usable feature")
expect_error("too many classes", df.assign(lbl=[f"c{i % 25}" for i in range(n)]).drop(columns=["plan"]), "lbl", "more than 20")

# 6. Flask flow (each client = one visitor with a private session)
from app import app, MODEL_DIR
def upload(c, data, name="d.csv"):
    return c.post("/", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data", follow_redirects=True)
def train(c, target="target"):
    r = c.post("/api/train", data={"target": target})
    for _ in range(400):
        s = c.get("/api/status").get_json()
        if s["state"] in ("done", "error"): break
        time.sleep(0.2)
    return r, s

c = app.test_client()
check("web: .txt rejected", b"Only .csv" in upload(c, b"a,b", "x.txt").data)
check("web: 1-column csv rejected", b"at least 2 columns" in upload(c, b"a\n1\n2\n").data)
check("web: valid upload -> configure", b"Choose the target" in upload(c, Path("sample_data/breast_cancer.csv").read_bytes()).data)
r, s = train(c)
check("web: live training finishes", r.status_code == 200 and s["state"] == "done", s["message"])
check("web: dashboard", b"BEST MODEL" in c.get("/results").data)
with c.session_transaction() as ss: sid1 = ss["sid"]
b = load_bundle(MODEL_DIR / f"{sid1}.joblib")
vals = {f["name"]: str(f["default"]) for f in b["features"]}
j = c.post("/api/predict", json=vals).get_json()
check("web: live predict + probabilities", j.get("prediction") in (0, 1) and abs(sum(j["probabilities"].values()) - 1) < 0.01, str(j))
check("web: predict bad input -> 400", c.post("/api/predict", json={**vals, "mean radius": "x"}).status_code == 400)
check("web: model download", c.get("/download").status_code == 200)

# 7. Multi-user isolation: a second visitor must not see or overwrite the first one's model
c2 = app.test_client()
check("multi: new visitor has no model", b"Upload. Compare. Deploy." in c2.get("/predict", follow_redirects=True).data)
upload(c2, Path("sample_data/diabetes.csv").read_bytes())
r2, s2 = train(c2)
check("multi: second visitor trains", s2["state"] == "done")
check("multi: visitor 1 still has classification model", b"classification" in c.get("/results").data)
check("multi: visitor 2 has regression model", b"regression" in c2.get("/results").data)

print(f"\n{sum(results)}/{len(results)} tests passed")