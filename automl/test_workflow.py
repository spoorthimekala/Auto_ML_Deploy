import io
import time
from pathlib import Path

from app import app, MODEL_DIR
from automl.pipeline import load_bundle

results = []

def check(name, condition, detail=""):
    passed = bool(condition)
    results.append(passed)

    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name}")

    if detail and not passed:
        print(f"  Details: {detail}")


def upload(c, data, name="d.csv"):
    return c.post(
        "/",
        data={"file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
        follow_redirects=True
    )


def train(c, target="target"):
    r = c.post("/api/train", data={"target": target})

    for _ in range(400):
        s = c.get("/api/status").get_json()

        if s["state"] in ("done", "error"):
            break

        time.sleep(0.2)

    return r, s