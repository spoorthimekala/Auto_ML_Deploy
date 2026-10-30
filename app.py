"""Flask web app: upload -> choose target -> live training -> dashboard -> live predict.
Every visitor gets a private session, so many people can use the deployed app at once."""
import os
import time
import uuid
from pathlib import Path

from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_file, session, url_for)

from automl import live
from automl.pipeline import load_bundle
from automl.utils import DataError, detect_task, load_csv

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

UPLOAD_DIR, MODEL_DIR = Path("uploads"), Path("models")
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def sid():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def upload_path():
    return UPLOAD_DIR / f"{sid()}.csv"


def model_path():
    return MODEL_DIR / f"{sid()}.joblib"


def cleanup(max_age_hours=2):
    """Delete other users' old files so the server disk stays small."""
    limit = time.time() - max_age_hours * 3600
    for folder, ext in ((UPLOAD_DIR, "*.csv"), (MODEL_DIR, "*.joblib")):
        for f in folder.glob(ext):
            if f.stat().st_mtime < limit:
                f.unlink(missing_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    cleanup()
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            flash("Please choose a CSV file.", "error")
        elif not f.filename.lower().endswith(".csv"):
            flash("Only .csv files are allowed.", "error")
        else:
            f.save(upload_path())
            try:
                load_csv(upload_path())
                return redirect(url_for("configure"))
            except DataError as e:
                flash(str(e), "error")
    return render_template("index.html", has_model=model_path().exists())


@app.route("/configure")
def configure():
    if not upload_path().exists():
        flash("Upload a dataset first.", "error")
        return redirect(url_for("index"))
    try:
        df = load_csv(upload_path())
    except DataError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))
    tasks = {c: detect_task(df[c].dropna()) for c in df.columns}
    return render_template(
        "configure.html", shape=df.shape, columns=list(df.columns), tasks=tasks,
        missing=int(df.isna().sum().sum()),
        preview=df.head(5).to_html(index=False, classes="table", border=0))


@app.route("/api/train", methods=["POST"])
def api_train():
    target = request.form.get("target", "")
    try:
        if not upload_path().exists():
            raise DataError("Upload a dataset first.")
        if target not in load_csv(upload_path()).columns:
            raise DataError(f"Target column '{target}' not found.")
        live.start_training(sid(), upload_path(), target, model_path())
    except DataError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True)


@app.route("/api/status")
def api_status():
    return jsonify(live.get_status(sid()))


@app.route("/results")
def results():
    try:
        return render_template("results.html", b=load_bundle(model_path()))
    except DataError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))


@app.route("/predict")
def predict():
    try:
        return render_template("predict.html", b=load_bundle(model_path()))
    except DataError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        pred, probs = live.predict_detail(load_bundle(model_path()), request.get_json(silent=True) or {})
        return jsonify(prediction=pred, probabilities=probs)
    except DataError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"Prediction failed: {e}"), 500


@app.route("/download")
def download():
    if not model_path().exists():
        flash("No trained model found.", "error")
        return redirect(url_for("index"))
    return send_file(model_path(), as_attachment=True, download_name="best_model.joblib")


@app.errorhandler(413)
def too_large(_):
    flash("File too large (max 10 MB).", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)