"""Per-user background training jobs (live progress) + prediction with probabilities."""
import threading

from automl import trainer
from automl.pipeline import predict_one, run_automl
from automl.utils import DataError, load_csv

JOBS = {}                         # session id -> job status dict
_lock = threading.Lock()
_slot = threading.Semaphore(1)    # train one dataset at a time (keeps memory low on free hosting)
_tls = threading.local()          # remembers which user the current thread works for


def _set(sid, **kw):
    with _lock:
        JOBS.setdefault(sid, {"state": "idle", "message": "", "progress": 0, "error": None}).update(kw)


def get_status(sid):
    with _lock:
        return dict(JOBS.get(sid, {"state": "idle", "message": "", "progress": 0, "error": None}))


def _hook(msg, i, total):
    sid = getattr(_tls, "sid", None)
    if sid:
        _set(sid, message=msg, progress=15 + int(70 * i / total))


trainer.progress_hook = _hook     # trainer reports progress through this function


def start_training(sid, path, target, model_path):
    if get_status(sid)["state"] in ("running", "queued"):
        raise DataError("Training is already in progress.")
    _set(sid, state="queued", message="Waiting for another user's training to finish...", progress=2, error=None)

    def work():
        _tls.sid = sid
        with _slot:
            try:
                _set(sid, state="running", message="Reading dataset...", progress=5)
                df = load_csv(path)
                _set(sid, message="Cleaning data & splitting train/test...", progress=10)
                run_automl(df, target, model_path)
                _set(sid, state="done", message="Done! Opening dashboard...", progress=100)
            except DataError as e:
                _set(sid, state="error", error=str(e))
            except Exception as e:
                _set(sid, state="error", error=f"Training failed: {e}")

    threading.Thread(target=work, daemon=True).start()


class _Capture:
    """Wraps the pipeline to grab the exact row predict_one builds."""
    def __init__(self, pipe):
        self.pipe, self.row = pipe, None

    def predict(self, df):
        self.row = df
        return self.pipe.predict(df)


def predict_detail(bundle, values):
    """Returns (prediction, {class: probability} or None)."""
    cap = _Capture(bundle["pipeline"])
    pred = predict_one({**bundle, "pipeline": cap}, values)
    probs = None
    pipe = bundle["pipeline"]
    if bundle["task"] == "classification" and hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(cap.row)[0]
        probs = {str(c): round(float(x), 4) for c, x in zip(pipe.classes_, p)}
    elif bundle["task"] == "regression":
        pred = round(pred, 4)
    return pred, probs