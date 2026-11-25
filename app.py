"""Server Flask: API data Yahoo Finance + pelatihan LSTM, dan menyajikan dashboard."""
import os
import threading
import time
import traceback
import uuid

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, jsonify, request

import data_source
from model import ALL_FEATURES, Params, train_and_forecast

DEFAULT_TICKER = os.environ.get("DEFAULT_TICKER", "XSSI.JK")
DEFAULT_START = os.environ.get("DEFAULT_START", "2022-03-01")

app = Flask(__name__, static_folder="static", static_url_path="")

_jobs: dict = {}
_jobs_lock = threading.Lock()
_train_lock = threading.Lock()  # satu pelatihan sekaligus supaya CPU tidak kewalahan


def _args():
    return (request.args.get("ticker", DEFAULT_TICKER).strip().upper(),
            request.args.get("start", DEFAULT_START))


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/config")
def config():
    return jsonify({"ticker": DEFAULT_TICKER, "start": DEFAULT_START, "features": ALL_FEATURES,
                    "defaults": Params().to_dict(), "limits": Params.LIMITS})


@app.get("/api/history")
def history():
    ticker, start = _args()
    try:
        h = data_source.get_history(ticker, start, force=request.args.get("refresh") == "1")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502
    df = h["df"]
    rows = [{"date": d.strftime("%Y-%m-%d"), "open": float(r.Open), "high": float(r.High),
             "low": float(r.Low), "close": float(r.Close),
             "volume": float(r.Volume) if "Volume" in df and r.Volume == r.Volume else None}
            for d, r in df.iterrows()]
    return jsonify({"ticker": ticker, "source": h["source"], "error": h["error"],
                    "fetched_at": h["fetched_at"], "rows": rows})


@app.get("/api/quote")
def quote():
    ticker, _ = _args()
    try:
        return jsonify(data_source.get_quote(ticker))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


def _run_job(job_id: str, ticker: str, start: str, params: Params):
    def update(**kw):
        with _jobs_lock:
            _jobs[job_id].update(kw)

    with _train_lock:
        try:
            update(status="loading_data")
            h = data_source.get_history(ticker, start)
            update(status="training", data_source=h["source"])

            def on_epoch(epoch, total, loss, val_loss):
                with _jobs_lock:
                    job = _jobs[job_id]
                    job["epoch"], job["epochs"] = epoch, total
                    job["live_loss"].append({"epoch": epoch, "loss": loss, "val_loss": val_loss})

            result = train_and_forecast(h["df"], params, on_epoch=on_epoch)
            result.update(ticker=ticker, start=start, data_source=h["source"],
                          last_date=h["df"].index[-1].strftime("%Y-%m-%d"))
            update(status="done", result=result, finished_at=time.time())
        except Exception as exc:
            traceback.print_exc()
            update(status="error", error=str(exc), finished_at=time.time())


@app.post("/api/train")
def train():
    body = request.get_json(silent=True) or {}
    ticker = str(body.get("ticker", DEFAULT_TICKER)).strip().upper()
    start = str(body.get("start", DEFAULT_START))
    params = Params.from_dict(body.get("params", {}))
    with _jobs_lock:
        running = [j for j in _jobs.values() if j["status"] in ("queued", "loading_data", "training")]
        if running:
            return jsonify({"error": "Masih ada pelatihan yang berjalan", "job_id": running[0]["id"]}), 409
        job_id = uuid.uuid4().hex[:10]
        _jobs[job_id] = {"id": job_id, "status": "queued", "epoch": 0, "epochs": params.epochs,
                         "live_loss": [], "ticker": ticker, "params": params.to_dict(),
                         "started_at": time.time()}
    threading.Thread(target=_run_job, args=(job_id, ticker, start, params), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def job(job_id):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            return jsonify({"error": "Job tidak ditemukan"}), 404
        out = dict(j)
        # selama training, kirim loss terbaru saja agar respons ringan
        if out["status"] != "done":
            out["live_loss"] = out["live_loss"][-300:]
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
