"""`nilmlite serve` — drive the no-code studio with REAL PyTorch.

A dependency-free stdlib HTTP server (no FastAPI) that:
  * serves docs/ (the studio + demo_data.json + ONNX) same-origin, and
  * exposes /train and /predict that run the actual nilmlite PyTorch models
    (Seq2Point/DAE/GRU/PatchTST) on CPU or CUDA.

In-browser inference stays ONNX (PyTorch-exported); training is genuine PyTorch.
"""
from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

import nilmlite as nl
from nilmlite.matrix import _concat

W, P = 99, 60
ALL = [("REDD · US", "data/redd", [1, 2], 3),
       ("UK-DALE · UK", "data/ukdale", [5], 2),
       ("iAWE · India", "data/iawe", [1], 1)]
SPEC = {n: dict(path=p, train=tr, test=te) for n, p, tr, te in ALL}
MODELS: dict = {}
DOCS = Path("docs")
_CTYPE = {".html": "text/html", ".json": "application/json", ".js": "text/javascript",
          ".css": "text/css", ".onnx": "application/octet-stream"}


def _train_xy(name, app):
    s = SPEC[name]
    return _concat(s["path"], s["train"], app, P, W)


def _test_slice(name, app, L=1500):
    s = SPEC[name]
    df = nl.resample_to(nl.Dataset(s["path"]).load(s["test"]), P)
    if app not in df.columns:
        return None
    mains = df["mains"].to_numpy().astype(float)
    truth = df[app].to_numpy().astype(float)
    bs, best = 0, -1
    for st in range(0, max(1, len(truth) - L), 200):
        sc = int((truth[st + W // 2: st + W // 2 + (L - W + 1)] > 15).sum())
        if sc > best:
            best, bs = sc, st
    return mains[bs:bs + L], truth[bs:bs + L]


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, b"")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            import importlib.util
            return self._send(200, json.dumps({"ok": True, "torch": importlib.util.find_spec("torch") is not None}))
        f = DOCS / ("studio.html" if path == "/" else path.lstrip("/"))
        if f.is_file():
            return self._send(200, f.read_bytes(), _CTYPE.get(f.suffix, "application/octet-stream"))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or "{}")
        if self.path == "/train":
            return self._train_stream(body)
        try:
            if self.path == "/predict":
                return self._send(200, json.dumps(self._predict(body)))
        except Exception as e:  # noqa: BLE001
            return self._send(400, json.dumps({"error": str(e)}))
        self._send(404, json.dumps({"error": "unknown route"}))

    def _train_stream(self, b):
        """Stream NDJSON training events: start → epoch* → done (or error)."""
        try:
            import torch
            from nilmlite.models_torch import ZOO
            ds, app = b["ds"], b["app"]
            arch, ep = b.get("arch", "Seq2Point"), int(b.get("epochs", 6))
            Xtr, ytr = _train_xy(ds, app)
            cap = min(12000, int(len(Xtr)))
        except Exception as e:  # noqa: BLE001
            return self._send(400, json.dumps({"error": str(e)}))

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(o):
            try:
                self.wfile.write((json.dumps(o) + "\n").encode())
                self.wfile.flush()
            except Exception:  # client went away
                pass

        backend = "cuda" if torch.cuda.is_available() else "cpu"
        emit({"type": "start", "arch": arch, "ds": ds, "app": app, "total": ep,
              "backend": backend, "windows": cap})
        t0 = time.time()
        model = ZOO[arch](window=W, epochs=ep, max_train=12000)
        try:
            model.fit(Xtr, ytr, on_epoch=lambda e, tot, loss: emit(
                {"type": "epoch", "epoch": e, "total": tot,
                 "loss": round(float(loss), 5), "elapsed": round(time.time() - t0, 1)}))
        except Exception as e:  # noqa: BLE001
            return emit({"type": "error", "error": str(e)})
        mid = uuid.uuid4().hex[:8]
        MODELS[mid] = {"model": model, "arch": arch, "trained_on": ds}
        emit({"type": "done", "model_id": mid, "params": int(model.n_params),
              "seconds": round(time.time() - t0, 1), "backend": backend,
              "arch": arch, "trained_on": ds})

    def _predict(self, b):
        h = MODELS.get(b["model_id"])
        if not h:
            raise ValueError("unknown model_id — train first")
        sl = _test_slice(b["ds"], b["app"])
        if sl is None:
            raise ValueError(f"{b['app']} not metered in {b['ds']}")
        mains, truth = sl
        X = nl.windows.sliding_windows(mains, W)
        pred = np.asarray(h["model"].predict(X)).ravel()
        half, n = W // 2, len(X)
        tt, mm = truth[half:half + n], mains[half:half + n]
        return {"label": f"{h['arch']} (trained on {h['trained_on']})", "ds": b["ds"], "app": b["app"],
                "mains": [round(float(x), 1) for x in mm], "truth": [round(float(x), 1) for x in tt],
                "pred": [round(float(x), 1) for x in pred],
                "mae": round(float(nl.metrics.mae(tt, pred)), 2),
                "f1": round(float(nl.metrics.f1(tt, pred)), 3)}


def serve(port: int = 8000, docs: str = "docs"):
    global DOCS
    DOCS = Path(docs)
    import importlib.util
    torch_ok = importlib.util.find_spec("torch") is not None
    print(f"nilmlite studio → http://localhost:{port}/")
    print(f"  real PyTorch training: {'ready (CPU/CUDA)' if torch_ok else 'install extras: pip install nilmlite[dl]'}")
    # single-threaded on purpose: torch training runs on the main thread
    # (a request worker thread segfaults with OpenMP on macOS)
    HTTPServer(("127.0.0.1", port), _H).serve_forever()
