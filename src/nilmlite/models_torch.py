"""Seq2Point deep model (optional `[dl]` extra).

Implements the same ``fit(X, y)`` / ``predict(X)`` / ``n_params`` interface as the
baselines, so it drops straight into ``evaluate``. ``export_onnx`` writes a
*self-contained* graph — input/output normalisation is baked in as constants, so
the ONNX takes a raw mains window (watts) and returns raw appliance power. That
file runs unchanged in the browser via onnxruntime-web or Rust tract/candle: the
training path (here) and the inference path (browser) share one artifact.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Seq2Point"]


def _build_net(window: int):
    import torch.nn as nn
    # Classic Seq2Point: 'valid' convs (no padding) keep CPU forward fast and the
    # flatten dim small; LazyLinear infers its input size on the first forward.
    return nn.Sequential(
        nn.Unflatten(1, (1, window)),                       # (B, window) -> (B, 1, window)
        nn.Conv1d(1, 30, 10), nn.ReLU(),
        nn.Conv1d(30, 30, 8), nn.ReLU(),
        nn.Conv1d(30, 40, 6), nn.ReLU(),
        nn.Conv1d(40, 50, 5), nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(128), nn.ReLU(),
        nn.Linear(128, 1),                                  # (B, 1)
    )


def _wrapped(net, stats):
    """nn.Module that normalises in, runs net, denormalises out, clamps >= 0."""
    import torch
    import torch.nn as nn

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            for k, v in stats.items():
                self.register_buffer(k, torch.tensor(float(v)))

        def forward(self, x):                                # x: (B, window) raw watts
            z = (x - self.mx) / self.sx
            out = self.net(z).squeeze(-1)                    # (B,)
            return torch.clamp(out * self.sy + self.my, min=0.0)

    return Wrapped()


class Seq2Point:
    def __init__(self, window: int = 99, epochs: int = 8, batch: int = 256,
                 lr: float = 1e-3, max_train: int = 200_000, seed: int = 0):
        self.window = window
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.max_train = max_train
        self.seed = seed
        self.model = None
        self.n_params: int | None = None

    def fit(self, X, y) -> "Seq2Point":
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.shape[0] > self.max_train:                     # subsample for CPU speed
            idx = rng.choice(X.shape[0], self.max_train, replace=False)
            X, y = X[idx], y[idx]

        stats = {"mx": X.mean(), "sx": X.std() + 1e-6,
                 "my": y.mean(), "sy": y.std() + 1e-6}
        net = _build_net(self.window)
        with torch.no_grad():                               # materialise LazyLinear
            net(torch.zeros(1, self.window))
        self.n_params = int(sum(p.numel() for p in net.parameters()))
        model = _wrapped(net, stats)
        model.train()

        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        ds = torch.utils.data.TensorDataset(Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        model.eval()
        self.model = model
        return self

    def predict(self, X) -> np.ndarray:
        import torch
        if self.model is None:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=np.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 4096):
                xb = torch.from_numpy(X[i:i + 4096])
                outs.append(self.model(xb).cpu().numpy())
        return np.concatenate(outs) if outs else np.empty(0, np.float32)

    def export_onnx(self, path: str) -> str:
        """Write a self-contained ONNX graph: raw mains window -> raw watts."""
        import torch
        if self.model is None:
            raise RuntimeError("call fit() first")
        dummy = torch.zeros(1, self.window, dtype=torch.float32)
        torch.onnx.export(
            self.model, dummy, str(path),
            input_names=["mains_window"], output_names=["power"],
            dynamic_axes={"mains_window": {0: "batch"}, "power": {0: "batch"}},
            opset_version=17, dynamo=False,   # legacy exporter: no onnxscript needed
        )
        return str(path)
