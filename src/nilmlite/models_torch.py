"""Deep model zoo (optional `[dl]` extra).

Every model maps a mains window -> midpoint appliance power and shares one
`fit(X, y)` / `predict(X)` / `n_params` / `export_onnx` interface, so they drop
straight into `evaluate` and the cross-dataset matrix. `export_onnx` bakes
input/output normalisation into the graph, so the same file runs in the browser
via onnxruntime-web. Training stays in Python; only inference crosses to WASM.

    Seq2Point  — classic 1-D conv stack (Zhang et al. 2018)
    DAE        — denoising conv autoencoder -> point
    GRU        — bidirectional recurrent -> point
    PatchTST   — patch embedding + transformer encoder (ICLR 2023), as a NILM head
"""
from __future__ import annotations

import numpy as np

__all__ = ["Seq2Point", "DAE", "GRU", "PatchTST", "ZOO"]


def _wrapped(net, stats):
    """nn.Module: normalise in -> net -> denormalise out -> clamp >= 0."""
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
            out = self.net(z).squeeze(-1)
            return torch.clamp(out * self.sy + self.my, min=0.0)

    return Wrapped()


class _Base:
    """Shared training / inference / ONNX-export machinery."""
    label = "model"

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

    def build_net(self):                                     # -> nn.Module taking (B, window)
        raise NotImplementedError

    def fit(self, X, y, on_epoch=None):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.shape[0] > self.max_train:
            idx = rng.choice(X.shape[0], self.max_train, replace=False)
            X, y = X[idx], y[idx]

        stats = {"mx": X.mean(), "sx": X.std() + 1e-6,
                 "my": y.mean(), "sy": y.std() + 1e-6}
        net = self.build_net()
        with torch.no_grad():                               # materialise lazy layers
            net(torch.zeros(1, self.window))
        self.n_params = int(sum(p.numel() for p in net.parameters()))
        model = _wrapped(net, stats)
        model.train()

        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        for ep in range(self.epochs):
            run, nb = 0.0, 0
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                run += float(loss); nb += 1
            if on_epoch is not None:
                on_epoch(ep + 1, self.epochs, run / max(1, nb))
        model.eval()
        self.model = model
        return self

    def predict(self, X):
        import torch
        if self.model is None:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=np.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 4096):
                outs.append(self.model(torch.from_numpy(X[i:i + 4096])).cpu().numpy())
        return np.concatenate(outs) if outs else np.empty(0, np.float32)

    def export_onnx(self, path: str) -> str:
        import torch
        if self.model is None:
            raise RuntimeError("call fit() first")
        dummy = torch.zeros(1, self.window, dtype=torch.float32)
        torch.onnx.export(
            self.model, dummy, str(path),
            input_names=["mains_window"], output_names=["power"],
            dynamic_axes={"mains_window": {0: "batch"}, "power": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        return str(path)


class Seq2Point(_Base):
    label = "Seq2Point"

    def build_net(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Unflatten(1, (1, self.window)),
            nn.Conv1d(1, 30, 10), nn.ReLU(),
            nn.Conv1d(30, 30, 8), nn.ReLU(),
            nn.Conv1d(30, 40, 6), nn.ReLU(),
            nn.Conv1d(40, 50, 5), nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, 1),
        )


class DAE(_Base):
    label = "DAE"

    def build_net(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Unflatten(1, (1, self.window)),
            nn.Conv1d(1, 8, 4, padding=2), nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )


class GRU(_Base):
    label = "GRU"

    def build_net(self):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, window):
                super().__init__()
                self.conv = nn.Conv1d(1, 16, 4, padding=2)
                self.gru = nn.GRU(16, 64, batch_first=True, bidirectional=True)
                self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(self, x):                            # (B, window)
                h = self.conv(x.unsqueeze(1)).transpose(1, 2)   # (B, T, 16)
                o, _ = self.gru(h)
                return self.head(o[:, -1])                   # (B, 1)

        return Net(self.window)


class PatchTST(_Base):
    label = "PatchTST"

    def __init__(self, *a, patch: int = 16, stride: int = 8, d: int = 64,
                 heads: int = 4, layers: int = 2, **k):
        super().__init__(*a, **k)
        self.patch, self.stride, self.d, self.heads, self.layers = patch, stride, d, heads, layers

    def build_net(self):
        import torch
        import torch.nn as nn

        patch, stride, d, heads, layers, window = (
            self.patch, self.stride, self.d, self.heads, self.layers, self.window)
        n_patches = (window - patch) // stride + 1

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Conv1d(1, d, kernel_size=patch, stride=stride)  # patchify
                self.pos = nn.Parameter(torch.zeros(1, n_patches, d))
                enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d,
                                                 batch_first=True, dropout=0.0)
                self.tr = nn.TransformerEncoder(enc, layers)
                self.head = nn.Linear(n_patches * d, 1)

            def forward(self, x):                            # (B, window)
                z = self.embed(x.unsqueeze(1)).transpose(1, 2)  # (B, n_patches, d)
                z = self.tr(z + self.pos)
                return self.head(z.flatten(1))               # (B, 1)

        return Net()


ZOO = {"Seq2Point": Seq2Point, "DAE": DAE, "GRU": GRU, "PatchTST": PatchTST}
