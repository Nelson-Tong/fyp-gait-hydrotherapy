# ── utils_scaler.py ─────────────────────────────────────
import json, numpy as np, pathlib

def fit_scaler(X, method="zscore", groups=None, eps=1e-8):
    """
    X : (N, W, F) ndarray  – complete window tensor from training split
    groups : list[list[int]] | None
        e.g. [[0,1,2], [3,4,5]] to couple (x,y,z) triplets.
        If None ⇒ treat each feature independently.
    Returns a dict {feat_idx: {"mean": ..., "std": ...}}  or {"max": ...}
    """
    F = X.shape[-1]
    if groups is None:
        groups = [[i] for i in range(F)]

    stats = {}
    for g in groups:
        slice_ = X[..., g].reshape(-1, len(g))
        if method == "zscore":
            mu  = slice_.mean(axis=0)
            std = slice_.std(axis=0) + eps
            for i, f in enumerate(g):
                stats[f] = {"mean": float(mu[i]), "std": float(std[i])}
        elif method == "max":
            mx = np.abs(slice_).max(axis=0) + eps
            for i, f in enumerate(g):
                stats[f] = {"max": float(mx[i])}
        else:
            raise ValueError("method must be 'zscore' or 'max'")
    return stats

def apply_scaler(X, stats, method="zscore"):
    F = X.shape[-1]
    for f in range(F):
        if method == "zscore":
            X[..., f] = (X[..., f] - stats[f]["mean"]) / stats[f]["std"]
        else:  # max-norm
            X[..., f] = X[..., f] / stats[f]["max"]
    return X

def save_stats(stats, path):
    pathlib.Path(path).write_text(json.dumps(stats, indent=2))

def load_stats(path):
    import json
    return json.loads(open(path).read())
# ────────────────────────────────────────────────────────
