"""Metrics, cost-based threshold and bootstrap used in notebooks 03 and 04."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score


def lift_at(y, p, frac=0.10):
    y = np.asarray(y); k = max(1, int(len(y) * frac))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return y[top].mean() / y.mean()


def metrics(y, p, thr):
    y = np.asarray(y); p = np.asarray(p); pred = p >= thr
    return {"pr_auc": average_precision_score(y, p), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "lift@10%": lift_at(y, p),
            "pr_auc_lift": average_precision_score(y, p) / y.mean(), "warned_%": 100 * pred.mean(),
            "base_rate": y.mean()}


def cost_per_1000(y, p, thr, c_fn, c_fp=1.0):
    y = np.asarray(y); pred = np.asarray(p) >= thr
    fn = ((~pred) & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    return 1000 * (c_fn * fn + c_fp * fp) / len(y)


def best_threshold(y, p, c_fn, c_fp=1.0, grid=None):
    grid = np.linspace(0.02, 0.9, 89) if grid is None else grid
    costs = [cost_per_1000(y, p, t, c_fn, c_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def bootstrap(y, p, thr, c_fn, n_boot=500, seed=0):
    """Percentile 95% intervals for PR AUC, precision, recall, lift@10% and cost, resampling trips."""
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); n = len(y)
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        m = metrics(y[i], p[i], thr); m["cost_per_1000"] = cost_per_1000(y[i], p[i], thr, c_fn)
        rows.append(m)
    b = pd.DataFrame(rows)
    return b.quantile([0.025, 0.975]).T.rename(columns={0.025: "low", 0.975: "high"})
