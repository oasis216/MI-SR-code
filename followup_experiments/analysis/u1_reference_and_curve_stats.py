"""(1) Own transcription of the published estimator's arithmetic (float32, K@H, torch.inverse,
eps=1e-5*m, sigma=50) on controlled inputs, vs float64 HKH. The external repository is NOT
executed. (2) Distribution of all stored trajectory values per model. No model or GPU."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

data_root, out = sys.argv[1:3]
M, SIGMA, EPS = 128, 50.0, 1e-5


def transcribed(x, y):
    def r(v):
        v = v.view(-1, 1)
        h = torch.eye(M) - (1.0 / M) * torch.ones(M, M)
        sq = (v * v).sum(1).view(-1, 1)
        d = torch.abs(sq.expand(M, M) - 2 * (v @ v.T) + sq.T.expand(M, M))
        kc = torch.exp(-d / (2.0 * SIGMA * SIGMA)).float() @ h
        return kc @ torch.inverse(kc + EPS * M * torch.eye(M))
    return float(torch.sum(r(x) * r(y).T))


def exact(x, y):
    h = torch.eye(M, dtype=torch.float64) - 1.0 / M
    ident = torch.eye(M, dtype=torch.float64)
    def r(v):
        v = v.double()
        k = h @ torch.exp(-(v[:, None] - v[None, :]) ** 2 / (2 * SIGMA * SIGMA)) @ h
        return k @ torch.linalg.solve(k + EPS * M * ident, ident)
    return float((r(x) * r(y).T).sum())


rng = np.random.default_rng(20261030)
vals, refs = [], []
for s in [0.5, 1, 2, 5, 10]:
    for trial in range(100):
        x = torch.from_numpy(rng.normal(0, s, M).astype(np.float32))
        y = torch.from_numpy(rng.normal(0, s, M).astype(np.float32))
        vals.append(transcribed(x, y)); refs.append(exact(x, y))
vals, refs = np.array(vals), np.array(refs)
res = {"transcribed_estimator": {"n": len(vals), "negative_frac": float((vals < 0).mean()),
       "min": float(vals.min()), "max": float(vals.max()), "float64_negative": int((refs < 0).sum()),
       "float64_min": float(refs.min()), "max_abs_diff": float(np.abs(vals - refs).max())}}
for model in sys.argv[3].split(","):
    store = pickle.load(open(Path(data_root) / model / "all_biases/all_candidate_curve_store.pkl", "rb"))
    edges = np.linspace(-1.0, 2.0, 30001)
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    total = neg = 0
    amin, amax, asum = np.inf, -np.inf, 0.0
    for r in store.values():
        for c in r["curves"].values():
            a = np.asarray(c, dtype=np.float32)
            if not a.size:
                continue
            hist += np.histogram(a, bins=edges)[0]
            total += a.size; neg += int((a < 0).sum()); asum += float(a.sum())
            amin, amax = min(amin, float(a.min())), max(amax, float(a.max()))
    del store
    cum = np.cumsum(hist) / hist.sum()
    q = lambda p: float(edges[1:][np.searchsorted(cum, p)])
    res[model] = {"points": int(total), "negative_frac": neg / total, "mean": asum / total, "median": q(.5),
                  "p01": q(.01), "p05": q(.05), "p95": q(.95), "p99": q(.99), "min": amin, "max": amax}
print(json.dumps(res, indent=1))
json.dump(res, open(out, "w"), indent=1)
