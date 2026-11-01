"""Synthetic check of candidate fixes for the HSIC-CCA sign problem (no model).

Same inputs as hsic_precision_check.py. Reference: float64, double centering HKH.
Fixes tested in float32: (1) double centering HKH; (2) expm1 kernel (K - 11^T has the
same centered matrix, without cancelling entries near 1) + HKH; (3) symmetric eigh
with eigenvalues clamped at 0, which makes the score non-negative by construction.
Also times one question's worth of float64 work (5 candidates x 256 steps).
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

SIGMA, EPS, M = 50.0, 1e-5, 128


def centered(v, dtype, mode):
    v = torch.as_tensor(v, dtype=dtype)
    h = torch.eye(M, dtype=dtype) - torch.ones(M, M, dtype=dtype) / M
    dist = (v.unsqueeze(-1) - v.unsqueeze(-2)) ** 2 / (2 * SIGMA * SIGMA)
    if mode == "kh":
        sq = (v * v).unsqueeze(-1)
        dist = torch.abs(sq - 2 * v.unsqueeze(-1) * v.unsqueeze(-2) + sq.T) / (2 * SIGMA * SIGMA)
        return torch.exp(-dist) @ h
    k = torch.expm1(-dist) if mode == "expm1" else torch.exp(-dist)
    return h @ k @ h


def score(x, y, dtype, mode, use_eigh=False):
    ident = torch.eye(M, dtype=dtype)
    if use_eigh:
        def spec(v):
            lam, u = torch.linalg.eigh(centered(v, dtype, "expm1"))
            lam = lam.clamp_min(0)
            return lam / (lam + EPS * M), u
        a, u = spec(x)
        b, w = spec(y)
        return float(a @ ((u.T @ w) ** 2) @ b)
    rx = centered(x, dtype, mode)
    ry = centered(y, dtype, mode)
    rx = rx @ torch.linalg.solve(rx + EPS * M * ident, ident)
    ry = ry @ torch.linalg.solve(ry + EPS * M * ident, ident)
    return float((rx * ry.T).sum())


rng = np.random.default_rng(20261030)
rows = []
for s in [0.5, 1, 2, 5, 10, 20, 50]:
    for outlier in [0, 50, 300]:
        for trial in range(20):
            def vec():
                v = rng.normal(0, s, M)
                if outlier:
                    v[rng.choice(M, 3, replace=False)] = rng.normal(0, outlier, 3)
                return v
            ht, yc = vec(), vec()
            for kind, y in [("indep", yc), ("related", 0.7 * ht + 0.3 * yc)]:
                ref = score(ht, y, torch.float64, "hkh")
                rows.append(dict(scale=s, outlier=outlier, kind=kind, ref=ref,
                                 submitted_kh32=score(ht, y, torch.float32, "kh") - ref,
                                 hkh32=score(ht, y, torch.float32, "hkh") - ref,
                                 expm1_hkh32=score(ht, y, torch.float32, "expm1") - ref,
                                 eigh32=score(ht, y, torch.float32, "expm1", True) - ref,
                                 eigh32_value=score(ht, y, torch.float32, "expm1", True)))
d = pd.DataFrame(rows)
cols = ["submitted_kh32", "hkh32", "expm1_hkh32", "eigh32"]
summ = pd.DataFrame({c: {"max_abs_err": d[c].abs().max(), "mean_abs_err": d[c].abs().mean(),
                         "negative_frac": float(((d[c] + d.ref) < 0).mean())} for c in cols}).T
summ["eigh32_min_value"] = d.eigh32_value.min()
print(summ.to_string())
print("ref negatives:", int((d.ref < 0).sum()), "of", len(d))

# float64 cost for one question: 5 candidates x 256 steps, 128x128 double-centered solves
x = torch.randn(256, M, dtype=torch.float64)
y = torch.randn(5, M, dtype=torch.float64)
h = torch.eye(M, dtype=torch.float64) - 1.0 / M
ident = torch.eye(M, dtype=torch.float64)
t0 = time.perf_counter()
def kern(v):
    return h @ torch.expm1(-(v.unsqueeze(-1) - v.unsqueeze(-2)) ** 2 / (2 * SIGMA * SIGMA)) @ h
kx = kern(x)
rx = kx @ torch.linalg.solve(kx + EPS * M * ident, ident.expand(256, -1, -1))
for j in range(5):
    ky = kern(y[j])
    ry = ky @ torch.linalg.solve(ky + EPS * M * ident, ident)
    _ = (rx * ry.T).sum(dim=(-2, -1))
print("float64 time per question (1 CPU thread): %.3f s" % (time.perf_counter() - t0))
d.to_csv(sys.argv[1], index=False)
summ.to_csv(sys.argv[2])
