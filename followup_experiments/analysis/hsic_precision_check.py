"""Synthetic precision check of the submitted HSIC-CCA implementation (no model).

Mirrors autonomous_harmed_mi_search.estimate_mi_curve: scalar RBF kernel over d'=128
paired coordinates, one-sided centering K@H, float32 solve, eps*m regularization.
Compares against float64 double-centered HKH reference.
"""
import json
import sys
import numpy as np
import torch

SIGMA, EPS, M = 50.0, 1e-5, 128


def score(x, y, dtype, double_center):
    x = torch.as_tensor(x, dtype=dtype)
    y = torch.as_tensor(y, dtype=dtype)
    h = torch.eye(M, dtype=dtype) - torch.ones(M, M, dtype=dtype) / M
    ident = torch.eye(M, dtype=dtype)

    def op(v):
        sq = (v * v).unsqueeze(-1)
        dist = torch.abs(sq - 2 * v.unsqueeze(-1) * v.unsqueeze(-2) + sq.T)
        k = torch.exp(-dist / (2 * SIGMA * SIGMA))
        kc = h @ k @ h if double_center else k @ h
        return kc @ torch.linalg.solve(kc + EPS * M * ident, ident)

    return float((op(x) * op(y).T).sum())


rng = np.random.default_rng(20261030)
rows = []
# hidden-state-like vectors: bulk N(0, s) plus a few large "massive activation" coordinates
for s in [0.5, 1, 2, 5, 10, 20, 50]:
    for outlier in [0, 50, 300]:
        for trial in range(20):
            def vec():
                v = rng.normal(0, s, M)
                if outlier:
                    v[rng.choice(M, 3, replace=False)] = rng.normal(0, outlier, 3)
                return v
            ht, yc = vec(), vec()
            yc_related = 0.7 * ht + 0.3 * yc  # partially dependent pair
            for kind, y in [("indep", yc), ("related", yc_related)]:
                rows.append(dict(scale=s, outlier=outlier, kind=kind,
                                 kh32=score(ht, y, torch.float32, False),
                                 hkh32=score(ht, y, torch.float32, True),
                                 kh64=score(ht, y, torch.float64, False),
                                 hkh64=score(ht, y, torch.float64, True)))
import pandas as pd
d = pd.DataFrame(rows)
d["err_kh32"] = d.kh32 - d.hkh64
d["err_hkh32"] = d.hkh32 - d.hkh64
summ = d.groupby(["scale", "outlier"]).agg(
    ref_mean=("hkh64", "mean"), ref_min=("hkh64", "min"), ref_max=("hkh64", "max"),
    kh32_neg=("kh32", lambda v: float((v < 0).mean())),
    kh32_abs_err=("err_kh32", lambda v: float(v.abs().mean())),
    kh32_max_abs_err=("err_kh32", lambda v: float(v.abs().max())),
    hkh32_max_abs_err=("err_hkh32", lambda v: float(v.abs().max())),
    kh64_vs_hkh64=("kh64", lambda v: 0.0)).reset_index()
summ["kh64_vs_hkh64_max_diff"] = d.assign(x=(d.kh64 - d.hkh64).abs()).groupby(["scale", "outlier"]).x.max().values
print(summ.drop(columns="kh64_vs_hkh64").to_string())
print("ref (float64 HKH) negative:", int((d.hkh64 < -1e-9).sum()), "of", len(d))
print("spearman kh32 vs ref:", d[["kh32", "hkh64"]].corr("spearman").iloc[0, 1])
d.to_csv(sys.argv[1], index=False)
summ.to_csv(sys.argv[2], index=False)
