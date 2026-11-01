"""U1 mechanism: (1) predicted error formula vs actual; (2) dimension sweep with low memory;
no external code executed. arg: output json."""
import gc, json, sys, time
import numpy as np, torch
torch.set_num_threads(1)
SIG, EPS = 50.0, 1e-5
F32, F64 = torch.float32, torch.float64
rng = np.random.default_rng(20261031)


def R(v, dt=F32, mode="kh"):
    m = v.numel()
    v = v.to(dt)
    sq = (v*v).unsqueeze(-1)
    k = torch.exp(-torch.abs(sq - 2*v.unsqueeze(-1)*v.unsqueeze(-2) + sq.T)/(2*SIG*SIG))
    h = torch.eye(m, dtype=dt) - torch.ones(m, m, dtype=dt)/m
    a = (k @ h) if mode == "kh" else (h @ k @ h)
    del k, h
    I = torch.eye(m, dtype=dt)
    s = torch.linalg.solve(a + EPS*m*I, I)
    return a @ s


def vec(m, s):
    return torch.from_numpy(rng.normal(0, s, m).astype(np.float32))


out = {}
# 1. err = tr(Rx32 Ry32) - tr(Rx64 Ry64) predicted by v_x^T (Ry32 1) + v_y^T (Rx32 1)
rows = []
for s in [0.5, 1, 2, 5, 10]:
    for _ in range(100):
        x, y = vec(128, s), vec(128, s)
        Rx, Ry = R(x).double(), R(y).double()
        Xe, Ye = R(x, F64), R(y, F64)
        vx, vy = Xe.sum(0)/128, Ye.sum(0)/128     # exact uncentred row component
        rx1, ry1 = Rx.sum(1), Ry.sum(1)               # should be exactly zero
        err = float((Rx*Ry.T).sum() - (Xe*Ye.T).sum())
        pred = float((vx*ry1).sum() + (vy*rx1).sum())
        rows.append((s, err, pred, float(vx.norm()), float(rx1.norm()), float((Xe*Ye.T).sum())))
a = np.array(rows)
out["err_formula"] = dict(n=len(a), pearson=float(np.corrcoef(a[:, 1], a[:, 2])[0, 1]), r2=float(1-((a[:, 1]-a[:, 2])**2).sum()/((a[:, 1]-a[:, 1].mean())**2).sum()),
    by_scale=[dict(s=s, vnorm=float(a[a[:, 0] == s, 3].mean()), R1norm=float(a[a[:, 0] == s, 4].mean()), abs_err=float(np.abs(a[a[:, 0] == s, 1]).mean()), abs_pred=float(np.abs(a[a[:, 0] == s, 2]).mean()), pearson=float(np.corrcoef(a[a[:, 0] == s, 1], a[a[:, 0] == s, 2])[0, 1])) for s in [0.5, 1, 2, 5, 10]])
print(json.dumps(out, indent=1), flush=True)

# 2. dimension sweep, submitted-style arithmetic, freeing memory
sweep = {}
for m, n in [(64, 100), (128, 100), (256, 40), (512, 20), (1024, 8), (2048, 3)]:
    for s in [1, 2]:
        t0 = time.time(); kh, hk, ex = [], [], []
        for _ in range(n):
            x, y = vec(m, s), vec(m, s)
            Rx = R(x); Ry = R(y)
            kh.append(float((Rx*Ry.T).sum())); del Rx, Ry
            if m <= 1024:
                Rx = R(x, mode="hkh"); Ry = R(y, mode="hkh")
                hk.append(float((Rx*Ry.T).sum())); del Rx, Ry
            if m <= 512:
                Rx = R(x, F64); Ry = R(y, F64)
                ex.append(float((Rx*Ry.T).sum())); del Rx, Ry
            gc.collect()
        kh = np.array(kh)
        sweep[f"m{m}_s{s}"] = dict(m=m, s=s, n=n, kh32_neg=float((kh < 0).mean()), kh32_min=float(kh.min()), kh32_mean=float(kh.mean()), kh32_max=float(kh.max()),
            hkh32_mean=float(np.mean(hk)) if hk else None, hkh32_neg=float((np.array(hk) < 0).mean()) if hk else None,
            exact_mean=float(np.mean(ex)) if ex else None, sec=round(time.time()-t0, 1))
        print(sweep[f"m{m}_s{s}"], flush=True)
        json.dump(dict(out, sweep=sweep), open(sys.argv[1], "w"), indent=1)
print("ALL_DONE")
