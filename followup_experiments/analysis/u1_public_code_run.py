"""Run the unmodified public estimator files on controlled inputs; compare to transcription of submitted code."""
import os
import importlib.util, json, sys, time
import numpy as np, torch
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("qian_mi", os.environ.get("MI_PEAKS_ROOT", "MI-Peaks") + "/src/mi_estimators.py")
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
rng = np.random.default_rng(20261030)
res = {}
def exact(x, y):
    m = len(x); h = torch.eye(m, dtype=torch.float64) - 1.0/m; I = torch.eye(m, dtype=torch.float64)
    def r(v):
        v = v.double(); k = h @ torch.exp(-(v[:,None]-v[None,:])**2/(2*2500.)) @ h
        return k @ torch.linalg.solve(k + 1e-5*m*I, I)
    return float((r(x)*r(y).T).sum())
for m, n, scales in [(128, 200, [0.5,1,2,5,10]), (4096, 4, [0.5,1,2,5])]:
    for s in scales:
        t0 = time.time(); v, e = [], []
        for i in range(n):
            x = torch.from_numpy(rng.normal(0,s,m).astype(np.float32)); y = torch.from_numpy(rng.normal(0,s,m).astype(np.float32))
            v.append(float(q.estimate_mi_hsic(x, y))); e.append(exact(x, y))
        v, e = np.array(v), np.array(e)
        res[f"m{m}_s{s}"] = dict(n=n, qian_neg=float((v<0).mean()), qian_min=float(v.min()), qian_mean=float(v.mean()), qian_max=float(v.max()),
                                 exact_neg=float((e<0).mean()), exact_mean=float(e.mean()), exact_min=float(e.min()), mae=float(np.abs(v-e).mean()), sec=time.time()-t0)
        print(m, s, res[f"m{m}_s{s}"], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
