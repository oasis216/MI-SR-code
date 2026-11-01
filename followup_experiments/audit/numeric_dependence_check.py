"""Small synthetic arithmetic audit; not a reproduction of submitted model curves."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import numpy as np

def score(x, y, dtype, double_center):
    n = len(x)
    h = np.eye(n, dtype=dtype) - np.ones((n, n), dtype=dtype) / n
    ident = np.eye(n, dtype=dtype)
    def op(v):
        v = v.astype(dtype)
        squared = (v*v)[:, None]
        distance = np.abs(squared - 2*v[:, None]*v[None, :] + squared.T)
        k = np.exp(-distance / (2*50.0**2)).astype(dtype)
        kc = k @ h
        if double_center:
            kc = h @ kc
        return kc @ np.linalg.solve(kc + (1e-5*n)*ident, ident)
    a, b = op(x), op(y)
    return float(np.sum(a*b.T))

rng = np.random.default_rng(321)
rows=[]
for scale in [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
    for trial in range(3):
        x,y = rng.normal(size=(2,128))*scale
        rows.append(dict(scale=scale,trial=trial,
                         kh_f32=score(x,y,np.float32,False),
                         hkh_f32=score(x,y,np.float32,True),
                         kh_f64=score(x,y,np.float64,False),
                         hkh_f64=score(x,y,np.float64,True)))
print(json.dumps(rows,indent=2))
