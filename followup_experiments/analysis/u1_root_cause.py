"""U1 root-cause checks on controlled inputs. No model, no GPU, no external repository executed.
transcribed() is my own transcription of the public estimator's arithmetic (read, not imported).
submitted functions are taken from the submission's autonomous_harmed_mi_search.py via ast."""
import ast, json, sys, time
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
SIG, EPS = 50.0, 1e-5
src = open(sys.argv[1], encoding="utf-8").read()
ns = {"np": np, "torch": torch}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in {"distmat_t", "kernelmat", "hsic_normalized_cca", "estimate_mi", "estimate_mi_curve"}:
        exec(compile(ast.Module([node], []), "submitted", "exec"), ns)
sub_single, sub_curve = ns["hsic_normalized_cca"], ns["estimate_mi_curve"]


def transcribed(x, y):
    m = x.numel()
    def r(v):
        v = v.view(-1, 1); h = torch.eye(m) - (1.0/m)*torch.ones(m, m)
        sq = (v*v).sum(1).view(-1, 1)
        d = torch.abs(sq.expand(m, m) - 2*(v @ v.T) + sq.T.expand(m, m))
        kc = torch.exp(-d/(2.0*SIG*SIG*1)).float() @ h
        return kc @ torch.inverse(kc + EPS*m*torch.eye(m))
    return float(torch.sum(r(x)*r(y).T))


def R(v, kdt=torch.float32, cdt=torch.float32, sdt=torch.float32, mode="kh"):
    """Kernel in kdt, centering in cdt, solve+product in sdt."""
    m = v.numel()
    v = v.to(kdt)
    sq = (v*v).unsqueeze(-1)
    d = torch.abs(sq - 2*v.unsqueeze(-1)*v.unsqueeze(-2) + sq.T)
    k = torch.exp(-d/(2*SIG*SIG)).to(cdt)
    h = torch.eye(m, dtype=cdt) - torch.ones(m, m, dtype=cdt)/m
    a = (k @ h) if mode == "kh" else (h @ k @ h)
    a = a.to(sdt); I = torch.eye(m, dtype=sdt)
    return a @ torch.linalg.solve(a + EPS*m*I, I)


def tr(a, b, dt=torch.float64):
    return float((a.to(dt)*b.to(dt).T).sum())


F32, F64 = torch.float32, torch.float64
rng = np.random.default_rng(20261030)
def vec(m, s): return torch.from_numpy(rng.normal(0, s, m).astype(np.float32))
out = {}

# A. algebraic identity in double precision
pairs = [(vec(128, s), vec(128, s)) for s in [0.5, 1, 2, 5, 10] for _ in range(100)]
t64 = np.array([tr(R(x, F64, F64, F64), R(y, F64, F64, F64)) for x, y in pairs])
h64 = np.array([tr(R(x, F64, F64, F64, "hkh"), R(y, F64, F64, F64, "hkh")) for x, y in pairs])
out["A_identity_f64"] = dict(n=len(pairs), max_abs_KH_minus_HKH=float(np.abs(t64-h64).max()), negatives=int((t64 < 0).sum()), min=float(t64.min()), mean=float(t64.mean()), max=float(t64.max()))


# B. stage attribution
def summarize(v):
    v = np.array(v)
    return dict(neg=float((v < 0).mean()), min=float(v.min()), mean=float(v.mean()), max=float(v.max()), mae_vs_exact=float(np.abs(v-t64).mean()), maxae=float(np.abs(v-t64).max()))


stages = {
 "submitted_all_f32(KH)": lambda x, y: float((R(x)*R(y).T).sum()),
 "kernel_f32_rest_f64": lambda x, y: tr(R(x, F32, F64, F64), R(y, F32, F64, F64)),
 "kernel+center_f32_rest_f64": lambda x, y: tr(R(x, F32, F32, F64), R(y, F32, F32, F64)),
 "kernel_f64_center+solve_f32": lambda x, y: tr(R(x, F64, F32, F32), R(y, F64, F32, F32)),
 "all_f32_except_final_sum_f64": lambda x, y: tr(R(x), R(y)),
 "HKH_all_f32": lambda x, y: float((R(x, mode="hkh")*R(y, mode="hkh").T).sum()),
 "exact_f64": lambda x, y: tr(R(x, F64, F64, F64), R(y, F64, F64, F64)),
}
out["B_stages"] = {k: summarize([f(x, y) for x, y in pairs]) for k, f in stages.items()}
print(json.dumps(out, indent=1), flush=True)

# C. non-normal structure and error decomposition
rows = []
for s in [0.5, 1, 2, 5, 10]:
    for _ in range(40):
        x, y = vec(128, s), vec(128, s); m = 128
        P = lambda a: a.mean(0, keepdim=True).expand_as(a)   # (1/m)11^T A: the part H removes on the left
        Rx64, Ry64 = R(x, F64, F64, F64), R(y, F64, F64, F64)
        Rx32, Ry32 = R(x).double(), R(y).double()
        Ex, Ey = Rx32-Rx64, Ry32-Ry64
        one = torch.ones(m, 1, dtype=F64)
        rows.append(dict(s=s, frob_R=float(Rx64.norm()), frob_HR=float((Rx64-P(Rx64)).norm()), frob_1v=float(P(Rx64).norm()),
            R1_exact=float((Rx64 @ one).norm()), R1_f32=float((Rx32 @ one).norm()),
            frob_E=float(Ex.norm()),
            exact=tr(Rx64, Ry64), f32=tr(Rx32, Ry32), term_Ex_Ry=tr(Ex, Ry64), term_Rx_Ey=tr(Rx64, Ey), term_Ex_Ey=tr(Ex, Ey)))
c = pd.DataFrame(rows); c["err"] = c.f32-c.exact
mabs = lambda v: float(np.abs(v).mean())
out["C_structure_by_scale"] = c.groupby("s").agg(frob_R=("frob_R", "median"), frob_HR=("frob_HR", "median"), frob_1v=("frob_1v", "median"), R1_f32=("R1_f32", "median"), R1_exact=("R1_exact", "median"), frob_E=("frob_E", "median"), abs_err=("err", mabs), ExRy=("term_Ex_Ry", mabs), RxEy=("term_Rx_Ey", mabs), ExEy=("term_Ex_Ey", mabs), neg=("f32", lambda v: float((v < 0).mean()))).reset_index().to_dict("records")
print(json.dumps(out["C_structure_by_scale"], indent=1), flush=True)

# D. transcription vs submitted code, dimension sweep
d = {}
for m, n, scales in [(128, 100, [0.5, 1, 2, 5, 10]), (1024, 6, [1, 2])]:
    for s in scales:
        t0 = time.time(); a, b, e = [], [], []
        for _ in range(n):
            x, y = vec(m, s), vec(m, s)
            a.append(transcribed(x, y)); b.append(sub_single(x, y, SIG))
            if m <= 1024:
                e.append(tr(R(x, F64, F64, F64), R(y, F64, F64, F64)))
        a, b = np.array(a), np.array(b); e = np.array(e) if e else np.array([np.nan])
        d[f"m{m}_s{s}"] = dict(n=n, transcribed_neg=float((a < 0).mean()), transcribed_min=float(a.min()), transcribed_mean=float(a.mean()), transcribed_max=float(a.max()), submitted_neg=float((b < 0).mean()), submitted_min=float(b.min()), submitted_mean=float(b.mean()), exact_mean=float(e.mean()), exact_neg=float((e < 0).mean()), transcribed_vs_submitted_mean_abs=float(np.abs(a-b).mean()), sec=time.time()-t0)
        print(m, s, d[f"m{m}_s{s}"], flush=True)
out["D_dim_sweep"] = d

# E. does dependence survive?
e_rows = []
for s in [0.5, 1, 2, 5]:
    for rho in [0, .1, .2, .3, .5, .7, .9]:
        for _ in range(30):
            x = rng.normal(0, s, 128); z = rng.normal(0, s, 128)
            y = rho*x + np.sqrt(1-rho*rho)*z
            xt, yt = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
            e_rows.append(dict(s=s, rho=rho, f32=float((R(xt)*R(yt).T).sum()), exact=tr(R(xt, F64, F64, F64), R(yt, F64, F64, F64)), hkh32=float((R(xt, mode="hkh")*R(yt, mode="hkh").T).sum())))
e = pd.DataFrame(e_rows); e["noise"] = e.f32-e.exact
out["E_by_scale"] = [dict(s=s, pearson_f32_exact=float(g.f32.corr(g.exact)), pearson_hkh32_exact=float(g.hkh32.corr(g.exact)), noise_sd=float(g.noise.std()), exact_sd=float(g.exact.std()), neg=float((g.f32 < 0).mean())) for s, g in e.groupby("s")]
out["E_by_rho_s1"] = [dict(rho=r, exact=float(g.exact.mean()), f32=float(g.f32.mean()), sd_noise=float(g.noise.std())) for r, g in e[e.s == 1].groupby("rho")]
choice = {}
for s in [1, 2]:
    for rho in [.3, .5, .7]:
        hit64 = hit32 = 0; n = 100
        for _ in range(n):
            x = rng.normal(0, s, 128)
            ys = [rho*x + np.sqrt(1-rho*rho)*rng.normal(0, s, 128)] + [rng.normal(0, s, 128) for _ in range(4)]
            xt = torch.from_numpy(x.astype(np.float32)); yts = [torch.from_numpy(v.astype(np.float32)) for v in ys]
            v32 = [float((R(xt)*R(v).T).sum()) for v in yts]
            v64 = [tr(R(xt, F64, F64, F64), R(v, F64, F64, F64)) for v in yts]
            hit32 += int(np.argmax(v32) == 0); hit64 += int(np.argmax(v64) == 0)
        choice[f"s{s}_rho{rho}"] = dict(f32=hit32/n, f64=hit64/n)
out["E_pick_related_of_5"] = choice

# F. determinism and batched path
x = np.stack([vec(128, 1).numpy() for _ in range(32)]); y = vec(128, 1).numpy()
curve = sub_curve(x, y, 128, SIG, 64)
curve8 = sub_curve(x, y, 128, SIG, 8)
single = np.array([sub_single(torch.from_numpy(x[i]), torch.from_numpy(y), SIG) for i in range(32)])
out["F_batched_vs_single"] = dict(max_abs_batched_vs_single=float(np.abs(curve-single).max()), mean_abs=float(np.abs(curve-single).mean()), max_abs_chunk64_vs_8=float(np.abs(curve-curve8).max()))
np.save(sys.argv[2]+".inputs.npy", np.vstack([x, y[None]])); np.save(sys.argv[2]+".curve_thread1.npy", curve)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k[0] in "EF"}, indent=1))
print("ALL_DONE")
