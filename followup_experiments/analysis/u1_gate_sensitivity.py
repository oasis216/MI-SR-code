"""U1: how much do trajectory numerics matter to the frozen gate? CPU only, stored outputs only.

Perturbs the stored trajectories, recomputes the curve-derived gate columns with the submission's
definitions, reruns the frozen gate and the published rule r>0. Columns that need new model probes
when a peak changes chunk (support_*, persistent, leap, span flags) are held at stored values, and
the share of peaks leaving their chunk is reported, so this is a partial analysis, not a rerun.

usage: u1_gate_sensitivity.py <deep-test-3 root> <model key> <out.json>
"""
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

root, model, out_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
d = root / "output_mi_sr_full_targeted_v5" / model / "all_biases"
art = root / "frozen_artifacts"
cfg = json.load(open(art / "frozen_gate_v002.json"))
cols = cfg["feature_columns"]
models = joblib.load(art / "frozen_gate_v002_models.joblib")
challenger_cfg = json.load(open(art / "frozen_challenger_v001.json"))
frame = pickle.load(open(d / "deploy_gate_features_before_unblinding.pkl", "rb")).reset_index(drop=True)
policy = pd.read_csv(d / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
frame = frame[frame.case_id.isin(policy.index)].reset_index(drop=True)
plan = pd.read_csv(d / "mi_sr_peak_features/peak_probe_plan.csv", encoding="utf-8-sig")
peaks_csv = pd.read_csv(d / "mi_sr_peak_features/candidate_peak_features.csv", encoding="utf-8-sig")
curves = pickle.load(open(d / "all_candidate_curve_store.pkl", "rb"))
labels = pd.read_csv(d / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id")
review = pd.read_csv(d / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id")

CURVE_COLS = ["candidate_mi_late_max", "candidate_mi_mean_max", "pair_mi_late_max", "pair_mi_max",
              "challenger_mi_candidate_late_mean", "challenger_mi_candidate_mean",
              "challenger_mi_candidate_minus_answer_late_mean", "challenger_mi_candidate_minus_answer_max",
              "challenger_mi_candidate_minus_answer_mean", "challenger_mi_candidate_minus_answer_peak_count",
              "challenger_mi_candidate_minus_answer_peak_prominence_max", "challenger_mi_candidate_peak_count",
              "challenger_mi_candidate_peak_prominence_max"]
PEAK_COLS = ["peak_prominence_max", "peak_prominence_mean", "challenger_peak_prominence", "challenger_peak_position"]
assert set(CURVE_COLS + PEAK_COLS) <= set(cols)
HELD = [c for c in cols if c not in CURVE_COLS + PEAK_COLS and any(k in c for k in
        ("support", "persistent", "leap", "semantic", "peak_", "mi_", "override", "persistence"))]

alts = plan.groupby("case_id")
ends = {}
for r in plan.itertuples(index=False):
    if r.peak_position > 0:
        ends[r.case_id] = int(round(r.peak_step / r.peak_position))
plan_map = {(r.case_id, r.candidate): (r.before_step, r.after_step, r.peak_step) for r in plan.itertuples(index=False)}


def stats(v):
    n = len(v)
    floor = max(float(np.std(v)) * 0.75, 1e-6)
    _, prop = find_peaks(v, prominence=floor)
    pr = prop.get("prominences", np.asarray([], dtype=float))
    return dict(mean=float(v.mean()), max=float(v.max()), late=float(np.array_split(v, 3)[2].mean()),
                count=int(len(pr)), prom=float(pr.max()) if len(pr) else 0.0)


def strongest(values, end_step):
    values = values[: end_step + 1]
    start = min(10, max(0, len(values) - 1))
    local = values[start:]
    points, prop = find_peaks(local, prominence=max(float(np.std(local)) * 0.25, 1e-6), distance=4)
    if len(points):
        best = int(np.argmax(prop["prominences"]))
        return int(points[best] + start), float(prop["prominences"][best])
    step = int(np.argmax(local) + start)
    return step, float(peak_prominences(values, np.asarray([step]))[0][0])


def recompute(transform):
    """Curve-derived columns for each case under transform(curve array)."""
    rows, moved, total = [], 0, 0
    for r in frame[["case_id", "model_answer", "learned_challenger"]].itertuples(index=False):
        cmap = {str(k).upper(): transform(np.asarray(v, dtype=float)) for k, v in curves[r.case_id]["curves"].items()}
        ans = cmap[r.model_answer]
        alt = [c for c in cmap if c != r.model_answer]
        s_c, s_p, prom = {}, {}, {}
        for c in alt:
            n = min(len(cmap[c]), len(ans))
            s_c[c] = stats(cmap[c][:n])
            s_p[c] = stats(cmap[c][:n] - ans[:n])
            step, prom[c] = strongest(cmap[c], ends.get(r.case_id, len(cmap[c]) - 1))
            before, after, _ = plan_map[(r.case_id, c)]
            moved += int(not before <= step <= after); total += 1
            if c == r.learned_challenger:
                pos = step / max(ends.get(r.case_id, len(cmap[c]) - 1), 1)
        ch = r.learned_challenger
        rows.append(dict(
            candidate_mi_late_max=max(s["late"] for s in s_c.values()), candidate_mi_mean_max=max(s["mean"] for s in s_c.values()),
            pair_mi_late_max=max(s["late"] for s in s_p.values()), pair_mi_max=max(s["max"] for s in s_p.values()),
            challenger_mi_candidate_late_mean=s_c[ch]["late"], challenger_mi_candidate_mean=s_c[ch]["mean"],
            challenger_mi_candidate_minus_answer_late_mean=s_p[ch]["late"], challenger_mi_candidate_minus_answer_max=s_p[ch]["max"],
            challenger_mi_candidate_minus_answer_mean=s_p[ch]["mean"], challenger_mi_candidate_minus_answer_peak_count=s_p[ch]["count"],
            challenger_mi_candidate_minus_answer_peak_prominence_max=s_p[ch]["prom"], challenger_mi_candidate_peak_count=s_c[ch]["count"],
            challenger_mi_candidate_peak_prominence_max=s_c[ch]["prom"],
            peak_prominence_max=max(prom.values()), peak_prominence_mean=float(np.mean(list(prom.values()))),
            challenger_peak_prominence=prom[ch], challenger_peak_position=pos))
    return pd.DataFrame(rows), moved / max(total, 1)


def score(values):
    x = frame[cols].copy()
    for c in CURVE_COLS + PEAK_COLS:
        x[c] = values[c].to_numpy()
    p = np.column_stack([models[t].predict_proba(x)[:, 1] for t in ("baseline_correct", "targeted_correct", "generic_correct")])
    return p[:, 1:].max(1) - p[:, 0]


ids = frame.case_id.to_numpy()
gold = labels.loc[ids, "gold"].astype(str).to_numpy()
base_ok = labels.loc[ids, "biased_correct"].astype(bool).to_numpy()
reviewed = review.reindex(ids)
# Same counting as the paper: an incomplete review keeps the first answer.
rev_ok = np.where(reviewed.reeval_complete.fillna(False).astype(bool).to_numpy(), reviewed.reeval_answer.astype(str).to_numpy() == gold, base_ok)
stored_r = policy.loc[ids, "review_score"].to_numpy()
stored_t = stored_r > 0


def outcome(r):
    t = r > 0
    final = np.where(t, rev_ok, base_ok)
    both = t & stored_t
    return dict(reviews=int(t.sum()), acc=float(final.mean()*100), corrected=int((t & rev_ok & ~base_ok).sum()), damaged=int((t & ~rev_ok & base_ok).sum()),
                decision_flip_pct=float((t != stored_t).mean()*100), r_pearson=float(np.corrcoef(r, stored_r)[0, 1]), r_mae=float(np.abs(r-stored_r).mean()), both=int(both.sum()))


res = dict(model=model, n=len(frame), held_columns=HELD, curve_columns=CURVE_COLS+PEAK_COLS, n_gate_columns=len(cols),
           challenger_feature_columns=challenger_cfg.get("feature_columns"))
base_vals, moved0 = recompute(lambda v: v)
res["verify_recompute_vs_stored_max_abs"] = {c: float(np.nanmax(np.abs(base_vals[c].to_numpy(float)-frame[c].to_numpy(float)))) for c in CURVE_COLS + PEAK_COLS}
res["verify_peak_left_chunk_unperturbed_pct"] = moved0*100
res["verify_r_max_abs"] = float(np.abs(score(frame[CURVE_COLS+PEAK_COLS])-stored_r).max())
res["published"] = outcome(stored_r)
print(json.dumps({k: res[k] for k in ("n", "verify_r_max_abs", "verify_peak_left_chunk_unperturbed_pct", "published")}), flush=True)
print(max(res["verify_recompute_vs_stored_max_abs"].items(), key=lambda kv: kv[1]), flush=True)

all_vals = np.concatenate([np.asarray(c, dtype=np.float32) for v in curves.values() for c in v["curves"].values() if len(c)])
neg = all_vals[all_vals < 0]
sigma = float(np.sqrt(np.mean(neg.astype(float)**2)))
res["real_curve"] = dict(points=int(all_vals.size), negative_pct=float(neg.size/all_vals.size*100), sigma_from_negatives=sigma, curve_sd=float(all_vals.std()),
    frac_above={str(k): float((all_vals > k*sigma).mean()*100) for k in (1, 2, 3, 5)}, frac_below={str(k): float((all_vals < -k*sigma).mean()*100) for k in (1, 2, 3, 5)})
curve_max = np.array([np.max(c) for v in curves.values() for c in v["curves"].values() if len(c)])
res["real_curve"]["curves_max_above_3sigma_pct"] = float((curve_max > 3*sigma).mean()*100)
del all_vals, neg
print(res["real_curve"], flush=True)

rng = np.random.default_rng(20261030)
res["perturb"] = {}
for name, sd in [("sd_0.02", 0.02), ("sd_sigma", sigma), ("sd_0.05", 0.05)]:
    rows, moved = [], []
    for draw in range(5):
        vals, mv = recompute(lambda v: v + rng.normal(0, sd, v.shape))
        rows.append(outcome(score(vals))); moved.append(mv*100)
        print(name, draw, rows[-1], round(mv*100, 1), flush=True)
    res["perturb"][name] = dict(sd=sd, peak_left_chunk_pct=float(np.mean(moved)), **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, acc_min=min(r["acc"] for r in rows), acc_max=max(r["acc"] for r in rows))
    json.dump(res, open(out_path, "w"), indent=1)
vals, mv = recompute(lambda v: np.maximum(v, 0.0))
res["clip_at_zero"] = dict(outcome(score(vals)), peak_left_chunk_pct=mv*100)
print("clip", res["clip_at_zero"], flush=True)
shuffled = []
for draw in range(5):
    perm = rng.permutation(len(frame))
    shuffled.append(outcome(score(frame[CURVE_COLS+PEAK_COLS].iloc[perm].reset_index(drop=True))))
res["shuffled_across_cases"] = {k: float(np.mean([r[k] for r in shuffled])) for k in shuffled[0]}
print("shuffled", res["shuffled_across_cases"], flush=True)
json.dump(res, open(out_path, "w"), indent=1)
print("ALL_DONE")
