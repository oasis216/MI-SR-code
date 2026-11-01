"""Compare each U1 variant with the published run, using the stored targeted reviews. Pre-specified primary test:
f64_hkh at r>0 passes if the source-question bootstrap 95% CI of the accuracy difference lies within +-0.5pp.
usage: u1_numeric_analysis.py --model qwen3_14b_reference --out <rerun dir>; writes <out>/u1_numeric_summary_<model>.json"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("U1_ROOT", str(Path(__file__).resolve().parents[2] / "original_experiments"))).resolve()
p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--out", required=True); a = p.parse_args()
src = ROOT / "output_mi_sr_full_targeted_v5" / a.model / "all_biases"
cols = json.load(open(ROOT / "frozen_artifacts/frozen_gate_v002.json"))["feature_columns"]
pol0 = pd.read_csv(src / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
labels = pd.read_csv(src / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id")
review = pd.read_csv(src / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id")
curves0 = pickle.load(open(src / "all_candidate_curve_store.pkl", "rb"))
plan0 = pd.read_csv(src / "mi_sr_peak_features/peak_probe_plan.csv", encoding="utf-8-sig").set_index(["case_id", "candidate"])
feat0 = pickle.load(open(src / "deploy_gate_features_before_unblinding.pkl", "rb")).set_index("case_id")
rng = np.random.default_rng(20261031)


def score(frame):
    return (frame[["p_targeted_correct", "p_generic_correct"]].max(axis=1) - frame.p_none_correct).to_numpy()


result = {"model": a.model, "primary_margin_pp": 0.5}
for variant in ("f32_rerun", "f64_hkh", "f32_hkh"):
    d = Path(a.out) / variant / a.model / "all_biases"
    pol = pd.read_csv(d / "frozen_mi_sr_policy_v3.csv", encoding="utf-8-sig").set_index("case_id")
    ids = pol.index.intersection(pol0.index).to_numpy()
    gold = labels.loc[ids, "gold"].astype(str).to_numpy(); base = labels.loc[ids, "biased_correct"].astype(bool).to_numpy()
    rv = review.reindex(ids)
    rev = np.where(rv.reeval_complete.fillna(False).astype(bool).to_numpy(), rv.reeval_answer.astype(str).to_numpy() == gold, base)
    gain = rev.astype(int) - base.astype(int)
    r0 = pol0.loc[ids, "review_score"].to_numpy(); r1 = score(pol.loc[ids])
    t0, t1 = r0 > 0, r1 > 0
    uq, inv = np.unique(labels.loc[ids, "q_idx"].to_numpy(), return_inverse=True); counts = np.bincount(inv)

    def compare(s1, s0):
        diff = np.bincount(inv, (s1.astype(int) - s0.astype(int)) * gain, len(uq))
        draws = rng.integers(0, len(uq), (4000, len(uq)))
        boot = diff[draws].sum(1) / counts[draws].sum(1) * 100
        return dict(reviews=int(s1.sum()), reviews_published=int(s0.sum()), acc=float((base.sum() + gain[s1].sum()) / len(ids) * 100), acc_published=float((base.sum() + gain[s0].sum()) / len(ids) * 100),
                    delta_pp=float((gain[s1].sum() - gain[s0].sum()) / len(ids) * 100), ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
                    corrected=int((s1 & rev & ~base).sum()), damaged=int((s1 & ~rev & base).sum()), corrected_published=int((s0 & rev & ~base).sum()), damaged_published=int((s0 & ~rev & base).sum()), decision_flip_pct=float((s1 != s0).mean() * 100))

    def top(score_values, k):
        s = np.zeros(len(ids), bool); s[np.argsort(-score_values, kind="stable")[:k]] = True
        return s

    res = {"n": len(ids), "r_pearson": float(np.corrcoef(r0, r1)[0, 1]), "at_r_gt_0": compare(t1, t0), "same_review_count": {}}
    for label, k in [("5%", .05), ("10%", .1), ("20%", .2), ("30%", .3), ("40%", .4), ("50%", .5), ("published", t0.mean())]:
        k = int(round(k * len(ids))); res["same_review_count"][label] = compare(top(r1, k), top(r0, k))
    store = pickle.load(open(d / "all_candidate_curve_store.pkl", "rb"))
    a0, a1, same_peak = [], [], []
    for c in ids:
        for k, v in store[c]["curves"].items():
            a0.append(np.asarray(curves0[c]["curves"][k], float)); a1.append(np.asarray(v, float))
    x0, x1 = np.concatenate(a0), np.concatenate(a1)
    del a0, a1
    res["curves"] = dict(points=int(x1.size), negative_pct=float((x1 < 0).mean() * 100), negative_pct_published=float((x0 < 0).mean() * 100), min=float(x1.min()), mean=float(x1.mean()), median=float(np.median(x1)), p99=float(np.quantile(x1, .99)),
                         pearson_with_published=float(np.corrcoef(x0, x1)[0, 1]), mean_abs_diff=float(np.abs(x0 - x1).mean()), exact_match_pct=float((x0.astype(np.float32) == x1.astype(np.float32)).mean() * 100))
    plan = pd.read_csv(d / "mi_sr_peak_features/peak_probe_plan.csv", encoding="utf-8-sig").set_index(["case_id", "candidate"])
    m = plan.join(plan0, rsuffix="_0", how="inner"); m = m[m.index.get_level_values(0).isin(ids)]
    res["peaks"] = dict(candidates=len(m), same_step_pct=float((m.peak_step == m.peak_step_0).mean() * 100), within_4_steps_pct=float(((m.peak_step - m.peak_step_0).abs() <= 4).mean() * 100),
                        same_chunk_pct=float(((m.before_step == m.before_step_0) & (m.after_step == m.after_step_0)).mean() * 100), mean_prominence=float(m.peak_prominence.mean()), mean_prominence_published=float(m.peak_prominence_0.mean()))
    feat = pickle.load(open(d / "deploy_gate_features_before_unblinding.pkl", "rb")).set_index("case_id")
    shift = {}
    for c in cols:
        f0, f1 = pd.to_numeric(feat0.loc[ids, c], errors="coerce").astype(float), pd.to_numeric(feat.loc[ids, c], errors="coerce").astype(float)
        sd = f0.std()
        shift[c] = 0.0 if not sd or np.isnan(sd) else float((f1.mean() - f0.mean()) / sd)
    res["feature_shift_top10_in_sd"] = dict(sorted(shift.items(), key=lambda kv: -abs(kv[1]))[:10])
    res["features_shifted_over_0.2sd"] = int(sum(abs(v) > .2 for v in shift.values()))
    res["challenger_same_pct"] = float((pol.loc[ids, "challenger"] == pol0.loc[ids, "challenger"]).mean() * 100)
    if variant == "f64_hkh":
        rr = res["at_r_gt_0"]
        result["primary_test_pass"] = bool(rr["ci_low"] >= -0.5 and rr["ci_high"] <= 0.5)
    result[variant] = res
    print(variant, json.dumps({k: res[k] for k in ("n", "r_pearson", "at_r_gt_0", "curves", "peaks", "challenger_same_pct")}, ensure_ascii=False), flush=True)
out = Path(a.out) / f"u1_numeric_summary_{a.model}.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
print("WROTE", out, "primary_test_pass =", result.get("primary_test_pass"))
