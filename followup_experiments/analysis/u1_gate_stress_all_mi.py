"""U1 stress test on stored outputs, CPU only. Destroys all MI-dependent gate columns by permuting them
jointly across cases, then reruns the frozen gate with the published rule r>0. It is not a rerun with
corrected numerics: it asks how far the result can move if those columns carry no case information.

usage: u1_gate_stress_all_mi.py <deep-test-3 root> <evidence dir> <out.csv>
"""
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

root, evid, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
art = root / "frozen_artifacts"
cols = json.load(open(art / "frozen_gate_v002.json"))["feature_columns"]
models = joblib.load(art / "frozen_gate_v002_models.joblib")
TARGETS = ("baseline_correct", "targeted_correct", "generic_correct")
rows = []
for model in ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]:
    d = root / "output_mi_sr_full_targeted_v5" / model / "all_biases"
    meta = json.load(open(evid / f"u1_gate_sensitivity_{model}.json"))
    curve_cols, held = meta["curve_columns"], meta["held_columns"]
    frame = pickle.load(open(d / "deploy_gate_features_before_unblinding.pkl", "rb")).reset_index(drop=True)
    policy = pd.read_csv(d / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    frame = frame[frame.case_id.isin(policy.index)].reset_index(drop=True)
    labels = pd.read_csv(d / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id")
    review = pd.read_csv(d / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id")
    ids = frame.case_id.to_numpy()
    q = labels.loc[ids, "q_idx"].to_numpy()
    gold = labels.loc[ids, "gold"].astype(str).to_numpy()
    base = labels.loc[ids, "biased_correct"].astype(bool).to_numpy()
    rv = review.reindex(ids)
    rev = np.where(rv.reeval_complete.fillna(False).astype(bool).to_numpy(), rv.reeval_answer.astype(str).to_numpy() == gold, base)
    stored = policy.loc[ids, "review_score"].to_numpy() > 0
    uq, inv = np.unique(q, return_inverse=True)
    counts = np.bincount(inv)
    rng = np.random.default_rng(20261030)

    def summarize(t, name, draw):
        final = np.where(t, rev, base); ref = np.where(stored, rev, base)
        diff = np.bincount(inv, final.astype(int) - ref.astype(int), len(uq))
        boot = diff[(draws := rng.integers(0, len(uq), (2000, len(uq))))].sum(1) / counts[draws].sum(1) * 100
        rows.append(dict(model=model, variant=name, draw=draw, n=len(t), reviews=int(t.sum()), acc=final.mean()*100, delta_pp=(final.mean()-ref.mean())*100,
                         ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975), corrected=int((t & rev & ~base).sum()), damaged=int((t & ~rev & base).sum()),
                         flip_pct=(t != stored).mean()*100, net_per_review=((t & rev & ~base).sum()-(t & ~rev & base).sum())/max(t.sum(), 1)*100))
        print(rows[-1], flush=True)

    def r(x):
        p = np.column_stack([models[c].predict_proba(x[cols])[:, 1] for c in TARGETS])
        return p[:, 1:].max(1) - p[:, 0]

    assert np.abs(r(frame) - policy.loc[ids, "review_score"].to_numpy()).max() < 1e-9
    summarize(stored, "published", 0)
    for name, group in [("shuffle_17_curve", curve_cols), ("shuffle_29_peak_probe", held), ("shuffle_all_46_mi", curve_cols + held)]:
        for draw in range(20):
            x = frame.copy()
            x[group] = frame[group].iloc[rng.permutation(len(frame))].to_numpy()
            summarize(r(x) > 0, name, draw)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
print("ALL_DONE")
