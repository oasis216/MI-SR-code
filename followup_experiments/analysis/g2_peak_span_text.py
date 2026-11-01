"""W2 follow-up: isolate the peak-span text features inside the frozen gate. Permutes only those columns
jointly across cases and compares accuracy at the same review count. Stored outcomes, CPU only.
usage: g2_peak_span_text.py <experiment root> <out.csv>"""
import json, os, pickle, sys
from pathlib import Path
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
import joblib
import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), sys.argv[2]
art = root / "frozen_artifacts"
cols = json.load(open(art / "frozen_gate_v002.json"))["feature_columns"]
models = joblib.load(art / "frozen_gate_v002_models.joblib")
TARGETS = ("baseline_correct", "targeted_correct", "generic_correct")
TEXT = ["challenger_mi_semantic_support", "challenger_peak_negative", "challenger_peak_positive", "challenger_peak_reasoning",
        "gate_mi_semantic", "mi_negative_count", "mi_override", "mi_positive_count", "mi_reasoning_count", "mi_semantic_count",
        "raw_mi_semantic_support", "selected_mi_semantic_support"]
assert set(TEXT) <= set(cols)
disc = set(json.load(open(Path(__file__).resolve().parents[1] / "run" / "discovery_q_idx.json")))
for t in TARGETS:
    coef = np.abs(np.ravel(models[t][-1].coef_))
    print(t, "weight share of text columns", round(float(sum(c for c, n in zip(coef, cols) if n in TEXT) / coef.sum()), 3), flush=True)
rows = []
for model in ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]:
    d = root / "output_mi_sr_full_targeted_v5" / model / "all_biases"
    frame = pickle.load(open(d / "deploy_gate_features_before_unblinding.pkl", "rb")).reset_index(drop=True)
    policy = pd.read_csv(d / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    frame = frame[frame.case_id.isin(policy.index)].reset_index(drop=True)
    labels = pd.read_csv(d / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id")
    review = pd.read_csv(d / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id")
    ids = frame.case_id.to_numpy()
    gold = labels.loc[ids, "gold"].astype(str).to_numpy()
    base = labels.loc[ids, "biased_correct"].astype(bool).to_numpy()
    rv = review.reindex(ids)
    rev = np.where(rv.reeval_complete.fillna(False).astype(bool).to_numpy(), rv.reeval_answer.astype(str).to_numpy() == gold, base)
    q = labels.loc[ids, "q_idx"].to_numpy()
    gain = rev.astype(int) - base.astype(int)
    rng = np.random.default_rng(20261101)
    def r(x):
        p = np.column_stack([models[c].predict_proba(x[cols])[:, 1] for c in TARGETS])
        return p[:, 1:].max(1) - p[:, 0]
    stored = r(frame)
    published_k = int((stored > 0).sum())
    shuffled = []
    for draw in range(20):
        x = frame.copy()
        x[TEXT] = frame[TEXT].iloc[rng.permutation(len(frame))].to_numpy()
        shuffled.append(r(x))
    print(model, "n", len(ids), "published k", published_k, "first", round(base.mean() * 100, 2), flush=True)
    for scope, keep in (("full", np.ones(len(ids), bool)), ("exclude95", ~np.isin(q, list(disc)))):
        idx = np.where(keep)[0]; g = gain[idx]; uq, inv = np.unique(q[idx], return_inverse=True); counts = np.bincount(inv)
        pk = int((stored[idx] > 0).sum())
        for label, k in [("20%", .20), ("30%", .30), ("40%", .40), ("50%", .50), ("published", pk / len(idx))]:
            k = int(round(k * len(idx)))
            order = np.argsort(-stored[idx], kind="stable")
            pick = np.zeros(len(idx), bool); pick[order[:k]] = True
            res = []
            for s in shuffled:
                alt = np.zeros(len(idx), bool); alt[np.argsort(-s[idx], kind="stable")[:k]] = True
                diff = np.bincount(inv, (pick.astype(int) - alt.astype(int)) * g, len(uq))
                draws = rng.integers(0, len(uq), (1000, len(uq)))
                boot = diff[draws].sum(1) / counts[draws].sum(1) * 100
                res.append(dict(lead=(g[pick].sum() - g[alt].sum()) / len(idx) * 100, low=np.quantile(boot, .025), high=np.quantile(boot, .975),
                                cor=int((g[alt] > 0).sum()), dmg=int((g[alt] < 0).sum())))
            a = pd.DataFrame(res).mean()
            rows.append(dict(model=model, scope=scope, budget=label, k=k, corrected=int((g[pick] > 0).sum()), damaged=int((g[pick] < 0).sum()),
                             shuf_corrected=a.cor, shuf_damaged=a.dmg, lead_pp=a.lead, ci_low=a.low, ci_high=a.high))
            print({k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
