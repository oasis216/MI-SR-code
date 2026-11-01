"""Does permuting the 46 MI-related gate columns hurt the ranking at tighter review budgets?
Frozen gate, stored outcomes, same review count on both sides. Also reports the share of absolute
standardized gate weight on these columns. CPU only. usage: <deep-test-3 root> <evidence dir> <out.csv>"""
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
for target in TARGETS:
    m = models[target]
    coef = np.abs(np.ravel(m[-1].coef_ if hasattr(m, "__getitem__") else m.coef_))
    meta = json.load(open(evid / "u1_gate_sensitivity_qwen3_14b_reference.json"))
    group = set(meta["curve_columns"] + meta["held_columns"])
    print(target, "weight share of 46 columns", round(float(sum(c for c, n in zip(coef, cols) if n in group) / coef.sum()), 3), flush=True)
for model in ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]:
    d = root / "output_mi_sr_full_targeted_v5" / model / "all_biases"
    meta = json.load(open(evid / f"u1_gate_sensitivity_{model}.json"))
    group = meta["curve_columns"] + meta["held_columns"]
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
    uq, inv = np.unique(labels.loc[ids, "q_idx"].to_numpy(), return_inverse=True)
    counts = np.bincount(inv)
    gain = rev.astype(int) - base.astype(int)
    rng = np.random.default_rng(20261030)

    def r(x):
        p = np.column_stack([models[c].predict_proba(x[cols])[:, 1] for c in TARGETS])
        return p[:, 1:].max(1) - p[:, 0]

    stored = r(frame)
    published_k = int((stored > 0).sum())
    shuffled = []
    for draw in range(20):
        x = frame.copy()
        x[group] = frame[group].iloc[rng.permutation(len(frame))].to_numpy()
        shuffled.append(r(x))
    order = np.argsort(-stored, kind="stable")
    for label, k in [("5%", .05), ("10%", .10), ("20%", .20), ("30%", .30), ("40%", .40), ("50%", .50), ("published", published_k / len(ids))]:
        k = int(round(k * len(ids)))
        pick = np.zeros(len(ids), bool); pick[order[:k]] = True
        res = []
        for s in shuffled:
            alt = np.zeros(len(ids), bool); alt[np.argsort(-s, kind="stable")[:k]] = True
            diff = np.bincount(inv, (pick.astype(int) - alt.astype(int)) * gain, len(uq))
            draws = rng.integers(0, len(uq), (1000, len(uq)))
            boot = diff[draws].sum(1) / counts[draws].sum(1) * 100
            res.append(dict(lead=(gain[pick].sum() - gain[alt].sum()) / len(ids) * 100, low=np.quantile(boot, .025), high=np.quantile(boot, .975),
                            cor=int((alt & rev & ~base).sum()), dmg=int((alt & ~rev & base).sum())))
        a = pd.DataFrame(res).mean()
        rows.append(dict(model=model, budget=label, k=k, acc=(base.sum() + gain[pick].sum()) / len(ids) * 100, corrected=int((pick & rev & ~base).sum()), damaged=int((pick & ~rev & base).sum()),
                         shuf_acc=(base.sum() + gain[pick].sum()) / len(ids) * 100 - a.lead, shuf_corrected=a.cor, shuf_damaged=a.dmg, lead_pp=a.lead, ci_low=a.low, ci_high=a.high,
                         net_per_100=(gain[pick].sum()) / k * 100, shuf_net_per_100=(a.cor - a.dmg) / k * 100))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
