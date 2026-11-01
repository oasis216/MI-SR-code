"""Locator refit on each model's own 95 development questions vs the frozen Qwen locator vs the highest-probability alternative.
First-pass errors on the other questions. Same recipe and features. No model. Post hoc for the transfer models.
usage: u7_self_fit.py <experiment root> <output dir>"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT, EV = Path(sys.argv[1]), Path(sys.argv[2])
DISC = set(json.load(open(Path(__file__).resolve().parents[1] / "run" / "discovery_q_idx.json")))
ART = ROOT / "frozen_artifacts"
loc = joblib.load(ART / "frozen_challenger_v001.joblib"); cols = json.load(open(ART / "frozen_challenger_v001.json"))["feature_columns"]
L = list("ABCDE")


def five(d, policy, labels):
    p = pd.read_csv(d / "checkpoint_probe_logits.csv"); p = p[p.checkpoint.isin(["prompt_end", "pre_final"])]
    rows = [(r.case_id, c, r.checkpoint, getattr(r, f"prob_{c}"), getattr(r, f"logit_{c}"), r.candidate_entropy_normalized) for r in p.itertuples(index=False) for c in L]
    x = pd.DataFrame(rows, columns=["case_id", "candidate", "cp", "prob", "logit", "entropy"])
    w = x.pivot_table(index=["case_id", "candidate"], columns="cp", values=["prob", "logit", "entropy"])
    w.columns = [f"{i}_{j}" for i, j in w.columns]; w = w.reset_index(); w = w[w.case_id.isin(policy.index)]
    for cp in ("prompt_end", "pre_final"):
        w[f"log_prob_{cp}"] = np.log(w[f"prob_{cp}"].clip(1e-8, 1)); g = w.groupby("case_id")[f"log_prob_{cp}"]
        w[f"z_log_prob_{cp}"] = (w[f"log_prob_{cp}"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        w[f"rank_prob_{cp}"] = w.groupby("case_id")[f"prob_{cp}"].rank(ascending=False)
    w["probe_log_growth"] = w.log_prob_pre_final - w.log_prob_prompt_end; w["probe_probability_growth"] = w.prob_pre_final - w.prob_prompt_end
    w["is_first"] = (w.candidate == w.case_id.map(policy.model_answer)).astype(int); w["is_gold"] = (w.candidate == w.case_id.map(labels.gold.astype(str))).astype(int); w["q_idx"] = w.case_id.map(labels.q_idx)
    return w


def ci(d, q):
    uq, inv = np.unique(q, return_inverse=True); s = np.bincount(inv, d, len(uq)); c = np.bincount(inv)
    dr = np.random.default_rng(1).integers(0, len(uq), (4000, len(uq))); b = s[dr].sum(1) / c[dr].sum(1) * 100
    return [round(float(x), 3) for x in np.quantile(b, [.025, .975])]


res = []
for m in ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]:
    d = ROOT / "output_mi_sr_full_targeted_v5" / m / "all_biases"
    pol = pd.read_csv(d / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    lab = pd.read_csv(d / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id").loc[pol.index]
    w = five(d, pol, lab); a = w[w.is_first == 0].copy()
    base = lab.biased_correct.astype(bool); wrong = set(base.index[~base])
    train = a[a.q_idx.isin(DISC) & a.case_id.isin(wrong)]
    self_fit = clone(loc).fit(train[cols], train.is_gold)
    a["frozen"] = loc.predict_proba(a[cols])[:, 1]; a["self"] = self_fit.predict_proba(a[cols])[:, 1]
    test = a[~a.q_idx.isin(DISC) & a.case_id.isin(wrong)]
    pick = lambda c: test.loc[test.groupby("case_id")[c].idxmax()].set_index("case_id").candidate
    ids = pick("self").index; gold = lab.loc[ids, "gold"].astype(str); q = lab.loc[ids, "q_idx"].to_numpy()
    fz, sf, tp = (pick("frozen").loc[ids] == gold).astype(int).to_numpy(), (pick("self").loc[ids] == gold).astype(int).to_numpy(), (pol.loc[ids, "raw_prefinal_challenger"].astype(str) == gold).astype(int).to_numpy()
    st = (pol.loc[ids, "challenger"].astype(str) == gold).astype(int).to_numpy()
    cr = pd.read_csv(d / "clean_reference_outputs.csv", encoding="utf-8-sig", usecols=["q_idx", "clean_answer", "clean_complete"]).drop_duplicates("q_idx").set_index("q_idx").reindex(q)
    bias_err = (cr.clean_complete.fillna(False).astype(bool).to_numpy() & (cr.clean_answer.astype(str).to_numpy() == gold.to_numpy()))
    split = {}
    for name, g in (("bias", bias_err), ("nocue", ~bias_err)):
        split[f"{name}_n"] = int(g.sum()); split[f"{name}_self"] = sf[g].mean() * 100; split[f"{name}_frozen"] = st[g].mean() * 100; split[f"{name}_top"] = tp[g].mean() * 100; split[f"{name}_self_vs_top"] = (sf[g] - tp[g]).mean() * 100; split[f"{name}_self_vs_top_ci"] = ci(sf[g] - tp[g], q[g])
    res.append(dict(**split, model=m, n=len(ids), train_cases=train.case_id.nunique(), published_hit=st.mean() * 100, frozen=fz.mean() * 100, self_fit=sf.mean() * 100, top_alt=tp.mean() * 100,
                    frozen_vs_top=(fz - tp).mean() * 100, frozen_vs_top_ci=ci(fz - tp, q), self_vs_top=(sf - tp).mean() * 100, self_vs_top_ci=ci(sf - tp, q), self_vs_frozen=(sf - fz).mean() * 100, self_vs_frozen_ci=ci(sf - fz, q)))
d = pd.DataFrame(res); d.to_csv(EV / "u7_self_fit.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220); print(d.round(2).T.to_string())
