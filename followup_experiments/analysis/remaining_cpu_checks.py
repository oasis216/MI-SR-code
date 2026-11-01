"""CPU checks for U3 (single five-option scorer), U4 (overfitting of the 74-feature gate), U7 (locator vs top alternative).
Stored outputs only. All fits use the frozen recipe on the 95 development questions; evaluation excludes them."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.signal import find_peaks
ROOT = Path(sys.argv[1]); AUDIT = Path(sys.argv[2]); OUT = Path(sys.argv[3])
DISC = set(json.load(open(AUDIT / "recovered_split_audit.json"))["matches"][0]["discovery_q_idx"])
art = ROOT / "frozen_artifacts"
gate = joblib.load(art / "frozen_gate_v002_models.joblib"); cols = json.load(open(art / "frozen_gate_v002.json"))["feature_columns"]
loc = joblib.load(art / "frozen_challenger_v001.joblib"); loc_cols = json.load(open(art / "frozen_challenger_v001.json"))["feature_columns"]
MODELS = ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]
L = list("ABCDE")
res = {}

def ci(d, q, n=4000, seed=1):
    uq, inv = np.unique(q, return_inverse=True); s = np.bincount(inv, d, len(uq)); c = np.bincount(inv)
    dr = np.random.default_rng(seed).integers(0, len(uq), (n, len(uq))); b = s[dr].sum(1) / c[dr].sum(1) * 100
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))

def load(m):
    d = ROOT / "output_mi_sr_full_targeted_v5" / m / "all_biases"
    pol = pd.read_csv(d / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    lab = pd.read_csv(d / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id").loc[pol.index]
    rv = pd.read_csv(d / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id").reindex(pol.index)
    gv = pd.read_csv(d / "generic_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id").reindex(pol.index) if (d / "generic_all_reevaluation.csv").exists() else None
    base = lab.biased_correct.astype(bool).to_numpy(); gold = lab.gold.astype(str).to_numpy()
    rev = np.where(rv.reeval_complete.fillna(False).astype(bool).to_numpy(), rv.reeval_answer.astype(str).to_numpy() == gold, base)
    gen = None if gv is None else np.where(gv.reeval_complete.fillna(False).astype(bool).to_numpy(), gv.reeval_answer.astype(str).to_numpy() == gold, base)
    return d, pol, lab, base, gold, rev, gen

def five_frame(d, pol, lab, m):
    p = pd.read_csv(d / "checkpoint_probe_logits.csv"); p = p[p.checkpoint.isin(["prompt_end", "pre_final"])]
    rows = []
    for r in p.itertuples(index=False):
        for c in L:
            rows.append((r.case_id, c, r.checkpoint, getattr(r, f"prob_{c}"), getattr(r, f"logit_{c}"), r.candidate_entropy_normalized))
    x = pd.DataFrame(rows, columns=["case_id", "candidate", "cp", "prob", "logit", "entropy"])
    w = x.pivot_table(index=["case_id", "candidate"], columns="cp", values=["prob", "logit", "entropy"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]; w = w.reset_index()
    w = w[w.case_id.isin(pol.index)]
    for cp in ("prompt_end", "pre_final"):
        w[f"log_prob_{cp}"] = np.log(w[f"prob_{cp}"].clip(1e-8, 1))
        g = w.groupby("case_id")[f"log_prob_{cp}"]
        w[f"z_log_prob_{cp}"] = (w[f"log_prob_{cp}"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        w[f"rank_prob_{cp}"] = w.groupby("case_id")[f"prob_{cp}"].rank(ascending=False)
    w["probe_log_growth"] = w.log_prob_pre_final - w.log_prob_prompt_end
    w["probe_probability_growth"] = w.prob_pre_final - w.prob_prompt_end
    w["is_first"] = (w.candidate == w.case_id.map(pol.model_answer)).astype(int)
    w["is_gold"] = (w.candidate == w.case_id.map(lab.gold.astype(str))).astype(int)
    w["q_idx"] = w.case_id.map(lab.q_idx)
    return w

# ---- U7 and U3
qd, qp, ql, qb, qg, qr, qgen = load(MODELS[0])
qw = five_frame(qd, qp, ql, MODELS[0])
feat5 = loc_cols + ["is_first"]
train = qw[qw.q_idx.isin(DISC)]
scorer = clone(loc).fit(train[feat5], train.is_gold)
print("scorer train", len(train), train.is_gold.mean(), flush=True)
u3, u7 = [], []
for m in MODELS:
    d, pol, lab, base, gold, rev, gen = load(m)
    q = lab.q_idx.to_numpy(); keep = ~np.isin(q, list(DISC))
    w = five_frame(d, pol, lab, m); w["s"] = scorer.predict_proba(w[feat5])[:, 1]
    ids = pol.index.to_numpy()
    top = w.loc[w.groupby("case_id").s.idxmax()].set_index("case_id").reindex(ids)
    first_s = w[w.is_first == 1].set_index("case_id").s.reindex(ids)
    alt = w[w.is_first == 0]; alt_top = alt.loc[alt.groupby("case_id").s.idxmax()].set_index("case_id").reindex(ids)
    margin = (alt_top.s - first_s).to_numpy()
    r0 = pol.review_score.to_numpy(); K = int((r0 > 0)[keep].sum())
    gain = rev.astype(int) - base.astype(int)
    wrong = keep & ~base
    locc = pol.challenger.astype(str).to_numpy(); rawc = pol.raw_prefinal_challenger.astype(str).to_numpy()
    h_loc = (locc == gold); h_raw = (rawc == gold); h_sc = (alt_top.candidate.astype(str).to_numpy() == gold)
    u7.append(dict(model=m, scope="excl95", n_wrong=int(wrong.sum()), locator=float(h_loc[wrong].mean()*100), top_alt=float(h_raw[wrong].mean()*100), diff=float((h_loc[wrong].mean()-h_raw[wrong].mean())*100), ci=ci(h_loc[wrong].astype(int)-h_raw[wrong].astype(int), q[wrong]), disagree_pct=float((locc[wrong] != rawc[wrong]).mean()*100),
                   locator_full=float(h_loc[~base].mean()*100), top_alt_full=float(h_raw[~base].mean()*100), scorer_alt=float(h_sc[wrong].mean()*100)))
    def sel(score, k):
        s = np.zeros(len(ids), bool); o = np.argsort(-score[keep], kind="stable")[:k]; s[np.where(keep)[0][o]] = True; return s
    out = dict(model=m, n=int(keep.sum()), K=K, first=float(base[keep].mean()*100), scorer_top_is_gold=float((top.candidate.astype(str).to_numpy()[keep] == gold[keep]).mean()*100), scorer_changes_answer_pct=float((top.is_first.to_numpy()[keep] == 0).mean()*100), scorer_alt_same_as_locator_pct=float((alt_top.candidate.astype(str).to_numpy()[keep] == locc[keep]).mean()*100))
    # as a direct replacement: answer = scorer's top option, no review
    out["scorer_answer_acc"] = out["scorer_top_is_gold"]
    # as a gate at the published threshold-free rule and at the same review count
    natural = keep & (top.is_first.to_numpy() == 0)
    out["natural_reviews"] = int(natural.sum()); out["natural_acc"] = float((base[keep].sum() + gain[natural].sum()) / keep.sum() * 100); out["natural_dmg"] = int((natural & ~rev & base).sum()); out["natural_cor"] = int((natural & rev & ~base).sum())
    sm, ss = sel(r0, K), sel(margin, K)
    out.update(misr_acc=float((base[keep].sum() + gain[sm].sum()) / keep.sum() * 100), same_k_scorer_acc=float((base[keep].sum() + gain[ss].sum()) / keep.sum() * 100), misr_cor=int((sm & rev & ~base).sum()), misr_dmg=int((sm & ~rev & base).sum()), scorer_cor=int((ss & rev & ~base).sum()), scorer_dmg=int((ss & ~rev & base).sum()))
    d_ = (sm.astype(int) - ss.astype(int)) * gain
    out["lead"] = float(d_[keep].sum() / keep.sum() * 100); out["lead_ci"] = ci(d_[keep], q[keep])
    for fr in (.2, .4, .5):
        k = int(round(fr * keep.sum())); a, b = sel(r0, k), sel(margin, k); dd = (a.astype(int) - b.astype(int)) * gain
        out[f"lead_{int(fr*100)}"] = float(dd[keep].sum() / keep.sum() * 100); out[f"lead_{int(fr*100)}_ci"] = ci(dd[keep], q[keep])
    u3.append(out); print(out, flush=True); print(u7[-1], flush=True)
res["u3"] = u3; res["u7"] = u7

# ---- U4
f = pickle.load(open(qd / "deploy_gate_features_before_unblinding.pkl", "rb")).set_index("case_id").loc[qp.index]
q = ql.q_idx.to_numpy(); dev = np.isin(q, list(DISC))
y = {"baseline_correct": qb, "targeted_correct": qr, "generic_correct": qgen}
u4 = {"n_dev": int(dev.sum()), "n_rest": int((~dev).sum()), "heads": {}}
for t, model in gate.items():
    X = f[cols]; yy = y[t].astype(int)
    p = model.predict_proba(X)[:, 1]
    oof = np.full(dev.sum(), np.nan); Xd, yd, gd = X[dev], yy[dev], q[dev]
    for a, b in GroupKFold(5).split(Xd, yd, gd):
        oof[b] = clone(model).fit(Xd.iloc[a], yd[a]).predict_proba(Xd.iloc[b])[:, 1]
    # effective degrees of freedom at the fitted solution
    Z = model[:-1].transform(Xd); lr = model[-1]; eta = lr.decision_function(Z); w = 1 / (1 + np.exp(eta)) * 1 / (1 + np.exp(-eta))
    cw = np.where(yd == 1, len(yd) / (2 * yd.sum()), len(yd) / (2 * (len(yd) - yd.sum())))
    A = (Z * (w * cw)[:, None]).T @ Z; lam = 1 / lr.C
    edf = float(np.trace(A @ np.linalg.inv(A + lam * np.eye(A.shape[0]))))
    u4["heads"][t] = dict(auc_dev_in_sample=float(roc_auc_score(yd, p[dev])), auc_dev_grouped_oof=float(roc_auc_score(yd, oof)), auc_rest=float(roc_auc_score(yy[~dev], p[~dev])), effective_df=edf, coef_l2=float(np.linalg.norm(lr.coef_)))
    print(t, u4["heads"][t], flush=True)
gain = qr.astype(int) - qb.astype(int); sel_ = qp.review_score.to_numpy() > 0
u4["gain_dev"] = float(((qb.sum() * 0 + (qb[dev].sum() + gain[dev & sel_].sum())) / dev.sum() - qb[dev].mean()) * 100); u4["gain_rest"] = float(((qb[~dev].sum() + gain[~dev & sel_].sum()) / (~dev).sum() - qb[~dev].mean()) * 100)
res["u4"] = u4; print(u4["gain_dev"], u4["gain_rest"], flush=True)
json.dump(res, open(OUT / "remaining_cpu_checks.json", "w"), indent=1, default=float)
print("DONE")
