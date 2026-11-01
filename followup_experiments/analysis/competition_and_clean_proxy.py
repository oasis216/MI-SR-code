"""(1) Is a bias-induced error visible as candidate competition? Endpoint probes only, so independent of the trajectory
numerics. (2) Closest evidence to a clean run: cases where the cue did not change the answer. Excludes the 95 questions."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT, AUDIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
DISC = set(json.load(open(AUDIT / "recovered_split_audit.json"))["matches"][0]["discovery_q_idx"])
L = list("ABCDE"); res = {}

def ci(d, q, seed=1):
    uq, inv = np.unique(q, return_inverse=True); s = np.bincount(inv, d, len(uq)); c = np.bincount(inv)
    dr = np.random.default_rng(seed).integers(0, len(uq), (4000, len(uq))); b = s[dr].sum(1) / c[dr].sum(1) * 100
    return round(float(np.quantile(b, .025)), 2), round(float(np.quantile(b, .975)), 2)

for m in ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]:
    d = ROOT / "output_mi_sr_full_targeted_v5" / m / "all_biases"
    pol = pd.read_csv(d / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    lab = pd.read_csv(d / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id").loc[pol.index]
    rv = pd.read_csv(d / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id").reindex(pol.index)
    p = pd.read_csv(d / "checkpoint_probe_logits.csv")
    pe = p[p.checkpoint == "prompt_end"].set_index("case_id").reindex(pol.index); pf = p[p.checkpoint == "pre_final"].set_index("case_id").reindex(pol.index)
    A, B = pe[[f"prob_{x}" for x in L]].to_numpy(float), pf[[f"prob_{x}" for x in L]].to_numpy(float)
    cr = pd.read_csv(d / "clean_reference_outputs.csv", encoding="utf-8-sig", usecols=["q_idx", "clean_answer", "clean_correct", "clean_complete"]).drop_duplicates("q_idx").set_index("q_idx").reindex(lab.q_idx)
    gold = lab.gold.astype(str).to_numpy(); ans = pol.model_answer.astype(str).to_numpy(); clean = cr.clean_answer.astype(str).to_numpy()
    gi = np.array([L.index(x) if x in L else -1 for x in gold]); ai = np.array([L.index(x) for x in ans]); idx = np.arange(len(pol))
    base = lab.biased_correct.astype(bool).to_numpy(); cc = clean == gold
    has_clean = cr.clean_complete.fillna(False).astype(bool).to_numpy() & np.isin(clean, L)
    rev = np.where(rv.reeval_complete.fillna(False).astype(bool).to_numpy(), rv.reeval_answer.astype(str).to_numpy() == gold, base)
    T = pol.review_score.to_numpy() > 0; q = lab.q_idx.to_numpy(); keep = ~np.isin(q, list(DISC)) & has_clean
    rank_f = (-B).argsort(1).argsort(1)[idx, gi] + 1; rank_e = (-A).argsort(1).argsort(1)[idx, gi] + 1
    ch = pol.challenger.astype(str).to_numpy(); raw = pol.raw_prefinal_challenger.astype(str).to_numpy()
    harmed = keep & cc & ~base; robust = keep & cc & base; rescued = keep & ~cc & base; stuck = keep & ~cc & ~base
    def grp(g):
        return dict(n=int(g.sum()), review_rate=float(T[g].mean() * 100), gold_prob_prompt_end=float(A[idx, gi][g].mean()), gold_prob_pre_final=float(B[idx, gi][g].mean()), answer_prob_pre_final=float(B[idx, ai][g].mean()),
                    gold_top2_pre_final_pct=float((rank_f[g] <= 2).mean() * 100), gold_top2_prompt_end_pct=float((rank_e[g] <= 2).mean() * 100), gold_top1_prompt_end_pct=float((rank_e[g] == 1).mean() * 100), gold_prob_falls_pct=float((B[idx, gi][g] < A[idx, gi][g]).mean() * 100),
                    answer_prob_rises_pct=float((B[idx, ai][g] > A[idx, ai][g]).mean() * 100), challenger_is_gold_pct=float((ch[g] == gold[g]).mean() * 100), first_answer_was_ahead_prompt_end_pct=float((A[idx, ai][g] >= A[idx, gi][g]).mean() * 100))
    r = dict(harmed=grp(harmed), robust=grp(robust), stuck=grp(stuck))
    h_t = harmed & T
    r["harmed_repair"] = dict(all=float((rev & T)[harmed].mean() * 100), reviewed=float(rev[h_t].mean() * 100), reviewed_challenger_gold=float(rev[h_t & (ch == gold)].mean() * 100), reviewed_challenger_not_gold=float(rev[h_t & (ch != gold)].mean() * 100), n_ch_gold=int((h_t & (ch == gold)).sum()), n_ch_not=int((h_t & (ch != gold)).sum()))
    r["harmed_vs_robust_review_diff"] = float(T[harmed].mean() * 100 - T[robust].mean() * 100)
    # clean proxy: cue did not change the answer
    same = keep & (ans == clean)
    gain = rev.astype(int) - base.astype(int); ga = gain * T
    def prox(g):
        return dict(n=int(g.sum()), pct_of_cases=float(g.sum() / keep.sum() * 100), review_rate=float(T[g].mean() * 100), first=float(base[g].mean() * 100), misr=float((base[g].sum() + ga[g].sum()) / g.sum() * 100), gain=float(ga[g].mean() * 100), ci=ci(ga[g], q[g]), corrected=int((T & rev & ~base)[g].sum()), damaged=int((T & ~rev & base)[g].sum()), damage_rate_of_correct=float((T & ~rev & base)[g].sum() / max(base[g].sum(), 1) * 100), at_gain=float(gain[g].mean() * 100), at_damaged=int((~rev & base)[g].sum()), at_damage_rate_of_correct=float((~rev & base)[g].sum() / max(base[g].sum(), 1) * 100))
    r["cue_no_effect"] = prox(same); r["cue_changed_answer"] = prox(keep & (ans != clean))
    r["clean_correct_kept"] = dict(n=int((keep & cc).sum()), first=float(base[keep & cc].mean() * 100), misr=float((base[keep & cc].sum() + ga[keep & cc].sum()) / (keep & cc).sum() * 100))
    r["clean_wrong"] = dict(n=int((keep & ~cc).sum()), first=float(base[keep & ~cc].mean() * 100), misr=float((base[keep & ~cc].sum() + ga[keep & ~cc].sum()) / (keep & ~cc).sum() * 100))
    res[m] = r
    print(m)
    for k, v in r.items():
        print(' ', k, {a: (round(x, 2) if isinstance(x, float) else x) for a, x in v.items()} if isinstance(v, dict) else round(v, 2), flush=True)
json.dump(res, open(OUT / "competition_and_clean_proxy.json", "w"), indent=1, default=float)
