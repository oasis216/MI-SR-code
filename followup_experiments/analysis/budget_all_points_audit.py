"""At every review count k: MI-SR ranking vs best/each simple gate, same targeted outcomes.
Question-cluster bootstrap at fixed budgets, full set and excluding the 95 development
questions. Frozen scores only, nothing tuned on evaluation data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, AUDIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
DISC = set(json.loads((AUDIT / "recovered_split_audit.json").read_text())["matches"][0]["discovery_q_idx"])
MODELS = ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]


def yes(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1"])


rows, sweep = [], []
for model in MODELS:
    f = pd.read_csv(ROOT / f"data/analysis_exports/{model}/{model}_full_case_level.csv")
    for c in ["baseline_correct", "targeted_correct", "mi_sr_correct", "selected"]:
        f[c] = yes(f[c])
    p = pd.read_csv(ROOT / f"output_mi_sr_full_targeted_v5/{model}/all_biases/checkpoint_probe_logits.csv")
    for scope in ["full", "exclude95"]:
        g0 = f if scope == "full" else f[~f.q_idx.isin(DISC)]
        scores = {}
        for cp in ["pre_final", "prompt_end"]:
            q = p[p.checkpoint == cp][["case_id"] + [f"prob_{x}" for x in "ABCDE"]]
            g = g0.merge(q, on="case_id", how="left", validate="one_to_one").reset_index(drop=True)
            a = g[[f"prob_{x}" for x in "ABCDE"]].to_numpy(float)
            idx = np.array(["ABCDE".index(x) for x in g.model_answer])
            pm = a[np.arange(len(a)), idx]
            alt = a.copy(); alt[np.arange(len(a)), idx] = -np.inf
            srt = np.sort(a, axis=1)
            scores[f"{cp}:low_answer_prob"] = -pm
            scores[f"{cp}:low_margin_vs_best_alt"] = alt.max(1) - pm
            scores[f"{cp}:high_entropy"] = -(a * np.log(np.maximum(a, 1e-30))).sum(1)
            scores[f"{cp}:low_top1_top2"] = -(srt[:, -1] - srt[:, -2])
        r = (np.maximum(g.p_targeted_correct, g.p_generic_correct) - g.p_none_correct).to_numpy()
        n = len(g); base = g.baseline_correct.to_numpy(); tgt = g.targeted_correct.to_numpy()
        ids = g.case_id.to_numpy()

        def acc_curve(s):
            o = np.lexsort((ids, -s))
            return (base.sum() + np.r_[0, (tgt[o].astype(int) - base[o].astype(int)).cumsum()]) / n, o

        mc, mo = acc_curve(r)
        curves = {k: acc_curve(v) for k, v in scores.items()}
        best = np.max([c for c, _ in curves.values()], axis=0)
        interior = np.arange(1, n)
        lead = mc[interior] - best[interior]
        below = interior[lead <= 0]
        sweep.append(dict(model=model, scope=scope, n=n, frac_k_above_best_simple=float((lead > 0).mean()),
                          min_lead_pp=float(100 * lead.min()), k_frac_at_min_lead=float(interior[lead.argmin()] / n),
                          mean_lead_pp=float(100 * (mc - best).mean()),
                          below_or_tie_k_frac_min=float(below.min() / n) if len(below) else None,
                          below_or_tie_k_frac_max=float(below.max() / n) if len(below) else None))
        uq, inv = np.unique(g.q_idx.to_numpy(), return_inverse=True)
        K = int(g.selected.sum())
        rng = np.random.default_rng(20261030)
        for fr, label in [(0.2, "20%"), (0.4, "40%"), (0.5, "50%"), (0.6, "60%"), (0.7, "70%"), (0.8, "80%"), (0.9, "90%"), (K / n, "published K")]:
            k = int(round(fr * n))
            sel_m = np.zeros(n, bool); sel_m[mo[:k]] = True
            fin_m = np.where(sel_m, tgt, base)
            bestname = max(curves, key=lambda c: curves[c][0][k])
            sel_s = np.zeros(n, bool); sel_s[curves[bestname][1][:k]] = True
            fin_s = np.where(sel_s, tgt, base)
            d = np.bincount(inv, fin_m.astype(int) - fin_s.astype(int), len(uq)); c = np.bincount(inv, minlength=len(uq))
            draws = rng.integers(0, len(uq), (4000, len(uq)))
            boot = d[draws].sum(1) / c[draws].sum(1) * 100
            rows.append(dict(model=model, scope=scope, budget=label, k=k, misr_acc=100 * fin_m.mean(), best_simple=bestname,
                             best_simple_acc=100 * fin_s.mean(), lead_pp=100 * (fin_m.mean() - fin_s.mean()),
                             ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
                             random_expected_acc=100 * (base.sum() + (k / n) * (tgt.astype(int) - base.astype(int)).sum()) / n,
                             misr_damaged=int((base & ~fin_m).sum()), simple_damaged=int((base & ~fin_s).sum()),
                             misr_corrected=int((~base & fin_m).sum()), simple_corrected=int((~base & fin_s).sum())))
d = pd.DataFrame(rows); s = pd.DataFrame(sweep)
d.to_csv(OUT / "budget_all_points_rows.csv", index=False, encoding="utf-8-sig")
s.to_csv(OUT / "budget_all_points_sweep.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(s.round(4).to_string()); print(d.round(2).to_string())
