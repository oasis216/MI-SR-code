"""Matched total time: a simple gate needs first pass + endpoints + its reviews, MI-SR needs first pass + full analysis
+ its reviews. Seconds come from the controlled serial benchmark (R2); stored reviews are reused. No model.
usage: g6_equal_time_audit.py <data root> <output dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, EV = Path(sys.argv[1]), Path(sys.argv[2])
DISC = set(json.load(open(Path(__file__).resolve().parents[1] / "run" / "discovery_q_idx.json")))
DIRS = {"qwen3_14b_reference": "qwen_results", "deepseek_r1_qwen14b": "deepseek_results", "mistral7b_instruct_v03": "mistral_results"}


def yes(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1"])


rows = []
for model, sub in DIRS.items():
    path = EV / sub / f"timing_summary_{model}.json"
    if not path.exists():
        continue
    t = json.load(open(path)); s = t["seconds"]
    review, endpoint, analysis, first = s["review_s"]["mean"], s["endpoint_s"]["mean"], s["analysis_all_s"]["mean"], s["first_pass_s"]["mean"]
    f = pd.read_csv(ROOT / f"data/analysis_exports/{model}/{model}_full_case_level.csv")
    for c in ["baseline_correct", "targeted_correct", "selected", "mi_sr_correct"]:
        f[c] = yes(f[c])
    p = pd.read_csv(ROOT / f"output_mi_sr_full_targeted_v5/{model}/all_biases/checkpoint_probe_logits.csv")
    rate = f.selected.mean()
    misr_s = first + analysis + rate * review
    frac = min(1.0, (misr_s - first - endpoint) / review)
    for scope, g0 in [("full", f), ("exclude95", f[~f.q_idx.isin(DISC)])]:
        for cp in ["pre_final", "prompt_end"]:
            q = p[p.checkpoint == cp][["case_id"] + [f"prob_{x}" for x in "ABCDE"]]
            g = g0.merge(q, on="case_id", how="left", validate="one_to_one").reset_index(drop=True)
            a = g[[f"prob_{x}" for x in "ABCDE"]].to_numpy(float)
            idx = np.array(["ABCDE".index(x) for x in g.model_answer])
            pm = a[np.arange(len(a)), idx]
            alt = a.copy(); alt[np.arange(len(a)), idx] = -np.inf
            srt = np.sort(a, axis=1)
            base, tgt, misr = g.baseline_correct.to_numpy(), g.targeted_correct.to_numpy(), g.mi_sr_correct.to_numpy()
            n, ids = len(g), g.case_id.to_numpy(); k = int(round(frac * n))
            uq, inv = np.unique(g.q_idx.to_numpy(), return_inverse=True); c = np.bincount(inv)
            for name, score in {"low_answer_prob": -pm, "low_margin_vs_best_alt": alt.max(1) - pm,
                                "high_entropy": -(a * np.log(np.maximum(a, 1e-30))).sum(1),
                                "low_top1_top2": -(srt[:, -1] - srt[:, -2])}.items():
                o = np.lexsort((ids, -score)); sel = np.zeros(n, bool); sel[o[:k]] = True
                fin = np.where(sel, tgt, base)
                gain = (tgt.astype(int) - base.astype(int))[o]
                best_k = int(np.argmax(np.cumsum(gain)[:k])) + 1 if k else 0  # most favourable review count within the budget, chosen in hindsight
                best_acc = (base.sum() + gain[:best_k].sum()) / n * 100
                d = np.bincount(inv, misr.astype(int) - fin.astype(int), len(uq))
                dr = np.random.default_rng(20261031).integers(0, len(uq), (4000, len(uq)))
                bt = d[dr].sum(1) / c[dr].sum(1) * 100
                rows.append(dict(model=model, scope=scope, gate=f"{cp}:{name}", misr_s=misr_s, at_s=first + endpoint + review, simple_review_frac=frac, k=k,
                                 misr_acc=misr.mean() * 100, simple_acc=fin.mean() * 100, lead=(misr.mean() - fin.mean()) * 100, ci_low=np.quantile(bt, .025), ci_high=np.quantile(bt, .975),
                                 misr_damaged=int((base & ~misr).sum()), simple_damaged=int((base & ~fin).sum()), best_within_budget_frac=best_k / n, best_within_budget_acc=best_acc))
d = pd.DataFrame(rows)
d.to_csv(EV / "g6_equal_time_audit.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(d.round(2).to_string())
