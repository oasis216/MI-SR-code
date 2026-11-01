"""Accuracy at every review budget: MI-SR ranking vs fixed-direction simple gates (no model).

Reviewed cases take the existing learned-challenger targeted outcome; others keep the
first answer. Rankings use frozen scores only; no threshold is tuned on evaluation data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODELS = ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]
FRACS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def yes(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1"])


rows = []
for model in MODELS:
    f = pd.read_csv(ROOT / f"data/analysis_exports/{model}/{model}_full_case_level.csv")
    for col in ["baseline_correct", "targeted_correct", "mi_sr_correct", "selected"]:
        f[col] = yes(f[col])
    p = pd.read_csv(ROOT / "output_mi_sr_full_targeted_v5"
                    / model / "all_biases/checkpoint_probe_logits.csv")
    p = p[p.checkpoint == "pre_final"][["case_id"] + [f"prob_{x}" for x in "ABCDE"]]
    g = f.merge(p, on="case_id", how="left", validate="one_to_one")
    a = g[[f"prob_{x}" for x in "ABCDE"]].to_numpy(float)
    idx = np.array(["ABCDE".index(x) for x in g.model_answer])
    pm = a[np.arange(len(a)), idx]
    srt = np.sort(a, axis=1)
    r = np.maximum(g.p_targeted_correct, g.p_generic_correct) - g.p_none_correct
    scores = {"MI-SR r": r.to_numpy(),
              "low answer prob": -pm,
              "high entropy": -(a * np.log(np.maximum(a, 1e-30))).sum(1),
              "low top1-top2 margin": -(srt[:, -1] - srt[:, -2])}
    base, tgt = g.baseline_correct.to_numpy(), g.targeted_correct.to_numpy()
    n = len(g)
    for name, s in scores.items():
        order = np.lexsort((g.case_id.to_numpy(), -s))
        gain = (tgt[order].astype(int) - base[order].astype(int)).cumsum()
        harm = (base[order] & ~tgt[order]).cumsum()
        for fr in FRACS + [float(g.selected.mean())]:
            k = int(round(fr * n))
            rows.append(dict(model=model, gate=name, review_frac=round(k / n, 4), k=k,
                             accuracy=(base.sum() + (gain[k - 1] if k else 0)) / n,
                             damaged=int(harm[k - 1]) if k else 0))
        # mean accuracy over budgets 0..n (area under the budget curve)
        rows.append(dict(model=model, gate=name, review_frac="AUC", k=n,
                         accuracy=float(((base.sum() + np.r_[0, gain]) / n).mean()), damaged=None))
d = pd.DataFrame(rows)
d.to_csv(OUT / "budget_frontier_audit.csv", index=False, encoding="utf-8-sig")
print(d.pivot_table(index=["model", "review_frac"], columns="gate", values="accuracy",
                    aggfunc="first").round(4).to_string())
