"""Per-question token accounting at matched total tokens. All Targeted and simple gates also
need the two endpoint probes (the locator reads them), which an earlier tally omitted.
Prefill and decode tokens are counted equally; review tokens are estimates (token_cost_audit)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
tc = json.load(open(OUT / "token_cost_audit.json"))
reuse_extra = {"qwen3_14b_reference": 87, "deepseek_r1_qwen14b": 93, "mistral7b_instruct_v03": 90}


def yes(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1"])


rows = []
for m, extra in reuse_extra.items():
    t = tc[m]; q = t["per_question"]
    fp = q["first_pass_prefill"] + q["first_pass_decode"]
    probes_reuse = 2 * t["probe_suffix_tokens"]
    probes_sub = q["misr_extra_endpoint_probes"]
    misr_rev = q["misr_review_prefill"] + q["misr_review_decode"]
    at_rev = q["all_targeted_review_prefill"] + q["all_targeted_review_decode"]
    misr_sub = fp + q["misr_extra_prefill_total"] + misr_rev
    misr_reuse = fp + extra + misr_rev
    per_review = at_rev * t["n"] / t["reviews_all"]
    frac = min(1.0, (misr_reuse - fp - probes_reuse) / per_review)
    f = pd.read_csv(ROOT / f"data/analysis_exports/{m}/{m}_full_case_level.csv")
    for c in ["baseline_correct", "targeted_correct", "mi_sr_correct"]:
        f[c] = yes(f[c])
    p = pd.read_csv(ROOT / f"output_mi_sr_full_targeted_v5/{m}/all_biases/checkpoint_probe_logits.csv")
    p = p[p.checkpoint == "pre_final"]
    g = f.merge(p[["case_id"] + [f"prob_{x}" for x in "ABCDE"]], on="case_id")
    a = g[[f"prob_{x}" for x in "ABCDE"]].to_numpy()
    idx = np.array(["ABCDE".index(x) for x in g.model_answer])
    o = np.lexsort((g.case_id.to_numpy(), a[np.arange(len(a)), idx]))
    k = int(round(frac * len(g))); sel = np.zeros(len(g), bool); sel[o[:k]] = True
    b, tg = g.baseline_correct.to_numpy(), g.targeted_correct.to_numpy()
    fin = np.where(sel, tg, b)
    rows.append(dict(model=m, misr_submitted=misr_sub, all_targeted_submitted=fp + probes_sub + at_rev,
                     ratio_submitted=misr_sub / (fp + probes_sub + at_rev), misr_reuse=misr_reuse,
                     all_targeted_reuse=fp + probes_reuse + at_rev,
                     reuse_change_pct=100 * (misr_reuse / (fp + probes_reuse + at_rev) - 1),
                     decode_misr=t["misr_total_decode"], decode_at=t["all_targeted_total_decode"],
                     simple_review_frac=frac, simple_acc=100 * fin.mean(), simple_damaged=int((b & ~fin).sum()),
                     misr_acc=100 * g.mi_sr_correct.mean(), misr_damaged=int((b & ~g.mi_sr_correct.to_numpy()).sum()),
                     at_acc=100 * tg.mean(), at_damaged=int((b & ~tg).sum())))
d = pd.DataFrame(rows)
d.to_csv(OUT / "equal_compute_audit.csv", index=False, encoding="utf-8-sig")
print(d.round(2).T.to_string())
