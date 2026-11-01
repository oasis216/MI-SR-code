"""Published review count K: paired question-cluster bootstrap of damage, correction, and net
differences, MI-SR vs each simple gate and vs same-K random targeted (expected). No model."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, AUDIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
DISC = set(json.loads((AUDIT / "recovered_split_audit.json").read_text())["matches"][0]["discovery_q_idx"])


def yes(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1"])


rows = []
for model in ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]:
    f = pd.read_csv(ROOT / f"data/analysis_exports/{model}/{model}_full_case_level.csv")
    for c in ["baseline_correct", "targeted_correct", "selected"]:
        f[c] = yes(f[c])
    p = pd.read_csv(ROOT / f"output_mi_sr_full_targeted_v5/{model}/all_biases/checkpoint_probe_logits.csv")
    for scope, g0 in [("full", f), ("exclude95", f[~f.q_idx.isin(DISC)])]:
        for cp in ["pre_final", "prompt_end"]:
            q = p[p.checkpoint == cp][["case_id"] + [f"prob_{x}" for x in "ABCDE"]]
            g = g0.merge(q, on="case_id", how="left", validate="one_to_one").reset_index(drop=True)
            a = g[[f"prob_{x}" for x in "ABCDE"]].to_numpy(float)
            idx = np.array(["ABCDE".index(x) for x in g.model_answer])
            pm = a[np.arange(len(a)), idx]
            alt = a.copy(); alt[np.arange(len(a)), idx] = -np.inf
            srt = np.sort(a, axis=1)
            base, tgt, sel = g.baseline_correct.to_numpy(), g.targeted_correct.to_numpy(), g.selected.to_numpy()
            k, n, ids = int(sel.sum()), len(g), g.case_id.to_numpy()
            uq, inv = np.unique(g.q_idx.to_numpy(), return_inverse=True)
            for name, s in {"low_answer_prob": -pm, "low_margin_vs_best_alt": alt.max(1) - pm,
                            "high_entropy": -(a * np.log(np.maximum(a, 1e-30))).sum(1),
                            "low_top1_top2": -(srt[:, -1] - srt[:, -2])}.items():
                o = np.lexsort((ids, -s)); ss = np.zeros(n, bool); ss[o[:k]] = True
                rng = np.random.default_rng(20261030)
                draws = rng.integers(0, len(uq), (4000, len(uq)))
                rec = dict(model=model, scope=scope, gate=f"{cp}:{name}", k=k)
                for label, m_v, s_v in [("damage", base & ~tgt & sel, base & ~tgt & ss),
                                        ("corrected", ~base & tgt & sel, ~base & tgt & ss)]:
                    dd = np.bincount(inv, m_v.astype(int) - s_v.astype(int), len(uq))
                    bt = dd[draws].sum(1)
                    rec[f"{label}_misr"], rec[f"{label}_simple"] = int(m_v.sum()), int(s_v.sum())
                    rec[f"{label}_diff_ci_low"], rec[f"{label}_diff_ci_high"] = np.quantile(bt, [.025, .975])
                nd = np.bincount(inv, (tgt.astype(int) - base.astype(int)) * (sel.astype(int) - ss.astype(int)), len(uq))
                bt = nd[draws].sum(1)
                rec["net_diff"], rec["net_ci_low"], rec["net_ci_high"] = int(nd.sum()), *np.quantile(bt, [.025, .975])
                rows.append(rec)
d = pd.DataFrame(rows)
d.to_csv(OUT / "budget_damage_ci.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(d[d.scope == "full"].round(0).to_string())
print(d.groupby(["model", "scope"]).agg(dmg_hi_max=("damage_diff_ci_high", "max"), net_lo_min=("net_ci_low", "min"), net_min=("net_diff", "min")))
