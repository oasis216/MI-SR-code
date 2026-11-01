"""Locator vs highest-probability alternative on first-pass errors, split by whether the cue caused the error.
No model. Excludes the 95 questions. usage: u7_by_error_type.py <data root> <output dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, EV = Path(sys.argv[1]), Path(sys.argv[2])
DISC = set(json.load(open(Path(__file__).resolve().parents[1] / "run" / "discovery_q_idx.json")))


def yes(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1"])


def ci(d, q):
    uq, inv = np.unique(q, return_inverse=True); s = np.bincount(inv, d, len(uq)); c = np.bincount(inv)
    dr = np.random.default_rng(1).integers(0, len(uq), (4000, len(uq))); b = s[dr].sum(1) / c[dr].sum(1) * 100
    return np.quantile(b, [.025, .975])


rows = []
for m in ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]:
    f = pd.read_csv(ROOT / f"data/analysis_exports/{m}/{m}_full_case_level.csv")
    f = f[~f.q_idx.isin(DISC)]
    base, cc, sel = yes(f.baseline_correct).to_numpy(), yes(f.clean_correct_value).to_numpy(), yes(f.selected).to_numpy()
    gold, loc, top = f.gold.astype(str).to_numpy(), f.challenger.astype(str).to_numpy(), f.raw_prefinal_challenger.astype(str).to_numpy()
    q = f.q_idx.to_numpy()
    for name, g in [("all errors", ~base), ("bias-induced", ~base & cc), ("no-cue errors", ~base & ~cc), ("bias-induced, reviewed", ~base & cc & sel), ("all errors, reviewed", ~base & sel)]:
        a, b = (loc[g] == gold[g]).astype(int), (top[g] == gold[g]).astype(int)
        lo, hi = ci(a - b, q[g])
        rows.append(dict(model=m, group=name, n=int(g.sum()), locator=a.mean() * 100, top_alt=b.mean() * 100, diff=(a - b).mean() * 100, lo=lo, hi=hi, same_pick=(loc[g] == top[g]).mean() * 100))
d = pd.DataFrame(rows)
d.to_csv(EV / "u7_by_error_type.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(d.round(2).to_string())
