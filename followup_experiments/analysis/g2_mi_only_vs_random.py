"""How far does routing with the MI-only gate branch alone go, against random review of the same
number of answers? Uses the frozen MI-only heads stored in the case-level files. No model. CPU only.
usage: g2_mi_only_vs_random.py <data root> <output csv>"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
root, out = sys.argv[1], sys.argv[2]
disc = set(json.load(open(Path(__file__).resolve().parents[1] / "run" / "discovery_q_idx.json")))
yes = lambda s: s.fillna(False).astype(str).str.lower().isin(['true', '1'])
rows = []
for m in ['qwen3_14b_reference', 'deepseek_r1_qwen14b', 'mistral7b_instruct_v03']:
    f = pd.read_csv(f'{root}/data/analysis_exports/{m}/{m}_full_case_level.csv', low_memory=False)
    for c in ['baseline_correct', 'targeted_correct']:
        f[c] = yes(f[c])
    for scope, g in (('full', f), ('exclude95', f[~f.q_idx.isin(disc)])):
        b = g.baseline_correct.to_numpy().astype(int); gain = g.targeted_correct.to_numpy().astype(int) - b; n = len(g)
        for name, (N, T, G) in {'mi_only': ('p_none_mi_only', 'p_targeted_mi_only', 'p_generic_mi_only'), 'complete': ('p_none_probe_plus_mi', 'p_targeted_probe_plus_mi', 'p_generic_probe_plus_mi')}.items():
            sel = (np.maximum(g[T].to_numpy(), g[G].to_numpy()) - g[N].to_numpy()) > 0
            k = int(sel.sum()); acc = (b.sum() + gain[sel].sum()) / n * 100; rnd = (b.sum() + gain.mean() * k) / n * 100
            uq, inv = np.unique(g.q_idx, return_inverse=True); cnt = np.bincount(inv)
            s_sel = np.bincount(inv, gain * sel, len(uq)); s_all = np.bincount(inv, gain, len(uq)); s_b = np.bincount(inv, b, len(uq))
            rng = np.random.default_rng(20261101); d = rng.integers(0, len(uq), (5000, len(uq)))
            N_ = cnt[d].sum(1); lead = (s_sel[d].sum(1) - s_all[d].sum(1) / N_ * (sel.mean() * N_)) / N_ * 100
            rows.append(dict(model=m, scope=scope, gate=name, n=n, k=k, review_rate=k / n * 100, first=b.mean() * 100, acc=acc, gain=acc - b.mean() * 100, random_same_k=rnd, lead_vs_random=acc - rnd, ci_low=np.quantile(lead, .025), ci_high=np.quantile(lead, .975), corrected=int((gain[sel] > 0).sum()), damaged=int((gain[sel] < 0).sum())))
d = pd.DataFrame(rows); d.to_csv(out, index=False, encoding='utf-8-sig')
pd.set_option('display.width', 220, 'display.max_columns', 30)
print(d.round(2).to_string())
