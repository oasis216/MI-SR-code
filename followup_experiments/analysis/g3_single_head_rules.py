"""Current rule max(pT, pG) > pN against rules using only pT or only pG. Stored targeted reviews, no model.
Question-cluster bootstrap of the accuracy difference. usage: g3_single_head_rules.py <data root> <output csv>"""
import sys
import numpy as np, pandas as pd
root, out = sys.argv[1], sys.argv[2]
yes = lambda s: s.fillna(False).astype(str).str.lower().isin(['true', '1'])
rows = []
for m in ['qwen3_14b_reference', 'deepseek_r1_qwen14b', 'mistral7b_instruct_v03']:
    f = pd.read_csv(f'{root}/data/analysis_exports/{m}/{m}_full_case_level.csv', low_memory=False)
    for c in ['baseline_correct', 'targeted_correct']:
        f[c] = yes(f[c])
    gain = f.targeted_correct.astype(int).to_numpy() - f.baseline_correct.astype(int).to_numpy()
    N, T, G = f.p_none_correct.to_numpy(), f.p_targeted_correct.to_numpy(), f.p_generic_correct.to_numpy()
    cur = np.maximum(T, G) > N
    uq, inv = np.unique(f.q_idx, return_inverse=True); cnt = np.bincount(inv)
    draws = np.random.default_rng(20261101).integers(0, len(uq), (5000, len(uq)))
    for name, alt in (('pT_only', T > N), ('pG_only', G > N)):
        d = np.bincount(inv, gain * (cur.astype(int) - alt.astype(int)), len(uq)); b = d[draws].sum(1) / cnt[draws].sum(1) * 100
        rows.append(dict(model=m, rule=name, reviews=int(alt.sum()), current_reviews=int(cur.sum()), accuracy=(f.baseline_correct.sum() + gain[alt].sum()) / len(f) * 100,
                         current_accuracy=(f.baseline_correct.sum() + gain[cur].sum()) / len(f) * 100, lead=d.sum() / len(f) * 100, ci_low=np.quantile(b, .025), ci_high=np.quantile(b, .975)))
d = pd.DataFrame(rows); d.to_csv(out, index=False, encoding='utf-8-sig'); print(d.round(2).to_string())
