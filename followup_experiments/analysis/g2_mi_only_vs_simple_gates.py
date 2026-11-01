"""Same review count: the frozen MI-only gate branch against each simple probability gate, using the stored
targeted reviews. Question-cluster bootstrap of the accuracy lead and damage difference. No model. CPU only.
usage: g2_mi_only_vs_simple_gates.py <data root> <experiment root> <output csv>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
DATA, EXP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
disc = set(json.load(open(Path(__file__).resolve().parents[1] / "run" / "discovery_q_idx.json")))
yes = lambda s: s.fillna(False).astype(str).str.lower().isin(['true', '1'])
rows = []
for model in ['qwen3_14b_reference', 'deepseek_r1_qwen14b', 'mistral7b_instruct_v03']:
    f = pd.read_csv(DATA / f'data/analysis_exports/{model}/{model}_full_case_level.csv', low_memory=False)
    for c in ['baseline_correct', 'targeted_correct']:
        f[c] = yes(f[c])
    p = pd.read_csv(EXP / f'output_mi_sr_full_targeted_v5/{model}/all_biases/checkpoint_probe_logits.csv')
    for scope, g0 in [('full', f), ('exclude95', f[~f.q_idx.isin(disc)])]:
        for cp in ['pre_final', 'prompt_end']:
            q = p[p.checkpoint == cp][['case_id'] + [f'prob_{x}' for x in 'ABCDE']]
            g = g0.merge(q, on='case_id', how='left', validate='one_to_one').reset_index(drop=True)
            a = g[[f'prob_{x}' for x in 'ABCDE']].to_numpy(float)
            idx = np.array(['ABCDE'.index(x) for x in g.model_answer])
            pm = a[np.arange(len(a)), idx]
            alt = a.copy(); alt[np.arange(len(a)), idx] = -np.inf
            srt = np.sort(a, axis=1)
            base, tgt = g.baseline_correct.to_numpy(), g.targeted_correct.to_numpy()
            sel = (np.maximum(g.p_targeted_mi_only, g.p_generic_mi_only) - g.p_none_mi_only).to_numpy() > 0
            k, n, ids = int(sel.sum()), len(g), g.case_id.to_numpy()
            gain = tgt.astype(int) - base.astype(int)
            uq, inv = np.unique(g.q_idx.to_numpy(), return_inverse=True); cnt = np.bincount(inv)
            for name, s in {'low_answer_prob': -pm, 'low_margin_vs_best_alt': alt.max(1) - pm,
                            'high_entropy': -(a * np.log(np.maximum(a, 1e-30))).sum(1),
                            'low_top1_top2': -(srt[:, -1] - srt[:, -2])}.items():
                o = np.lexsort((ids, -s)); ss = np.zeros(n, bool); ss[o[:k]] = True
                rng = np.random.default_rng(20261101); draws = rng.integers(0, len(uq), (4000, len(uq)))
                dd = np.bincount(inv, gain * (sel.astype(int) - ss.astype(int)), len(uq))
                boot = dd[draws].sum(1) / cnt[draws].sum(1) * 100
                dm = np.bincount(inv, (base & ~tgt & sel).astype(int) - (base & ~tgt & ss).astype(int), len(uq))
                bd = dm[draws].sum(1)
                rows.append(dict(model=model, scope=scope, gate=f'{cp}:{name}', k=k, mi_only_acc=(base.sum() + gain[sel].sum()) / n * 100, simple_acc=(base.sum() + gain[ss].sum()) / n * 100,
                                 lead=gain[sel].sum() / n * 100 - gain[ss].sum() / n * 100, ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975),
                                 damage_mi_only=int((base & ~tgt & sel).sum()), damage_simple=int((base & ~tgt & ss).sum()), damage_ci_low=np.quantile(bd, .025), damage_ci_high=np.quantile(bd, .975)))
d = pd.DataFrame(rows); d.to_csv(OUT, index=False, encoding='utf-8-sig')
pd.set_option('display.width', 220, 'display.max_columns', 30, 'display.max_rows', 60)
print(d[d.scope == 'exclude95'].round(2).to_string())
e = d[d.scope == 'exclude95']
print(e.groupby('model').agg(lead_min=('lead', 'min'), lead_max=('lead', 'max'), ci_pos=('ci_low', lambda s: int((s > 0).sum())), ci_neg=('ci_high', lambda s: int((s < 0).sum())), dmg_sig=('damage_ci_high', lambda s: int((s < 0).sum()))).round(2))
