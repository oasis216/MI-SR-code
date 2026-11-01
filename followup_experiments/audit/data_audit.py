"""Read-only audit of submitted CSVs; never imports original project/joblib files."""
import os
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MI_SR_DATA_ROOT", "."))
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True, parents=True)
SEED = 20261025
REPS = 10000

def truth(x):
    return x.fillna(False).astype(str).str.lower().isin(['true','1'])

def bootstrap(frame, a, b):
    # Resample entire source-question clusters and retain case-weighted estimand.
    sums = (a.astype(int)-b.astype(int)).groupby(frame.q_idx).sum().to_numpy()
    ns = frame.groupby('q_idx').size().to_numpy()
    rng = np.random.default_rng(SEED)
    draws=[]
    for start in range(0,REPS,500):
        ind=rng.integers(0,len(ns),(min(500,REPS-start),len(ns)))
        draws.extend((sums[ind].sum(axis=1)/ns[ind].sum(axis=1)*100).tolist())
    return np.quantile(draws,[.025,.975]).tolist()

def manifest(n):
    questions=np.arange(n)
    np.random.default_rng(20260719).shuffle(questions)
    d={'seed':20260719,'group_column':'q_idx','discovery_q_idx':sorted(map(int,questions[:95])),
       'holdout_q_idx':sorted(map(int,questions[95:])), 'same_source_question_bias_variants_kept_together':True}
    h=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return d,h

target=json.loads((Path(__file__).resolve().parents[2]/'original_experiments/frozen_artifacts/frozen_artifact_manifest.json').read_text())['split_hash']
matches=[]
for n in range(95,1274):
    d,h=manifest(n)
    if h==target:
        d['split_hash']=h
        d['recovery_note']='Reconstructed from packaged algorithm assuming contiguous original q_idx 0..n-1; full SHA-256 equals packaged frozen manifest.'
        d['source_question_n']=n
        matches.append(d)
(OUT/'recovered_split_audit.json').write_text(json.dumps({'target_hash':target,'matches':matches},indent=2),encoding='utf-8')
discovery=set(matches[0]['discovery_q_idx']) if len(matches)==1 else None

audit={}; policy_rows=[]; subset_rows=[]; paired_rows=[]; clean_rows=[]; ablation_rows=[]; localization_rows=[]; matched_rows=[]; comparison_rows=[]; risk_rows=[]
models=['qwen3_14b_reference','deepseek_r1_qwen14b','mistral7b_instruct_v03']
for model in models:
    path=ROOT/f'data/analysis_exports/{model}/{model}_full_case_level.csv'
    f=pd.read_csv(path)
    for c in ['baseline_correct','targeted_correct','generic_correct','mi_sr_correct','selected','clean_correct_value','clean_complete_value','challenger_is_gold']:
        f[c]=truth(f[c])
    probs=['p_none_correct','p_targeted_correct','p_generic_correct']
    assert not f[probs].isna().any().any()
    assert truth(f.biased_complete_value).all()
    assert f.case_id.is_unique
    clean_inconsistent=f.groupby('q_idx')[['clean_correct_value','clean_complete_value']].nunique().gt(1).any().any()
    pn,pt,pg=(f[c] for c in probs)
    mask=(np.maximum(pt,pg)>pn)
    assert mask.equals(f.selected)
    assert f.mi_sr_correct.equals(f.baseline_correct.where(~mask,f.targeted_correct))
    scopes={'full':f}
    if discovery is not None:
        scopes['exclude_discovery95']=f[~f.q_idx.isin(discovery)]
        scopes['discovery95']=f[f.q_idx.isin(discovery)]
    scopes['q_idx_ge_127']=f[f.q_idx.ge(127)]
    scopes['q_idx_ge_637']=f[f.q_idx.ge(637)]
    audit[model]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(f),
        'questions':int(f.q_idx.nunique()),'columns':list(f.columns),'clean_inconsistent_within_question':bool(clean_inconsistent),
        'missing':f.isna().sum()[f.isna().sum().gt(0)].to_dict(),
        'discovery_questions_present':int(f[f.q_idx.isin(discovery)].q_idx.nunique()) if discovery else None,
        'discovery_rows_present':int(f.q_idx.isin(discovery).sum()) if discovery else None}
    for scope,g in scopes.items():
        b,t,gen=g.baseline_correct,g.targeted_correct,g.generic_correct
        pn,pt,pg=(g[c] for c in probs)
        selected=np.maximum(pt,pg)>pn
        sw=int((selected&~b).sum()); sc=int((selected&b).sum()); nw=int((~selected&~b).sum()); nc=int((~selected&b).sum())
        risk_rows.append(dict(model=model,scope=scope,n=len(g),selected_wrong=sw,selected_correct=sc,none_wrong=nw,none_correct=nc,
            selected_wrong_rate=sw/(sw+sc),none_wrong_rate=nw/(nw+nc),
            wrong_case_recall=sw/(sw+nw),odds_ratio=sw*nc/(sc*nw)))
        tg=pt>pn
        gg=pg>pn
        # T wins ties against G; N wins any tie at its own utility.
        use_t=(pt>pn)&(pt>=pg)
        use_g=(pg>pn)&(pg>pt)
        finals={'First':b,'Published_max_gate_always_T':b.where(~selected,t),
            'T_only_gate':b.where(~tg,t),'G_only_gate_always_T':b.where(~gg,t),
            'Argmax_N_T_G':b.where(~use_t,t).where(~use_g,gen),'All_Targeted':t,'All_Generic':gen,
            'Published_gate_Generic':b.where(~selected,gen)}
        for strategy,final in finals.items():
            if strategy=='First': review_n=0
            elif strategy in ['All_Targeted','All_Generic']:review_n=len(g)
            elif strategy=='T_only_gate':review_n=int(tg.sum())
            elif strategy=='G_only_gate_always_T':review_n=int(gg.sum())
            else:review_n=int(selected.sum())
            corr=int((~b&final).sum()); harm=int((b&~final).sum())
            lo,hi=bootstrap(g,final,b) if scope in ['full','exclude_discovery95'] else [None,None]
            policy_rows.append(dict(model=model,scope=scope,strategy=strategy,n=len(g),questions=g.q_idx.nunique(),review_n=review_n,
                review_rate=review_n/len(g),correct=int(final.sum()),accuracy=float(final.mean()),corrected=corr,damaged=harm,
                net=corr-harm,gain_pp=float((final.mean()-b.mean())*100),cluster_boot_ci_low_pp=lo,cluster_boot_ci_high_pp=hi))
        if scope in ['full','exclude_discovery95']:
            for comparator in ['T_only_gate','Argmax_N_T_G','All_Targeted','All_Generic','Published_gate_Generic']:
                lo,hi=bootstrap(g,finals['Published_max_gate_always_T'],finals[comparator])
                comparison_rows.append(dict(model=model,scope=scope,comparison='Published_minus_'+comparator,
                    difference_pp=float((finals['Published_max_gate_always_T'].mean()-finals[comparator].mean())*100),
                    ci_low_pp=lo,ci_high_pp=hi))
        for group,sel in {'pG_only_trigger':gg&~tg,'pT_only_trigger':tg&~gg,'both_trigger':tg&gg,'none_trigger':~tg&~gg}.items():
            z=g[sel]; bb=b[sel]; tt=t[sel]; ge=gen[sel]
            subset_rows.append(dict(model=model,scope=scope,group=group,n=len(z),n_over_all=len(z)/len(g),
                first_correct=int(bb.sum()),targeted_correct=int(tt.sum()),generic_correct=int(ge.sum()),
                corrected_T=int((~bb&tt).sum()),damaged_T=int((bb&~tt).sum()),net_T=int(tt.sum()-bb.sum()),
                corrected_G=int((~bb&ge).sum()),damaged_G=int((bb&~ge).sum()),net_G=int(ge.sum()-bb.sum()),
                exact_pT_eq_pN=int((pt[sel]==pn[sel]).sum())))
        for variant in ['probe_only','mi_only','probe_plus_mi']:
            choose=g[[f'p_targeted_{variant}',f'p_generic_{variant}']].max(axis=1)>g[f'p_none_{variant}']
            result=b.where(~choose,t)
            lo,hi=bootstrap(g,finals['Published_max_gate_always_T'],result) if scope in ['full','exclude_discovery95'] else [None,None]
            ablation_rows.append(dict(model=model,scope=scope,variant=variant,n=len(g),review_n=int(choose.sum()),
                accuracy=float(result.mean()),gain_vs_first_pp=float((result.mean()-b.mean())*100),
                full_minus_variant_pp=float((finals['Published_max_gate_always_T'].mean()-result.mean())*100),
                full_minus_variant_ci_low_pp=lo,full_minus_variant_ci_high_pp=hi))
            if scope in ['full','exclude_discovery95']:
                score=g[[f'p_targeted_{variant}',f'p_generic_{variant}']].max(axis=1)-g[f'p_none_{variant}']
                rank_frame=pd.DataFrame({'score':score,'case_id':g.case_id}).sort_values(['score','case_id'],ascending=[False,True])
                match=pd.Series(g.index.isin(rank_frame.index[:int(selected.sum())]),index=g.index)
                match_result=b.where(~match,t)
                lo,hi=bootstrap(g,finals['Published_max_gate_always_T'],match_result)
                matched_rows.append(dict(model=model,scope=scope,variant=variant,n=len(g),review_n=int(match.sum()),
                    accuracy=float(match_result.mean()),corrected=int((~b&match_result).sum()),damaged=int((b&~match_result).sum()),
                    full_minus_variant_pp=float((finals['Published_max_gate_always_T'].mean()-match_result.mean())*100),
                    full_minus_variant_ci_low_pp=lo,full_minus_variant_ci_high_pp=hi,
                    note='Top-K by score with K equal to published rule, case_id deterministic tie-break; retrospective matched-coverage diagnostic, no gold used to choose K.'))
        wrong=~b
        raw=g.raw_prefinal_challenger.eq(g.gold)
        learned=g.challenger.eq(g.gold)
        localization_rows.append(dict(model=model,scope=scope,wrong_n=int(wrong.sum()),learned_hits=int(learned[wrong].sum()),
            learned_rate=float(learned[wrong].mean()),raw_hits=int(raw[wrong].sum()),raw_rate=float(raw[wrong].mean())))
        # Strict paired cohort requires completed clean + biased first-pass.
        p=g[g.clean_complete_value].copy()
        c=p.clean_correct_value; b0=p.baseline_correct; m=p.mi_sr_correct
        harmed=c&~b0; robust=c&b0; recovered=~c&b0; bw=~c&~b0
        drop=float((c.mean()-b0.mean())*100); gain=float((m.mean()-b0.mean())*100)
        paired_rows.append(dict(model=model,scope=scope,n=len(p),questions=p.q_idx.nunique(),
            clean_correct=int(c.sum()),clean_accuracy=float(c.mean()),first_correct=int(b0.sum()),first_accuracy=float(b0.mean()),
            mi_sr_correct=int(m.sum()),mi_sr_accuracy=float(m.mean()),bias_drop_pp=drop,mi_gain_pp=gain,
            net_drop_recovery_percent=gain/drop*100 if drop else None,
            harmed_n=int(harmed.sum()),harmed_repaired=int((harmed&m).sum()),harmed_repair_rate=float(m[harmed].mean()),
            robust_n=int(robust.sum()),robust_damaged=int((robust&~m).sum()),
            recovered_n=int(recovered.sum()),recovered_damaged=int((recovered&~m).sum()),
            baseline_wrong_n=int(bw.sum()),baseline_wrong_repaired=int((bw&m).sum()),
            net_gain_harmed_robust=int((harmed&m).sum()-(robust&~m).sum()),
            net_gain_other=int((bw&m).sum()-(recovered&~m).sum())))
        q=g.drop_duplicates('q_idx'); qc=q[q.clean_complete_value]
        clean_rows.append(dict(model=model,scope=scope,represented_questions=len(q),clean_complete=int(q.clean_complete_value.sum()),
            clean_correct=int(q.clean_correct_value.sum()),clean_accuracy_all_represented_questions=float(q.clean_correct_value.mean()),
            clean_accuracy_completed_questions=float(qc.clean_correct_value.mean()),
            note='Derived from distinct q_idx in evaluable biased-case CSV, not standalone original clean file.'))

(OUT/'data_schema_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
for name,rows in [('gate_rules',policy_rows),('gate_trigger_subgroups',subset_rows),('paired_clean_bias',paired_rows),
                  ('clean_question_accuracy',clean_rows),('gate_ablation',ablation_rows),('challenger_localization',localization_rows),
                  ('gate_ablation_matched_coverage',matched_rows),('paired_rule_comparisons',comparison_rows),('risk_enrichment',risk_rows)]:
    pd.DataFrame(rows).to_csv(OUT/f'audit_{name}.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'split_matches':[{'source_n':m['source_question_n'],'discovery_n':len(m['discovery_q_idx'])} for m in matches],
                  'audit':{k:{a:v[a] for a in ['rows','questions','discovery_questions_present','discovery_rows_present']} for k,v in audit.items()}},indent=2))
print(pd.DataFrame(policy_rows).query("scope == 'exclude_discovery95' and strategy in ['First','Published_max_gate_always_T','T_only_gate','Argmax_N_T_G']").to_string(index=False))
print(pd.DataFrame(subset_rows).query("scope in ['full','exclude_discovery95'] and group == 'pG_only_trigger'").to_string(index=False))
print(pd.DataFrame(clean_rows).query("scope == 'full'").to_string(index=False))
print(pd.DataFrame(paired_rows).query("scope == 'full'").to_string(index=False))
