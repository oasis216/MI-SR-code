"""Read only named CSV/JSON ZIP members; compare fixed-direction simple gates."""
import os
from pathlib import Path
import zipfile, json, hashlib, io, csv
import numpy as np
import pandas as pd
ROOT=Path(os.environ.get("MI_SR_DATA_ROOT", "."))
ARCHIVE=Path(os.environ.get("MI_SR_ARCHIVE_ZIP", "raw_archive-20260729.zip"))
OUT=Path(__file__).resolve().parent
DISC=set(json.loads((OUT/'recovered_split_audit.json').read_text())['matches'][0]['discovery_q_idx'])
MODELS=['qwen3_14b_reference','deepseek_r1_qwen14b','mistral7b_instruct_v03']
def yes(s):return s.fillna(False).astype(str).str.lower().isin(['true','1'])
def ci(g,a,b):
    d=(a.astype(int)-b.astype(int)).groupby(g.q_idx).sum().to_numpy()
    n=g.groupby('q_idx').size().to_numpy();rng=np.random.default_rng(20261025);draws=[]
    for _ in range(20):
        inds=rng.integers(0,len(n),(500,len(n)))
        draws.extend((d[inds].sum(axis=1)/n[inds].sum(axis=1)*100).tolist())
    return np.quantile(draws,[.025,.975]).tolist()
records=[]; validation={}; manifests={}; inventory=[]
for archive in ARCHIVE.parent.glob('*20260729.zip'):
    with zipfile.ZipFile(archive) as z:
        inventory.append({'archive':str(archive),'bytes':archive.stat().st_size,'entries':len(z.infolist()),
            'split_manifest_matches':[n for n in z.namelist() if 'split_manifest' in n],
            'probe_members':[{'name':n,'bytes':z.getinfo(n).file_size} for n in z.namelist() if n.endswith('checkpoint_probe_logits.csv')]})
with zipfile.ZipFile(ARCHIVE) as z:
    for model in MODELS:
        stem=f'output_mi_sr_full_targeted_v5/{model}/all_biases/'
        member=stem+'checkpoint_probe_logits.csv';raw=z.read(member)
        probes=pd.read_csv(io.BytesIO(raw))
        # Retain archive provenance, including the actual frozen v002 head hashes.
        for name in ['run_manifest_full_targeted.json','deploy_policy_manifest_v3.json','cache_migration_manifest.json']:
            if stem+name in z.namelist():manifests[model+'/'+name]=json.loads(z.read(stem+name))
        f=pd.read_csv(ROOT/f'data/analysis_exports/{model}/{model}_full_case_level.csv')
        for col in ['baseline_correct','targeted_correct','mi_sr_correct','selected']:f[col]=yes(f[col])
        f=f[~f.q_idx.isin(DISC)].copy()
        k=int(f.selected.sum())
        assert not probes.duplicated(['case_id','checkpoint']).any()
        validation[model]={'member':member,'member_sha256':hashlib.sha256(raw).hexdigest(),'probe_rows':len(probes),
            'checkpoint_counts':probes.checkpoint.value_counts().to_dict(),'primary_n':len(f),'review_k':k,'checkpoints':{}}
        for checkpoint in ['pre_final','prompt_end']:
            p=probes[probes.checkpoint==checkpoint].copy()
            cols=['case_id','q_idx','model_answer','candidate_entropy']+[f'prob_{x}' for x in 'ABCDE']
            g=f.merge(p[cols],on='case_id',suffixes=('','_probe'),how='left',validate='one_to_one')
            assert g.q_idx.eq(g.q_idx_probe).all()
            assert g.model_answer.eq(g.model_answer_probe).all()
            a=g[[f'prob_{x}' for x in 'ABCDE']].to_numpy(float)
            assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
            assert np.max(np.abs(a.sum(axis=1)-1))<1e-5
            idx=np.array(['ABCDE'.index(x) for x in g.model_answer])
            pm=a[np.arange(len(a)),idx]
            alt=a.copy();alt[np.arange(len(a)),idx]=-np.inf
            maxalt=alt.max(axis=1)
            entropy=-(a*np.log(np.maximum(a,1e-30))).sum(axis=1)
            sort_probs=np.sort(a,axis=1)
            scores={'low_answer_probability':-pm,'low_answer_vs_best_alt_margin':maxalt-pm,
                    'high_candidate_entropy':entropy,'low_top1_top2_margin':-(sort_probs[:,-1]-sort_probs[:,-2])}
            validation[model]['checkpoints'][checkpoint]={'rows_joined':len(g),'max_probability_sum_error':float(np.max(np.abs(a.sum(axis=1)-1))),
                    'max_entropy_error':float(np.max(np.abs(entropy-g.candidate_entropy))),
                    'raw_prefinal_argmax_agreement':float(pd.Series(np.array(list('ABCDE'))[alt.argmax(axis=1)],index=g.index).eq(g.raw_prefinal_challenger).mean()) if checkpoint=='pre_final' else None}
            for name,score in scores.items():
                ranks=pd.DataFrame({'score':score,'case_id':g.case_id}).sort_values(['score','case_id'],ascending=[False,True])
                choose=pd.Series(g.index.isin(ranks.index[:k]),index=g.index)
                final=g.baseline_correct.where(~choose,g.targeted_correct)
                lo,hi=ci(g,g.mi_sr_correct,final)
                corrected=int((~g.baseline_correct&final).sum());damaged=int((g.baseline_correct&~final).sum())
                # Independently verify rankings through stdlib sorted on the scalar scores.
                inds=sorted(range(len(g)),key=lambda i:(-float(score[i]),g.case_id.iloc[i]))[:k]
                assert set(inds)==set(ranks.index[:k])
                records.append(dict(model=model,scope='exclude_discovery95',checkpoint=checkpoint,gate=name,n=len(g),review_n=k,
                    correct=int(final.sum()),accuracy=float(final.mean()),corrected=corrected,damaged=damaged,net=corrected-damaged,
                    published_accuracy=float(g.mi_sr_correct.mean()),published_minus_simple_pp=float((g.mi_sr_correct.mean()-final.mean())*100),
                    diff_ci_low_pp=lo,diff_ci_high_pp=hi,score_ties_at_cut=int((score==score[ranks.index[k-1]]).sum()),
                    note='Fixed direction, same K as published gate, case_id tie-break. Existing learned-challenger targeted outcomes held fixed. Candidate-letter entropy, not semantic entropy.'))
    for n in ['output_paper_full_v1/protocol_manifest.json',
              'output_mi_sr_full_targeted_v5/qwen3_14b_reference/all_biases/v005_exploratory_archive/deploy_policy_manifest_v3.json',
              'output_mi_sr_full_targeted_v5/qwen3_14b_reference/all_biases/v005_exploratory_archive/analysis_v3/result_assessment.json']:
        if n in z.namelist():manifests[n]=json.loads(z.read(n))
pd.DataFrame(records).to_csv(OUT/'audit_simple_gates_matched_coverage.csv',index=False,encoding='utf-8-sig')
(OUT/'archive_probe_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
(OUT/'archive_selected_manifests.json').write_text(json.dumps(manifests,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'archive_inventory.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False),encoding='utf-8')
print(pd.DataFrame(records).query("checkpoint == 'pre_final'")[['model','gate','n','review_n','accuracy','corrected','damaged','published_minus_simple_pp','diff_ci_low_pp','diff_ci_high_pp']].to_string(index=False))
