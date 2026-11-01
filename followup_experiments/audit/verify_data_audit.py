"""Independent stdlib-only verification of recovered split and key audit counts."""
import os
from pathlib import Path
import csv, json, hashlib
P=Path(os.environ.get("MI_SR_DATA_ROOT", "."))
O=Path(__file__).resolve().parent
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def yes(v):return str(v).lower() in ('true','1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((O/'recovered_split_audit.json').read_text())['matches'][0]
payload={k:m[k] for k in ('seed','group_column','discovery_q_idx','holdout_q_idx','same_source_question_bias_variants_kept_together')}
h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
frozen=json.loads((Path(__file__).resolve().parents[2]/'original_experiments/frozen_artifacts/frozen_artifact_manifest.json').read_text())
assert h==frozen['split_hash']==m['split_hash']
disc=set(m['discovery_q_idx']);hold=set(m['holdout_q_idx'])
assert len(disc)==95 and len(hold)==32 and not (disc&hold) and disc|hold==set(range(127))
latest=json.loads((P/'data/LATEST.json').read_text(encoding='utf-8'))
reported=rows(O/'audit_gate_rules.csv');matched=rows(O/'audit_gate_ablation_matched_coverage.csv')
verified=[]
table4_audit=[]
for label,entry in latest['case_files'].items():
    name=Path(entry['path']).name
    p=list((P/'data/analysis_exports').glob('*/'+name))[0]
    model=p.parent.name
    allrows=rows(p)
    assert len(allrows)==entry['rows'] and sha(p)==entry['sha256']
    rs=[r for r in allrows if int(r['q_idx']) not in disc]
    n=len(rs); nq=len({r['q_idx'] for r in rs})
    get=lambda key:next(r for r in reported if r['model']==model and r['scope']=='exclude_discovery95' and r['strategy']==key)
    count=lambda key:sum(yes(r[key]) for r in rs)
    assert n==int(get('First')['n']) and nq==1178
    assert count('baseline_correct')==int(get('First')['correct'])
    assert count('mi_sr_correct')==int(get('Published_max_gate_always_T')['correct'])
    assert count('targeted_correct')==int(get('All_Targeted')['correct'])
    chosen=[r for r in rs if max(float(r['p_targeted_correct']),float(r['p_generic_correct']))>float(r['p_none_correct'])]
    assert len(chosen)==int(get('Published_max_gate_always_T')['review_n'])
    k=len(chosen)
    scored=sorted(rs,key=lambda r:(-(max(float(r['p_targeted_probe_only']),float(r['p_generic_probe_only']))-float(r['p_none_probe_only'])),r['case_id']))
    selected={r['case_id'] for r in scored[:k]}
    correct=sum(yes(r['targeted_correct']) if r['case_id'] in selected else yes(r['baseline_correct']) for r in rs)
    mr=next(r for r in matched if r['model']==model and r['scope']=='exclude_discovery95' and r['variant']=='probe_only')
    assert abs(correct/n-float(mr['accuracy']))<1e-12
    verified.append({'model':model,'input_sha256':sha(p),'all_rows':len(allrows),'removed_rows':len(allrows)-n,
                    'retained_rows':n,'retained_questions':nq,'matched_probe_only_correct':correct,'passed':True})
    if model=='qwen3_14b_reference':
        published={r['Gate input']:r for r in rows(P/'data/table4_qwen_gate_ablation.csv')}
        source={r['strategy']:r for r in rows(p.parent/'gate_ablations.csv') if r['scope']=='Full-100%'}
        for v,label,source_label in [('probe_only','Probe-only','Probe-only Gate'),('mi_only','MI-only','MI-only Gate'),('probe_plus_mi','Probe + MI','Probe+MI Gate')]:
            corr=damage=rn=0
            for r in allrows:
                choose=max(float(r['p_targeted_'+v]),float(r['p_generic_'+v]))>float(r['p_none_'+v])
                b=yes(r['baseline_correct']); end=yes(r['targeted_correct']) if choose else b
                rn+=choose;corr+=(not b and end);damage+=(b and not end)
            bsum=sum(yes(r['baseline_correct']) for r in allrows)
            acc=(bsum+corr-damage)/len(allrows)*100
            assert corr==int(source[source_label]['corrected']) and damage==int(source[source_label]['damaged'])
            table4_audit.append({'variant':v,'case_level_review_n':rn,'case_level_corrected':corr,'case_level_damaged':damage,'case_level_accuracy_percent':acc,
                'source_analysis_corrected':int(source[source_label]['corrected']),'source_analysis_accuracy_percent':float(source[source_label]['accuracy'])*100,
                'published_table4_corrected':int(published[label]['Corrected']),'published_table4_damaged':int(published[label]['Damaged']),
                'published_table4_accuracy_percent':float(published[label]['Final accuracy (%)']),
                'published_matches_case_level':corr==int(published[label]['Corrected']) and abs(acc-float(published[label]['Final accuracy (%)']))<1e-10})
out={'split_sha256':h,'discovery_n':len(disc),'old_holdout_n':len(hold),'verified':verified,
     'method':'stdlib csv and independent Python sorted; no pandas/numpy/joblib/model execution'}
(O/'audit_independent_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(O/'audit_table4_consistency.json').write_text(json.dumps(table4_audit,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
