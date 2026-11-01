"""Collect audited tables into compact JSON for the follow-up analysis."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parent
tables={n:pd.read_csv(P/f'audit_{n}.csv') for n in ['gate_rules','gate_trigger_subgroups','paired_clean_bias','clean_question_accuracy',
        'gate_ablation','gate_ablation_matched_coverage','challenger_localization','paired_rule_comparisons','risk_enrichment']}
if (P/'audit_simple_gates_matched_coverage.csv').exists():
    tables['simple_gates_matched_coverage']=pd.read_csv(P/'audit_simple_gates_matched_coverage.csv')
out={'scope':'Exclude the recovered exact 95 fitting source questions; 1178 source questions remain.',
     'statistical_method':'10000 source-question cluster bootstrap replicates, seed 20261025, case-weighted ratio of sum differences / sum case counts; percentile CI. No multiple-comparison adjustment. Retrospective diagnostics.',
     'models':{}}
for model in tables['gate_rules'].model.unique():
    out['models'][model]={}
    for name,df in tables.items():
        sub=df[(df.model==model)&(df.scope=='exclude_discovery95')].copy()
        if name=='gate_trigger_subgroups':sub=sub[sub.group=='pG_only_trigger']
        out['models'][model][name]=sub.drop(columns=['model','scope']).to_dict('records')
out['limits']=[
    'Reconstructed split reproduces frozen SHA-256, but the original split_manifest file is absent from the submission. Authors should recover and archive the original manifest.',
    'Removing the fitted 95 questions does not establish untouched confirmatory evaluation: frozen_gate_v002/v005 expressly record old-v1 holdout exposure and v005 freezing after a Qwen20% audit failure.',
    'Original ZIP full-run manifest confirms final run used frozen v002 heads (config margin .08) with zero-margin binary max(pT,pG)-pN execution; v005 margin .05 is retained in an exploratory archive. Selection chronology for final zero threshold must be recovered before claiming prespecification.',
    'All proposed rules reuse fixed pre-existing T/G review outcomes, which is valid for cache-based routing counterfactuals, not a prospectively registered evaluation.',
    'pG-only subgroup here is pG>pN and pT<=pN; no exact pT=pN ties occurred. T-only and current rules have different review coverage.',
    'Equal-budget ablation uses per-score Top-K at K chosen by the published full rule, tie-broken by case_id, without gold selection. It is a retrospective matched-coverage diagnostic.',
    'Clean-question accuracy is reconstructed by unique q_idx from the biased-case CSV; all 1273 source questions occur. Clean completion is checked. No clean MI-SR review outcomes are present.',
    'Clean/bias paired rows exclude incomplete clean output. Net gap recovery is an aggregate accounting fraction, not a causal bias-specific treatment effect.',
    'Raw-prefinal challenger hit rates can be audited, but targeted generation using the raw challenger is absent; end-to-end challenger-replacement results require new generation.',
    'Original ZIP probability CSV allows pre-final and prompt-end fixed-direction simple gates at matched K; these hold the learned challenger and its cached targeted outcomes fixed. Candidate-letter entropy is not semantic entropy.',
    'The independent verifier checks input hashes, split hash, retained counts, published accuracy, All Targeted accuracy, and matched Probe-only sorting using stdlib CSV; it does not independently recompute all bootstrap draws.'
]
(P/'audit_summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
print(P/'audit_summary.json')
