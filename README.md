# MI-SR code

Code for MI-SR: locating a competing answer and selectively reviewing first-pass
answers of language models on medical multiple-choice questions with cognitive
bias cues. This release is anonymous.

## Contents

| Path | What it holds |
|---|---|
| `original_experiments/` | Inference, feature extraction, frozen-policy deployment, repair and analysis scripts behind the reported results |
| `original_experiments/frozen_artifacts/` | Fixed locator and gate parameters |
| `followup_experiments/run/` | Follow-up GPU experiments; `run_all.sh` lists their order |
| `followup_experiments/analysis/` | Follow-up CPU analyses of stored results |
| `followup_experiments/audit/` | Read-only audits of the case-level results |
| `followup_experiments/tests/` | Smoke tests that replace the model with a small fake |

Not included: model weights, the BiasMedQA dataset, and model outputs.

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/carlwharris/cog-bias-med-LLMs upstream
ln -s upstream/med_llm_bias.py upstream/models.py upstream/utils.py upstream/data_clean .
cp upstream/.api_config.json api_config.json
export MI_SR_REPO_ROOT="$PWD"
```

`med_llm_bias.py`, `models.py`, `utils.py` and the questions come from that
repository and are not redistributed here. `models.py` reads `api_config.json` when
imported, so the template above is enough; these experiments call no paid API. The
experiments read `data_clean/questions/US/test_with_bias_irrel.jsonl` from its
dataset release. Obtain data and model weights under their own licenses.

| Variable | Meaning |
|---|---|
| `MI_SR_REPO_ROOT` | Directory holding `med_llm_bias.py` and `data_clean/`, as set above |
| `MI_SR_WORK_DIR` | Working directory of `original_experiments/` |
| `MI_SR_MODEL_ROOT` | Directory holding `DeepSeek-R1-Distill-Qwen-14B` and `Mistral-7B-Instruct-v0.3` |
| `MI_SR_QWEN_MODEL_PATH` | Qwen3-14B weights |
| `MI_SR_ARTIFACT_DIR` | Frozen artifacts; defaults to `original_experiments/frozen_artifacts` |
| `U1_ROOT` | Directory with the modules and `output_mi_sr_full_targeted_v5/`; defaults to `original_experiments/` |
| `MI_SR_WORK_ROOT` | Outputs of the follow-up runs; defaults to `work/` |
| `MI_SR_DATA_ROOT` | Directory containing `data/analysis_exports/<model>/<model>_full_case_level.csv` |
| `MI_SR_ARCHIVE_ZIP` | Archive of raw run outputs, read by two audits |
| `MI_PEAKS_ROOT` | Clone of the public MI-Peaks code, used only to compare estimators |

## Original experiments

The final runs grew in stages, each resuming from the caches of the previous one:

```bash
cd original_experiments
# Fit and freeze the locator and gate on the 95 development source questions
python freeze_qwen_challenger_gate_models.py
python freeze_gate_v2_ablation_models.py

# Qwen3-14B and DeepSeek-R1-Distill-Qwen-14B: 20% -> 50% -> all questions
python mi_sr_full_runner_v3.py --model-key qwen3_14b_reference --gpu-id 0
python prepare_mi_sr_50pct.py --model-key qwen3_14b_reference
python mi_sr_50pct_runner.py --model-key qwen3_14b_reference --gpu-id 0
python mi_sr_full_targeted_runner.py --model-key qwen3_14b_reference --gpu-id 0

# Mistral-7B-Instruct-v0.3: 10% -> all questions
python mi_sr_10pct_targeted_runner.py --model-key mistral7b_instruct_v03 --gpu-id 0
python mi_sr_full_targeted_runner.py --model-key mistral7b_instruct_v03 --gpu-id 0

# Every model: generic review control, repair of incomplete cases, reports
python mi_sr_full_generic_runner.py --model-key qwen3_14b_reference --gpu-id 0
python repair_incomplete_full_runs.py --model-key qwen3_14b_reference --gpu-id 0
python analyze_full_binary_targeted_reports.py --model-key qwen3_14b_reference
```

The frozen parameters are already in `frozen_artifacts/`, so the two freeze steps
are only needed to refit them. The 95 development questions are listed in
`followup_experiments/run/discovery_q_idx.json`.

## Follow-up experiments

| Question | GPU run | CPU analysis |
|---|---|---|
| Numerical origin of negative dependence values; effect of double-precision HKH | `u1_preflight`, `u1_numeric_rerun`, `u1_numeric_analysis` | `u1_root_cause`, `u1_error_formula_and_dims`, `u1_gate_sensitivity`, `u1_reference_and_curve_stats`, `u1_cpu_instruction_check`, `u1_public_code_run`, `u1_run_public_code`, `hsic_precision_check`, `hsic_fix_check` |
| Pipeline on unbiased questions | `r3_clean_run` | `r3_strata`, `g5_threshold_curve`, `g5_loss_decomposition`, `g5_gate_with_clean`, `g5_cue_detector` |
| Comparison with simple probability and token gates | `g4_token_probability_baseline` | `budget_all_points_audit`, `budget_damage_ci`, `budget_frontier_audit`, `audit/audit_archive_simple_gates` |
| Total latency, tokens and memory | `u1_timing_benchmark` | `g6_equal_time_audit`, `equal_compute_audit`, `token_cost_audit` |
| Contribution of MI features | | `u1_gate_stress_all_mi`, `u1_stress_by_budget`, `g2_mi_only_vs_random`, `g2_mi_only_vs_simple_gates`, `g2_peak_span_text` |
| Option representation and letter weighting | `r8_f32`, `r789_extra_experiments --task r8` | `peak_mention_audit`, `r8_ci` |
| Prompt arms on selected cases | `r789_extra_experiments --task r7` | `r7_strata` |
| Single five-option scorer | `r789_extra_experiments --task r9` | `remaining_cpu_checks` |
| Overfitting of the 74-feature gate | | `remaining_cpu_checks` |
| Locator against the highest-probability alternative | | `remaining_cpu_checks`, `u7_by_error_type`, `u7_self_fit` |
| Role of the generic-review head in the routing rule | | `g3_pg_signal`, `g3_single_head_rules` |
| Candidate competition | | `competition_and_clean_proxy` |
| Consistency of case-level results and tables | | `audit/data_audit`, `audit/verify_data_audit`, `audit/build_audit_summary`, `audit/numeric_dependence_check`, `audit/static_curve_buffer_audit` |

Analyses take their paths as command-line arguments or from the variables above.
Audits write next to the script. Run `audit/data_audit.py` first.

Relative to the versions that produced the results, the follow-up scripts differ
only in path constants, comments and file locations. `run_all.sh` is written for
this release and states the order of the steps.

## Tests

```bash
cd followup_experiments/tests
python r3789_smoke_fake_model.py r3 ../run qwen3_14b_reference smoke_out
```

The smoke tests read `original_experiments/output_mi_sr_full_targeted_v5/` and use
one GPU. Their numbers are meaningless; they only check that the code runs.

## License

MIT, for the code in this repository. Upstream code, data and model weights keep
their own licenses.
