#!/bin/bash
# Runs the follow-up GPU experiments one at a time, in the order used for the reported results.
# Requires the environment described in README.md. A step that fails stops the script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
A="${MI_SR_WORK_ROOT:-$PWD/work}"
PY="${PYTHON:-python}"
export U1_ROOT="${U1_ROOT:-$(cd "$HERE/../../original_experiments" && pwd)}"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=qwen3_14b_reference; D=deepseek_r1_qwen14b; M=mistral7b_instruct_v03
cd "$HERE"; mkdir -p "$A"
for m in $Q $D $M; do
  $PY -W ignore u1_preflight.py --model $m --out "$A/u1_numeric_rerun"
  $PY -W ignore u1_numeric_rerun.py --model $m --out "$A/u1_numeric_rerun"      # double-precision HKH recomputation
  $PY -W ignore u1_numeric_analysis.py --model $m --out "$A/u1_numeric_rerun"
  $PY -W ignore g4_token_probability_baseline.py --model $m --out "$A/u1_token"  # token-probability gates
  $PY -W ignore u1_timing_benchmark.py --model $m --out "$A/u1_timing"           # controlled serial latency
  $PY -W ignore r8_f32.py --task r8 --model $m --out "$A/u1_extra_f32"           # option-representation variants
  $PY -W ignore r3_clean_run.py --model $m --out "$A/u1_clean"                   # same pipeline on unbiased questions
done
$PY -W ignore r789_extra_experiments.py --task r7 --model $Q --out "$A/u1_extra" # prompt arms on selected cases
$PY -W ignore r789_extra_experiments.py --task r9 --model $Q --out "$A/u1_extra" # single five-option scorer, end to end
