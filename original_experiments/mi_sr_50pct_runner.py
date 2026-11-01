#!/usr/bin/env python3
"""Extend one frozen MI-SR run from 20% to 50% without Generic generation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mi_sr_full_runner_v2 as legacy
import run_frozen_holdout as runner
from cross_model_runtime import load_model_and_tokenizer, model_spec
from mi_sr_deploy_v3 import (
    ABLATION_CONFIG_PATH,
    ABLATION_MODEL_PATH,
    CHALLENGER_CONFIG_PATH,
    CHALLENGER_MODEL_PATH,
    GATE_CONFIG_PATH,
    GATE_MODEL_PATH,
    build_deploy_policy,
)
from mi_sr_full_runner_v3 import audit_review_cache
from mi_sr_peak_features import extract_peak_features


OUTPUT_ROOT = ROOT / "output_mi_sr_50pct_v4"
SOURCE_QUESTION_LIMIT = 637
EXPECTED_CASES = SOURCE_QUESTION_LIMIT * 7
MODELS = ("qwen3_14b_reference", "deepseek_r1_qwen14b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def log(run_dir: Path, stage: str, state: str, **extra) -> None:
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "state": state,
        **extra,
    }
    with (run_dir / "stage_log_50pct.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def configure(run_dir: Path, model_key: str) -> pd.DataFrame:
    full_plan = legacy._build_full_plan(run_dir)
    plan = full_plan[full_plan.q_idx.lt(SOURCE_QUESTION_LIMIT)].copy()
    if len(plan) != EXPECTED_CASES:
        raise RuntimeError(f"Expected {EXPECTED_CASES} active cases, found {len(plan)}")
    runner.OUTPUT_DIR = run_dir
    runner.RANDOM_SEED = legacy.RANDOM_SEED
    runner.PROMPT_VARIANT = "mi_sr_complete"
    if model_key == "deepseek_r1_qwen14b":
        runner.MAX_NEW_TOKENS = 640
        runner.RETRY_NEW_TOKENS = 768
    else:
        runner.MAX_NEW_TOKENS = 384
        runner.RETRY_NEW_TOKENS = 512
    runner.REEVAL_MAX_NEW_TOKENS = runner.MAX_NEW_TOKENS
    runner.MI_MAX_STEPS = 256
    runner.MI_MAX_DIM = 128
    runner.TARGET_MODE = "full_option"
    runner.build_plan = lambda: plan.copy()
    return plan


def write_manifest(run_dir: Path, model_key: str, load_metadata: dict) -> None:
    artifacts = {
        path.name: sha256(path)
        for path in (
            CHALLENGER_CONFIG_PATH,
            CHALLENGER_MODEL_PATH,
            GATE_CONFIG_PATH,
            GATE_MODEL_PATH,
            ABLATION_CONFIG_PATH,
            ABLATION_MODEL_PATH,
        )
    }
    payload = {
        "protocol_version": "mi_sr_v4_incremental_50pct_targeted_only",
        "model_key": model_key,
        "model_path": str(model_spec(model_key)["path"]),
        "load_metadata": load_metadata,
        "n_source_questions": SOURCE_QUESTION_LIMIT,
        "n_bias_types": 7,
        "n_cases": EXPECTED_CASES,
        "source_question_fraction": SOURCE_QUESTION_LIMIT / 1273,
        "reused_source_questions": 255,
        "incremental_source_questions": SOURCE_QUESTION_LIMIT - 255,
        "prompt_variant": runner.PROMPT_VARIANT,
        "generation_limits": {
            "max_new_tokens": runner.MAX_NEW_TOKENS,
            "retry_new_tokens": runner.RETRY_NEW_TOKENS,
            "reeval_max_new_tokens": runner.REEVAL_MAX_NEW_TOKENS,
            "mi_max_steps": runner.MI_MAX_STEPS,
        },
        "frozen_artifact_hashes": artifacts,
        "gold_available_to_decision_path": False,
        "reviews_generated": ["targeted"],
        "generic_role": "existing 20% control cache preserved; no 30% extension",
    }
    path = run_dir / "run_manifest_50pct.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

        # The physical GPU index may change after a machine restart without
        # changing the model, protocol, or frozen artifacts. Keep the original
        # manifest immutable while excluding this runtime-only field from the
        # resume compatibility check.
        comparable_existing = json.loads(json.dumps(existing))
        comparable_payload = json.loads(json.dumps(payload))
        comparable_existing.get("load_metadata", {}).pop("gpu_id", None)
        comparable_payload.get("load_metadata", {}).pop("gpu_id", None)
        if comparable_existing != comparable_payload:
            raise RuntimeError(f"Immutable 50% manifest mismatch: {path}")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(model_key: str, gpu_id: int = 0) -> dict:
    run_dir = OUTPUT_ROOT / model_key / "all_biases"
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = configure(run_dir, model_key)
    model, tokenizer, load_metadata = load_model_and_tokenizer(model_key, gpu_id)
    write_manifest(run_dir, model_key, load_metadata)

    log(run_dir, "first_pass_curves_probes", "started")
    runner.run_all_feature_stages(model, tokenizer, include_clean=False)
    log(run_dir, "first_pass_curves_probes", "completed")

    log(run_dir, "clean_references", "started")
    clean = legacy._run_clean_references(
        model, tokenizer, run_dir, SOURCE_QUESTION_LIMIT
    )
    log(run_dir, "clean_references", "completed", rows=len(clean))

    log(run_dir, "mi_peak_features", "started")
    peaks = extract_peak_features(model, tokenizer, run_dir)
    log(
        run_dir,
        "mi_peak_features",
        "completed",
        cases=int(peaks.case_id.nunique()),
        rows=len(peaks),
    )

    log(run_dir, "deploy_policy", "started")
    policy, _ = build_deploy_policy(run_dir)
    cache_audit = audit_review_cache(run_dir, policy)
    log(
        run_dir,
        "deploy_policy",
        "completed",
        cases=len(policy),
        cache_audit=cache_audit,
    )

    log(run_dir, "targeted_reviews", "started")
    runner.run_reviews(model, tokenizer, review_types=("targeted",))
    log(run_dir, "targeted_reviews", "completed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    marker = run_dir / "targeted_50pct_complete.json"
    result = {
        "model_key": model_key,
        "source_questions": SOURCE_QUESTION_LIMIT,
        "planned_cases": len(plan),
        "policy_cases": len(policy),
        "targeted_only": True,
    }
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=MODELS)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.model_key, args.gpu_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
