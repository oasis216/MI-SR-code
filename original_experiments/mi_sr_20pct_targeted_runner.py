#!/usr/bin/env python3
"""Run the frozen 20% MI-SR protocol with Targeted/None execution only."""

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
from cross_model_runtime import MODEL_SPECS, load_model_and_tokenizer, model_spec
from mi_sr_deploy_v3 import (
    ABLATION_CONFIG_PATH,
    ABLATION_MODEL_PATH,
    CHALLENGER_CONFIG_PATH,
    CHALLENGER_MODEL_PATH,
    GATE_CONFIG_PATH,
    GATE_MODEL_PATH,
    build_deploy_policy,
)
from mi_sr_peak_features import extract_peak_features


OUTPUT_ROOT = ROOT / "output_mi_sr_20pct_targeted_v4"
SOURCE_QUESTION_LIMIT = 255
EXPECTED_CASES = SOURCE_QUESTION_LIMIT * 7
SUPPORTED_MODELS = tuple(MODEL_SPECS)


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
    with (run_dir / "stage_log_20pct.jsonl").open("a", encoding="utf-8") as handle:
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
    frozen_paths = (
        CHALLENGER_CONFIG_PATH,
        CHALLENGER_MODEL_PATH,
        GATE_CONFIG_PATH,
        GATE_MODEL_PATH,
        ABLATION_CONFIG_PATH,
        ABLATION_MODEL_PATH,
    )
    payload = {
        "protocol_version": "mi_sr_v4_20pct_binary_targeted_none",
        "model_key": model_key,
        "model_path": str(model_spec(model_key)["path"]),
        "load_metadata": load_metadata,
        "n_source_questions": SOURCE_QUESTION_LIMIT,
        "n_bias_types": 7,
        "n_cases": EXPECTED_CASES,
        "source_question_fraction": SOURCE_QUESTION_LIMIT / 1273,
        "prompt_variant": runner.PROMPT_VARIANT,
        "generation_limits": {
            "max_new_tokens": runner.MAX_NEW_TOKENS,
            "retry_new_tokens": runner.RETRY_NEW_TOKENS,
            "reeval_max_new_tokens": runner.REEVAL_MAX_NEW_TOKENS,
            "mi_max_steps": runner.MI_MAX_STEPS,
            "mi_max_dim": runner.MI_MAX_DIM,
        },
        "frozen_artifact_hashes": {path.name: sha256(path) for path in frozen_paths},
        "gold_available_to_decision_path": False,
        "execution_actions": ["targeted", "none"],
        "binary_execution_rule": (
            "retain only when the frozen Gate selects None; map every review "
            "decision to Targeted"
        ),
        "reviews_generated": ["targeted"],
        "generic_role": "not generated; control-only and excluded from deployment",
        "targeted_template_sha256": hashlib.sha256(
            runner.TARGETED_REVIEW_TEMPLATE.encode("utf-8")
        ).hexdigest(),
    }
    path = run_dir / "run_manifest_20pct.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = json.loads(json.dumps(existing))
        comparable_payload = json.loads(json.dumps(payload))
        comparable_existing.get("load_metadata", {}).pop("gpu_id", None)
        comparable_payload.get("load_metadata", {}).pop("gpu_id", None)
        if comparable_existing != comparable_payload:
            raise RuntimeError(f"Immutable 20% manifest mismatch: {path}")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_binary_policy(run_dir: Path, policy: pd.DataFrame) -> pd.DataFrame:
    binary = policy.copy()
    binary["source_gate_action"] = binary["action"]
    binary["action"] = binary["action"].where(
        binary["action"].eq("none"), "targeted"
    )
    unexpected = set(binary.action) - {"targeted", "none"}
    if unexpected:
        raise RuntimeError(f"Unexpected binary actions: {sorted(unexpected)}")
    expected_none = int(policy.action.eq("none").sum())
    if int(binary.action.eq("none").sum()) != expected_none:
        raise RuntimeError("Binary Gate changed the frozen None partition")
    binary.to_csv(
        run_dir / "binary_deploy_policy.csv", index=False, encoding="utf-8-sig"
    )
    return binary


def audit_targeted_cache(run_dir: Path, policy: pd.DataFrame) -> dict:
    path = run_dir / "targeted_all_reevaluation.csv"
    if not path.exists():
        return {"kept": 0, "rejected": 0}
    expected = policy.set_index("case_id")[["model_answer", "challenger"]]
    frame = pd.read_csv(path)
    valid_case = frame.case_id.astype(str).isin(expected.index.astype(str))
    valid_answer = frame.model_answer.astype(str).eq(
        frame.case_id.map(expected.model_answer).astype(str)
    )
    valid_challenger = frame.challenger.astype(str).eq(
        frame.case_id.map(expected.challenger).astype(str)
    )
    valid = valid_case & valid_answer & valid_challenger
    rejected = frame[~valid].copy()
    if len(rejected):
        rejected.to_csv(
            run_dir / "targeted_rejected_cache_20pct.csv",
            index=False,
            encoding="utf-8-sig",
        )
    kept = frame[valid].drop_duplicates("case_id", keep="last")
    kept.to_csv(path, index=False, encoding="utf-8-sig")
    return {"kept": len(kept), "rejected": len(rejected)}


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

    log(run_dir, "binary_deploy_policy", "started")
    policy, _ = build_deploy_policy(run_dir)
    binary = write_binary_policy(run_dir, policy)
    cache_audit = audit_targeted_cache(run_dir, policy)
    log(
        run_dir,
        "binary_deploy_policy",
        "completed",
        cases=len(binary),
        action_counts=binary.action.value_counts().to_dict(),
        source_action_counts=binary.source_gate_action.value_counts().to_dict(),
        targeted_cache_audit=cache_audit,
    )

    log(run_dir, "targeted_reviews", "started")
    runner.run_reviews(model, tokenizer, review_types=("targeted",))
    log(run_dir, "targeted_reviews", "completed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {
        "model_key": model_key,
        "source_questions": SOURCE_QUESTION_LIMIT,
        "planned_cases": len(plan),
        "policy_cases": len(binary),
        "binary_action_counts": binary.action.value_counts().to_dict(),
        "targeted_only": True,
    }
    (run_dir / "targeted_20pct_complete.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    run(args.model_key, args.gpu_id)


if __name__ == "__main__":
    main()
