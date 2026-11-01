#!/usr/bin/env python3
"""Resume frozen MI-SR caches to their requested Targeted-only ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mi_sr_20pct_targeted_runner as base
from cross_model_runtime import model_spec


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_mi_sr_full_targeted_v5"
GATE_THRESHOLD = 0.0
MODELS = (
    "qwen3_14b_reference",
    "deepseek_r1_qwen14b",
    "mistral7b_instruct_v03",
)
SOURCE_QUESTION_LIMITS = {
    "qwen3_14b_reference": 1273,
    "deepseek_r1_qwen14b": 1273,
    "mistral7b_instruct_v03": 1273,
}
# These globals are set per runner process before delegating to the shared pipeline.
SOURCE_QUESTION_LIMIT = 1273
EXPECTED_CASES = SOURCE_QUESTION_LIMIT * 7
SEED_DIRS = {
    "qwen3_14b_reference": ROOT
    / "output_mi_sr_50pct_v4/qwen3_14b_reference/all_biases",
    "deepseek_r1_qwen14b": ROOT
    / "output_mi_sr_50pct_v4/deepseek_r1_qwen14b/all_biases",
    "mistral7b_instruct_v03": ROOT
    / "output_mi_sr_10pct_targeted_v4/mistral7b_instruct_v03/all_biases",
}
SEED_QUESTION_COUNTS = {
    "qwen3_14b_reference": 637,
    "deepseek_r1_qwen14b": 637,
    "mistral7b_instruct_v03": 127,
}


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
    with (run_dir / "stage_log_full.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def seed_cache(model_key: str) -> Path:
    source = SEED_DIRS[model_key]
    target = OUTPUT_ROOT / model_key / "all_biases"
    if not source.exists():
        raise FileNotFoundError(f"Missing seed cache: {source}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        state = "seed_cache_copied"
    else:
        state = "existing_full_cache_preserved"
    seed_labels = pd.read_csv(source / "first_pass_evaluation_labels.csv")
    payload = {
        "model_key": model_key,
        "source": str(source),
        "target": str(target),
        "state": state,
        "seed_question_count": SEED_QUESTION_COUNTS[model_key],
        "seed_first_pass_cases": int(seed_labels.case_id.nunique()),
        "raw_and_review_caches_preserved": True,
    }
    manifest = target / "cache_seed_full_manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "state"}
        expected = {key: value for key, value in payload.items() if key != "state"}
        if comparable != expected:
            raise RuntimeError(f"Full seed manifest mismatch: {manifest}")
    else:
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return target


def write_manifest(run_dir: Path, model_key: str, load_metadata: dict) -> None:
    frozen_paths = (
        base.CHALLENGER_CONFIG_PATH,
        base.CHALLENGER_MODEL_PATH,
        base.GATE_CONFIG_PATH,
        base.GATE_MODEL_PATH,
        base.ABLATION_CONFIG_PATH,
        base.ABLATION_MODEL_PATH,
    )
    payload = {
        "protocol_version": "mi_sr_v5_full_binary_targeted_none_zero_margin",
        "model_key": model_key,
        "model_path": str(model_spec(model_key)["path"]),
        "load_metadata": load_metadata,
        "n_source_questions": SOURCE_QUESTION_LIMIT,
        "n_bias_types": 7,
        "n_cases": EXPECTED_CASES,
        "seed_source": str(SEED_DIRS[model_key]),
        "seed_question_count": SEED_QUESTION_COUNTS[model_key],
        "prompt_variant": base.runner.PROMPT_VARIANT,
        "generation_limits": {
            "max_new_tokens": base.runner.MAX_NEW_TOKENS,
            "retry_new_tokens": base.runner.RETRY_NEW_TOKENS,
            "reeval_max_new_tokens": base.runner.REEVAL_MAX_NEW_TOKENS,
            "mi_max_steps": base.runner.MI_MAX_STEPS,
            "mi_max_dim": base.runner.MI_MAX_DIM,
        },
        "frozen_artifact_hashes": {path.name: sha256(path) for path in frozen_paths},
        "gold_available_to_decision_path": False,
        "execution_actions": ["targeted", "none"],
        "gate_threshold": GATE_THRESHOLD,
        "binary_execution_rule": (
            "execute Targeted iff max(p_targeted_correct, p_generic_correct) "
            "minus p_none_correct is strictly greater than zero; tie retains"
        ),
        "reviews_generated": ["targeted"],
        "generic_role": "latent historical utility head only; never executed or generated",
        "targeted_template_sha256": hashlib.sha256(
            base.runner.TARGETED_REVIEW_TEMPLATE.encode("utf-8")
        ).hexdigest(),
    }
    path = run_dir / "run_manifest_full_targeted.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = json.loads(json.dumps(existing))
        comparable_payload = json.loads(json.dumps(payload))
        comparable_existing.get("load_metadata", {}).pop("gpu_id", None)
        comparable_payload.get("load_metadata", {}).pop("gpu_id", None)
        if comparable_existing != comparable_payload:
            existing_questions = int(existing.get("n_source_questions", 0))
            payload_questions = int(payload.get("n_source_questions", 0))
            if existing_questions < payload_questions:
                archive = (
                    run_dir
                    / f"run_manifest_full_targeted_superseded_{existing_questions}q.json"
                )
                if not archive.exists():
                    path.replace(archive)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return
            raise RuntimeError(f"Immutable full-run manifest mismatch: {path}")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_binary_policy(run_dir: Path, policy: pd.DataFrame) -> pd.DataFrame:
    binary = policy.copy()
    binary["source_gate_action"] = binary.action
    binary["review_score"] = (
        binary[["p_targeted_correct", "p_generic_correct"]].max(axis=1)
        - binary.p_none_correct
    )
    binary["action"] = np.where(
        binary.review_score.gt(GATE_THRESHOLD), "targeted", "none"
    )
    if set(binary.action) - {"targeted", "none"}:
        raise RuntimeError("Binary policy contains an unexpected action")
    binary.to_csv(
        run_dir / "binary_deploy_policy.csv", index=False, encoding="utf-8-sig"
    )
    return binary


def run(model_key: str, gpu_id: int) -> dict:
    global SOURCE_QUESTION_LIMIT, EXPECTED_CASES

    SOURCE_QUESTION_LIMIT = SOURCE_QUESTION_LIMITS[model_key]
    EXPECTED_CASES = SOURCE_QUESTION_LIMIT * 7
    run_dir = seed_cache(model_key)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.SOURCE_QUESTION_LIMIT = SOURCE_QUESTION_LIMIT
    base.EXPECTED_CASES = EXPECTED_CASES
    base.write_manifest = write_manifest
    base.write_binary_policy = write_binary_policy
    base.log = log
    result = base.run(model_key, gpu_id)
    transient_marker = run_dir / "targeted_20pct_complete.json"
    final_marker = run_dir / "targeted_full_complete.json"
    if transient_marker.exists():
        transient_marker.replace(final_marker)
    result.update(
        {
            "protocol_version": "mi_sr_v5_full_binary_targeted_none_zero_margin",
            "seed_question_count": SEED_QUESTION_COUNTS[model_key],
            "target_question_count": SOURCE_QUESTION_LIMIT,
            "incremental_question_count": SOURCE_QUESTION_LIMIT
            - SEED_QUESTION_COUNTS[model_key],
        }
    )
    final_marker.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=MODELS)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args()
    if args.seed_only:
        print(seed_cache(args.model_key))
        return
    print(json.dumps(run(args.model_key, args.gpu_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
