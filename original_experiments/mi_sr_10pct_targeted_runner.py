#!/usr/bin/env python3
"""Run a 10% zero-threshold Targeted/None MI-SR pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mi_sr_20pct_targeted_runner as base
from cross_model_runtime import MODEL_SPECS, model_spec


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_mi_sr_10pct_targeted_v4"
SOURCE_QUESTION_LIMIT = 127
EXPECTED_CASES = SOURCE_QUESTION_LIMIT * 7
GATE_THRESHOLD = 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        "protocol_version": "mi_sr_v4_10pct_zero_threshold_targeted_none",
        "model_key": model_key,
        "model_path": str(model_spec(model_key)["path"]),
        "load_metadata": load_metadata,
        "n_source_questions": SOURCE_QUESTION_LIMIT,
        "n_bias_types": 7,
        "n_cases": EXPECTED_CASES,
        "source_question_fraction": SOURCE_QUESTION_LIMIT / 1273,
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
            "minus p_none_correct is at least zero; otherwise retain"
        ),
        "reviews_generated": ["targeted"],
        "generic_role": "latent historical utility head only; never executed or generated",
        "targeted_template_sha256": hashlib.sha256(
            base.runner.TARGETED_REVIEW_TEMPLATE.encode("utf-8")
        ).hexdigest(),
    }
    path = run_dir / "run_manifest_10pct.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = json.loads(json.dumps(existing))
        comparable_payload = json.loads(json.dumps(payload))
        comparable_existing.get("load_metadata", {}).pop("gpu_id", None)
        comparable_payload.get("load_metadata", {}).pop("gpu_id", None)
        if comparable_existing != comparable_payload:
            raise RuntimeError(f"Immutable 10% manifest mismatch: {path}")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_zero_threshold_policy(run_dir: Path, policy: pd.DataFrame) -> pd.DataFrame:
    binary = policy.copy()
    binary["source_gate_action"] = binary.action
    binary["review_score"] = (
        binary[["p_targeted_correct", "p_generic_correct"]].max(axis=1)
        - binary.p_none_correct
    )
    binary["action"] = np.where(
        binary.review_score.ge(GATE_THRESHOLD), "targeted", "none"
    )
    binary.to_csv(
        run_dir / "binary_deploy_policy.csv", index=False, encoding="utf-8-sig"
    )
    return binary


def run(model_key: str, gpu_id: int) -> dict:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.SOURCE_QUESTION_LIMIT = SOURCE_QUESTION_LIMIT
    base.EXPECTED_CASES = EXPECTED_CASES
    base.write_manifest = write_manifest
    base.write_binary_policy = write_zero_threshold_policy
    result = base.run(model_key, gpu_id)
    run_dir = OUTPUT_ROOT / model_key / "all_biases"
    old_marker = run_dir / "targeted_20pct_complete.json"
    new_marker = run_dir / "targeted_10pct_complete.json"
    if old_marker.exists():
        old_marker.replace(new_marker)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    run(args.model_key, args.gpu_id)


if __name__ == "__main__":
    main()
