#!/usr/bin/env python3
"""Generate full-run Generic review controls for MI-SR model caches.

This runner does not change the deployed Targeted/None policy. It only fills
`generic_all_reevaluation.csv` as a counterfactual control, with checkpointed
resume semantics inherited from `run_frozen_holdout.run_reviews`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

import mi_sr_20pct_targeted_runner as base
import mi_sr_full_targeted_runner as full
from cross_model_runtime import load_model_and_tokenizer, model_spec


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = full.OUTPUT_ROOT
SUPPORTED_MODELS = full.MODELS


def log(run_dir: Path, stage: str, state: str, **extra) -> None:
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "state": state,
        **extra,
    }
    with (run_dir / "stage_log_full_generic.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def configure_full_generic(model_key: str) -> Path:
    source_questions = full.SOURCE_QUESTION_LIMITS[model_key]
    run_dir = OUTPUT_ROOT / model_key / "all_biases"
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing full-run cache: {run_dir}")
    if not (run_dir / "frozen_policy_scores_before_unblinding.csv").exists():
        raise FileNotFoundError(
            "Generic reviews require frozen_policy_scores_before_unblinding.csv"
        )
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.SOURCE_QUESTION_LIMIT = source_questions
    base.EXPECTED_CASES = source_questions * 7
    base.configure(run_dir, model_key)
    return run_dir


def write_manifest(run_dir: Path, model_key: str, load_metadata: dict) -> None:
    payload = {
        "protocol_version": "mi_sr_v5_full_generic_control",
        "model_key": model_key,
        "model_path": str(model_spec(model_key)["path"]),
        "load_metadata": load_metadata,
        "n_source_questions": full.SOURCE_QUESTION_LIMITS[model_key],
        "n_bias_types": 7,
        "target_cases": int(base.EXPECTED_CASES),
        "control_role": "All Generic counterfactual review only",
        "deployment_policy_changed": False,
        "generic_template_sha256": hashlib.sha256(
            base.runner.GENERIC_REVIEW_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "generic_template": base.runner.GENERIC_REVIEW_TEMPLATE,
    }
    path = run_dir / "run_manifest_full_generic_control.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = json.loads(json.dumps(existing))
        comparable_payload = json.loads(json.dumps(payload))
        comparable_existing.get("load_metadata", {}).pop("gpu_id", None)
        comparable_payload.get("load_metadata", {}).pop("gpu_id", None)
        if comparable_existing != comparable_payload:
            archive = run_dir / (
                "run_manifest_full_generic_control_superseded_"
                f"{int(time.time())}.json"
            )
            path.replace(archive)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(model_key: str, gpu_id: int) -> dict:
    run_dir = configure_full_generic(model_key)
    model, tokenizer, load_metadata = load_model_and_tokenizer(model_key, gpu_id)
    write_manifest(run_dir, model_key, load_metadata)

    log(run_dir, "generic_reviews", "started")
    base.runner.run_reviews(model, tokenizer, review_types=("generic",))
    log(run_dir, "generic_reviews", "completed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    path = run_dir / "generic_all_reevaluation.csv"
    rows = 0
    unique_cases = 0
    if path.exists():
        import pandas as pd

        frame = pd.read_csv(path, usecols=["case_id"])
        rows = len(frame)
        unique_cases = int(frame.case_id.nunique())
    result = {
        "model_key": model_key,
        "source_questions": full.SOURCE_QUESTION_LIMITS[model_key],
        "generic_rows": rows,
        "generic_unique_cases": unique_cases,
        "output": str(path),
    }
    (run_dir / "generic_full_complete.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.model_key, args.gpu_id)


if __name__ == "__main__":
    main()
