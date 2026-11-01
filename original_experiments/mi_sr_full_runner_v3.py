"""Resume one model to the frozen 20% MI-SR v3 evaluation endpoint."""

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
from cross_model_gate_threshold_sweep import analyze_thresholds
from mi_sr_deploy_v3 import (
    ABLATION_CONFIG_PATH,
    ABLATION_MODEL_PATH,
    CHALLENGER_CONFIG_PATH,
    CHALLENGER_MODEL_PATH,
    GATE_CONFIG_PATH,
    GATE_MODEL_PATH,
    build_deploy_policy,
)
from mi_sr_full_analysis_v3 import analyze
from mi_sr_peak_features import extract_peak_features


OUTPUT_ROOT = ROOT / "output_mi_sr_20pct_v3"
SOURCE_QUESTION_LIMIT = 255
EXPECTED_CASES = SOURCE_QUESTION_LIMIT * 7


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
    with (run_dir / "stage_log_v3.jsonl").open("a", encoding="utf-8") as handle:
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


def write_manifest(
    run_dir: Path, model_key: str, load_metadata: dict, plan: pd.DataFrame
) -> None:
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
        "protocol_version": "mi_sr_v3_learned_challenger_three_way_gate",
        "model_key": model_key,
        "model_path": str(model_spec(model_key)["path"]),
        "load_metadata": load_metadata,
        "n_source_questions": SOURCE_QUESTION_LIMIT,
        "n_bias_types": 7,
        "n_cases": len(plan),
        "source_question_fraction": SOURCE_QUESTION_LIMIT / 1273,
        "prompt_variant": runner.PROMPT_VARIANT,
        "generation_limits": {
            "max_new_tokens": runner.MAX_NEW_TOKENS,
            "retry_new_tokens": runner.RETRY_NEW_TOKENS,
            "reeval_max_new_tokens": runner.REEVAL_MAX_NEW_TOKENS,
            "mi_max_steps": runner.MI_MAX_STEPS,
        },
        "frozen_artifact_hashes": artifacts,
        "action_margin": json.loads(
            GATE_CONFIG_PATH.read_text(encoding="utf-8")
        )["absolute_action_margin"],
        "gold_available_to_decision_path": False,
        "all_targeted_and_generic_generated_for_shared_counterfactuals": True,
        "targeted_template_sha256": hashlib.sha256(
            runner.TARGETED_REVIEW_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "generic_template_sha256": hashlib.sha256(
            runner.GENERIC_REVIEW_TEMPLATE.encode("utf-8")
        ).hexdigest(),
    }
    path = run_dir / "run_manifest_v3.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable = [
            "protocol_version",
            "model_key",
            "n_source_questions",
            "n_cases",
            "prompt_variant",
            "generation_limits",
            "frozen_artifact_hashes",
            "action_margin",
            "targeted_template_sha256",
            "generic_template_sha256",
        ]
        for key in immutable:
            if existing.get(key) != payload.get(key):
                raise RuntimeError(f"Immutable v3 manifest mismatch: {key}")
    else:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def audit_review_cache(run_dir: Path, policy: pd.DataFrame) -> dict:
    expected = policy.set_index("case_id")[["model_answer", "challenger"]]
    summary = {}
    for review_type in ("targeted", "generic"):
        path = run_dir / f"{review_type}_all_reevaluation.csv"
        if not path.exists():
            summary[review_type] = {"kept": 0, "rejected": 0}
            continue
        frame = pd.read_csv(path)
        valid_case = frame.case_id.astype(str).isin(expected.index.astype(str))
        valid_answer = frame.model_answer.astype(str).eq(
            frame.case_id.map(expected.model_answer).astype(str)
        )
        if review_type == "targeted":
            valid_challenger = frame.challenger.astype(str).eq(
                frame.case_id.map(expected.challenger).astype(str)
            )
        else:
            valid_challenger = pd.Series(True, index=frame.index)
        valid = valid_case & valid_answer & valid_challenger
        rejected = frame[~valid].copy()
        if len(rejected):
            rejected.to_csv(
                run_dir / f"{review_type}_rejected_cache_v3.csv",
                index=False,
                encoding="utf-8-sig",
            )
        kept = frame[valid].drop_duplicates("case_id", keep="last")
        kept.to_csv(path, index=False, encoding="utf-8-sig")
        summary[review_type] = {"kept": len(kept), "rejected": len(rejected)}
    return summary


def run(model_key: str, gpu_id: int = 0) -> dict:
    run_dir = OUTPUT_ROOT / model_key / "all_biases"
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = configure(run_dir, model_key)
    model, tokenizer, load_metadata = load_model_and_tokenizer(model_key, gpu_id)
    write_manifest(run_dir, model_key, load_metadata, plan)

    log(run_dir, "first_pass_curves_probes", "started")
    runner.run_all_feature_stages(model, tokenizer, include_clean=False)
    log(run_dir, "first_pass_curves_probes", "completed")

    log(run_dir, "clean_references", "started")
    clean = legacy._run_clean_references(
        model, tokenizer, run_dir, SOURCE_QUESTION_LIMIT
    )
    log(
        run_dir,
        "clean_references",
        "completed",
        complete=int(clean.clean_complete.fillna(False).astype(bool).sum()),
        rows=len(clean),
    )

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
        action_counts=policy.action.value_counts().to_dict(),
        review_cache_audit=cache_audit,
    )

    log(run_dir, "counterfactual_reviews", "started")
    runner.run_all_reviews(model, tokenizer)
    log(run_dir, "counterfactual_reviews", "completed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(run_dir, "analysis_v3", "started")
    summary = analyze(run_dir, model_key)
    log(
        run_dir,
        "analysis_v3",
        "completed",
        proceed_to_next_model=summary["proceed_to_next_model"],
    )
    log(run_dir, "gate_threshold_sweep", "started")
    threshold_summary = analyze_thresholds(run_dir, model_key)
    log(
        run_dir,
        "gate_threshold_sweep",
        "completed",
        selected_threshold=threshold_summary["selected_threshold"],
    )
    summary["adaptive_gate_threshold"] = threshold_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        choices=(
            "qwen3_14b_reference",
            "gemma3_12b",
            "phi4",
            "deepseek_r1_qwen14b",
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.model_key, args.gpu_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
