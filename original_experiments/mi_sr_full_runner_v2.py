"""Run the frozen MI-SR v2 paper protocol for one model on one GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

import autonomous_harmed_mi_search as mi_exp
import run_frozen_holdout as runner
from cross_model_runtime import load_model_and_tokenizer, model_spec
from mi_sr_full_analysis import analyze
from mi_sr_peak_features import answer_support_table, extract_peak_features
from mi_sr_protocol import FROZEN_CONFIG, build_frozen_policy, config_dict


ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / "output_mi_sr_full_v2"
RANDOM_SEED = 20260810
BIAS_TYPES = tuple(mi_exp.BIAS_TYPES)
EXPECTED_QUESTIONS = 1273


def _log(run_dir: Path, stage: str, state: str, **extra) -> None:
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "state": state,
        **extra,
    }
    with (run_dir / "stage_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _build_full_plan(run_dir: Path) -> pd.DataFrame:
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    if len(questions) != EXPECTED_QUESTIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_QUESTIONS} source questions, found {len(questions)}"
        )
    rows = []
    position = 0
    for q_idx in range(len(questions)):
        for bias_index, bias_type in enumerate(BIAS_TYPES):
            rows.append(
                {
                    "case_id": f"q{q_idx:04d}_b{bias_index}",
                    "split": "paper_full_all_biases",
                    "split_position": position,
                    "q_idx": q_idx,
                    "bias_type": bias_type,
                    "record_seed": (
                        RANDOM_SEED
                        + position * 17
                        + q_idx * 31
                        + bias_index * 101
                    ),
                }
            )
            position += 1
    plan = pd.DataFrame(rows)
    plan_path = run_dir / "experiment_plan.csv"
    if plan_path.exists():
        existing = pd.read_csv(plan_path)
        if not existing.equals(plan):
            raise RuntimeError(f"Immutable experiment plan mismatch: {plan_path}")
    else:
        plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    return plan


def _configure(
    run_dir: Path, model_key: str, source_question_limit: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_plan = _build_full_plan(run_dir)
    if source_question_limit is None:
        plan = full_plan.copy()
    else:
        if not 1 <= source_question_limit <= EXPECTED_QUESTIONS:
            raise ValueError(
                f"source_question_limit must be in [1, {EXPECTED_QUESTIONS}]"
            )
        plan = full_plan[full_plan.q_idx.lt(source_question_limit)].copy()
    runner.OUTPUT_DIR = run_dir
    runner.RANDOM_SEED = RANDOM_SEED
    runner.PROMPT_VARIANT = "mi_sr_complete"
    if model_key == "deepseek_r1_qwen14b":
        # This model usually needs more than 384 tokens. Starting at 640 avoids
        # regenerating the identical greedy prefix during a completeness retry.
        runner.MAX_NEW_TOKENS = 640
        runner.RETRY_NEW_TOKENS = 768
    else:
        runner.MAX_NEW_TOKENS = 384
        runner.RETRY_NEW_TOKENS = 512
    # Avoid regenerating an identical greedy prefix when a 224-token review
    # misses the mandatory final-answer line.
    runner.REEVAL_MAX_NEW_TOKENS = runner.MAX_NEW_TOKENS
    runner.MI_MAX_STEPS = 256
    runner.MI_MAX_DIM = 128
    runner.TARGET_MODE = "full_option"
    runner.build_plan = lambda: plan.copy()
    return full_plan, plan


def _run_clean_references(
    model, tokenizer, run_dir: Path, source_question_limit: int
) -> pd.DataFrame:
    """Generate one clean answer per source question, not once per bias type."""

    path = run_dir / "clean_reference_outputs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    completed = {int(row["q_idx"]) for row in rows}
    questions = mi_exp.load_questions(runner.DATA_JSONL)

    def checkpoint() -> None:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    for q_idx in range(source_question_limit):
        if q_idx in completed:
            continue
        source = SimpleNamespace(
            q_idx=q_idx,
            bias_type=BIAS_TYPES[0],
            record_seed=RANDOM_SEED + 20_000_000 + q_idx * 31,
        )
        record = runner.record_for_plan(questions, source)
        prompt = mi_exp.build_prompt(tokenizer, record, "clean", "mi_sr_complete")
        try:
            generated = mi_exp.generate_complete(
                model,
                tokenizer,
                prompt,
                runner.MAX_NEW_TOKENS,
                runner.RETRY_NEW_TOKENS,
            )
            rows.append(
                {
                    "q_idx": q_idx,
                    "gold": record["gold"],
                    "clean_answer": generated["answer"],
                    "clean_correct": bool(
                        generated["complete"]
                        and generated["answer"] == record["gold"]
                    ),
                    "clean_complete": bool(generated["complete"]),
                    "prompt": prompt,
                    "gen_text": generated["gen_text"],
                }
            )
        except RuntimeError as error:
            rows.append(
                {
                    "q_idx": q_idx,
                    "gold": record["gold"],
                    "clean_answer": None,
                    "clean_correct": False,
                    "clean_complete": False,
                    "prompt": prompt,
                    "gen_text": "",
                    "generation_error": repr(error),
                }
            )
        completed.add(q_idx)
        if len(completed) % 5 == 0:
            checkpoint()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"clean references: {len(completed)}/{source_question_limit}",
                flush=True,
            )
    checkpoint()
    return pd.DataFrame(rows)


def _run_raw_delta_reviews(model, tokenizer, run_dir: Path, plan: pd.DataFrame) -> None:
    path = run_dir / "raw_prefinal_delta_reevaluation.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {str(row["case_id"]) for row in rows}
    policy = pd.read_csv(run_dir / "frozen_mi_sr_policy.csv")
    jobs = policy[policy.mi_override.fillna(False).astype(bool)].copy()
    if jobs.empty:
        return
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    plan_lookup = plan.set_index("case_id")
    for index, row in enumerate(jobs.itertuples(index=False), 1):
        case_id = str(row.case_id)
        if case_id in done:
            continue
        source = plan_lookup.loc[case_id]
        record = runner.record_for_plan(questions, source)
        prompt = runner.build_targeted_review_prompt(
            tokenizer,
            record,
            str(row.model_answer),
            str(row.raw_prefinal_challenger),
        )
        generated = mi_exp.generate_complete(
            model,
            tokenizer,
            prompt,
            runner.REEVAL_MAX_NEW_TOKENS,
            runner.RETRY_NEW_TOKENS,
        )
        rows.append(
            {
                "case_id": case_id,
                "review_type": "targeted_raw_prefinal_delta",
                "model_answer": row.model_answer,
                "challenger": row.raw_prefinal_challenger,
                "reeval_answer": generated["answer"],
                "reeval_complete": generated["complete"],
                "prompt": prompt,
                "gen_text": generated["gen_text"],
            }
        )
        done.add(case_id)
        if index % 5 == 0 or index == len(jobs):
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"raw pre-final delta reviews: {len(done)}/{len(jobs)}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def run(
    model_key: str,
    gpu_id: int,
    output_root: Path,
    source_question_limit: int | None = None,
    stop_after: str | None = None,
) -> dict:
    run_dir = output_root.resolve() / model_key / "all_biases"
    run_dir.mkdir(parents=True, exist_ok=True)
    full_plan, plan = _configure(run_dir, model_key, source_question_limit)
    active_questions = (
        EXPECTED_QUESTIONS
        if source_question_limit is None
        else source_question_limit
    )
    spec = model_spec(model_key)
    model, tokenizer, load_metadata = load_model_and_tokenizer(model_key, gpu_id)
    manifest = {
        "protocol": config_dict(FROZEN_CONFIG),
        "model_key": model_key,
        "model_path": str(spec["path"]),
        "load_metadata": load_metadata,
        "gpu_id": gpu_id,
        "split": "paper_full_all_biases",
        "n_planned": len(full_plan),
        "n_source_questions": EXPECTED_QUESTIONS,
        "n_bias_types": len(BIAS_TYPES),
        "active_phase": {
            "source_questions": active_questions,
            "biased_cases": len(plan),
            "fraction_of_full": active_questions / EXPECTED_QUESTIONS,
        },
        "clean_generated_once_per_question": True,
        "prompt_variant": runner.PROMPT_VARIANT,
        "mi_curve_implementation": "cpu_batched_exact_v1",
        "generation_limits": {
            "max_new_tokens": runner.MAX_NEW_TOKENS,
            "retry_new_tokens": runner.RETRY_NEW_TOKENS,
            "reeval_max_new_tokens": runner.REEVAL_MAX_NEW_TOKENS,
            "mi_max_steps": runner.MI_MAX_STEPS,
        },
        "chat_template_adapter": (
            "close_forced_think_before_visible_reasoning"
            if model_key == "deepseek_r1_qwen14b"
            else "none"
        ),
        "gold_available_to_decision_path": False,
        "generic_is_comparator_not_main_action": True,
        "review_templates": {
            "targeted_required_placeholders": [
                "model_answer",
                "challenger",
                "question",
                "options",
            ],
            "generic_required_placeholders": [
                "model_answer",
                "question",
                "options",
            ],
            "targeted_sha256": hashlib.sha256(
                runner.TARGETED_REVIEW_TEMPLATE.encode("utf-8")
            ).hexdigest(),
            "generic_sha256": hashlib.sha256(
                runner.GENERIC_REVIEW_TEMPLATE.encode("utf-8")
            ).hexdigest(),
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "chat_template_adapter" not in existing:
            existing["chat_template_adapter"] = manifest["chat_template_adapter"]
        if "generation_limits" not in existing:
            existing["generation_limits"] = manifest["generation_limits"]
        else:
            for key, value in manifest["generation_limits"].items():
                existing["generation_limits"].setdefault(key, value)
        if "mi_curve_implementation" not in existing:
            existing["mi_curve_implementation"] = manifest[
                "mi_curve_implementation"
            ]
        existing["active_phase"] = manifest["active_phase"]
        if (
            existing.get("chat_template_adapter") == manifest["chat_template_adapter"]
            and existing.get("generation_limits") == manifest["generation_limits"]
        ):
            manifest_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        for key in (
            "protocol",
            "model_key",
            "split",
            "n_planned",
            "prompt_variant",
            "mi_curve_implementation",
            "chat_template_adapter",
            "generation_limits",
            "review_templates",
        ):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"Immutable run manifest mismatch: {key}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _log(run_dir, "first_pass_curves_probes", "started")
    runner.run_all_feature_stages(model, tokenizer, include_clean=False)
    _log(run_dir, "first_pass_curves_probes", "completed")

    _log(run_dir, "clean_references", "started")
    clean = _run_clean_references(model, tokenizer, run_dir, active_questions)
    _log(
        run_dir,
        "clean_references",
        "completed",
        completed=int(clean.clean_complete.fillna(False).astype(bool).sum()),
        planned=active_questions,
    )
    if stop_after == "clean_references":
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        summary = {
            "status": "paused",
            "stop_after": stop_after,
            "clean_reference_rows": len(clean),
        }
        _log(run_dir, "run", "paused", **summary)
        return summary

    _log(run_dir, "mi_peak_checkpoints", "started")
    candidate_features = extract_peak_features(model, tokenizer, run_dir)
    answer_support = answer_support_table(run_dir)
    policy = build_frozen_policy(candidate_features, answer_support)
    policy.to_csv(run_dir / "frozen_mi_sr_policy.csv", index=False, encoding="utf-8-sig")
    policy[["case_id", "model_answer", "ratio_mi_challenger"]].rename(
        columns={"ratio_mi_challenger": "challenger"}
    ).to_csv(
        run_dir / "frozen_policy_scores_before_unblinding.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _log(
        run_dir,
        "mi_peak_checkpoints",
        "completed",
        eligible_cases=len(policy),
        ratio_mi_overrides=int(policy.mi_override.sum()),
        reviews_selected=int(policy.gate_targeted.sum()),
    )

    _log(run_dir, "counterfactual_reviews", "started")
    runner.run_all_reviews(model, tokenizer)
    _run_raw_delta_reviews(model, tokenizer, run_dir, plan)
    _log(run_dir, "counterfactual_reviews", "completed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log(run_dir, "analysis", "started")
    summary = analyze(run_dir)
    _log(run_dir, "analysis", "completed", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-question-limit", type=int)
    parser.add_argument(
        "--stop-after",
        choices=("clean_references",),
        help="Persist the completed stage and exit before later stages.",
    )
    args = parser.parse_args()
    run(
        args.model_key,
        args.gpu_id,
        args.output_root,
        source_question_limit=args.source_question_limit,
        stop_after=args.stop_after,
    )


if __name__ == "__main__":
    main()
