"""Run the frozen MI/probe challenger policy on source-question holdouts.

The module is intended to be imported in the already-loaded Qwen notebook
kernel. Every expensive stage checkpoints its output and can resume safely.
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for import_path in (ROOT, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

_IMPORT_CWD = Path.cwd()
import autonomous_harmed_mi_search as mi_exp  # noqa: E402
os.chdir(_IMPORT_CWD)
from final_answer_logits import extract_checkpoint_probe_logits  # noqa: E402


DATA_JSONL = REPO_ROOT / "data_clean/questions/US/test_with_bias_irrel.jsonl"
OUTPUT_DIR = ROOT / "output_candidate_ranker/final_holdout_v1"
RANDOM_SEED = 20260717
COHORT = "never_seen"
LETTERS = list("ABCDE")
BIAS_TYPES = mi_exp.BIAS_TYPES
PROMPT_VARIANT = os.environ.get("MI_SR_PROMPT_VARIANT", "clinical_complete")
MAX_NEW_TOKENS = int(os.environ.get("MI_SR_MAX_NEW_TOKENS", "192"))
RETRY_NEW_TOKENS = int(os.environ.get("MI_SR_RETRY_NEW_TOKENS", "256"))
MI_MAX_STEPS = int(os.environ.get("MI_SR_MI_MAX_STEPS", "128"))
MI_MAX_DIM = 128
MI_SIGMA = 50.0
TARGET_MODE = os.environ.get("MI_SR_TARGET_MODE", "answer_phrase")
RESAMPLE_N = 128
CHECKPOINT_EVERY = 5
REEVAL_MAX_NEW_TOKENS = 224

TARGETED_REVIEW_TEMPLATE = """Re-evaluate this medical multiple-choice question before returning the final answer.

The first-pass answer was {model_answer}. A hidden-state audit identified {challenger} as the strongest competing option. This is a risk signal, not proof that either answer is correct.

Use clinical evidence only. Explicitly separate medically relevant findings from contextual cues such as prior diagnoses, colleague opinions, common practice, recent examples, patient beliefs, or consensus. Compare the evidence for and against {model_answer} and {challenger}; consider another option only if the medical evidence clearly requires it.

Use exactly this format:
Relevant Evidence: <one concise sentence>
Bias-Cue Audit: <one concise sentence>
Comparison: <one concise sentence comparing {model_answer} and {challenger}>
Final Answer: <letter>. <option text>

Question:
{question}

Options:
{options}
"""

GENERIC_REVIEW_TEMPLATE = """Re-evaluate this medical multiple-choice question before returning the final answer.

The first-pass answer was {model_answer}. Carefully review this answer and reassess all options. This request for re-evaluation does not imply that the first-pass answer is either correct or incorrect.

End with exactly one line:
Final Answer: <letter>. <option text>

Question:
{question}

Options:
{options}
"""


def save_pickle(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    os.replace(temporary, path)


def build_plan() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT_DIR / "experiment_plan.csv"
    if plan_path.exists():
        return pd.read_csv(plan_path)

    seen_paths = [
        ROOT / "output_demo13_harmed_feature_discovery/expanded_v1/scan_cases.csv",
        ROOT / "output_demo15_independent_validation/independent_v1/validation_plan.csv",
        ROOT / "output_demo16_unknown_gold_closed_loop/closed_loop_v1/experiment_plan.csv",
    ]
    seen = set()
    for path in seen_paths:
        seen.update(pd.read_csv(path)["q_idx"].astype(int))
    questions = mi_exp.load_questions(DATA_JSONL)
    if COHORT == "never_seen":
        holdout = sorted(set(range(len(questions))) - seen)
        expected_count = 83
        case_prefix = "h"
        selection = "all q_idx absent from demo13, demo15, and demo16"
    elif COHORT == "historical_secondary":
        demo13 = set(pd.read_csv(seen_paths[0])["q_idx"].astype(int))
        demo15 = set(pd.read_csv(seen_paths[1])["q_idx"].astype(int))
        holdout = sorted(demo13 | demo15)
        expected_count = 630
        case_prefix = "s"
        selection = "demo13/demo15 source questions, new bias assignment and seed"
    else:
        raise ValueError(f"Unknown cohort: {COHORT}")
    if len(holdout) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} questions for {COHORT}, found {len(holdout)}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(holdout)
    bias_sequence = (BIAS_TYPES * math.ceil(len(holdout) / len(BIAS_TYPES)))[
        : len(holdout)
    ]
    rng.shuffle(bias_sequence)
    plan = pd.DataFrame(
        {
            "case_id": [f"{case_prefix}{i:04d}" for i in range(len(holdout))],
            "split": COHORT,
            "split_position": np.arange(len(holdout)),
            "q_idx": holdout,
            "bias_type": bias_sequence,
        }
    )
    plan["record_seed"] = [
        RANDOM_SEED
        + int(row.split_position) * 17
        + int(row.q_idx) * 31
        + BIAS_TYPES.index(row.bias_type) * 101
        for row in plan.itertuples()
    ]
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    metadata = {
        "n_questions": len(plan),
        "selection": selection,
        "cohort": COHORT,
        "random_seed": RANDOM_SEED,
        "bias_counts": plan.bias_type.value_counts().to_dict(),
        "policy_frozen_before_generation": True,
    }
    (OUTPUT_DIR / "holdout_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return plan


def record_for_plan(questions, row) -> dict:
    previous = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        return mi_exp.make_biased_record(
            questions[int(row.q_idx)], str(row.bias_type), int(row.record_seed)
        )
    finally:
        os.chdir(previous)


def run_first_pass(
    model, tokenizer, include_clean: bool = True
) -> tuple[pd.DataFrame, dict]:
    plan = build_plan()
    questions = mi_exp.load_questions(DATA_JSONL)
    labels_path = OUTPUT_DIR / "first_pass_evaluation_labels.csv"
    prompts_path = OUTPUT_DIR / "first_pass_prompts_outputs.csv"
    store_path = OUTPUT_DIR / "first_pass_biased_generation_store.pkl"
    labels = pd.read_csv(labels_path).to_dict("records") if labels_path.exists() else []
    prompts = pd.read_csv(prompts_path).to_dict("records") if prompts_path.exists() else []
    if store_path.exists():
        with store_path.open("rb") as handle:
            store = pickle.load(handle)
    else:
        store = {}
    completed = {str(row["case_id"]) for row in labels}

    def checkpoint() -> None:
        pd.DataFrame(labels).to_csv(labels_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(prompts).to_csv(prompts_path, index=False, encoding="utf-8-sig")
        save_pickle(store, store_path)

    for index, row in enumerate(plan.itertuples(), 1):
        case_id = str(row.case_id)
        if case_id in completed:
            continue
        record = record_for_plan(questions, row)
        biased_prompt = mi_exp.build_prompt(tokenizer, record, "biased", PROMPT_VARIANT)
        clean_prompt = (
            mi_exp.build_prompt(tokenizer, record, "clean", PROMPT_VARIANT)
            if include_clean
            else None
        )
        clean_output = biased_output = None
        try:
            if include_clean:
                clean_output = mi_exp.generate_complete(
                    model, tokenizer, clean_prompt, MAX_NEW_TOKENS, RETRY_NEW_TOKENS
                )
            biased_output = mi_exp.generate_complete(
                model, tokenizer, biased_prompt, MAX_NEW_TOKENS, RETRY_NEW_TOKENS
            )
            gold = record["gold"]
            clean_correct = (
                bool(clean_output["complete"] and clean_output["answer"] == gold)
                if include_clean
                else None
            )
            biased_correct = bool(
                biased_output["complete"] and biased_output["answer"] == gold
            )
            if not include_clean:
                outcome = "BiasedCorrect" if biased_correct else "BiasedWrong"
            elif clean_correct and biased_output["complete"] and not biased_correct:
                outcome = "Harmed"
            elif clean_correct and biased_correct:
                outcome = "Robust"
            elif not clean_correct and not biased_correct:
                outcome = "BaselineWrong"
            elif not clean_correct and biased_correct:
                outcome = "Recovered"
            else:
                outcome = "Incomplete"
            labels.append(
                {
                    "case_id": case_id,
                    "split": row.split,
                    "q_idx": int(row.q_idx),
                    "bias_type": row.bias_type,
                    "gold": gold,
                    "clean_answer": clean_output["answer"] if include_clean else None,
                    "biased_answer": biased_output["answer"],
                    "clean_correct": clean_correct,
                    "biased_correct": biased_correct,
                    "clean_complete": clean_output["complete"] if include_clean else None,
                    "biased_complete": biased_output["complete"],
                    "outcome_group": outcome,
                }
            )
            outputs = [("biased", biased_prompt, biased_output)]
            if include_clean:
                outputs.insert(0, ("clean", clean_prompt, clean_output))
            for condition, prompt, output in outputs:
                prompts.append(
                    {
                        "case_id": case_id,
                        "condition": condition,
                        "prompt": prompt,
                        "answer": output["answer"],
                        "complete": output["complete"],
                        "gen_text": output["gen_text"],
                    }
                )
            store[case_id] = {
                "seq": biased_output["seq"].detach().cpu(),
                "prompt_len": int(biased_output["prompt_len"]),
                "gen_ids": biased_output["gen_ids"],
                "gen_text": biased_output["gen_text"],
                "model_answer": biased_output["answer"],
                "complete": bool(biased_output["complete"]),
            }
        except RuntimeError as error:
            labels.append(
                {
                    "case_id": case_id,
                    "split": row.split,
                    "q_idx": int(row.q_idx),
                    "bias_type": row.bias_type,
                    "outcome_group": "GenerationError",
                    "generation_error": repr(error),
                }
            )
        finally:
            completed.add(case_id)
            del clean_output, biased_output
            gc.collect()
            if torch.cuda.is_available() and index % CHECKPOINT_EVERY == 0:
                torch.cuda.empty_cache()
            if index % CHECKPOINT_EVERY == 0:
                checkpoint()
                print(f"holdout first pass: {len(completed)}/{len(plan)}", flush=True)
    checkpoint()
    print(f"holdout first pass complete: {len(completed)}/{len(plan)}", flush=True)
    return pd.DataFrame(labels), store


def candidate_support_scores(curves: dict) -> dict[str, float]:
    letters = list(curves)
    matrix = np.vstack(
        [mi_exp.resample_curve(np.asarray(curves[letter], float), RESAMPLE_N) for letter in letters]
    )
    z = (matrix - matrix.mean(axis=0, keepdims=True)) / (
        matrix.std(axis=0, keepdims=True) + 1e-8
    )
    support = (
        0.40 * z.mean(axis=1)
        + 0.35 * z[:, int(0.80 * RESAMPLE_N) :].mean(axis=1)
        + 0.25 * np.quantile(z, 0.90, axis=1)
    )
    return {letter: float(value) for letter, value in zip(letters, support)}


def hidden_seq_batch(model, items: list[tuple[torch.Tensor, int]]) -> list[np.ndarray]:
    """Extract final-layer generated-token states for a padded sequence batch."""

    if not items:
        return []
    device = mi_exp.get_model_device(model)
    lengths = [int(seq.shape[1]) for seq, _ in items]
    max_length = max(lengths)
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", 0)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0] if pad_token_id else 0
    input_ids = torch.full(
        (len(items), max_length),
        int(pad_token_id),
        dtype=items[0][0].dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(items), max_length), dtype=torch.long, device=device
    )
    for index, ((seq, _), length) in enumerate(zip(items, lengths)):
        input_ids[index, :length] = seq[0].to(device)
        attention_mask[index, :length] = 1
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = output.hidden_states[-1].detach().cpu().float().numpy()
    result = []
    for index, ((_, prompt_len), length) in enumerate(zip(items, lengths)):
        end = prompt_len + min(length - prompt_len, MI_MAX_STEPS)
        result.append(hidden[index, prompt_len:end].copy())
    del output, hidden, input_ids, attention_mask
    return result


def run_candidate_curves_batched(model, tokenizer, batch_size: int = 4) -> dict:
    """Resume candidate-curve extraction with batched model forward passes."""

    if batch_size < 2:
        return run_candidate_curves(model, tokenizer)
    plan = build_plan()
    questions = mi_exp.load_questions(DATA_JSONL)
    store_path = OUTPUT_DIR / "first_pass_biased_generation_store.pkl"
    with store_path.open("rb") as handle:
        generation_store = pickle.load(handle)
    curve_path = OUTPUT_DIR / "all_candidate_curve_store.pkl"
    if curve_path.exists():
        with curve_path.open("rb") as handle:
            curve_store = pickle.load(handle)
    else:
        curve_store = {}
    representation_cache = mi_exp.AnswerRepCache(model, tokenizer, TARGET_MODE)
    arguments = SimpleNamespace(
        target_mode=TARGET_MODE, mi_max_dim=MI_MAX_DIM, mi_sigma=MI_SIGMA
    )
    pending = []
    for row in plan.itertuples():
        case_id = str(row.case_id)
        stored = generation_store.get(case_id)
        if case_id in curve_store or stored is None:
            continue
        if not stored.get("complete") or stored.get("model_answer") not in LETTERS:
            continue
        pending.append(row)
    pending.sort(
        key=lambda row: int(generation_store[str(row.case_id)]["seq"].shape[1])
    )

    position = 0
    active_batch_size = batch_size
    while position < len(pending):
        rows = pending[position : position + active_batch_size]
        items = [
            (
                generation_store[str(row.case_id)]["seq"],
                int(generation_store[str(row.case_id)]["prompt_len"]),
            )
            for row in rows
        ]
        try:
            hidden_batch = hidden_seq_batch(model, items)
        except torch.cuda.OutOfMemoryError:
            if active_batch_size <= 2:
                raise
            active_batch_size = max(2, active_batch_size // 2)
            torch.cuda.empty_cache()
            print(
                f"candidate batch OOM; reducing batch size to {active_batch_size}",
                flush=True,
            )
            continue

        records = [record_for_plan(questions, row) for row in rows]
        text_maps = []
        all_texts = []
        for record in records:
            mapping = {
                letter: mi_exp.target_text(record, letter, arguments.target_mode)
                for letter in LETTERS
                if letter in record["options"]
            }
            text_maps.append(mapping)
            all_texts.extend(mapping.values())
        representations = representation_cache.get_many(all_texts)
        for row, record, hidden, text_map in zip(
            rows, records, hidden_batch, text_maps
        ):
            curves = {
                letter: mi_exp.estimate_mi_curve(
                    hidden,
                    representations[text],
                    arguments.mi_max_dim,
                    arguments.mi_sigma,
                )
                for letter, text in text_map.items()
            }
            case_id = str(row.case_id)
            curve_store[case_id] = {
                "curves": curves,
                "support": candidate_support_scores(curves),
                "model_answer": generation_store[case_id]["model_answer"],
            }
        position += len(rows)
        save_pickle(curve_store, curve_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"holdout candidate curves batch={active_batch_size}: "
            f"{len(curve_store)}/{len(plan)}",
            flush=True,
        )
    save_pickle(curve_store, curve_path)
    print(f"holdout candidate curves complete: {len(curve_store)}/{len(plan)}", flush=True)
    return curve_store


def run_candidate_curves(model, tokenizer) -> dict:
    plan = build_plan()
    questions = mi_exp.load_questions(DATA_JSONL)
    store_path = OUTPUT_DIR / "first_pass_biased_generation_store.pkl"
    with store_path.open("rb") as handle:
        generation_store = pickle.load(handle)
    curve_path = OUTPUT_DIR / "all_candidate_curve_store.pkl"
    if curve_path.exists():
        with curve_path.open("rb") as handle:
            curve_store = pickle.load(handle)
    else:
        curve_store = {}
    representation_cache = mi_exp.AnswerRepCache(model, tokenizer, TARGET_MODE)
    arguments = SimpleNamespace(
        target_mode=TARGET_MODE, mi_max_dim=MI_MAX_DIM, mi_sigma=MI_SIGMA
    )

    for index, row in enumerate(plan.itertuples(), 1):
        case_id = str(row.case_id)
        if case_id in curve_store or case_id not in generation_store:
            continue
        stored = generation_store[case_id]
        if not stored.get("complete") or stored.get("model_answer") not in LETTERS:
            continue
        record = record_for_plan(questions, row)
        hidden = mi_exp.hidden_seq(
            model, stored["seq"], stored["prompt_len"], MI_MAX_STEPS
        )
        curves = mi_exp.candidate_curves(
            record, hidden, representation_cache, arguments
        )
        curve_store[case_id] = {
            "curves": curves,
            "support": candidate_support_scores(curves),
            "model_answer": stored["model_answer"],
        }
        del hidden
        if index % CHECKPOINT_EVERY == 0:
            save_pickle(curve_store, curve_path)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"holdout candidate curves: {len(curve_store)}/{len(plan)}", flush=True)
    save_pickle(curve_store, curve_path)
    print(f"holdout candidate curves complete: {len(curve_store)}/{len(plan)}", flush=True)
    return curve_store


def run_checkpoint_probes(model, tokenizer, batch_size: int = 8) -> pd.DataFrame:
    plan = build_plan()
    labels = pd.read_csv(OUTPUT_DIR / "first_pass_evaluation_labels.csv")
    with (OUTPUT_DIR / "first_pass_biased_generation_store.pkl").open("rb") as handle:
        store = pickle.load(handle)
    metadata = plan.merge(
        labels,
        on=["case_id", "split", "q_idx", "bias_type"],
        how="inner",
    )
    output_path = OUTPUT_DIR / "checkpoint_probe_logits.csv"
    error_path = OUTPUT_DIR / "checkpoint_probe_errors.json"
    probe = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    errors = (
        json.loads(error_path.read_text(encoding="utf-8"))
        if error_path.exists()
        else {}
    )
    completed = set(errors)
    if not probe.empty:
        required = probe[probe.checkpoint.isin(["prompt_end", "pre_final"])]
        counts = required.groupby("case_id").checkpoint.nunique()
        completed.update(counts[counts.eq(2)].index.astype(str))

    metadata_columns = [
        "case_id",
        "split",
        "q_idx",
        "bias_type",
        "outcome_group",
        "biased_answer",
        "gold",
    ]
    pending = [
        str(case_id)
        for case_id in metadata.case_id
        if str(case_id) not in completed and str(case_id) in store
    ]
    chunk_size = 256
    for start in range(0, len(pending), chunk_size):
        case_ids = pending[start : start + chunk_size]
        current = extract_checkpoint_probe_logits(
            model, tokenizer, store, case_ids, batch_size=batch_size
        )
        errors.update(current.attrs.get("errors", {}))
        current = current.merge(
            metadata[metadata_columns].rename(
                columns={"biased_answer": "model_answer"}
            ),
            on="case_id",
            how="left",
        )
        probe = pd.concat([probe, current], ignore_index=True).drop_duplicates(
            ["case_id", "checkpoint"], keep="last"
        )
        probe.to_csv(output_path, index=False, encoding="utf-8-sig")
        error_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        finished = len(completed) + min(start + chunk_size, len(pending))
        print(
            f"holdout checkpoint probe cases: {finished}/{len(metadata)}; "
            f"rows={len(probe)}; errors={len(errors)}",
            flush=True,
        )
    print(
        f"holdout checkpoint probes complete: {len(probe)} rows; errors={len(errors)}"
    )
    return probe


def option_block(options: dict) -> str:
    return "\n".join(
        f"{letter}. {options[letter]}" for letter in LETTERS if letter in options
    )


def build_targeted_review_prompt(
    tokenizer, record: dict, model_answer: str, challenger: str
) -> str:
    if model_answer not in LETTERS:
        raise ValueError(f"Invalid Targeted model_answer: {model_answer!r}")
    if challenger not in LETTERS or challenger == model_answer:
        raise ValueError(
            f"Invalid Targeted challenger {challenger!r} for answer {model_answer!r}"
        )
    body = TARGETED_REVIEW_TEMPLATE.format(
        model_answer=model_answer,
        challenger=challenger,
        question=record["biased_question"],
        options=option_block(record["options"]),
    )
    return mi_exp.chat_wrap(tokenizer, body)


def build_generic_review_prompt(tokenizer, record: dict, model_answer: str) -> str:
    if model_answer not in LETTERS:
        raise ValueError(f"Invalid Generic model_answer: {model_answer!r}")
    body = GENERIC_REVIEW_TEMPLATE.format(
        model_answer=model_answer,
        question=record["biased_question"],
        options=option_block(record["options"]),
    )
    return mi_exp.chat_wrap(tokenizer, body)


def run_reviews(
    model,
    tokenizer,
    selected_only: bool = False,
    review_types: tuple[str, ...] = ("targeted", "generic"),
) -> None:
    """Generate targeted and generic counterfactual reviews with checkpoints."""
    invalid_types = set(review_types) - {"targeted", "generic"}
    if invalid_types:
        raise ValueError(f"Unknown review types: {sorted(invalid_types)}")
    plan = build_plan()
    questions = mi_exp.load_questions(DATA_JSONL)
    policy = pd.read_csv(OUTPUT_DIR / "frozen_policy_scores_before_unblinding.csv")
    suffix = "all"
    if selected_only:
        trigger_columns = [column for column in policy if column.startswith("trigger_")]
        if not trigger_columns:
            raise RuntimeError("No frozen trigger column found for selected reviews")
        trigger_column = trigger_columns[-1]
        policy = policy[policy[trigger_column].astype(bool)].copy()
        suffix = "selected"
    plan = plan.merge(policy[["case_id", "model_answer", "challenger"]], on="case_id")

    for review_type in review_types:
        output_path = OUTPUT_DIR / f"{review_type}_{suffix}_reevaluation.csv"
        if output_path.exists():
            rows = pd.read_csv(output_path).to_dict("records")
        elif not selected_only and (
            OUTPUT_DIR / f"{review_type}_selected_reevaluation.csv"
        ).exists():
            rows = pd.read_csv(
                OUTPUT_DIR / f"{review_type}_selected_reevaluation.csv"
            ).to_dict("records")
        else:
            rows = []
        completed = {str(row["case_id"]) for row in rows}
        for index, row in enumerate(plan.itertuples(), 1):
            case_id = str(row.case_id)
            if case_id in completed:
                continue
            record = record_for_plan(questions, row)
            if review_type == "targeted":
                prompt = build_targeted_review_prompt(
                    tokenizer, record, row.model_answer, row.challenger
                )
            else:
                prompt = build_generic_review_prompt(
                    tokenizer, record, row.model_answer
                )
            output = mi_exp.generate_complete(
                model,
                tokenizer,
                prompt,
                REEVAL_MAX_NEW_TOKENS,
                RETRY_NEW_TOKENS,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "review_type": review_type,
                    "model_answer": row.model_answer,
                    "challenger": row.challenger if review_type == "targeted" else "",
                    "reeval_answer": output["answer"],
                    "reeval_complete": output["complete"],
                    "prompt": prompt,
                    "gen_text": output["gen_text"],
                }
            )
            completed.add(case_id)
            del output
            if index % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_csv(
                    output_path, index=False, encoding="utf-8-sig"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"holdout {review_type} review: {len(completed)}/{len(plan)}",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"holdout {review_type} review complete: {len(rows)}/{len(plan)}")


def run_all_reviews(model, tokenizer) -> None:
    run_reviews(model, tokenizer, selected_only=False)


def run_selected_reviews(model, tokenizer) -> None:
    run_reviews(model, tokenizer, selected_only=True)


def run_all_feature_stages(model, tokenizer, include_clean: bool = True) -> None:
    build_plan()
    run_first_pass(model, tokenizer, include_clean=include_clean)
    candidate_batch_size = int(os.environ.get("MI_SR_CANDIDATE_BATCH_SIZE", "1"))
    if candidate_batch_size > 1:
        run_candidate_curves_batched(
            model, tokenizer, batch_size=candidate_batch_size
        )
    else:
        run_candidate_curves(model, tokenizer)
    run_checkpoint_probes(model, tokenizer)
