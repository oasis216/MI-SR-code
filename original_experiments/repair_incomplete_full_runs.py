#!/usr/bin/env python3
"""Repair incomplete full-run MI-SR cases without changing the protocol.

The original successful rows remain immutable. Only first-pass, clean-reference,
or review generations that are missing or fail the mandatory final-answer format
are regenerated. Newly complete first-pass cases then traverse the original
candidate-curve, Probe, MI-peak, Challenger, and Gate pipeline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import autonomous_harmed_mi_search as mi_exp
import mi_sr_full_runner_v2 as legacy
import mi_sr_full_targeted_runner as full_runner
import run_frozen_holdout as runner
from cross_model_runtime import load_model_and_tokenizer
from mi_sr_deploy_v3 import build_deploy_policy
from mi_sr_peak_features import extract_peak_features


RUN_ROOT = ROOT / "output_mi_sr_full_targeted_v5"
BACKUP_ROOT = ROOT / "output_mi_sr_full_targeted_v5_before_repair_20260726"
REVISION = "incomplete_repair_v20260726"
LETTERS = tuple("ABCDE")
MODEL_KEYS = (
    "qwen3_14b_reference",
    "deepseek_r1_qwen14b",
    "mistral7b_instruct_v03",
)
REPAIR_TOKEN_LIMITS = {
    "qwen3_14b_reference": 1024,
    "deepseek_r1_qwen14b": 1280,
    "mistral7b_instruct_v03": 1024,
}
FINAL_LINE_WITH_NEWLINE = re.compile(
    r"(?:^|\n)\s*\**Final\s+Answer\s*[:\-]\s*\(?[A-E]\)?\s*"
    r"[\),.:：-]\s*\S[^\n]{1,}\n",
    flags=re.I,
)
REPAIRED_FINAL_LINE_AT_END = re.compile(
    r"Final\s+Answer\s*[:\-]\s*\(?[A-E]\)?\s*"
    r"[\),.:：-]\s*\S[^\n]*$",
    flags=re.I,
)
CONTRADICTORY_FINAL_LINE = re.compile(
    r"\b(?:incorrect|not\s+(?:the\s+)?correct|not\s+(?:the\s+)?primary|"
    r"not\s+(?:the\s+)?best|cannot\s+be\s+determined|none\s+of\s+the\s+above|"
    r"not\s+provided|insufficient\s+information)\b",
    flags=re.I,
)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1"})


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def repair_answer_complete(text: str | None) -> bool:
    normalized = (text or "").strip().strip("*").strip()
    matches = list(
        re.finditer(
            r"Final\s+Answer\s*[:\-]\s*\(?[A-E]\)?[^\n]*$",
            normalized,
            flags=re.I,
        )
    )
    if not matches:
        return False
    final_line = matches[-1].group(0)
    if CONTRADICTORY_FINAL_LINE.search(final_line):
        return False
    return bool(
        mi_exp.final_answer_complete(normalized)
        or REPAIRED_FINAL_LINE_AT_END.search(normalized)
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".repair.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def atomic_pickle(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".repair.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def revision_dir(run_dir: Path) -> Path:
    path = run_dir / "revisions" / REVISION
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_audit_path(run_dir: Path, stage: str) -> Path:
    return revision_dir(run_dir) / f"{stage}_attempts.csv"


def load_stage_audit(run_dir: Path, stage: str) -> list[dict]:
    path = stage_audit_path(run_dir, stage)
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def save_stage_audit(run_dir: Path, stage: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(["stage", "case_id"], keep="last")
    atomic_csv(frame, stage_audit_path(run_dir, stage))


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def backup_run(model_key: str) -> Path:
    source = RUN_ROOT / model_key / "all_biases"
    target = BACKUP_ROOT / model_key / "all_biases"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


class FlexibleFinalLineStoppingCriteria(StoppingCriteria):
    """Stop after a syntactically valid final-answer line reaches a newline."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.done_lengths: dict[int, int] = {}

    def __call__(self, input_ids, scores, **kwargs):
        done = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        for index in range(input_ids.shape[0]):
            if index in self.done_lengths:
                done[index] = True
                continue
            generated = input_ids[index, self.prompt_length :]
            if generated.numel() < 5:
                continue
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            matches = list(FINAL_LINE_WITH_NEWLINE.finditer(text))
            final_line = matches[-1].group(0) if matches else ""
            if matches and not CONTRADICTORY_FINAL_LINE.search(final_line):
                self.done_lengths[index] = int(generated.numel())
                done[index] = True
        return done


def generate_repair(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    device = mi_exp.get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    criteria = StoppingCriteriaList(
        [
            mi_exp.FinalOptionStoppingCriteria(tokenizer, prompt),
            FlexibleFinalLineStoppingCriteria(tokenizer, prompt_len),
        ]
    )
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos,
            pad_token_id=pad,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria=criteria,
        )
    seq = outputs.sequences
    gen_ids = seq[0, prompt_len:].detach().cpu().tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    result = {
        "seq": seq.detach().cpu(),
        "prompt_len": prompt_len,
        "gen_ids": gen_ids,
        "gen_text": gen_text,
        "answer": mi_exp.parse_answer_letter(gen_text),
        "complete": repair_answer_complete(gen_text),
    }
    del inputs, outputs, seq
    return result


def generate_repair_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int
) -> list[dict]:
    if len(prompts) == 1:
        return [generate_repair(model, tokenizer, prompts[0], max_new_tokens)]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
            mi_exp.get_model_device(model)
        )
    finally:
        tokenizer.padding_side = original_padding_side
    prompt_width = int(inputs["input_ids"].shape[1])
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    flexible = FlexibleFinalLineStoppingCriteria(tokenizer, prompt_width)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos,
            pad_token_id=pad,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria=StoppingCriteriaList([flexible]),
        )
    results = []
    for index in range(len(prompts)):
        generated = outputs.sequences[index, prompt_width:].detach().cpu().tolist()
        stop_length = flexible.done_lengths.get(index)
        if stop_length is None and eos is not None and eos in generated:
            stop_length = generated.index(eos) + 1
        if stop_length is not None:
            generated = generated[:stop_length]
        prompt_ids = inputs["input_ids"][index][
            inputs["attention_mask"][index].bool()
        ].detach().cpu()
        sequence = torch.cat(
            [prompt_ids, torch.tensor(generated, dtype=prompt_ids.dtype)]
        ).unsqueeze(0)
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(
            {
                "seq": sequence,
                "prompt_len": int(prompt_ids.numel()),
                "gen_ids": generated,
                "gen_text": gen_text,
                "answer": mi_exp.parse_answer_letter(gen_text),
                "complete": repair_answer_complete(gen_text),
            }
        )
    del inputs, outputs
    return results


def audit_counts(run_dir: Path) -> dict:
    labels = pd.read_csv(run_dir / "first_pass_evaluation_labels.csv")
    clean = pd.read_csv(run_dir / "clean_reference_outputs.csv")
    targeted = pd.read_csv(run_dir / "targeted_all_reevaluation.csv")
    generic = pd.read_csv(run_dir / "generic_all_reevaluation.csv")
    policy = pd.read_csv(run_dir / "binary_deploy_policy.csv")
    policy_cases = int(policy.case_id.nunique())
    targeted_complete = int(as_bool(targeted.reeval_complete).sum())
    generic_complete = int(as_bool(generic.reeval_complete).sum())
    return {
        "planned": int(labels.case_id.nunique()),
        "first_pass_complete": int(as_bool(labels.biased_complete).sum()),
        "first_pass_incomplete": int((~as_bool(labels.biased_complete)).sum()),
        "policy_cases": policy_cases,
        "clean_complete": int(as_bool(clean.clean_complete).sum()),
        "clean_incomplete": int(1273 - as_bool(clean.clean_complete).sum()),
        "targeted_complete": targeted_complete,
        "targeted_incomplete": policy_cases - targeted_complete,
        "generic_complete": generic_complete,
        "generic_incomplete": policy_cases - generic_complete,
    }


def configure_run(run_dir: Path, model_key: str) -> pd.DataFrame:
    _, plan = legacy._configure(run_dir, model_key, source_question_limit=None)
    if len(plan) != 8911:
        raise RuntimeError(f"{model_key}: expected 8911 planned cases, found {len(plan)}")
    return plan


def repair_first_pass(
    model,
    tokenizer,
    run_dir: Path,
    plan: pd.DataFrame,
    token_limit: int,
    max_cases: int | None,
) -> tuple[list[str], list[dict]]:
    labels_path = run_dir / "first_pass_evaluation_labels.csv"
    prompts_path = run_dir / "first_pass_prompts_outputs.csv"
    store_path = run_dir / "first_pass_biased_generation_store.pkl"
    labels = pd.read_csv(labels_path)
    prompts = pd.read_csv(prompts_path)
    store = load_pickle(store_path)
    invalid = ~as_bool(labels.biased_complete) | ~labels.biased_answer.fillna("").isin(
        LETTERS
    )
    pending = labels.loc[invalid, "case_id"].astype(str).tolist()
    if max_cases is not None:
        pending = pending[:max_cases]
    plan_lookup = plan.set_index("case_id")
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    repaired: list[str] = []
    audit = load_stage_audit(run_dir, "first_pass")
    attempted = {str(row["case_id"]) for row in audit}
    pending = [case_id for case_id in pending if case_id not in attempted]
    if max_cases is not None:
        pending = pending[:max_cases]

    def checkpoint() -> None:
        atomic_csv(labels, labels_path)
        atomic_csv(prompts, prompts_path)
        atomic_pickle(store, store_path)
        save_stage_audit(run_dir, "first_pass", audit)

    jobs = []
    for case_id in pending:
        plan_row = plan_lookup.loc[case_id]
        source = SimpleNamespace(**plan_row.to_dict())
        record = runner.record_for_plan(questions, source)
        prompt = mi_exp.build_prompt(tokenizer, record, "biased", "mi_sr_complete")
        old_prompt_rows = prompts[
            prompts.case_id.astype(str).eq(case_id) & prompts.condition.eq("biased")
        ]
        if len(old_prompt_rows) != 1:
            raise RuntimeError(f"{case_id}: expected one stored biased prompt")
        old_prompt = str(old_prompt_rows.iloc[0].prompt)
        if old_prompt != prompt:
            raise RuntimeError(f"{case_id}: reconstructed prompt differs from archive")
        jobs.append((case_id, record, prompt, str(old_prompt_rows.iloc[0].gen_text)))

    processed = 0
    batch_size = 4
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        results = generate_repair_batch(
            model, tokenizer, [item[2] for item in batch], token_limit
        )
        for (case_id, record, prompt, old_text), result in zip(batch, results):
            processed += 1
            success = bool(result["complete"] and result["answer"] in LETTERS)
            audit.append(
                {
                    "stage": "first_pass",
                    "case_id": case_id,
                    "old_complete": False,
                    "old_answer": labels.loc[
                        labels.case_id.astype(str).eq(case_id), "biased_answer"
                    ].iloc[0],
                    "old_text_sha256": sha256_text(old_text),
                    "new_complete": success,
                    "new_answer": result["answer"],
                    "new_text_sha256": sha256_text(result["gen_text"]),
                    "new_tokens": len(result["gen_ids"]),
                }
            )
            if success:
                mask = labels.case_id.astype(str).eq(case_id)
                labels.loc[mask, "gold"] = record["gold"]
                labels.loc[mask, "biased_answer"] = result["answer"]
                labels.loc[mask, "biased_complete"] = True
                labels.loc[mask, "biased_correct"] = result["answer"] == record["gold"]
                labels.loc[mask, "outcome_group"] = (
                    "BiasedCorrect"
                    if result["answer"] == record["gold"]
                    else "BiasedWrong"
                )
                prompt_mask = prompts.case_id.astype(str).eq(
                    case_id
                ) & prompts.condition.eq("biased")
                prompts.loc[prompt_mask, "answer"] = result["answer"]
                prompts.loc[prompt_mask, "complete"] = True
                prompts.loc[prompt_mask, "gen_text"] = result["gen_text"]
                store[case_id] = {
                    "seq": result["seq"],
                    "prompt_len": result["prompt_len"],
                    "gen_ids": result["gen_ids"],
                    "gen_text": result["gen_text"],
                    "model_answer": result["answer"],
                    "complete": True,
                }
                repaired.append(case_id)
            print(
                f"first-pass repair {processed}/{len(pending)}: "
                f"success={len(repaired)} case={case_id}",
                flush=True,
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if processed % 8 == 0 or processed == len(pending):
            checkpoint()

    checkpoint()
    repaired_all = [
        str(row["case_id"])
        for row in audit
        if str(row.get("new_complete", "")).lower() in {"true", "1"}
    ]
    return sorted(set(repaired_all)), audit


def invalidate_derived_rows(run_dir: Path, case_ids: list[str]) -> None:
    if not case_ids:
        return
    affected = set(map(str, case_ids))
    curve_path = run_dir / "all_candidate_curve_store.pkl"
    curves = load_pickle(curve_path)
    for case_id in affected:
        curves.pop(case_id, None)
    atomic_pickle(curves, curve_path)

    probe_path = run_dir / "checkpoint_probe_logits.csv"
    probes = pd.read_csv(probe_path)
    probes = probes[~probes.case_id.astype(str).isin(affected)].copy()
    atomic_csv(probes, probe_path)
    error_path = run_dir / "checkpoint_probe_errors.json"
    errors = json.loads(error_path.read_text(encoding="utf-8")) if error_path.exists() else {}
    errors = {key: value for key, value in errors.items() if str(key) not in affected}
    error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    chunk_path = run_dir / "mi_sr_peak_features/chunk_checkpoint_probes.csv"
    if chunk_path.exists():
        chunks = pd.read_csv(chunk_path)
        chunks = chunks[~chunks.pilot_case_id.astype(str).isin(affected)].copy()
        atomic_csv(chunks, chunk_path)


def rebuild_features_and_policy(
    model, tokenizer, run_dir: Path, repaired_case_ids: list[str]
) -> pd.DataFrame:
    invalidate_derived_rows(run_dir, repaired_case_ids)
    runner.run_candidate_curves_batched(model, tokenizer, batch_size=4)
    runner.run_checkpoint_probes(model, tokenizer, batch_size=8)
    extract_peak_features(model, tokenizer, run_dir, force=True)
    policy, _ = build_deploy_policy(run_dir)
    binary = full_runner.write_binary_policy(run_dir, policy)
    return binary


def repair_clean(
    model,
    tokenizer,
    run_dir: Path,
    token_limit: int,
    max_cases: int | None,
) -> list[dict]:
    path = run_dir / "clean_reference_outputs.csv"
    frame = pd.read_csv(path)
    pending = frame.loc[~as_bool(frame.clean_complete), "q_idx"].astype(int).tolist()
    if max_cases is not None:
        pending = pending[:max_cases]
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    audit = load_stage_audit(run_dir, "clean")
    attempted = {str(row["case_id"]) for row in audit}
    pending = [
        q_idx for q_idx in pending if f"q{q_idx:04d}" not in attempted
    ]
    if max_cases is not None:
        pending = pending[:max_cases]

    def checkpoint() -> None:
        atomic_csv(frame, path)
        save_stage_audit(run_dir, "clean", audit)

    for position, q_idx in enumerate(pending, 1):
        mask = frame.q_idx.eq(q_idx)
        prompt = str(frame.loc[mask, "prompt"].iloc[0])
        result = generate_repair(model, tokenizer, prompt, token_limit)
        success = bool(result["complete"] and result["answer"] in LETTERS)
        audit.append(
            {
                "stage": "clean",
                "case_id": f"q{q_idx:04d}",
                "old_complete": False,
                "old_answer": frame.loc[mask, "clean_answer"].iloc[0],
                "new_complete": success,
                "new_answer": result["answer"],
                "new_tokens": len(result["gen_ids"]),
            }
        )
        if success:
            gold = str(frame.loc[mask, "gold"].iloc[0])
            frame.loc[mask, "clean_answer"] = result["answer"]
            frame.loc[mask, "clean_complete"] = True
            frame.loc[mask, "clean_correct"] = result["answer"] == gold
            frame.loc[mask, "gen_text"] = result["gen_text"]
        print(
            f"clean repair {position}/{len(pending)}: success={success} q_idx={q_idx}",
            flush=True,
        )
        del result
        if position % 10 == 0:
            checkpoint()
    checkpoint()
    return audit


def repair_reviews(
    model,
    tokenizer,
    run_dir: Path,
    plan: pd.DataFrame,
    policy: pd.DataFrame,
    review_type: str,
    token_limit: int,
    max_cases: int | None,
) -> list[dict]:
    path = run_dir / f"{review_type}_all_reevaluation.csv"
    frame = pd.read_csv(path)
    policy = policy.copy()
    policy["case_id"] = policy.case_id.astype(str)
    frame["case_id"] = frame.case_id.astype(str)
    expected = set(policy.case_id)
    complete = as_bool(frame.reeval_complete)
    existing_complete = set(frame.loc[complete, "case_id"])
    pending = sorted(expected - existing_complete)
    if max_cases is not None:
        pending = pending[:max_cases]
    plan_lookup = plan.set_index("case_id")
    policy_lookup = policy.set_index("case_id")
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    audit = load_stage_audit(run_dir, review_type)
    attempted = {str(row["case_id"]) for row in audit}
    pending = [case_id for case_id in pending if case_id not in attempted]
    if max_cases is not None:
        pending = pending[:max_cases]

    def checkpoint() -> None:
        current = frame.drop_duplicates("case_id", keep="last").sort_values("case_id")
        atomic_csv(current, path)
        save_stage_audit(run_dir, review_type, audit)

    jobs = []
    for case_id in pending:
        plan_row = plan_lookup.loc[case_id]
        source = SimpleNamespace(**plan_row.to_dict())
        record = runner.record_for_plan(questions, source)
        policy_row = policy_lookup.loc[case_id]
        if review_type == "targeted":
            prompt = runner.build_targeted_review_prompt(
                tokenizer,
                record,
                str(policy_row.model_answer),
                str(policy_row.challenger),
            )
        else:
            prompt = runner.build_generic_review_prompt(
                tokenizer, record, str(policy_row.model_answer)
            )
        jobs.append((case_id, policy_row, prompt))

    processed = 0
    batch_size = 4
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        results = generate_repair_batch(
            model, tokenizer, [item[2] for item in batch], token_limit
        )
        for (case_id, policy_row, prompt), result in zip(batch, results):
            processed += 1
            success = bool(result["complete"] and result["answer"] in LETTERS)
            audit.append(
                {
                    "stage": review_type,
                    "case_id": case_id,
                    "old_complete": False,
                    "new_complete": success,
                    "new_answer": result["answer"],
                    "new_tokens": len(result["gen_ids"]),
                }
            )
            if success:
                replacement = {
                    "case_id": case_id,
                    "review_type": review_type,
                    "model_answer": str(policy_row.model_answer),
                    "challenger": (
                        str(policy_row.challenger)
                        if review_type == "targeted"
                        else ""
                    ),
                    "reeval_answer": result["answer"],
                    "reeval_complete": True,
                    "prompt": prompt,
                    "gen_text": result["gen_text"],
                }
                frame = frame[~frame.case_id.eq(case_id)].copy()
                frame = pd.concat(
                    [frame, pd.DataFrame([replacement])], ignore_index=True
                )
            print(
                f"{review_type} repair {processed}/{len(pending)}: "
                f"success={success} case={case_id}",
                flush=True,
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if processed % 8 == 0 or processed == len(pending):
            checkpoint()

    frame = frame.drop_duplicates("case_id", keep="last").sort_values("case_id")
    checkpoint()
    return audit


def write_revision_manifest(
    model_key: str,
    run_dir: Path,
    before: dict,
    after: dict,
    audit_rows: list[dict],
) -> Path:
    current_revision_dir = revision_dir(run_dir)
    audit_path = current_revision_dir / "repair_attempts.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")
    payload = {
        "revision": REVISION,
        "model_key": model_key,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "before": before,
        "after": after,
        "repair_token_limit": REPAIR_TOKEN_LIMITS[model_key],
        "prompt_changed": False,
        "decoding_changed": False,
        "decoding_mode": "greedy",
        "repair_change": (
            "longer continuation limit and flexible stop after a syntactically "
            "complete final-answer line"
        ),
        "successful_existing_rows_overwritten": False,
        "backup": str(BACKUP_ROOT / model_key / "all_biases"),
        "audit_csv": str(audit_path),
    }
    manifest_path = current_revision_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def run_repair(model_key: str, gpu_id: int, max_cases: int | None) -> dict:
    run_dir = RUN_ROOT / model_key / "all_biases"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    backup_dir = backup_run(model_key)
    before = audit_counts(backup_dir)
    plan = configure_run(run_dir, model_key)
    model, tokenizer, _ = load_model_and_tokenizer(model_key, gpu_id)
    token_limit = REPAIR_TOKEN_LIMITS[model_key]
    audit_rows: list[dict] = []

    repaired_case_ids, first_audit = repair_first_pass(
        model, tokenizer, run_dir, plan, token_limit, max_cases
    )
    audit_rows.extend(first_audit)
    current = audit_counts(run_dir)
    curves = load_pickle(run_dir / "all_candidate_curve_store.pkl")
    features_current = (
        current["policy_cases"] == current["first_pass_complete"]
        and len(curves) == current["first_pass_complete"]
    )
    if features_current:
        policy = pd.read_csv(run_dir / "binary_deploy_policy.csv")
    else:
        policy = rebuild_features_and_policy(
            model, tokenizer, run_dir, repaired_case_ids
        )
    audit_rows.extend(
        repair_clean(model, tokenizer, run_dir, token_limit, max_cases)
    )
    audit_rows.extend(
        repair_reviews(
            model,
            tokenizer,
            run_dir,
            plan,
            policy,
            "targeted",
            token_limit,
            max_cases,
        )
    )
    audit_rows.extend(
        repair_reviews(
            model,
            tokenizer,
            run_dir,
            plan,
            policy,
            "generic",
            token_limit,
            max_cases,
        )
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = audit_counts(run_dir)
    manifest = write_revision_manifest(
        model_key, run_dir, before, after, audit_rows
    )
    return {
        "model_key": model_key,
        "before": before,
        "after": after,
        "repaired_first_pass_cases": len(repaired_case_ids),
        "manifest": str(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=MODEL_KEYS)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        keys = [args.model_key] if args.model_key else list(MODEL_KEYS)
        result = {
            key: audit_counts(RUN_ROOT / key / "all_biases") for key in keys
        }
    else:
        if not args.model_key:
            parser.error("--model-key is required unless --audit-only is used")
        result = run_repair(args.model_key, args.gpu_id, args.max_cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
