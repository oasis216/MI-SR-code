"""Paper-scale current-dataset experiment with immutable question-level splits.

This module is imported into the already-loaded Qwen notebook kernel.  Each
expensive stage checkpoints through ``run_frozen_holdout`` and can be resumed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_IMPORT_CWD = Path.cwd()
import autonomous_harmed_mi_search as mi_exp  # noqa: E402
os.chdir(_IMPORT_CWD)
import run_frozen_holdout as runner  # noqa: E402


EXPERIMENT_ROOT = Path(
    os.environ.get("MI_SR_EXPERIMENT_ROOT", ROOT / "output_paper_full_v1")
).resolve()
DISCOVERY_DIR = EXPERIMENT_ROOT / "discovery"
VALIDATION_DIR = EXPERIMENT_ROOT / "validation"
CLEAN_DIR = EXPERIMENT_ROOT / "clean_reference"
FROZEN_DIR = EXPERIMENT_ROOT / "frozen"
ANALYSIS_DIR = EXPERIMENT_ROOT / "analysis"
DATA_JSONL = runner.DATA_JSONL
RANDOM_SEED = 20260810
DISCOVERY_N = 255
BIAS_TYPES = tuple(mi_exp.BIAS_TYPES)
CHECKPOINT_EVERY = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.equals(frame):
            raise RuntimeError(f"Immutable plan mismatch: {path}")
        return existing
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def build_protocol() -> dict[str, pd.DataFrame]:
    """Create immutable source-question split and all three execution plans."""
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    questions = mi_exp.load_questions(DATA_JSONL)
    if len(questions) != 1273:
        raise RuntimeError(f"Expected 1273 source questions, found {len(questions)}")
    q_indices = np.arange(len(questions), dtype=int)
    gold = np.asarray([str(question["answer_idx"]) for question in questions])
    discovery_q, validation_q = train_test_split(
        q_indices,
        train_size=DISCOVERY_N,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=gold,
    )
    discovery_q = np.sort(discovery_q)
    validation_q = np.sort(validation_q)
    split = pd.DataFrame(
        {
            "q_idx": q_indices,
            "gold": gold,
            "question_split": np.where(
                np.isin(q_indices, discovery_q), "discovery", "validation"
            ),
        }
    )
    _write_once(split, EXPERIMENT_ROOT / "question_split.csv")

    discovery_rows = []
    position = 0
    for q_idx in discovery_q:
        for bias_index, bias_type in enumerate(BIAS_TYPES):
            discovery_rows.append(
                {
                    "case_id": f"d{int(q_idx):04d}_{bias_index}",
                    "split": "paper_discovery",
                    "split_position": position,
                    "q_idx": int(q_idx),
                    "bias_type": bias_type,
                    "record_seed": (
                        RANDOM_SEED
                        + position * 17
                        + int(q_idx) * 31
                        + bias_index * 101
                    ),
                }
            )
            position += 1
    discovery = pd.DataFrame(discovery_rows)
    _write_once(discovery, DISCOVERY_DIR / "experiment_plan.csv")

    rng = np.random.default_rng(RANDOM_SEED + 1)
    shuffled_validation = validation_q.copy()
    rng.shuffle(shuffled_validation)
    bias_sequence = (list(BIAS_TYPES) * math.ceil(len(validation_q) / len(BIAS_TYPES)))[
        : len(validation_q)
    ]
    rng.shuffle(bias_sequence)
    validation = pd.DataFrame(
        {
            "case_id": [f"v{i:04d}" for i in range(len(validation_q))],
            "split": "paper_validation",
            "split_position": np.arange(len(validation_q)),
            "q_idx": shuffled_validation,
            "bias_type": bias_sequence,
        }
    )
    validation["record_seed"] = [
        RANDOM_SEED
        + 1_000_000
        + int(row.split_position) * 17
        + int(row.q_idx) * 31
        + BIAS_TYPES.index(str(row.bias_type)) * 101
        for row in validation.itertuples()
    ]
    _write_once(validation, VALIDATION_DIR / "experiment_plan.csv")

    clean = pd.DataFrame(
        {
            "case_id": [f"c{i:04d}" for i in q_indices],
            "q_idx": q_indices,
            "question_split": split["question_split"],
            "record_seed": RANDOM_SEED + 2_000_000 + q_indices * 31,
        }
    )
    _write_once(clean, CLEAN_DIR / "clean_plan.csv")

    if set(discovery_q) & set(validation_q):
        raise RuntimeError("Question leakage between discovery and validation")
    manifest_path = EXPERIMENT_ROOT / "protocol_manifest.json"
    manifest = {
        "protocol_version": "paper_full_v1",
        "random_seed": RANDOM_SEED,
        "n_source_questions": len(questions),
        "discovery_unique_questions": len(discovery_q),
        "validation_unique_questions": len(validation_q),
        "discovery_biased_cases": len(discovery),
        "validation_biased_cases": len(validation),
        "clean_reference_cases": len(clean),
        "discovery_bias_counts": discovery["bias_type"].value_counts().to_dict(),
        "validation_bias_counts": validation["bias_type"].value_counts().to_dict(),
        "question_overlap": 0,
        "primary_gate": "no-bias-type grouped-OOF one-SE absolute action advantage",
        "validation_fixed_review_fraction": False,
        "plans": {
            "question_split": "question_split.csv",
            "discovery": "discovery/experiment_plan.csv",
            "validation": "validation/experiment_plan.csv",
            "clean": "clean_reference/clean_plan.csv",
        },
        "plan_sha256": {
            "question_split": sha256(EXPERIMENT_ROOT / "question_split.csv"),
            "discovery": sha256(DISCOVERY_DIR / "experiment_plan.csv"),
            "validation": sha256(VALIDATION_DIR / "experiment_plan.csv"),
            "clean": sha256(CLEAN_DIR / "clean_plan.csv"),
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "random_seed",
            "discovery_unique_questions",
            "validation_unique_questions",
            "discovery_biased_cases",
            "validation_biased_cases",
        ):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"Immutable manifest mismatch: {key}")
        if existing.get("plan_sha256") not in (None, manifest["plan_sha256"]):
            raise RuntimeError("Immutable plan hash mismatch")
        if "plan_sha256" not in existing:
            existing["plan_sha256"] = manifest["plan_sha256"]
            manifest_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        manifest = existing
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"split": split, "discovery": discovery, "validation": validation, "clean": clean}


def _configure(output_dir: Path, plan_name: str) -> None:
    plans = build_protocol()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.OUTPUT_DIR = output_dir
    runner.RANDOM_SEED = RANDOM_SEED
    runner.build_plan = lambda: plans[plan_name].copy()


def run_discovery_features(model, tokenizer) -> None:
    _configure(DISCOVERY_DIR, "discovery")
    runner.run_all_feature_stages(model, tokenizer, include_clean=False)


def run_validation_features(model, tokenizer) -> None:
    _configure(VALIDATION_DIR, "validation")
    runner.run_all_feature_stages(model, tokenizer, include_clean=False)


def run_discovery_reviews(model, tokenizer) -> None:
    _configure(DISCOVERY_DIR, "discovery")
    runner.run_all_reviews(model, tokenizer)


def run_validation_reviews(model, tokenizer) -> None:
    _configure(VALIDATION_DIR, "validation")
    runner.run_all_reviews(model, tokenizer)


def run_clean_references(model, tokenizer) -> None:
    plans = build_protocol()
    plan = plans["clean"]
    questions = mi_exp.load_questions(DATA_JSONL)
    output_path = CLEAN_DIR / "clean_reference_outputs.csv"
    rows = pd.read_csv(output_path).to_dict("records") if output_path.exists() else []
    completed = {str(row["case_id"]) for row in rows}

    def checkpoint() -> None:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    for index, row in enumerate(plan.itertuples(), 1):
        case_id = str(row.case_id)
        if case_id in completed:
            continue
        previous = Path.cwd()
        try:
            os.chdir(REPO_ROOT)
            record = mi_exp.make_biased_record(
                questions[int(row.q_idx)], "confirmation", int(row.record_seed)
            )
        finally:
            os.chdir(previous)
        prompt = mi_exp.build_prompt(tokenizer, record, "clean", runner.PROMPT_VARIANT)
        output = mi_exp.generate_complete(
            model, tokenizer, prompt, runner.MAX_NEW_TOKENS, runner.RETRY_NEW_TOKENS
        )
        rows.append(
            {
                "case_id": case_id,
                "q_idx": int(row.q_idx),
                "question_split": row.question_split,
                "gold": record["gold"],
                "clean_answer": output["answer"],
                "clean_correct": bool(output["complete"] and output["answer"] == record["gold"]),
                "clean_complete": bool(output["complete"]),
                "prompt": prompt,
                "gen_text": output["gen_text"],
            }
        )
        completed.add(case_id)
        del output
        if index % CHECKPOINT_EVERY == 0:
            checkpoint()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"paper clean reference: {len(completed)}/{len(plan)}", flush=True)
    checkpoint()
    print(f"paper clean reference complete: {len(rows)}/{len(plan)}", flush=True)


def status() -> dict:
    plans = build_protocol()
    result = {
        "planned": {name: len(frame) for name, frame in plans.items()},
    }
    for name, directory in (
        ("discovery", DISCOVERY_DIR),
        ("validation", VALIDATION_DIR),
    ):
        result[name] = {}
        for filename in (
            "first_pass_evaluation_labels.csv",
            "checkpoint_probe_logits.csv",
            "targeted_all_reevaluation.csv",
            "generic_all_reevaluation.csv",
        ):
            path = directory / filename
            if not path.exists():
                result[name][filename] = {"rows": 0, "unique_cases": 0}
                continue
            frame = pd.read_csv(path)
            result[name][filename] = {
                "rows": len(frame),
                "unique_cases": int(frame["case_id"].nunique())
                if "case_id" in frame
                else 0,
            }
        curve_path = directory / "all_candidate_curve_store.pkl"
        if curve_path.exists():
            with curve_path.open("rb") as handle:
                result[name]["candidate_curve_cases"] = len(pickle.load(handle))
        else:
            result[name]["candidate_curve_cases"] = 0
    clean_path = CLEAN_DIR / "clean_reference_outputs.csv"
    result["clean_completed"] = len(pd.read_csv(clean_path)) if clean_path.exists() else 0
    return result


build_protocol()
