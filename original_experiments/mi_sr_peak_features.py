"""Extract frozen candidate MI-peak checkpoints for a paper-scale run.

The raw first-pass generations and MI curves are never rewritten.  This module
adds candidate-specific probes before/after the strongest MI peak chunk and
emits the canonical feature table consumed by :mod:`mi_sr_protocol`.
"""

from __future__ import annotations

import gc
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from final_answer_logits import find_final_answer_token_index
from run_mi_aligned_confidence_leap_pilot import (
    extract_probes,
    strongest_peak,
    token_chunks,
)
from search_untrained_dual_challenger import NEG, POS


LETTERS = tuple("ABCDE")
REASONING_RE = re.compile(
    r"\b(wait|however|therefore|thus|hence|but|instead|alternatively|"
    r"reconsider|actually|although|whereas|because|if|then|suggests|"
    r"supports|rules? out|unlikely|likely|evidence|diagnos)\w*\b",
    re.I,
)


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_peak_probe_plan(
    source_dir: Path,
    tokenizer,
    force: bool = False,
) -> pd.DataFrame:
    """Locate the strongest pre-commitment MI peak chunk for every alternative."""

    output_dir = source_dir / "mi_sr_peak_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "peak_probe_plan.csv"
    if path.exists() and not force:
        return pd.read_csv(path)

    labels = pd.read_csv(source_dir / "first_pass_evaluation_labels.csv")
    store = _load_pickle(source_dir / "first_pass_biased_generation_store.pkl")
    curves = _load_pickle(source_dir / "all_candidate_curve_store.pkl")
    eligible = labels[
        labels.biased_complete.fillna(False).astype(bool)
        & labels.biased_answer.fillna("").astype(str).isin(LETTERS)
    ]

    rows = []
    for label in eligible.itertuples(index=False):
        case_id = str(label.case_id)
        if case_id not in curves or case_id not in store:
            continue
        generated = [int(value) for value in store[case_id]["gen_ids"]]
        curve_map = curves[case_id]["curves"]
        curve_length = min(len(value) for value in curve_map.values())
        try:
            final_step = find_final_answer_token_index(generated, tokenizer)
        except Exception:
            final_step = curve_length - 1
        end_step = min(curve_length - 1, max(2, int(final_step) - 1))
        boundaries = token_chunks(tokenizer, generated, end_step)
        for candidate, values in curve_map.items():
            candidate = str(candidate).upper()
            if candidate == str(label.biased_answer).upper():
                continue
            peak_step, prominence = strongest_peak(np.asarray(values), end_step)
            interval = max(
                0,
                min(
                    len(boundaries) - 2,
                    int(np.searchsorted(boundaries, peak_step, side="right") - 1),
                ),
            )
            before = int(boundaries[interval])
            after = int(boundaries[interval + 1])
            next_step = int(boundaries[min(interval + 2, len(boundaries) - 1)])
            span = tokenizer.decode(
                generated[before : min(len(generated), after + 1)],
                skip_special_tokens=True,
            ).strip()
            positive = bool(POS.search(span))
            negative = bool(NEG.search(span))
            rows.append(
                {
                    "case_id": case_id,
                    "candidate": candidate,
                    "model_answer": str(label.biased_answer).upper(),
                    "peak_step": int(peak_step),
                    "peak_prominence": float(prominence),
                    "before_step": before,
                    "after_step": after,
                    "next_step": next_step,
                    "peak_position": float(peak_step / max(end_step, 1)),
                    "peak_span": span,
                    "peak_positive": positive,
                    "peak_negative": negative,
                    "mi_semantic_support": positive and not negative,
                    "peak_reasoning": bool(REASONING_RE.search(span)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result


def _probe_plan(peak_plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in peak_plan.itertuples(index=False):
        for role, step in (
            ("mi_before", row.before_step),
            ("mi_after", row.after_step),
            ("mi_next", row.next_step),
        ):
            rows.append(
                {
                    "pilot_case_id": row.case_id,
                    "step": int(step),
                    "request_role": role,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["pilot_case_id", "step"])


def extract_peak_features(
    model,
    tokenizer,
    source_dir: Path,
    force: bool = False,
    preserve_existing_probes: bool = True,
) -> pd.DataFrame:
    """Create the canonical per-candidate feature table with checkpoints."""

    source_dir = Path(source_dir).resolve()
    output_dir = source_dir / "mi_sr_peak_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "candidate_peak_features.csv"
    if feature_path.exists() and not force:
        existing = pd.read_csv(feature_path)
        curve_store = _load_pickle(source_dir / "all_candidate_curve_store.pkl")
        expected_cases = len(curve_store)
        existing_cases = existing.case_id.astype(str).nunique()
        if existing_cases == expected_cases:
            return existing
        # A later phase extended the generation/curve stores. Rebuild this
        # derived view from the complete raw stores; raw MI curves are retained.
        force = True

    peak_plan = build_peak_probe_plan(source_dir, tokenizer, force=force)
    if peak_plan.empty:
        raise RuntimeError(f"No eligible MI peak checkpoints in {source_dir}")
    store = _load_pickle(source_dir / "first_pass_biased_generation_store.pkl")
    request_plan = _probe_plan(peak_plan)
    request_plan.to_csv(output_dir / "probe_requests.csv", index=False)
    if force and not preserve_existing_probes:
        probe_path = output_dir / "chunk_checkpoint_probes.csv"
        if probe_path.exists():
            probe_path.unlink()
    probes = extract_probes(model, tokenizer, output_dir, store, request_plan)
    checkpoint = pd.read_csv(source_dir / "checkpoint_probe_logits.csv")
    labels = pd.read_csv(source_dir / "first_pass_evaluation_labels.csv")

    probe_lookup = probes.set_index(["pilot_case_id", "step"])
    checkpoint_lookup = checkpoint.set_index(["case_id", "checkpoint"])
    label_lookup = labels.set_index("case_id")
    rows = []
    for row in peak_plan.itertuples(index=False):
        case_id = str(row.case_id)
        candidate = str(row.candidate)
        before = float(
            probe_lookup.loc[(case_id, int(row.before_step)), f"prob_{candidate}"]
        )
        after = float(
            probe_lookup.loc[(case_id, int(row.after_step)), f"prob_{candidate}"]
        )
        next_support = float(
            probe_lookup.loc[(case_id, int(row.next_step)), f"prob_{candidate}"]
        )
        prompt = float(
            checkpoint_lookup.loc[(case_id, "prompt_end"), f"prob_{candidate}"]
        )
        final = float(
            checkpoint_lookup.loc[(case_id, "pre_final"), f"prob_{candidate}"]
        )
        label = label_lookup.loc[case_id]
        rows.append(
            {
                "case_id": case_id,
                "q_idx": int(label.q_idx),
                "bias_type": str(label.bias_type),
                "candidate": candidate,
                "model_answer": str(row.model_answer),
                "support_prompt": prompt,
                "support_before": before,
                "support_after": after,
                "support_next": next_support,
                "support_final": final,
                "persistent_support": min(after, next_support, final),
                "confidence_leap": after - before,
                "peak_prominence": float(row.peak_prominence),
                "peak_position": float(row.peak_position),
                "mi_semantic_support": bool(row.mi_semantic_support),
                "peak_positive": bool(row.peak_positive),
                "peak_negative": bool(row.peak_negative),
                "peak_reasoning": bool(row.peak_reasoning),
                "peak_span": str(row.peak_span),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(feature_path, index=False, encoding="utf-8-sig")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def answer_support_table(source_dir: Path) -> pd.DataFrame:
    """Extract first-answer support at prompt-end and pre-final checkpoints."""

    checkpoint = pd.read_csv(Path(source_dir) / "checkpoint_probe_logits.csv")
    labels = pd.read_csv(Path(source_dir) / "first_pass_evaluation_labels.csv")
    answers = labels.set_index("case_id").biased_answer.astype(str)
    pivot = checkpoint.set_index(["case_id", "checkpoint"])
    rows = []
    for case_id, model_answer in answers.items():
        if model_answer not in LETTERS:
            continue
        try:
            prompt = float(pivot.loc[(case_id, "prompt_end"), f"prob_{model_answer}"])
            final = float(pivot.loc[(case_id, "pre_final"), f"prob_{model_answer}"])
        except KeyError:
            continue
        rows.append(
            {
                "case_id": case_id,
                "answer_prompt_support": prompt,
                "answer_final_support": final,
            }
        )
    return pd.DataFrame(rows)
