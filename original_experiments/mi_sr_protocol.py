"""Frozen, training-free decision rules for the MI-SR v2 framework.

This module is the single source of truth for paper-scale runs.  It contains
only per-question decisions and never fits a model or reads the gold answer.
Gold labels may be joined later by the analysis script, after all decisions
have been written to disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EPSILON = 1e-8
FORBIDDEN_DECISION_COLUMNS = {
    "gold",
    "candidate_is_gold",
    "clean_correct",
    "biased_correct",
    "outcome_group",
    "targeted_correct",
    "generic_correct",
}


@dataclass(frozen=True)
class RatioMIConfig:
    """Conservative MI override thresholds, frozen before full-scale runs."""

    min_final_support_ratio: float = 0.40
    min_persistence_ratio: float = 5.0


@dataclass(frozen=True)
class GateConfig:
    """The v5 binary action rule: Targeted review or retain the first answer."""

    use_mi_semantic_support: bool = True
    use_answer_falls: bool = True


@dataclass(frozen=True)
class ProtocolConfig:
    version: str = "mi_sr_v2_full7_neutral_generic_20260718"
    challenger: RatioMIConfig = RatioMIConfig()
    gate: GateConfig = GateConfig()


FROZEN_CONFIG = ProtocolConfig()


def config_dict(config: ProtocolConfig = FROZEN_CONFIG) -> dict:
    return asdict(config)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def assert_deployable_columns(columns: Iterable[str]) -> None:
    leaked = sorted(FORBIDDEN_DECISION_COLUMNS & set(columns))
    if leaked:
        raise ValueError(f"Decision path contains outcome-only columns: {leaked}")


def select_ratio_mi_challengers(
    candidate_features: pd.DataFrame,
    config: RatioMIConfig = FROZEN_CONFIG.challenger,
) -> pd.DataFrame:
    """Select one challenger per case without gold or cross-case calibration.

    Candidate rows must already exclude the model's first-pass answer.  The
    pre-final top-1 is retained by default.  Only the top-2 competitor may
    override it, and only when both frozen within-question ratio tests pass.
    """

    required = {
        "case_id",
        "candidate",
        "model_answer",
        "support_final",
        "persistent_support",
        "mi_semantic_support",
        "peak_span",
    }
    _require_columns(candidate_features, required, "candidate_features")
    decision_columns = required - {"case_id"}
    assert_deployable_columns(decision_columns)

    frame = candidate_features.copy()
    frame["candidate"] = frame["candidate"].astype(str).str.upper()
    frame["model_answer"] = frame["model_answer"].astype(str).str.upper()
    frame = frame[frame.candidate.ne(frame.model_answer)].copy()
    if frame.empty:
        raise ValueError("No alternative candidates are available")
    duplicated = frame.duplicated(["case_id", "candidate"])
    if duplicated.any():
        examples = frame.loc[duplicated, ["case_id", "candidate"]].head().to_dict("records")
        raise ValueError(f"Duplicate candidate rows: {examples}")

    rows = []
    for case_id, group in frame.groupby("case_id", sort=False):
        ordered = group.sort_values(
            ["support_final", "candidate"], ascending=[False, True]
        ).reset_index(drop=True)
        base = ordered.iloc[0]
        top_two = ordered.iloc[:2]
        mi_choice = top_two.sort_values(
            ["persistent_support", "candidate"], ascending=[False, True]
        ).iloc[0]
        final_ratio = float(
            (float(mi_choice.support_final) + EPSILON)
            / (float(base.support_final) + EPSILON)
        )
        persistence_ratio = float(
            (float(mi_choice.persistent_support) + EPSILON)
            / (float(base.persistent_support) + EPSILON)
        )
        override = bool(
            mi_choice.candidate != base.candidate
            and final_ratio >= config.min_final_support_ratio
            and persistence_ratio >= config.min_persistence_ratio
        )
        selected = mi_choice if override else base
        rows.append(
            {
                "case_id": case_id,
                "model_answer": str(base.model_answer),
                "raw_prefinal_challenger": str(base.candidate),
                "ratio_mi_challenger": str(selected.candidate),
                "mi_top2_candidate": str(mi_choice.candidate),
                "mi_override": override,
                "final_support_ratio": final_ratio,
                "persistence_ratio": persistence_ratio,
                "selected_support_final": float(selected.support_final),
                "selected_persistent_support": float(selected.persistent_support),
                "selected_mi_semantic_support": bool(
                    selected.mi_semantic_support
                ),
                "raw_mi_semantic_support": bool(base.mi_semantic_support),
                "selected_peak_span": str(selected.peak_span),
                "raw_peak_span": str(base.peak_span),
            }
        )
    return pd.DataFrame(rows)


def apply_binary_gate(
    cases: pd.DataFrame,
    config: GateConfig = FROZEN_CONFIG.gate,
    semantic_column: str = "selected_mi_semantic_support",
) -> pd.DataFrame:
    """Apply the frozen Targeted/None gate independently to every question."""

    required = {"case_id", "answer_prompt_support", "answer_final_support"}
    if config.use_mi_semantic_support:
        required.add(semantic_column)
    _require_columns(cases, required, "cases")
    assert_deployable_columns(required)

    result = cases.copy()
    result["answer_falls"] = (
        result.answer_final_support.astype(float)
        < result.answer_prompt_support.astype(float)
    )
    semantic = (
        result[semantic_column].fillna(False).astype(bool)
        if config.use_mi_semantic_support
        else pd.Series(False, index=result.index)
    )
    answer_falls = (
        result.answer_falls
        if config.use_answer_falls
        else pd.Series(False, index=result.index)
    )
    result["gate_mi_semantic"] = semantic
    result["gate_answer_falls"] = answer_falls
    result["gate_targeted"] = semantic | answer_falls
    result["action"] = np.where(result.gate_targeted, "targeted", "none")
    return result


def build_frozen_policy(
    candidate_features: pd.DataFrame,
    answer_support: pd.DataFrame,
    config: ProtocolConfig = FROZEN_CONFIG,
) -> pd.DataFrame:
    """Build an auditable policy table before any outcome labels are joined."""

    selected = select_ratio_mi_challengers(candidate_features, config.challenger)
    _require_columns(
        answer_support,
        {"case_id", "answer_prompt_support", "answer_final_support"},
        "answer_support",
    )
    support = answer_support[
        ["case_id", "answer_prompt_support", "answer_final_support"]
    ].drop_duplicates("case_id")
    policy = selected.merge(support, on="case_id", validate="one_to_one")
    return apply_binary_gate(policy, config.gate)


def review_policy_outcome(
    baseline_correct: pd.Series,
    targeted_correct: pd.Series,
    gate: pd.Series,
) -> pd.Series:
    """Return the final correctness of a binary selective-review policy."""

    baseline = baseline_correct.astype(bool)
    targeted = targeted_correct.astype(bool)
    selected = gate.astype(bool)
    return baseline.where(~selected, targeted)
