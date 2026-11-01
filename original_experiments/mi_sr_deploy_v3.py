"""Deploy frozen Challenger v001 and Three-way Gate v002 without gold labels.

This module converts one completed MI-SR feature cache into a policy table.
It never reads correctness, gold answers, outcome groups, or review outputs.
The same Qwen-fitted artifacts are used unchanged for every target model.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mi_sr_peak_features import answer_support_table
from mi_sr_protocol import build_frozen_policy
from optimize_qwen_challenger_gate_loop import (
    EPSILON,
    LETTERS,
    curve_statistics,
    within_case_rank,
    within_case_z,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(
    os.environ.get("MI_SR_ARTIFACT_DIR", ROOT / "frozen_artifacts")
)
CHALLENGER_CONFIG_PATH = ARTIFACT_DIR / "frozen_challenger_v001.json"
CHALLENGER_MODEL_PATH = ARTIFACT_DIR / "frozen_challenger_v001.joblib"
GATE_CONFIG_PATH = ARTIFACT_DIR / "frozen_gate_v002.json"
GATE_MODEL_PATH = ARTIFACT_DIR / "frozen_gate_v002_models.joblib"
ABLATION_CONFIG_PATH = ARTIFACT_DIR / "frozen_gate_v002_ablation_manifest.json"
ABLATION_MODEL_PATH = ARTIFACT_DIR / "frozen_gate_v002_ablation_models.joblib"
FORBIDDEN = {
    "gold",
    "baseline_correct",
    "targeted_correct",
    "generic_correct",
    "biased_correct",
    "clean_correct",
    "outcome_group",
    "challenger_is_gold",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _assert_deployable(columns, context: str) -> None:
    leaked = sorted(FORBIDDEN & set(columns))
    if leaked:
        raise RuntimeError(f"Outcome leakage in {context}: {leaked}")


def build_candidate_features(run_dir: Path) -> pd.DataFrame:
    """Recreate the exact frozen Challenger feature schema from raw caches."""

    run_dir = Path(run_dir).resolve()
    peaks = pd.read_csv(
        run_dir / "mi_sr_peak_features/candidate_peak_features.csv"
    )
    probes = pd.read_csv(run_dir / "checkpoint_probe_logits.csv")
    curves = _load_pickle(run_dir / "all_candidate_curve_store.pkl")
    probe_rows = []
    for row in probes.itertuples(index=False):
        for candidate in LETTERS:
            probe_rows.append(
                {
                    "case_id": str(row.case_id),
                    "candidate": candidate,
                    f"prob_{row.checkpoint}": float(
                        getattr(row, f"prob_{candidate}")
                    ),
                    f"logit_{row.checkpoint}": float(
                        getattr(row, f"logit_{candidate}")
                    ),
                    f"entropy_{row.checkpoint}": float(
                        row.candidate_entropy_normalized
                    ),
                }
            )
    probe_wide = (
        pd.DataFrame(probe_rows)
        .groupby(["case_id", "candidate"], as_index=False)
        .first()
    )
    frame = peaks.merge(
        probe_wide, on=["case_id", "candidate"], validate="one_to_one"
    )
    frame = frame[frame.candidate.ne(frame.model_answer)].copy()

    curve_rows = []
    for row in frame[["case_id", "candidate", "model_answer"]].itertuples(
        index=False
    ):
        candidate_curve = np.asarray(
            curves[row.case_id]["curves"][row.candidate], dtype=float
        )
        answer_curve = np.asarray(
            curves[row.case_id]["curves"][row.model_answer], dtype=float
        )
        length = min(len(candidate_curve), len(answer_curve))
        candidate_curve = candidate_curve[:length]
        answer_curve = answer_curve[:length]
        item = {"case_id": row.case_id, "candidate": row.candidate}
        item.update(curve_statistics(candidate_curve, "mi_candidate"))
        item.update(curve_statistics(answer_curve, "mi_answer"))
        item.update(
            curve_statistics(
                candidate_curve - answer_curve, "mi_candidate_minus_answer"
            )
        )
        curve_rows.append(item)
    frame = frame.merge(
        pd.DataFrame(curve_rows),
        on=["case_id", "candidate"],
        validate="one_to_one",
    )

    for checkpoint in ("prompt_end", "pre_final"):
        probability = frame[f"prob_{checkpoint}"].clip(EPSILON, 1.0)
        frame[f"log_prob_{checkpoint}"] = np.log(probability)
        frame[f"z_log_prob_{checkpoint}"] = within_case_z(
            frame, f"log_prob_{checkpoint}"
        )
        frame[f"rank_prob_{checkpoint}"] = within_case_rank(
            frame, f"prob_{checkpoint}"
        )
    frame["probe_log_growth"] = (
        frame.log_prob_pre_final - frame.log_prob_prompt_end
    )
    frame["probe_probability_growth"] = (
        frame.prob_pre_final - frame.prob_prompt_end
    )
    for column in (
        "support_before",
        "support_after",
        "support_next",
        "support_final",
        "persistent_support",
        "confidence_leap",
        "peak_prominence",
        "peak_position",
    ):
        frame[f"z_{column}"] = within_case_z(frame, column)
        frame[f"rank_{column}"] = within_case_rank(frame, column)
    for column in (
        "mi_semantic_support",
        "peak_positive",
        "peak_negative",
        "peak_reasoning",
    ):
        frame[column] = frame[column].fillna(False).astype(bool).astype(int)
    counts = frame.groupby("case_id").size()
    if not counts.eq(4).all():
        raise RuntimeError(
            f"Expected four alternatives per case: {counts.value_counts().to_dict()}"
        )
    return frame.reset_index(drop=True)


def select_challengers(
    candidate_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _read_json(CHALLENGER_CONFIG_PATH)
    columns = list(config["feature_columns"])
    _assert_deployable(columns, "Challenger")
    missing = sorted(set(columns) - set(candidate_features.columns))
    if missing:
        raise RuntimeError(f"Missing Challenger features: {missing}")
    model = joblib.load(CHALLENGER_MODEL_PATH)
    frame = candidate_features.copy()
    frame["challenger_score"] = model.predict_proba(frame[columns])[:, 1]
    ranked = frame.sort_values(
        ["case_id", "challenger_score", "candidate"],
        ascending=[True, False, True],
    ).copy()
    ranked["challenger_rank"] = ranked.groupby("case_id").cumcount() + 1
    top = ranked[ranked.challenger_rank.eq(1)].copy()
    second = ranked[ranked.challenger_rank.eq(2)][
        ["case_id", "challenger_score"]
    ].rename(columns={"challenger_score": "challenger_second_score"})
    top = top.merge(second, on="case_id", validate="one_to_one")
    top["challenger_score_margin"] = (
        top.challenger_score - top.challenger_second_score
    )
    raw = (
        frame.sort_values(
            ["case_id", "prob_pre_final", "candidate"],
            ascending=[True, False, True],
        )
        .drop_duplicates("case_id")
        [["case_id", "candidate"]]
        .rename(columns={"candidate": "raw_prefinal_challenger"})
    )
    selection = top[
        [
            "case_id",
            "q_idx",
            "bias_type",
            "model_answer",
            "candidate",
            "challenger_score",
            "challenger_second_score",
            "challenger_score_margin",
        ]
    ].rename(columns={"candidate": "challenger"})
    selection = selection.merge(raw, on="case_id", validate="one_to_one")
    return selection, top


def _case_probe_features(
    probes: pd.DataFrame,
    eligible_case_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate checkpoint probes for cases with deployable MI features.

    A model can finish generation without producing the mandatory final-answer
    line. Such cases remain in the raw checkpoint cache for completion-rate
    accounting, but they have no candidate curves or peak features and cannot
    enter the frozen MI-SR policy. Filter to the feature-complete case set
    before indexing the answer letter.
    """

    probes = probes.copy()
    probes["case_id"] = probes.case_id.astype(str)
    if eligible_case_ids is not None:
        eligible = {str(case_id) for case_id in eligible_case_ids}
        probes = probes[probes.case_id.isin(eligible)].copy()
    probes["model_answer"] = (
        probes.model_answer.fillna("").astype(str).str.strip().str.upper()
    )
    invalid = probes[~probes.model_answer.isin(LETTERS)]
    if not invalid.empty:
        examples = invalid[["case_id", "model_answer"]].drop_duplicates().head(5)
        raise RuntimeError(
            "Invalid model answers remain after deployable-case filtering: "
            f"{examples.to_dict('records')}"
        )
    rows = []
    for row in probes.itertuples(index=False):
        probabilities = np.asarray(
            [getattr(row, f"prob_{letter}") for letter in LETTERS], dtype=float
        )
        answer_index = LETTERS.index(str(row.model_answer))
        alternatives = np.delete(probabilities, answer_index)
        ordered = np.sort(alternatives)
        rows.append(
            {
                "case_id": row.case_id,
                "checkpoint": row.checkpoint,
                "answer_probability": probabilities[answer_index],
                "best_alternative_probability": alternatives.max(),
                "second_alternative_probability": ordered[-2],
                "alternative_gap": ordered[-1] - ordered[-2],
                "candidate_entropy": row.candidate_entropy_normalized,
                "answer_rank": 1
                + int(np.sum(probabilities > probabilities[answer_index])),
                "best_alternative_minus_answer": (
                    alternatives.max() - probabilities[answer_index]
                ),
            }
        )
    result = pd.DataFrame(rows).pivot(index="case_id", columns="checkpoint")
    result.columns = [
        f"{name}_{checkpoint}" for name, checkpoint in result.columns
    ]
    result = result.reset_index()
    required = {
        f"answer_probability_{checkpoint}"
        for checkpoint in ("prompt_end", "pre_final")
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise RuntimeError(f"Missing checkpoint probe features: {missing}")
    if eligible_case_ids is not None:
        expected = {str(case_id) for case_id in eligible_case_ids}
        observed = set(result.case_id.astype(str))
        if observed != expected:
            raise RuntimeError(
                "Checkpoint/MI case mismatch after filtering: "
                f"missing={len(expected - observed)}, extra={len(observed - expected)}"
            )
    return result


def build_gate_features(
    run_dir: Path,
    candidates: pd.DataFrame,
    challenger_top: pd.DataFrame,
) -> pd.DataFrame:
    run_dir = Path(run_dir).resolve()
    peaks = pd.read_csv(
        run_dir / "mi_sr_peak_features/candidate_peak_features.csv"
    )
    probes = pd.read_csv(run_dir / "checkpoint_probe_logits.csv")
    old_policy = build_frozen_policy(peaks, answer_support_table(run_dir))
    metadata = peaks[
        ["case_id", "q_idx", "bias_type", "model_answer"]
    ].drop_duplicates("case_id")
    base = metadata.merge(
        old_policy,
        on=["case_id", "model_answer"],
        validate="one_to_one",
    )
    probe_features = _case_probe_features(
        probes,
        eligible_case_ids=set(base.case_id.astype(str)),
    )
    aggregate = candidates.groupby("case_id", as_index=False).agg(
        mi_semantic_count=("mi_semantic_support", "sum"),
        mi_positive_count=("peak_positive", "sum"),
        mi_negative_count=("peak_negative", "sum"),
        mi_reasoning_count=("peak_reasoning", "sum"),
        peak_prominence_max=("peak_prominence", "max"),
        peak_prominence_mean=("peak_prominence", "mean"),
        persistent_support_max=("persistent_support", "max"),
        persistent_support_mean=("persistent_support", "mean"),
        confidence_leap_max=("confidence_leap", "max"),
        confidence_leap_mean=("confidence_leap", "mean"),
        candidate_mi_mean_max=("mi_candidate_mean", "max"),
        candidate_mi_late_max=("mi_candidate_late_mean", "max"),
        pair_mi_late_max=("mi_candidate_minus_answer_late_mean", "max"),
        pair_mi_max=("mi_candidate_minus_answer_max", "max"),
    )
    selected_columns = [
        "case_id",
        "candidate",
        "challenger_score",
        "challenger_second_score",
        "challenger_score_margin",
        "support_prompt",
        "support_before",
        "support_after",
        "support_next",
        "support_final",
        "persistent_support",
        "confidence_leap",
        "peak_prominence",
        "peak_position",
        "mi_semantic_support",
        "peak_positive",
        "peak_negative",
        "peak_reasoning",
        "mi_candidate_mean",
        "mi_candidate_late_mean",
        "mi_candidate_peak_count",
        "mi_candidate_peak_prominence_max",
        "mi_candidate_minus_answer_mean",
        "mi_candidate_minus_answer_late_mean",
        "mi_candidate_minus_answer_max",
        "mi_candidate_minus_answer_peak_count",
        "mi_candidate_minus_answer_peak_prominence_max",
    ]
    selected = challenger_top[selected_columns].copy()
    selected["challenger_winner_oof_score"] = selected.challenger_score
    selected = selected.rename(
        columns={
            column: f"challenger_{column}"
            for column in selected.columns
            if column
            not in {
                "case_id",
                "candidate",
                "challenger_score",
                "challenger_second_score",
                "challenger_score_margin",
                "challenger_winner_oof_score",
            }
        }
    ).rename(columns={"candidate": "learned_challenger"})
    frame = (
        base.merge(probe_features, on="case_id", validate="one_to_one")
        .merge(aggregate, on="case_id", validate="one_to_one")
        .merge(selected, on="case_id", validate="one_to_one")
    )
    for name in (
        "answer_probability",
        "best_alternative_probability",
        "second_alternative_probability",
        "alternative_gap",
        "candidate_entropy",
        "answer_rank",
        "best_alternative_minus_answer",
    ):
        frame[f"{name}_change"] = (
            frame[f"{name}_pre_final"] - frame[f"{name}_prompt_end"]
        )
    return frame


def _predict_actions(
    frame: pd.DataFrame,
    models: dict,
    columns: list[str],
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    _assert_deployable(columns, "Gate")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing Gate features: {missing}")
    targets = ("baseline_correct", "targeted_correct", "generic_correct")
    probabilities = np.column_stack(
        [models[target].predict_proba(frame[columns])[:, 1] for target in targets]
    )
    action_index = np.argmax(probabilities, axis=1)
    best_review = probabilities[:, 1:].max(axis=1)
    action_index = np.where(
        best_review >= probabilities[:, 0] + margin, action_index, 0
    )
    return probabilities, np.asarray(("none", "targeted", "generic"))[action_index]


def build_deploy_policy(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and persist all decisions before joining any outcome labels."""

    run_dir = Path(run_dir).resolve()
    candidates = build_candidate_features(run_dir)
    challenger, challenger_top = select_challengers(candidates)
    gate_features = build_gate_features(run_dir, candidates, challenger_top)
    gate_config = _read_json(GATE_CONFIG_PATH)
    gate_models = joblib.load(GATE_MODEL_PATH)
    margin = float(gate_config["absolute_action_margin"])
    probabilities, actions = _predict_actions(
        gate_features,
        gate_models,
        list(gate_config["feature_columns"]),
        margin,
    )
    predictions = gate_features[["case_id"]].copy()
    predictions["p_none_correct"] = probabilities[:, 0]
    predictions["p_targeted_correct"] = probabilities[:, 1]
    predictions["p_generic_correct"] = probabilities[:, 2]
    predictions["action"] = actions
    policy = challenger.merge(predictions, on="case_id", validate="one_to_one")

    ablation_config = _read_json(ABLATION_CONFIG_PATH)
    ablation_models = joblib.load(ABLATION_MODEL_PATH)
    for feature_set in ("probe_only", "mi_only", "probe_plus_mi"):
        current_probabilities, current_actions = _predict_actions(
            gate_features,
            ablation_models[feature_set],
            list(ablation_config["feature_sets"][feature_set]),
            float(ablation_config["absolute_action_margin"]),
        )
        current = gate_features[["case_id"]].copy()
        current[f"action_{feature_set}"] = current_actions
        for index, target in enumerate(("none", "targeted", "generic")):
            current[f"p_{target}_{feature_set}"] = current_probabilities[:, index]
        policy = policy.merge(current, on="case_id", validate="one_to_one")

    if set(policy.action.unique()) - {"none", "targeted", "generic"}:
        raise RuntimeError(f"Invalid actions: {policy.action.value_counts().to_dict()}")
    policy.to_csv(
        run_dir / "frozen_mi_sr_policy_v3.csv", index=False, encoding="utf-8-sig"
    )
    policy[
        ["case_id", "model_answer", "challenger", "action"]
    ].to_csv(
        run_dir / "frozen_policy_scores_before_unblinding.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_features.to_pickle(run_dir / "deploy_gate_features_before_unblinding.pkl")
    plan_path = run_dir / "experiment_plan.csv"
    planned_case_ids = (
        set(pd.read_csv(plan_path).case_id.astype(str))
        if plan_path.exists()
        else set(policy.case_id.astype(str))
    )
    policy_case_ids = set(policy.case_id.astype(str))
    manifest = {
        "version": "mi_sr_v3_learned_challenger_three_way_gate",
        "n_cases": len(policy),
        "n_planned_cases": len(planned_case_ids),
        "n_evaluable_cases": len(policy_case_ids),
        "n_excluded_incomplete_or_feature_missing": len(
            planned_case_ids - policy_case_ids
        ),
        "deployment_coverage": len(policy_case_ids) / max(len(planned_case_ids), 1),
        "excluded_case_ids": sorted(planned_case_ids - policy_case_ids),
        "challenger_config": str(CHALLENGER_CONFIG_PATH),
        "gate_config": str(GATE_CONFIG_PATH),
        "gate_ablation_config": str(ABLATION_CONFIG_PATH),
        "gold_available_to_decision_path": False,
        "action_counts": {
            str(key): int(value) for key, value in policy.action.value_counts().items()
        },
    }
    (run_dir / "deploy_policy_manifest_v3.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return policy, gate_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result, _ = build_deploy_policy(args.run_dir)
    print(result.action.value_counts().to_string())
