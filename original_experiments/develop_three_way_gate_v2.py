"""Freeze a three-way Gate from Qwen Discovery grouped-OOF predictions.

This module reuses the immutable split and features produced by
``optimize_qwen_challenger_gate_loop.py``.  It predicts the correctness of
retaining the first answer, Targeted review, and Generic review separately.
The frozen action is the review with higher predicted correctness only when it
exceeds the predicted correctness of None by an absolute 0.08 margin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_qwen_challenger_gate_loop import (
    OUTPUT,
    SOURCE,
    ModelSpec,
    build_model,
    gate_feature_sets,
    grouped_oof,
    load_candidate_features,
    load_case_features,
    split_manifest,
    write_json,
)


VERSION = "gate_v002_three_way"
FEATURE_SET = "probe_plus_mi"
MODEL_SPEC = ModelSpec("logistic_C0.01", "logistic", 0.01)
ACTION_MARGIN = 0.08
TARGETS = ("baseline_correct", "targeted_correct", "generic_correct")
ACTIONS = np.asarray(("none", "targeted", "generic"))


def choose_actions(probabilities: np.ndarray) -> np.ndarray:
    action_index = np.argmax(probabilities, axis=1)
    best_review = probabilities[:, 1:].max(axis=1)
    action_index = np.where(
        best_review >= probabilities[:, 0] + ACTION_MARGIN,
        action_index,
        0,
    )
    return action_index


def summarize(frame: pd.DataFrame, action_index: np.ndarray) -> pd.DataFrame:
    rows = []
    baseline = frame.baseline_correct.astype(bool).to_numpy()
    outcomes = np.column_stack(
        [
            baseline,
            frame.targeted_correct.astype(bool).to_numpy(),
            frame.generic_correct.astype(bool).to_numpy(),
        ]
    )
    final = outcomes[np.arange(len(frame)), action_index]
    for index, action in enumerate(ACTIONS):
        selected = action_index == index
        rows.append(
            {
                "action": action,
                "n": int(selected.sum()),
                "fraction": float(selected.mean()),
                "first_pass_wrong": int((~baseline[selected]).sum()),
                "first_pass_wrong_rate": float((~baseline[selected]).mean())
                if selected.any()
                else np.nan,
                "first_pass_correct_rate": float(baseline[selected].mean())
                if selected.any()
                else np.nan,
                "action_accuracy": float(final[selected].mean())
                if selected.any()
                else np.nan,
            }
        )
    rows.append(
        {
            "action": "ALL",
            "n": len(frame),
            "fraction": 1.0,
            "first_pass_wrong": int((~baseline).sum()),
            "first_pass_wrong_rate": float((~baseline).mean()),
            "first_pass_correct_rate": float(baseline.mean()),
            "action_accuracy": float(final.mean()),
        }
    )
    return pd.DataFrame(rows)


def load_discovery() -> tuple[pd.DataFrame, list[str], dict]:
    cases = pd.read_csv(SOURCE / "analysis/case_level_results.csv")
    split = split_manifest(cases)
    discovery_q = set(split["discovery_q_idx"])
    candidates = load_candidate_features(cases)
    selected = pd.read_csv(OUTPUT / "challenger_v001_discovery_oof.csv")
    frame = load_case_features(
        cases[cases.q_idx.isin(discovery_q)],
        candidates[candidates.q_idx.isin(discovery_q)],
        selected,
    ).reset_index(drop=True)
    columns = gate_feature_sets(frame)[FEATURE_SET]
    return frame, columns, split


def discovery() -> None:
    frame, columns, split = load_discovery()
    probabilities = np.column_stack(
        [
            grouped_oof(frame, columns, target, MODEL_SPEC)
            for target in TARGETS
        ]
    )
    action_index = choose_actions(probabilities)
    for index, target in enumerate(TARGETS):
        frame[f"oof_p_{target}"] = probabilities[:, index]
    frame["oof_action"] = ACTIONS[action_index]
    summary = summarize(frame, action_index)
    frame.to_csv(OUTPUT / "gate_v002_discovery_oof.csv", index=False)
    summary.to_csv(OUTPUT / "gate_v002_discovery_summary.csv", index=False)
    config = {
        "version": VERSION,
        "feature_set": FEATURE_SET,
        "feature_columns": columns,
        "model": {
            "name": MODEL_SPEC.name,
            "family": MODEL_SPEC.family,
            "parameter": MODEL_SPEC.parameter,
        },
        "targets": list(TARGETS),
        "action_rule": (
            "Choose the review action with the higher predicted correctness "
            "only when it exceeds predicted None correctness by >=0.08; "
            "otherwise choose None."
        ),
        "absolute_action_margin": ACTION_MARGIN,
        "split_hash": split["split_hash"],
        "development_summary": summary.to_dict("records"),
        "holdout_status": (
            "The old v1 holdout was already revealed; v2 requires a new-source "
            "confirmation cohort for confirmatory claims."
        ),
    }
    write_json(OUTPUT / "frozen_gate_v002.json", config)
    print(summary.to_string(index=False))


def exploratory_old_holdout() -> None:
    frame, columns, split = load_discovery()
    cases = pd.read_csv(SOURCE / "analysis/case_level_results.csv")
    holdout_q = set(split["holdout_q_idx"])
    candidates = load_candidate_features(cases)
    selected = pd.read_csv(OUTPUT / "holdout_once_case_results.csv").rename(
        columns={"wrong_probability": "score"}
    )
    # Rebuild the selected-candidate table from the already frozen Challenger.
    selected_candidates = pd.read_csv(
        OUTPUT / "holdout_once_challenger_results.csv"
    )
    all_rows = candidates[candidates.q_idx.isin(holdout_q)].copy()
    challenger_config = json.loads(
        (OUTPUT / "frozen_challenger_v001.json").read_text(encoding="utf-8")
    )
    challenger_model = build_model(
        ModelSpec(**{
            "name": challenger_config["model"]["name"],
            "family": challenger_config["model"]["family"],
            "parameter": challenger_config["model"]["parameter"],
        })
    )
    train = candidates[
        candidates.q_idx.isin(set(split["discovery_q_idx"]))
        & candidates.wrong.eq(1)
    ]
    challenger_columns = challenger_config["feature_columns"]
    challenger_model.fit(train[challenger_columns], train.target)
    scores = challenger_model.predict_proba(all_rows[challenger_columns])[:, 1]
    all_rows["winner_oof_score"] = scores
    ranked = all_rows.assign(score=scores).sort_values(
        ["case_id", "score", "candidate"], ascending=[True, False, True]
    )
    ranked["score_rank"] = ranked.groupby("case_id").cumcount() + 1
    top = ranked[ranked.score_rank.eq(1)][
        ["case_id", "q_idx", "candidate", "gold", "model_answer", "target", "score"]
    ].copy()
    second = ranked[ranked.score_rank.eq(2)][["case_id", "score"]].rename(
        columns={"score": "second_score"}
    )
    top = top.merge(second, on="case_id")
    top["score_margin"] = top.score - top.second_score
    selected_features = top.merge(
        all_rows.drop(columns=["gold", "target", "model_answer"]),
        on=["case_id", "q_idx", "candidate"],
        validate="one_to_one",
    )
    holdout = load_case_features(
        cases[cases.q_idx.isin(holdout_q)], all_rows, selected_features
    ).reset_index(drop=True)
    probabilities = []
    for target in TARGETS:
        model = build_model(MODEL_SPEC)
        model.fit(frame[columns], frame[target])
        probabilities.append(model.predict_proba(holdout[columns])[:, 1])
    probability_matrix = np.column_stack(probabilities)
    action_index = choose_actions(probability_matrix)
    summary = summarize(holdout, action_index)
    for index, target in enumerate(TARGETS):
        holdout[f"p_{target}"] = probability_matrix[:, index]
    holdout["action"] = ACTIONS[action_index]
    holdout.to_csv(OUTPUT / "gate_v002_old_holdout_exploratory.csv", index=False)
    summary.to_csv(
        OUTPUT / "gate_v002_old_holdout_exploratory_summary.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    discovery()
    exploratory_old_holdout()
