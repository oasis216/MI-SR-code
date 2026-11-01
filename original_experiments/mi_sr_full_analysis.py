"""Reproducible paper tables for one frozen MI-SR v2 full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from mi_sr_protocol import review_policy_outcome


SEED = 20260718
N_CLUSTER_RANDOMIZATIONS = 20_000


def _review_outcome(
    reviews: pd.DataFrame,
    labels: pd.DataFrame,
    answer_column: str = "reeval_answer",
    complete_column: str = "reeval_complete",
) -> pd.Series:
    gold = labels.set_index("case_id").gold.astype(str)
    baseline = labels.set_index("case_id").biased_correct.astype(bool)
    frame = reviews.drop_duplicates("case_id").set_index("case_id")
    complete = frame[complete_column].fillna(False).astype(bool)
    parsed = frame[answer_column].astype(str).eq(frame.index.map(gold))
    outcome = parsed.where(complete, frame.index.map(baseline)).astype(bool)
    return outcome.reindex(labels.case_id).set_axis(labels.index)


def _cluster_sign_flip_p(
    delta: np.ndarray,
    groups: pd.Series,
    repetitions: int = N_CLUSTER_RANDOMIZATIONS,
) -> float:
    frame = pd.DataFrame(
        {"delta": np.asarray(delta, dtype=int), "group": groups.to_numpy()}
    )
    cluster_delta = frame.groupby("group").delta.sum().to_numpy(dtype=int)
    observed = abs(int(cluster_delta.sum()))
    if observed == 0 or not len(cluster_delta):
        return 1.0
    rng = np.random.default_rng(SEED + len(cluster_delta))
    exceed = 0
    completed = 0
    batch_size = 1000
    while completed < repetitions:
        current = min(batch_size, repetitions - completed)
        signs = rng.choice((-1, 1), size=(current, len(cluster_delta)))
        simulated = np.abs(signs @ cluster_delta)
        exceed += int(np.sum(simulated >= observed))
        completed += current
    return float((exceed + 1) / (repetitions + 1))


def _paired(
    name: str,
    main: pd.Series,
    other: pd.Series,
    groups: pd.Series,
) -> dict:
    main = main.astype(bool)
    other = other.astype(bool)
    main_only = int((main & ~other).sum())
    other_only = int((~main & other).sum())
    discordant = main_only + other_only
    return {
        "comparison": name,
        "main_only_correct": main_only,
        "comparator_only_correct": other_only,
        "paired_exact_p": (
            float(binomtest(main_only, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
        "n_question_clusters": int(groups.nunique()),
        "cluster_sign_flip_p": _cluster_sign_flip_p(
            main.astype(int).to_numpy() - other.astype(int).to_numpy(),
            groups,
        ),
    }


def _policy_row(
    method: str,
    baseline: pd.Series,
    result: pd.Series,
    gate: pd.Series,
) -> dict:
    baseline = baseline.astype(bool)
    result = result.astype(bool)
    gate = gate.astype(bool)
    corrected = int((~baseline & result).sum())
    damaged = int((baseline & ~result).sum())
    reviews = int(gate.sum())
    return {
        "method": method,
        "n_cases": len(result),
        "reviews": reviews,
        "review_rate": float(gate.mean()),
        "accuracy": float(result.mean()),
        "corrected": corrected,
        "damaged": damaged,
        "net_corrected": corrected - damaged,
        "net_per_100_reviews": 100.0 * (corrected - damaged) / max(reviews, 1),
    }


def _same_budget_random(
    baseline: pd.Series,
    targeted: pd.Series,
    observed_gate: pd.Series,
    strata: pd.Series,
    observed_accuracy: float,
    repetitions: int = 100_000,
) -> dict:
    base = baseline.to_numpy(bool)
    treatment = targeted.to_numpy(bool)
    gate = observed_gate.to_numpy(bool)
    strata = strata.astype(str).to_numpy()
    budget = int(gate.sum())
    gains = treatment.astype(int) - base.astype(int)
    rng = np.random.default_rng(SEED)
    values = np.empty(repetitions, dtype=float)
    stratum_jobs = []
    for stratum in sorted(set(strata)):
        indices = np.flatnonzero(strata == stratum)
        stratum_jobs.append((indices, int(gate[indices].sum())))
    for index in range(repetitions):
        gain = 0
        for indices, stratum_budget in stratum_jobs:
            if stratum_budget:
                selected = rng.choice(indices, stratum_budget, replace=False)
                gain += int(gains[selected].sum())
        values[index] = (int(base.sum()) + gain) / len(base)
    return {
        "method": "same_budget_random_targeted",
        "n_cases": len(base),
        "reviews": budget,
        "review_rate": budget / len(base),
        "accuracy": float(values.mean()),
        "accuracy_q05": float(np.quantile(values, 0.05)),
        "accuracy_q95": float(np.quantile(values, 0.95)),
        "monte_carlo_p": float(
            (1 + np.sum(values >= observed_accuracy)) / (repetitions + 1)
        ),
        "randomization_preserves_bias_counts": True,
    }


def analyze(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    output = run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(run_dir / "first_pass_evaluation_labels.csv")
    clean = pd.read_csv(run_dir / "clean_reference_outputs.csv")
    clean_columns = [
        "q_idx",
        "clean_answer",
        "clean_correct",
        "clean_complete",
    ]
    labels = labels.drop(
        columns=[column for column in clean_columns[1:] if column in labels],
        errors="ignore",
    ).merge(clean[clean_columns], on="q_idx", how="left", validate="many_to_one")
    clean_correct = labels.clean_correct.fillna(False).astype(bool)
    clean_complete = labels.clean_complete.fillna(False).astype(bool)
    biased_correct = labels.biased_correct.fillna(False).astype(bool)
    biased_complete = labels.biased_complete.fillna(False).astype(bool)
    labels["outcome_group"] = np.select(
        [
            clean_complete & biased_complete & clean_correct & biased_correct,
            clean_complete & biased_complete & clean_correct & ~biased_correct,
            clean_complete & biased_complete & ~clean_correct & ~biased_correct,
            clean_complete & biased_complete & ~clean_correct & biased_correct,
        ],
        ["Robust", "Harmed", "BaselineWrong", "Recovered"],
        default="Incomplete",
    )
    labels = labels[
        labels.biased_complete.fillna(False).astype(bool)
        & labels.biased_answer.fillna("").astype(str).str.fullmatch(r"[A-E]")
    ].reset_index(drop=True)
    policy = pd.read_csv(run_dir / "frozen_mi_sr_policy.csv").set_index("case_id")
    labels = labels[labels.case_id.isin(policy.index)].reset_index(drop=True)
    policy = policy.reindex(labels.case_id).set_axis(labels.index)
    targeted_reviews = pd.read_csv(run_dir / "targeted_all_reevaluation.csv")
    generic_reviews = pd.read_csv(run_dir / "generic_all_reevaluation.csv")

    baseline = labels.biased_correct.astype(bool)
    targeted = _review_outcome(targeted_reviews, labels)
    generic = _review_outcome(generic_reviews, labels)
    raw_targeted = targeted.copy()
    delta_path = run_dir / "raw_prefinal_delta_reevaluation.csv"
    if delta_path.exists():
        delta = pd.read_csv(delta_path)
        if len(delta):
            delta_outcome = _review_outcome(
                delta,
                labels,
                answer_column="reeval_answer",
                complete_column="reeval_complete",
            )
            changed = policy.mi_override.fillna(False).astype(bool)
            raw_targeted.loc[changed] = np.asarray(
                [
                    bool(value) if pd.notna(value) else False
                    for value in delta_outcome.loc[changed]
                ],
                dtype=bool,
            )

    full_gate = policy.gate_targeted.fillna(False).astype(bool)
    answer_gate = policy.answer_falls.fillna(False).astype(bool)
    semantic_gate = policy.selected_mi_semantic_support.fillna(False).astype(bool)
    raw_gate = (
        policy.raw_mi_semantic_support.fillna(False).astype(bool) | answer_gate
    )
    full_result = review_policy_outcome(baseline, targeted, full_gate)
    raw_result = review_policy_outcome(baseline, raw_targeted, raw_gate)
    answer_only_result = review_policy_outcome(baseline, targeted, answer_gate)
    semantic_only_result = review_policy_outcome(baseline, targeted, semantic_gate)

    main_rows = [
        _policy_row("biased_first_pass", baseline, baseline, pd.Series(False, index=labels.index)),
        _policy_row("all_generic", baseline, generic, pd.Series(True, index=labels.index)),
        _policy_row("all_targeted", baseline, targeted, pd.Series(True, index=labels.index)),
        _policy_row("MI-SR", baseline, full_result, full_gate),
    ]
    random_control = _same_budget_random(
        baseline,
        targeted,
        full_gate,
        labels.bias_type,
        observed_accuracy=float(full_result.mean()),
    )
    main = pd.DataFrame([main_rows[0], main_rows[1], main_rows[2], random_control, main_rows[3]])
    main.to_csv(output / "main_policy_comparison.csv", index=False)

    ablation = pd.DataFrame(
        [
            _policy_row("MI-SR", baseline, full_result, full_gate),
            _policy_row("w/o_challenger_MI", baseline, raw_result, raw_gate),
            _policy_row("w/o_gate_MI", baseline, answer_only_result, answer_gate),
            _policy_row("MI_semantic_only", baseline, semantic_only_result, semantic_gate),
        ]
    )
    ablation.to_csv(output / "minimal_module_ablation.csv", index=False)

    joined = labels[["case_id", "q_idx", "bias_type", "gold", "outcome_group"]].copy()
    joined = pd.concat([joined, policy.reset_index(drop=True)], axis=1)
    joined["baseline_correct"] = baseline
    joined["targeted_correct"] = targeted
    joined["generic_correct"] = generic
    joined["mi_sr_correct"] = full_result
    joined["challenger_is_gold"] = joined.ratio_mi_challenger.eq(joined.gold)
    joined["raw_challenger_is_gold"] = joined.raw_prefinal_challenger.eq(joined.gold)
    joined.to_csv(output / "case_level_results.csv", index=False, encoding="utf-8-sig")

    wrong = joined[~joined.baseline_correct].copy()
    localization = pd.DataFrame(
        [
            {
                "method": "raw_prefinal",
                "n_initial_wrong": len(wrong),
                "challenger_gold": int(wrong.raw_challenger_is_gold.sum()),
                "challenger_gold_rate": float(wrong.raw_challenger_is_gold.mean()),
            },
            {
                "method": "Ratio-MI",
                "n_initial_wrong": len(wrong),
                "challenger_gold": int(wrong.challenger_is_gold.sum()),
                "challenger_gold_rate": float(wrong.challenger_is_gold.mean()),
            },
        ]
    )
    localization.to_csv(output / "challenger_localization.csv", index=False)

    gate_strata = (
        joined.groupby("action", as_index=False)
        .agg(
            n=("case_id", "size"),
            first_pass_accuracy=("baseline_correct", "mean"),
            challenger_gold_rate=("challenger_is_gold", "mean"),
            targeted_accuracy=("targeted_correct", "mean"),
            mi_sr_accuracy=("mi_sr_correct", "mean"),
        )
        .sort_values("action")
    )
    gate_strata.to_csv(output / "gate_risk_strata.csv", index=False)

    conditional_review = (
        wrong.groupby("challenger_is_gold", as_index=False)
        .agg(
            n_initial_wrong=("case_id", "size"),
            targeted_correct=("targeted_correct", "sum"),
            targeted_correction_rate=("targeted_correct", "mean"),
            generic_correction_rate=("generic_correct", "mean"),
        )
        .sort_values("challenger_is_gold")
    )
    conditional_review.to_csv(
        output / "review_success_by_challenger_gold.csv", index=False
    )

    outcome_groups = (
        joined.groupby("outcome_group", as_index=False)
        .agg(
            n=("case_id", "size"),
            review_rate=("gate_targeted", "mean"),
            challenger_gold_rate=("challenger_is_gold", "mean"),
            first_pass_accuracy=("baseline_correct", "mean"),
            all_targeted_accuracy=("targeted_correct", "mean"),
            mi_sr_accuracy=("mi_sr_correct", "mean"),
        )
        .sort_values("outcome_group")
    )
    outcome_groups.to_csv(output / "results_by_outcome_group.csv", index=False)

    by_bias = (
        joined.groupby("bias_type", as_index=False)
        .agg(
            n=("case_id", "size"),
            first_pass_accuracy=("baseline_correct", "mean"),
            review_rate=("gate_targeted", "mean"),
            challenger_gold_rate=("challenger_is_gold", "mean"),
            mi_sr_accuracy=("mi_sr_correct", "mean"),
        )
        .sort_values("bias_type")
    )
    by_bias.to_csv(output / "results_by_bias.csv", index=False)

    tests = pd.DataFrame(
        [
            _paired("MI-SR_vs_first_pass", full_result, baseline, labels.q_idx),
            _paired("MI-SR_vs_all_generic", full_result, generic, labels.q_idx),
            _paired("MI-SR_vs_all_targeted", full_result, targeted, labels.q_idx),
            _paired(
                "MI-SR_vs_without_challenger_MI",
                full_result,
                raw_result,
                labels.q_idx,
            ),
            _paired(
                "MI-SR_vs_without_gate_MI",
                full_result,
                answer_only_result,
                labels.q_idx,
            ),
            _paired(
                "Ratio-MI_vs_raw_localization",
                wrong.challenger_is_gold,
                wrong.raw_challenger_is_gold,
                wrong.q_idx,
            ),
        ]
    )
    tests.to_csv(output / "paired_tests.csv", index=False)

    summary = {
        "n_evaluable": len(labels),
        "n_source_questions": int(labels.q_idx.nunique()),
        "n_bias_types": int(labels.bias_type.nunique()),
        "clean_accuracy": float(clean.clean_correct.fillna(False).astype(bool).mean()),
        "biased_first_pass_accuracy": float(baseline.mean()),
        "harmed_cases": int(labels.outcome_group.eq("Harmed").sum()),
        "mi_sr_accuracy": float(full_result.mean()),
        "mi_sr_reviews": int(full_gate.sum()),
        "mi_sr_review_rate": float(full_gate.mean()),
        "mi_sr_net_corrected": int((~baseline & full_result).sum() - (baseline & ~full_result).sum()),
        "ratio_mi_overrides": int(policy.mi_override.fillna(False).astype(bool).sum()),
        "analysis_files": sorted(path.name for path in output.glob("*.csv")),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
