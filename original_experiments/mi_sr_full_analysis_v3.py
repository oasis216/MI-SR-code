"""Paper-oriented analysis for learned Challenger + three-way Gate v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from mi_sr_full_analysis import _paired, _review_outcome


SEED = 20260719
RANDOM_REPETITIONS = 20_000
ROOT = Path(__file__).resolve().parent
DISCOVERY_SPLIT = (
    ROOT / "output_challenger_gate_loop_v20260719" / "split_manifest.json"
)


def apply_actions(
    baseline: pd.Series,
    targeted: pd.Series,
    generic: pd.Series,
    actions: pd.Series,
) -> pd.Series:
    result = baseline.astype(bool).copy()
    targeted_mask = actions.astype(str).eq("targeted")
    generic_mask = actions.astype(str).eq("generic")
    result.loc[targeted_mask] = targeted.loc[targeted_mask].astype(bool)
    result.loc[generic_mask] = generic.loc[generic_mask].astype(bool)
    return result


def policy_row(
    method: str,
    baseline: pd.Series,
    result: pd.Series,
    actions: pd.Series,
) -> dict:
    baseline = baseline.astype(bool)
    result = result.astype(bool)
    reviewed = actions.astype(str).ne("none")
    corrected = int((~baseline & result).sum())
    damaged = int((baseline & ~result).sum())
    review_n = int(reviewed.sum())
    return {
        "method": method,
        "n_cases": len(result),
        "targeted_n": int(actions.astype(str).eq("targeted").sum()),
        "generic_n": int(actions.astype(str).eq("generic").sum()),
        "review_n": review_n,
        "review_rate": float(reviewed.mean()),
        "accuracy": float(result.mean()),
        "corrected": corrected,
        "damaged": damaged,
        "net_corrected": corrected - damaged,
        "net_per_100_reviews": (
            100.0 * (corrected - damaged) / max(review_n, 1)
        ),
    }


def same_budget_random_mixed(
    frame: pd.DataFrame,
    baseline: pd.Series,
    targeted: pd.Series,
    generic: pd.Series,
    actions: pd.Series,
) -> dict:
    base = baseline.astype(bool).to_numpy()
    targeted_values = targeted.astype(bool).to_numpy()
    generic_values = generic.astype(bool).to_numpy()
    action_values = actions.astype(str).to_numpy()
    strata = frame.bias_type.astype(str).to_numpy()
    jobs = []
    for bias_type in sorted(set(strata)):
        indices = np.flatnonzero(strata == bias_type)
        targeted_n = int(np.sum(action_values[indices] == "targeted"))
        generic_n = int(np.sum(action_values[indices] == "generic"))
        jobs.append((indices, targeted_n, generic_n))
    observed = apply_actions(
        pd.Series(base),
        pd.Series(targeted_values),
        pd.Series(generic_values),
        pd.Series(action_values),
    ).mean()
    rng = np.random.default_rng(SEED + len(frame))
    accuracies = np.empty(RANDOM_REPETITIONS, dtype=float)
    for repetition in range(RANDOM_REPETITIONS):
        result = base.copy()
        for indices, targeted_n, generic_n in jobs:
            shuffled = rng.permutation(indices)
            targeted_indices = shuffled[:targeted_n]
            generic_indices = shuffled[
                targeted_n : targeted_n + generic_n
            ]
            result[targeted_indices] = targeted_values[targeted_indices]
            result[generic_indices] = generic_values[generic_indices]
        accuracies[repetition] = result.mean()
    return {
        "method": "same_budget_random_mixed_review",
        "n_cases": len(frame),
        "targeted_n": int(np.sum(action_values == "targeted")),
        "generic_n": int(np.sum(action_values == "generic")),
        "review_n": int(np.sum(action_values != "none")),
        "review_rate": float(np.mean(action_values != "none")),
        "accuracy": float(accuracies.mean()),
        "accuracy_q05": float(np.quantile(accuracies, 0.05)),
        "accuracy_q95": float(np.quantile(accuracies, 0.95)),
        "observed_policy_accuracy": float(observed),
        "monte_carlo_p": float(
            (1 + np.sum(accuracies >= observed)) / (RANDOM_REPETITIONS + 1)
        ),
        "preserves_bias_specific_action_counts": True,
    }


def gate_strata(
    frame: pd.DataFrame,
    actions: pd.Series,
    result: pd.Series,
    label: str,
) -> pd.DataFrame:
    current = frame.copy()
    current["evaluated_action"] = actions.astype(str).to_numpy()
    current["evaluated_result"] = result.astype(bool).to_numpy()
    current["challenger_is_gold"] = current.challenger.eq(current.gold)
    rows = []
    for action in ("targeted", "generic", "none"):
        group = current[current.evaluated_action.eq(action)]
        baseline = group.baseline_correct.astype(bool)
        final = group.evaluated_result.astype(bool)
        wrong = group[~baseline]
        rows.append(
            {
                "method": label,
                "action": action,
                "n": len(group),
                "fraction": len(group) / max(len(current), 1),
                "first_pass_wrong": int((~baseline).sum()),
                "first_pass_wrong_rate": float((~baseline).mean())
                if len(group)
                else np.nan,
                "first_pass_correct_rate": float(baseline.mean())
                if len(group)
                else np.nan,
                "challenger_gold_rate_all": float(
                    group.challenger_is_gold.mean()
                )
                if len(group)
                else np.nan,
                "challenger_gold_rate_within_wrong": float(
                    wrong.challenger_is_gold.mean()
                )
                if len(wrong)
                else np.nan,
                "action_accuracy": float(final.mean()) if len(group) else np.nan,
                "corrected": int((~baseline & final).sum()),
                "damaged": int((baseline & ~final).sum()),
            }
        )
    return pd.DataFrame(rows)


def prepare_labels(run_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(run_dir / "first_pass_evaluation_labels.csv")
    clean = pd.read_csv(run_dir / "clean_reference_outputs.csv")
    labels = labels.drop(
        columns=["clean_answer", "clean_correct", "clean_complete"],
        errors="ignore",
    ).merge(
        clean[["q_idx", "clean_answer", "clean_correct", "clean_complete"]],
        on="q_idx",
        how="left",
        validate="many_to_one",
    )
    clean_correct = labels.clean_correct.fillna(False).astype(bool)
    biased_correct = labels.biased_correct.fillna(False).astype(bool)
    clean_complete = labels.clean_complete.fillna(False).astype(bool)
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
    return labels[
        labels.biased_complete.fillna(False).astype(bool)
        & labels.biased_answer.fillna("").astype(str).str.fullmatch(r"[A-E]")
    ].reset_index(drop=True)


def analyze_scope(
    frame: pd.DataFrame,
    scope_name: str,
    output: Path,
) -> dict:
    baseline = frame.baseline_correct.astype(bool)
    targeted = frame.targeted_correct.astype(bool)
    generic = frame.generic_correct.astype(bool)
    actions = frame.action.astype(str)
    full = apply_actions(baseline, targeted, generic, actions)
    all_targeted_actions = pd.Series("targeted", index=frame.index)
    all_generic_actions = pd.Series("generic", index=frame.index)
    none_actions = pd.Series("none", index=frame.index)

    main = pd.DataFrame(
        [
            policy_row("first_pass", baseline, baseline, none_actions),
            policy_row("all_generic", baseline, generic, all_generic_actions),
            policy_row("all_targeted", baseline, targeted, all_targeted_actions),
            policy_row("MI-SR-v3", baseline, full, actions),
        ]
    )
    random_control = same_budget_random_mixed(
        frame, baseline, targeted, generic, actions
    )
    main.to_csv(output / f"{scope_name}_main_policy_comparison.csv", index=False)
    pd.DataFrame([random_control]).to_csv(
        output / f"{scope_name}_same_budget_random.csv", index=False
    )

    ablation_rows = []
    strata = []
    policy_results = {"MI-SR-v3": full}
    for feature_set, method in (
        ("probe_only", "Gate Probe-only"),
        ("mi_only", "Gate MI-only"),
        ("probe_plus_mi", "Gate Probe+MI"),
    ):
        current_actions = frame[f"action_{feature_set}"].astype(str)
        current_result = apply_actions(
            baseline, targeted, generic, current_actions
        )
        policy_results[method] = current_result
        ablation_rows.append(
            policy_row(method, baseline, current_result, current_actions)
        )
        strata.append(
            gate_strata(frame, current_actions, current_result, method)
        )
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(
        output / f"{scope_name}_minimal_gate_ablation.csv", index=False
    )
    strata_frame = pd.concat(strata, ignore_index=True)
    strata_frame.to_csv(
        output / f"{scope_name}_gate_risk_strata.csv", index=False
    )

    wrong = frame[~baseline].copy()
    wrong["learned_hit"] = wrong.challenger.eq(wrong.gold)
    wrong["raw_hit"] = wrong.raw_prefinal_challenger.eq(wrong.gold)
    learned_hits = int(wrong.learned_hit.sum())
    localization = pd.DataFrame(
        [
            {
                "method": "raw_prefinal",
                "n_initial_wrong": len(wrong),
                "challenger_gold": int(wrong.raw_hit.sum()),
                "challenger_gold_rate": float(wrong.raw_hit.mean()),
                "one_sided_p_vs_25pct": float(
                    binomtest(
                        int(wrong.raw_hit.sum()),
                        len(wrong),
                        0.25,
                        alternative="greater",
                    ).pvalue
                ),
            },
            {
                "method": "learned_challenger_v001",
                "n_initial_wrong": len(wrong),
                "challenger_gold": learned_hits,
                "challenger_gold_rate": float(wrong.learned_hit.mean()),
                "one_sided_p_vs_25pct": float(
                    binomtest(
                        learned_hits,
                        len(wrong),
                        0.25,
                        alternative="greater",
                    ).pvalue
                ),
            },
        ]
    )
    localization.to_csv(
        output / f"{scope_name}_challenger_localization.csv", index=False
    )

    tests = [
        _paired("MI-SR-v3_vs_first_pass", full, baseline, frame.q_idx),
        _paired("MI-SR-v3_vs_all_generic", full, generic, frame.q_idx),
        _paired("MI-SR-v3_vs_all_targeted", full, targeted, frame.q_idx),
    ]
    for method, result in policy_results.items():
        if method == "MI-SR-v3":
            continue
        tests.append(
            _paired(f"MI-SR-v3_vs_{method}", full, result, frame.q_idx)
        )
    pd.DataFrame(tests).to_csv(
        output / f"{scope_name}_paired_tests.csv", index=False
    )

    full_strata = strata_frame[strata_frame.method.eq("Gate Probe+MI")].set_index(
        "action"
    )
    probe_strata = strata_frame[strata_frame.method.eq("Gate Probe-only")].set_index(
        "action"
    )
    targeted_wrong = float(full_strata.loc["targeted", "first_pass_wrong_rate"])
    generic_wrong = float(full_strata.loc["generic", "first_pass_wrong_rate"])
    none_wrong = float(full_strata.loc["none", "first_pass_wrong_rate"])
    probe_targeted_wrong = float(
        probe_strata.loc["targeted", "first_pass_wrong_rate"]
    )
    full_row = main[main.method.eq("MI-SR-v3")].iloc[0]
    probe_row = ablation[ablation.method.eq("Gate Probe-only")].iloc[0]
    assessment = {
        "scope": scope_name,
        "n_cases": len(frame),
        "n_source_questions": int(frame.q_idx.nunique()),
        "baseline_accuracy": float(baseline.mean()),
        "challenger_gold_rate_within_wrong": float(wrong.learned_hit.mean()),
        "challenger_p_vs_25pct": float(
            localization.loc[
                localization.method.eq("learned_challenger_v001"),
                "one_sided_p_vs_25pct",
            ].iloc[0]
        ),
        "targeted_wrong_rate": targeted_wrong,
        "generic_wrong_rate": generic_wrong,
        "none_wrong_rate": none_wrong,
        "probe_only_targeted_wrong_rate": probe_targeted_wrong,
        "mi_targeted_wrong_rate_gain": targeted_wrong - probe_targeted_wrong,
        "probe_only_policy_accuracy": float(probe_row.accuracy),
        "mi_policy_accuracy_gain": float(full_row.accuracy - probe_row.accuracy),
        "net_corrected": int(full_row.net_corrected),
        "policy_accuracy": float(full_row.accuracy),
        "checks": {
            "challenger_significantly_above_25pct": bool(
                wrong.learned_hit.mean() > 0.25
                and localization.loc[
                    localization.method.eq("learned_challenger_v001"),
                    "one_sided_p_vs_25pct",
                ].iloc[0]
                < 0.05
            ),
            "targeted_wrong_rate_at_least_58pct": targeted_wrong >= 0.58,
            "targeted_is_highest_risk": (
                targeted_wrong > generic_wrong and targeted_wrong > none_wrong
            ),
            "mi_improves_targeted_risk_by_2pp": (
                targeted_wrong - probe_targeted_wrong >= 0.02
            ),
            "end_to_end_net_nonnegative": int(full_row.net_corrected) >= 0,
        },
    }
    assessment["all_checks_pass"] = all(assessment["checks"].values())
    return assessment


def analyze(run_dir: Path, model_key: str) -> dict:
    run_dir = Path(run_dir).resolve()
    output = run_dir / "analysis_v3"
    output.mkdir(parents=True, exist_ok=True)
    labels = prepare_labels(run_dir)
    policy = pd.read_csv(run_dir / "frozen_mi_sr_policy_v3.csv")
    labels = labels[labels.case_id.isin(policy.case_id)].reset_index(drop=True)
    policy = policy.set_index("case_id").reindex(labels.case_id).reset_index()
    for review_type in ("targeted", "generic"):
        reviews = pd.read_csv(run_dir / f"{review_type}_all_reevaluation.csv")
        missing = set(labels.case_id.astype(str)) - set(reviews.case_id.astype(str))
        if missing:
            raise RuntimeError(
                f"{review_type} reviews incomplete: missing {len(missing)} cases"
            )
    targeted = _review_outcome(
        pd.read_csv(run_dir / "targeted_all_reevaluation.csv"), labels
    )
    generic = _review_outcome(
        pd.read_csv(run_dir / "generic_all_reevaluation.csv"), labels
    )
    # q_idx/bias_type/model_answer are already frozen in the pre-unblinding
    # policy. Keep label-only columns here to avoid pandas _x/_y metadata.
    frame = labels[["case_id", "gold", "outcome_group"]].copy()
    frame = frame.merge(policy, on="case_id", validate="one_to_one")
    frame["baseline_correct"] = labels.biased_correct.astype(bool).to_numpy()
    frame["targeted_correct"] = targeted.to_numpy(bool)
    frame["generic_correct"] = generic.to_numpy(bool)
    frame.to_csv(output / "case_level_results_v3.csv", index=False)

    scopes = {"all_20pct": frame}
    if model_key == "qwen3_14b_reference":
        split = json.loads(DISCOVERY_SPLIT.read_text(encoding="utf-8"))
        discovery_q = set(split["discovery_q_idx"])
        scopes["primary_unseen"] = frame[~frame.q_idx.isin(discovery_q)].copy()
    else:
        scopes["primary_unseen"] = frame.copy()
    assessments = {
        name: analyze_scope(current, name, output)
        for name, current in scopes.items()
    }
    summary = {
        "model_key": model_key,
        "primary_scope": "primary_unseen",
        "assessments": assessments,
        "proceed_to_next_model": bool(
            assessments["primary_unseen"]["all_checks_pass"]
        ),
    }
    (output / "result_assessment.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.run_dir, args.model_key), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
