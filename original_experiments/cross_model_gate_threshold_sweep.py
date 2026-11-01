"""Calibrate and evaluate the frozen MI-SR Gate threshold per model.

The Gate and Challenger weights remain frozen. Only one scalar action margin is
selected on a model-specific calibration split, then evaluated once on held-out
source questions. Cached counterfactual reviews make the sweep inference-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_model_runtime import MODEL_SPECS
from develop_three_way_gate_v2 import load_discovery
from optimize_qwen_challenger_gate_loop import gate_feature_sets
from qwen_gate_threshold_sweep import (
    THRESHOLDS,
    matched_threshold_ablation,
    oof_probabilities,
    paired_variant_tests,
    render_mi_ablation_report,
    sweep,
)


ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "output_challenger_gate_loop_v20260719/split_manifest.json"
NO_REVIEW_THRESHOLD = 1.01
VARIANTS = ("probe_only", "mi_only", "probe_plus_mi")
PROBABILITY_COLUMNS = {
    variant: [
        f"p_none_{variant}",
        f"p_targeted_{variant}",
        f"p_generic_{variant}",
    ]
    for variant in VARIANTS
}


def as_no_review_row(row: pd.Series) -> pd.Series:
    """Build the fail-closed operating point without touching held-out labels."""
    result = row.copy()
    result["threshold"] = NO_REVIEW_THRESHOLD
    result["review_n"] = 0
    result["review_rate"] = 0.0
    result["targeted_n"] = 0
    result["targeted_first_pass_wrong_n"] = 0
    result["targeted_first_pass_wrong_rate"] = np.nan
    result["generic_n"] = 0
    result["generic_first_pass_wrong_n"] = 0
    result["generic_first_pass_wrong_rate"] = np.nan
    result["none_n"] = int(row.n_cases)
    result["none_first_pass_wrong_n"] = int(row.first_pass_wrong_n)
    result["none_first_pass_wrong_rate"] = (
        row.first_pass_wrong_n / row.n_cases if row.n_cases else np.nan
    )
    result["final_correct_n"] = int(row.first_pass_correct_n)
    result["final_accuracy"] = row.first_pass_accuracy
    result["accuracy_gain_pp"] = 0.0
    result["corrected"] = 0
    result["damaged"] = 0
    result["net_corrected"] = 0
    result["net_per_100_reviews"] = 0.0
    return result


def select_calibrated_variants(
    summary: pd.DataFrame,
    calibration_scope: str,
) -> tuple[pd.DataFrame, dict]:
    """Select one scalar threshold per branch without touching held-out data."""
    rows = []
    rules = {}
    for variant in VARIANTS:
        calibration = summary[
            summary.scope.eq(calibration_scope)
            & summary.gate_variant.eq(variant)
        ].copy()
        eligible = calibration[
            calibration.risk_ordered
            & calibration.targeted_first_pass_wrong_rate.ge(0.58)
        ]
        rule = "risk_ordered_and_targeted_wrong_rate_ge_0.58"
        if eligible.empty:
            eligible = calibration[calibration.risk_ordered].copy()
            rule = "fallback_risk_ordered"
        if eligible.empty:
            selected = as_no_review_row(calibration.iloc[0])
            rule = "fallback_no_review_no_valid_risk_order"
        else:
            selected = eligible.sort_values(
                ["final_accuracy", "net_per_100_reviews", "review_rate"],
                ascending=[False, False, True],
            ).iloc[0]
        unseen_candidates = summary[
            summary.scope.eq("primary_unseen")
            & summary.gate_variant.eq(variant)
            & summary.threshold.eq(selected.threshold)
        ]
        if selected.threshold == NO_REVIEW_THRESHOLD:
            unseen = as_no_review_row(
                summary[
                    summary.scope.eq("primary_unseen")
                    & summary.gate_variant.eq(variant)
                ].iloc[0]
            )
        else:
            unseen = unseen_candidates.iloc[0]
        rules[variant] = rule
        rows.append({
            "gate_variant": variant,
            "selection_rule": rule,
            "selected_threshold": float(selected.threshold),
            "calibration_final_accuracy": selected.final_accuracy,
            "calibration_review_rate": selected.review_rate,
            "calibration_targeted_wrong_rate": (
                selected.targeted_first_pass_wrong_rate
            ),
            "unseen_final_accuracy": unseen.final_accuracy,
            "unseen_accuracy_gain_pp": unseen.accuracy_gain_pp,
            "unseen_review_n": int(unseen.review_n),
            "unseen_review_rate": unseen.review_rate,
            "unseen_targeted_n": int(unseen.targeted_n),
            "unseen_targeted_wrong_rate": (
                unseen.targeted_first_pass_wrong_rate
            ),
            "unseen_generic_n": int(unseen.generic_n),
            "unseen_generic_wrong_rate": unseen.generic_first_pass_wrong_rate,
            "unseen_none_n": int(unseen.none_n),
            "unseen_none_wrong_rate": unseen.none_first_pass_wrong_rate,
            "unseen_corrected": int(unseen.corrected),
            "unseen_damaged": int(unseen.damaged),
            "unseen_net_corrected": int(unseen.net_corrected),
            "unseen_net_per_100_reviews": unseen.net_per_100_reviews,
        })
    return pd.DataFrame(rows), rules


def validate(
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    cases: pd.DataFrame,
    calibration_scope: str,
) -> dict:
    failures = []
    expected_scopes = {"all_20pct", "primary_unseen", calibration_scope}
    if set(summary.scope) != expected_scopes:
        failures.append(
            f"scope mismatch: {sorted(set(summary.scope))} != "
            f"{sorted(expected_scopes)}"
        )
    expected_rows = len(expected_scopes) * len(VARIANTS) * len(THRESHOLDS)
    if len(summary) != expected_rows:
        failures.append(f"summary rows {len(summary)} != {expected_rows}")
    if len(strata) != expected_rows * 3:
        failures.append(f"strata rows {len(strata)} != {expected_rows * 3}")
    for row in summary.itertuples(index=False):
        if row.targeted_n + row.generic_n + row.none_n != row.n_cases:
            failures.append(
                f"action count mismatch {row.scope}/{row.gate_variant}/"
                f"{row.threshold}"
            )
        if row.final_correct_n != row.first_pass_correct_n + row.net_corrected:
            failures.append(
                f"net identity mismatch {row.scope}/{row.gate_variant}/"
                f"{row.threshold}"
            )
    for (scope, variant), group in summary.groupby(["scope", "gate_variant"]):
        if not group.sort_values("threshold").review_n.diff().fillna(0).le(0).all():
            failures.append(f"non-monotone reviews {scope}/{variant}")
    official = summary[
        summary.scope.eq("all_20pct")
        & summary.gate_variant.eq("probe_plus_mi")
        & summary.threshold.eq(0.08)
    ].iloc[0]
    expected_actions = cases.action.value_counts().to_dict()
    reproduced = {
        "targeted": int(official.targeted_n),
        "generic": int(official.generic_n),
        "none": int(official.none_n),
    }
    if reproduced != expected_actions:
        failures.append(
            f"frozen 0.08 policy mismatch {reproduced} != {expected_actions}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "passed": True,
        "summary_rows": len(summary),
        "strata_rows": len(strata),
        "threshold_count": len(THRESHOLDS),
        "official_margin_008_action_counts_reproduced": reproduced,
    }


def format_rate(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.2%}"


def render_threshold_report(
    model_name: str,
    calibration_scope: str,
    calibration_source: str,
    calibrated: pd.DataFrame,
    full_unseen: pd.DataFrame,
) -> str:
    full = calibrated.set_index("gate_variant").loc["probe_plus_mi"]
    posthoc = full_unseen.sort_values(
        ["final_accuracy", "net_per_100_reviews", "review_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    lines = [
        "| 阈值 | T/G/N数量 | T/G/N错题率 | 复核率 | 最终准确率 | "
        "纠正/改错/净纠正 | 每100次复核净纠正 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in full_unseen.sort_values("threshold").itertuples(index=False):
        lines.append(
            f"| {row.threshold:.2f} | {row.targeted_n}/{row.generic_n}/"
            f"{row.none_n} | {format_rate(row.targeted_first_pass_wrong_rate)}/"
            f"{format_rate(row.generic_first_pass_wrong_rate)}/"
            f"{format_rate(row.none_first_pass_wrong_rate)} | "
            f"{row.review_rate:.2%} | {row.final_accuracy:.2%} | "
            f"{row.corrected}/{row.damaged}/{row.net_corrected} | "
            f"{row.net_per_100_reviews:.2f} |"
        )
    return f"""# {model_name} 20% Gate阈值全指标汇总

## 固定协议

- 扫描阈值：0.00至0.20，步长0.01。
- Challenger与三路Gate权重完全冻结，只允许校准一个绝对动作收益阈值。
- 校准范围：`{calibration_scope}`；概率来源：`{calibration_source}`。
- 每个特征分支分别在校准集选阈值，随后冻结并在160道未见源题上评估。
- 未见集事后最佳阈值只报告灵敏度上界，不用于主结果。

## Probe+MI校准结果

- 校准阈值：**{full.selected_threshold:.2f}**
- 未见集首答准确率：**{full.unseen_final_accuracy - full.unseen_accuracy_gain_pp / 100:.2%}**
- 未见集最终准确率：**{full.unseen_final_accuracy:.2%}**
- 准确率提升：**{full.unseen_accuracy_gain_pp:.2f}个百分点**
- 复核率：**{full.unseen_review_rate:.2%}**
- 净纠正：**{full.unseen_net_corrected}例**

## 独立未见集全部阈值

{chr(10).join(lines)}

## 事后灵敏度上界

直接在未见集挑选的最高准确率阈值为{posthoc.threshold:.2f}，准确率{posthoc.final_accuracy:.2%}。该值不能替代校准后冻结的主结果。
"""


def analyze_thresholds(
    run_dir: Path,
    model_key: str,
    output_dir: Path | None = None,
) -> dict:
    run_dir = Path(run_dir).resolve()
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key: {model_key}")
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else run_dir / "analysis_v3/gate_threshold_sweep"
    )
    output.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(run_dir / "analysis_v3/case_level_results_v3.csv")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    calibration_q = set(split["discovery_q_idx"])
    calibration_cases = cases[cases.q_idx.isin(calibration_q)].copy()
    unseen = cases[~cases.q_idx.isin(calibration_q)].copy()
    if calibration_cases.q_idx.nunique() != 95 or unseen.q_idx.nunique() != 160:
        raise RuntimeError(
            "Expected 95 calibration and 160 unseen source questions, found "
            f"{calibration_cases.q_idx.nunique()} and {unseen.q_idx.nunique()}"
        )

    summaries = []
    strata_rows = []
    for scope, frame in (("all_20pct", cases), ("primary_unseen", unseen)):
        for variant, columns in PROBABILITY_COLUMNS.items():
            current_summary, current_strata = sweep(
                frame, frame[columns].to_numpy(float), scope, variant
            )
            summaries.append(current_summary)
            strata_rows.append(current_strata)

    if model_key == "qwen3_14b_reference":
        calibration_scope = "discovery_grouped_oof"
        calibration_source = "Qwen Discovery grouped-OOF Gate probabilities"
        discovery, _, _ = load_discovery()
        discovery = discovery.rename(columns={"learned_challenger": "challenger"})
        feature_sets = gate_feature_sets(discovery)
        for variant in VARIANTS:
            probabilities = oof_probabilities(discovery, feature_sets[variant])
            current_summary, current_strata = sweep(
                discovery, probabilities, calibration_scope, variant
            )
            summaries.append(current_summary)
            strata_rows.append(current_strata)
    else:
        calibration_scope = "model_calibration"
        calibration_source = (
            "Frozen Qwen-trained Gate probabilities on target-model calibration cases"
        )
        for variant, columns in PROBABILITY_COLUMNS.items():
            current_summary, current_strata = sweep(
                calibration_cases,
                calibration_cases[columns].to_numpy(float),
                calibration_scope,
                variant,
            )
            summaries.append(current_summary)
            strata_rows.append(current_strata)

    summary = pd.concat(summaries, ignore_index=True)
    strata = pd.concat(strata_rows, ignore_index=True)
    validation = validate(summary, strata, cases, calibration_scope)
    calibrated, selection_rules = select_calibrated_variants(
        summary, calibration_scope
    )
    matched = matched_threshold_ablation(summary)
    paired = paired_variant_tests(unseen, calibrated)
    display_name = MODEL_SPECS[model_key]["display_name"]

    summary.to_csv(output / "gate_threshold_sweep_full_metrics.csv", index=False)
    strata.to_csv(output / "gate_threshold_action_strata.csv", index=False)
    calibrated.to_csv(output / "gate_adaptive_variant_ablation.csv", index=False)
    matched.to_csv(output / "gate_threshold_mi_matched_ablation.csv", index=False)
    paired.to_csv(output / "gate_adaptive_variant_paired_tests.csv", index=False)
    for scope in ("all_20pct", "primary_unseen", calibration_scope):
        summary[
            summary.scope.eq(scope)
            & summary.gate_variant.eq("probe_plus_mi")
        ].to_csv(output / f"gate_threshold_{scope}.csv", index=False)

    full_unseen = summary[
        summary.scope.eq("primary_unseen")
        & summary.gate_variant.eq("probe_plus_mi")
    ].copy()
    threshold_report = render_threshold_report(
        display_name,
        calibration_scope,
        calibration_source,
        calibrated,
        full_unseen,
    )
    (output / f"{model_key}-20pct-Gate阈值全指标汇总.md").write_text(
        threshold_report, encoding="utf-8"
    )
    (output / f"{model_key}-Gate阈值与MI消融联合分析.md").write_text(
        render_mi_ablation_report(
            calibrated,
            matched,
            paired,
            model_name=display_name,
            calibration_label=calibration_scope,
        ),
        encoding="utf-8",
    )
    full = calibrated.set_index("gate_variant").loc["probe_plus_mi"]
    adaptive = {
        "model_key": model_key,
        "model_name": display_name,
        "threshold_range": [float(THRESHOLDS.min()), float(THRESHOLDS.max())],
        "threshold_step": 0.01,
        "calibration_scope": calibration_scope,
        "calibration_probability_source": calibration_source,
        "selection_rules": selection_rules,
        "selected_threshold": float(full.selected_threshold),
        "probe_plus_mi_primary_unseen": full.to_dict(),
        "posthoc_unseen_best_is_sensitivity_only": True,
    }
    (output / "adaptive_threshold_selection.json").write_text(
        json.dumps(adaptive, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation.update({
        "model_key": model_key,
        "calibration_scope": calibration_scope,
        "calibration_source": calibration_source,
        "selected_probe_plus_mi_threshold": float(full.selected_threshold),
    })
    (output / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return adaptive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or (
        ROOT / "output_mi_sr_20pct_v3" / args.model_key / "all_biases"
    )
    print(json.dumps(
        analyze_thresholds(run_dir, args.model_key, args.output_dir),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
