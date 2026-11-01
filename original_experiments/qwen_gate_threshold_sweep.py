"""Complete Qwen Gate threshold sweep with grouped-OOF calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from develop_three_way_gate_v2 import MODEL_SPEC, TARGETS, load_discovery
from optimize_qwen_challenger_gate_loop import gate_feature_sets, grouped_oof


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_qwen_gate_threshold_sweep_v20260719"
CASE_RESULTS = (
    ROOT
    / "output_mi_sr_20pct_v3"
    / "qwen3_14b_reference"
    / "all_biases"
    / "analysis_v3"
    / "case_level_results_v3.csv"
)
SPLIT = (
    ROOT / "output_challenger_gate_loop_v20260719" / "split_manifest.json"
)
THRESHOLDS = np.round(np.arange(0.00, 0.201, 0.01), 2)
ACTIONS = np.asarray(["none", "targeted", "generic"])


def choose_actions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    best_index = np.argmax(probabilities, axis=1)
    best_review_gain = probabilities[:, 1:].max(axis=1) - probabilities[:, 0]
    best_index = np.where(best_review_gain >= threshold, best_index, 0)
    return ACTIONS[best_index]


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def sweep_row(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    scope: str,
    variant: str,
) -> tuple[dict, list[dict]]:
    actions = choose_actions(probabilities, threshold)
    baseline = frame.baseline_correct.astype(bool).to_numpy()
    targeted = frame.targeted_correct.astype(bool).to_numpy()
    generic = frame.generic_correct.astype(bool).to_numpy()
    outcomes = np.column_stack([baseline, targeted, generic])
    action_index = pd.Categorical(
        actions, categories=["none", "targeted", "generic"]
    ).codes
    final = outcomes[np.arange(len(frame)), action_index]
    corrected_mask = (~baseline) & final
    damaged_mask = baseline & (~final)
    reviewed = actions != "none"
    challenger_gold = frame.challenger.astype(str).eq(frame.gold.astype(str)).to_numpy()
    total_wrong = int((~baseline).sum())

    strata_rows = []
    wide = {}
    for action in ("targeted", "generic", "none"):
        selected = actions == action
        n = int(selected.sum())
        wrong_n = int(((~baseline) & selected).sum())
        correct_n = int((baseline & selected).sum())
        final_correct_n = int((final & selected).sum())
        corrected_n = int((corrected_mask & selected).sum())
        damaged_n = int((damaged_mask & selected).sum())
        challenger_gold_n = int((challenger_gold & selected).sum())
        challenger_gold_wrong_n = int(
            (challenger_gold & selected & (~baseline)).sum()
        )
        item = {
            "scope": scope,
            "gate_variant": variant,
            "threshold": threshold,
            "action": action,
            "n": n,
            "fraction": safe_rate(n, len(frame)),
            "first_pass_wrong_n": wrong_n,
            "first_pass_wrong_rate": safe_rate(wrong_n, n),
            "first_pass_correct_n": correct_n,
            "first_pass_correct_rate": safe_rate(correct_n, n),
            "wrong_recall": safe_rate(wrong_n, total_wrong),
            "challenger_gold_n_all": challenger_gold_n,
            "challenger_gold_rate_all": safe_rate(challenger_gold_n, n),
            "challenger_gold_n_within_wrong": challenger_gold_wrong_n,
            "challenger_gold_rate_within_wrong": safe_rate(
                challenger_gold_wrong_n, wrong_n
            ),
            "final_correct_n": final_correct_n,
            "action_accuracy": safe_rate(final_correct_n, n),
            "corrected": corrected_n,
            "damaged": damaged_n,
            "net_corrected": corrected_n - damaged_n,
        }
        strata_rows.append(item)
        for key, value in item.items():
            if key not in {"scope", "gate_variant", "threshold", "action"}:
                wide[f"{action}_{key}"] = value

    review_n = int(reviewed.sum())
    corrected = int(corrected_mask.sum())
    damaged = int(damaged_mask.sum())
    final_correct_n = int(final.sum())
    reviewed_wrong_n = int(((~baseline) & reviewed).sum())
    all_targeted_correct = int(targeted.sum())
    all_generic_correct = int(generic.sum())
    summary = {
        "scope": scope,
        "gate_variant": variant,
        "threshold": threshold,
        "n_cases": len(frame),
        "n_source_questions": int(frame.q_idx.nunique()),
        "first_pass_correct_n": int(baseline.sum()),
        "first_pass_wrong_n": total_wrong,
        "first_pass_accuracy": float(baseline.mean()),
        "review_n": review_n,
        "review_rate": safe_rate(review_n, len(frame)),
        "reviewed_first_pass_wrong_n": reviewed_wrong_n,
        "reviewed_first_pass_wrong_rate": safe_rate(reviewed_wrong_n, review_n),
        "wrong_recall_by_review": safe_rate(reviewed_wrong_n, total_wrong),
        "final_correct_n": final_correct_n,
        "final_accuracy": safe_rate(final_correct_n, len(frame)),
        "accuracy_gain_pp": 100.0 * (final.mean() - baseline.mean()),
        "corrected": corrected,
        "damaged": damaged,
        "net_corrected": corrected - damaged,
        "net_per_100_reviews": 100.0 * (corrected - damaged) / max(review_n, 1),
        "all_targeted_correct_n": all_targeted_correct,
        "all_targeted_accuracy": safe_rate(all_targeted_correct, len(frame)),
        "all_targeted_corrected": int(((~baseline) & targeted).sum()),
        "all_targeted_damaged": int((baseline & (~targeted)).sum()),
        "all_generic_correct_n": all_generic_correct,
        "all_generic_accuracy": safe_rate(all_generic_correct, len(frame)),
        "all_generic_corrected": int(((~baseline) & generic).sum()),
        "all_generic_damaged": int((baseline & (~generic)).sum()),
    }
    summary.update(wide)
    summary["targeted_minus_generic_wrong_rate_pp"] = 100.0 * (
        summary["targeted_first_pass_wrong_rate"]
        - summary["generic_first_pass_wrong_rate"]
    )
    summary["targeted_minus_none_wrong_rate_pp"] = 100.0 * (
        summary["targeted_first_pass_wrong_rate"]
        - summary["none_first_pass_wrong_rate"]
    )
    summary["generic_minus_none_wrong_rate_pp"] = 100.0 * (
        summary["generic_first_pass_wrong_rate"]
        - summary["none_first_pass_wrong_rate"]
    )
    summary["risk_ordered"] = bool(
        summary["targeted_first_pass_wrong_rate"]
        > summary["generic_first_pass_wrong_rate"]
        > summary["none_first_pass_wrong_rate"]
    )
    return summary, strata_rows


def sweep(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    scope: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    strata = []
    for threshold in THRESHOLDS:
        summary, current_strata = sweep_row(
            frame, probabilities, float(threshold), scope, variant
        )
        summaries.append(summary)
        strata.extend(current_strata)
    return pd.DataFrame(summaries), pd.DataFrame(strata)


def oof_probabilities(
    frame: pd.DataFrame, columns: list[str]
) -> np.ndarray:
    return np.column_stack(
        [grouped_oof(frame, columns, target, MODEL_SPEC) for target in TARGETS]
    )


def render_markdown(
    calibration: pd.DataFrame,
    unseen: pd.DataFrame,
    all_cases: pd.DataFrame,
    selected_threshold: float,
) -> str:
    def markdown_table(frame: pd.DataFrame) -> str:
        headers = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(map(str, row)) + " |")
        return "\n".join(lines)

    selected = unseen.copy()
    columns = [
        "threshold", "targeted_n", "targeted_first_pass_wrong_rate",
        "generic_n", "generic_first_pass_wrong_rate", "none_n",
        "none_first_pass_wrong_rate", "review_rate", "final_accuracy",
        "corrected", "damaged", "net_corrected", "net_per_100_reviews",
    ]
    selected = selected[columns]
    for column in [
        "targeted_first_pass_wrong_rate", "generic_first_pass_wrong_rate",
        "none_first_pass_wrong_rate", "review_rate", "final_accuracy",
    ]:
        selected[column] = (100 * selected[column]).map(lambda x: f"{x:.2f}%")
    selected["net_per_100_reviews"] = selected["net_per_100_reviews"].map(
        lambda x: f"{x:.2f}"
    )
    selected.columns = [
        "阈值", "Targeted n", "Targeted错题率", "Generic n",
        "Generic错题率", "None n", "None错题率", "复核率",
        "最终准确率", "纠正", "改错", "净纠正", "每100次复核净纠正",
    ]
    best_unseen = unseen.sort_values(
        ["final_accuracy", "net_per_100_reviews", "review_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    best_all = all_cases.sort_values(
        ["final_accuracy", "net_per_100_reviews", "review_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    return rf"""# Qwen 20% Gate 阈值全指标汇总

## 口径

- 阈值范围：0.00–0.20，步长0.01，共{len(THRESHOLDS)}个候选。
- 自适应阈值只能由95道Discovery题的grouped-OOF结果选择；160道独立未见题只用于验证。
- 每个case的Targeted和Generic反事实均已缓存，因此阈值扫描不增加模型推理成本。
- 完整逐阈值宽表、动作长表和三特征分支消融表位于本目录CSV文件。

## 校准结果

在风险顺序成立、Targeted错题率不低于58%的约束下，Discovery grouped-OOF按最终准确率优先、每100次复核净收益次优、复核率再次优先，选择阈值 **{selected_threshold:.2f}**。

## 模型级自适应规则

对模型 $m$ 的校准集，在候选阈值集合 $\mathcal{{T}}=\{{0.00,0.01,\ldots,0.20\}}$ 上遍历。若最佳复核动作相对None的预测正确率增益不低于 $\tau$，则执行该复核动作，否则保留首答：

$$
a_i(\tau)=
\begin{{cases}}
\arg\max_{{a\in\{{T,G\}}}} p_i(a), & \max(p_i(T),p_i(G))-p_i(N)\ge\tau,\\
N, & \text{{otherwise}}.
\end{{cases}}
$$

在风险顺序 $R_T>R_G>R_N$ 且 $R_T\ge 0.58$ 的约束下，先最大化校准准确率，再最大化每100次复核净纠正，最后最小化复核率，得到模型专属阈值 $\tau_m^*$。只冻结这个标量；Challenger和Gate回归权重均不重新训练。

## 独立未见集全部21个阈值

{markdown_table(selected)}

## 仅用于描述的事后上界

- 独立未见集直接事后挑选的最高准确率阈值为{best_unseen.threshold:.2f}，准确率{best_unseen.final_accuracy:.2%}。该数值不能作为确认性主结果。
- 全255题直接事后挑选的最高准确率阈值为{best_all.threshold:.2f}，准确率{best_all.final_accuracy:.2%}。该数值同样只能描述灵敏度。

## 方法学结论

阈值降低通常扩大复核覆盖、提高总纠正数，但会降低Targeted错题纯度和单位复核效率。论文中的“模型自适应Gate”应表述为小型校准集上冻结阈值，再在独立评估集测试，而不是在测试集上选择最优阈值。

Qwen中，Discovery grouped-OOF选择0.03；独立未见集上0.01与0.03均达到64.64%的并列最高准确率，0.03的每100次复核净纠正更高（8.24对7.69）。这是支持自适应阈值的首个内部证据，但在其他模型完成“校准后冻结→独立测试”前，不能声称跨模型有效性已经得到证明。
"""


def metric_dictionary() -> str:
    return """# Gate 阈值扫描指标字典

## 标识与总体指标

| 字段 | 中文含义 |
| --- | --- |
| scope | 数据口径：Discovery grouped-OOF、Qwen独立未见集或全部20% |
| gate_variant | Probe-only、MI-only或Probe+MI |
| threshold | 最佳复核动作相对None所需的最小预测正确率增益 |
| n_cases / n_source_questions | case数量 / 源问题数量；同题偏见变体按源题聚类 |
| first_pass_correct_n / first_pass_wrong_n / first_pass_accuracy | 首轮正确数、错误数和准确率 |
| review_n / review_rate | 被分配到Targeted或Generic的数量和比例 |
| reviewed_first_pass_wrong_n / reviewed_first_pass_wrong_rate | 所有复核对象中的首轮错题数量和纯度 |
| wrong_recall_by_review | 全部首轮错题中有多少进入任一种复核 |
| final_correct_n / final_accuracy / accuracy_gain_pp | Gate执行后的正确数、准确率及相对首轮百分点增益 |
| corrected / damaged / net_corrected | 复核改对数、改错数及二者之差 |
| net_per_100_reviews | 每100次复核带来的净纠正数量 |
| all_targeted_* / all_generic_* | 所有case统一执行某动作时的反事实准确率、纠正和改错，用作上界/对照 |
| targeted_minus_generic_wrong_rate_pp | Targeted与Generic首轮错题率差，单位为百分点 |
| targeted_minus_none_wrong_rate_pp | Targeted与None首轮错题率差，主要风险分离指标 |
| generic_minus_none_wrong_rate_pp | Generic与None首轮错题率差 |
| risk_ordered | 是否满足Targeted错题率 > Generic错题率 > None错题率 |

## 动作前缀字段

宽表中的 `targeted_`、`generic_`、`none_` 使用完全相同的后缀定义：

| 后缀 | 中文含义 |
| --- | --- |
| n / fraction | 该动作的case数量和占比 |
| first_pass_wrong_n / first_pass_wrong_rate | 该动作组首轮错误数和错误纯度 |
| first_pass_correct_n / first_pass_correct_rate | 该动作组首轮正确数和正确纯度 |
| wrong_recall | 该动作组覆盖了全部首轮错题的比例 |
| challenger_gold_n_all / challenger_gold_rate_all | 该动作全部case中Challenger等于gold的数量和比例 |
| challenger_gold_n_within_wrong / challenger_gold_rate_within_wrong | 仅首轮错题中Challenger等于gold的数量和比例 |
| final_correct_n / action_accuracy | 执行该动作后的正确数和正确率；None即保留首答 |
| corrected / damaged / net_corrected | 该动作单独产生的改对、改错和净纠正 |

## 文件关系

- `qwen_gate_threshold_sweep_full_metrics.csv`：189行×80列；3个数据口径×3种Gate特征分支×21个阈值。
- `qwen_gate_threshold_action_strata.csv`：上述每行拆为Targeted/Generic/None三行动作记录。
- `qwen_gate_threshold_{scope}.csv`：只保留完整Probe+MI Gate，便于论文直接制表。
- `qwen_gate_adaptive_variant_ablation.csv`：三个特征分支分别经Discovery校准后，在独立未见集上的公平消融结果。
- `qwen_gate_threshold_mi_matched_ablation.csv`：相同阈值下Probe+MI相对Probe-only/MI-only的逐点比较。
- `qwen_gate_adaptive_variant_paired_tests.csv`：独立未见集上的聚类Bootstrap区间与McNemar配对检验。
"""


def select_variant_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    """Calibrate each ablation branch independently, then read unseen metrics."""
    rows = []
    for variant in ("probe_only", "mi_only", "probe_plus_mi"):
        calibration = summary[
            summary.scope.eq("discovery_grouped_oof")
            & summary.gate_variant.eq(variant)
        ].copy()
        eligible = calibration[
            calibration.risk_ordered
            & calibration.targeted_first_pass_wrong_rate.ge(0.58)
        ]
        selected = eligible.sort_values(
            ["final_accuracy", "net_per_100_reviews", "review_rate"],
            ascending=[False, False, True],
        ).iloc[0]
        unseen = summary[
            summary.scope.eq("primary_unseen")
            & summary.gate_variant.eq(variant)
            & summary.threshold.eq(selected.threshold)
        ].iloc[0]
        rows.append({
            "gate_variant": variant,
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
    return pd.DataFrame(rows)


def matched_threshold_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare feature branches at the same operating threshold."""
    unseen = summary[summary.scope.eq("primary_unseen")].copy()
    metrics = [
        "final_accuracy", "review_rate", "targeted_first_pass_wrong_rate",
        "targeted_minus_none_wrong_rate_pp", "net_corrected",
        "net_per_100_reviews",
    ]
    rows = []
    for threshold in THRESHOLDS:
        current = unseen[unseen.threshold.eq(threshold)].set_index(
            "gate_variant"
        )
        row = {"threshold": float(threshold)}
        for variant in ("probe_only", "mi_only", "probe_plus_mi"):
            for metric in metrics:
                row[f"{variant}_{metric}"] = current.loc[variant, metric]
        row["mi_gain_vs_probe_accuracy_pp"] = 100.0 * (
            row["probe_plus_mi_final_accuracy"]
            - row["probe_only_final_accuracy"]
        )
        row["mi_gain_vs_probe_targeted_wrong_rate_pp"] = 100.0 * (
            row["probe_plus_mi_targeted_first_pass_wrong_rate"]
            - row["probe_only_targeted_first_pass_wrong_rate"]
        )
        row["mi_gain_vs_probe_risk_gap_pp"] = (
            row["probe_plus_mi_targeted_minus_none_wrong_rate_pp"]
            - row["probe_only_targeted_minus_none_wrong_rate_pp"]
        )
        row["mi_gain_vs_probe_net_corrected"] = (
            row["probe_plus_mi_net_corrected"]
            - row["probe_only_net_corrected"]
        )
        row["mi_gain_vs_probe_review_rate_pp"] = 100.0 * (
            row["probe_plus_mi_review_rate"]
            - row["probe_only_review_rate"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def final_outcomes(
    frame: pd.DataFrame,
    variant: str,
    threshold: float,
) -> np.ndarray:
    columns = [
        f"p_none_{variant}", f"p_targeted_{variant}",
        f"p_generic_{variant}",
    ]
    probabilities = frame[columns].to_numpy(float)
    actions = choose_actions(probabilities, threshold)
    action_index = pd.Categorical(
        actions, categories=["none", "targeted", "generic"]
    ).codes
    outcomes = frame[
        ["baseline_correct", "targeted_correct", "generic_correct"]
    ].astype(bool).to_numpy()
    return outcomes[np.arange(len(frame)), action_index].astype(int)


def exact_mcnemar_pvalue(full_wins: int, ablation_wins: int) -> float:
    discordant = full_wins + ablation_wins
    if discordant == 0:
        return 1.0
    lower = min(full_wins, ablation_wins)
    tail = sum(
        math.comb(discordant, index) for index in range(lower + 1)
    ) / (2.0 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_variant_tests(
    unseen: pd.DataFrame,
    calibrated: pd.DataFrame,
) -> pd.DataFrame:
    threshold_by_variant = calibrated.set_index(
        "gate_variant"
    ).selected_threshold.to_dict()
    full = final_outcomes(
        unseen, "probe_plus_mi", threshold_by_variant["probe_plus_mi"]
    )
    rng = np.random.default_rng(20260719)
    rows = []
    for ablation in ("probe_only", "mi_only"):
        comparator = final_outcomes(
            unseen, ablation, threshold_by_variant[ablation]
        )
        difference = full - comparator
        blocks = pd.DataFrame({
            "q_idx": unseen.q_idx.to_numpy(), "difference": difference,
        }).groupby("q_idx").difference.agg(["sum", "count"])
        draw = rng.integers(
            0, len(blocks), size=(50000, len(blocks)), endpoint=False
        )
        bootstrap = (
            blocks["sum"].to_numpy()[draw].sum(axis=1)
            / blocks["count"].to_numpy()[draw].sum(axis=1)
            * 100.0
        )
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        full_wins = int(((full == 1) & (comparator == 0)).sum())
        ablation_wins = int(((full == 0) & (comparator == 1)).sum())
        rows.append({
            "comparison": f"probe_plus_mi_vs_{ablation}",
            "probe_plus_mi_threshold": threshold_by_variant["probe_plus_mi"],
            "ablation_threshold": threshold_by_variant[ablation],
            "accuracy_difference_pp": 100.0 * difference.mean(),
            "cluster_bootstrap_ci95_lower_pp": lower,
            "cluster_bootstrap_ci95_upper_pp": upper,
            "probe_plus_mi_only_correct_n": full_wins,
            "ablation_only_correct_n": ablation_wins,
            "mcnemar_exact_p": exact_mcnemar_pvalue(
                full_wins, ablation_wins
            ),
        })
    return pd.DataFrame(rows)


def render_mi_ablation_report(
    calibrated: pd.DataFrame,
    matched: pd.DataFrame,
    paired_tests: pd.DataFrame,
    model_name: str = "Qwen",
    calibration_label: str = "Discovery grouped-OOF",
) -> str:
    label = {
        "probe_only": "Probe-only",
        "mi_only": "MI-only",
        "probe_plus_mi": "Probe+MI",
    }
    calibrated_lines = [
        "| 分支 | Discovery校准阈值 | 未见集准确率 | 相对首答提升 | 复核率 | "
        "Targeted错题率 | T/G/N错题率 | 净纠正 | 每100次复核净纠正 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in calibrated.itertuples(index=False):
        calibrated_lines.append(
            f"| {label[row.gate_variant]} | {row.selected_threshold:.2f} | "
            f"{row.unseen_final_accuracy:.2%} | "
            f"{row.unseen_accuracy_gain_pp:.2f} pp | "
            f"{row.unseen_review_rate:.2%} | "
            f"{row.unseen_targeted_wrong_rate:.2%} | "
            f"{row.unseen_targeted_wrong_rate:.2%} / "
            f"{row.unseen_generic_wrong_rate:.2%} / "
            f"{row.unseen_none_wrong_rate:.2%} | "
            f"{row.unseen_net_corrected} | "
            f"{row.unseen_net_per_100_reviews:.2f} |"
        )
    test_lines = [
        "| 配对比较 | 准确率差 | 95%聚类Bootstrap CI | 独有正确数 | "
        "对照独有正确数 | McNemar exact p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paired_tests.itertuples(index=False):
        test_lines.append(
            f"| {row.comparison} | {row.accuracy_difference_pp:.2f} pp | "
            f"[{row.cluster_bootstrap_ci95_lower_pp:.2f}, "
            f"{row.cluster_bootstrap_ci95_upper_pp:.2f}] pp | "
            f"{row.probe_plus_mi_only_correct_n} | "
            f"{row.ablation_only_correct_n} | {row.mcnemar_exact_p:.3f} |"
        )
    joint = matched[
        matched.mi_gain_vs_probe_accuracy_pp.gt(0)
        & matched.mi_gain_vs_probe_targeted_wrong_rate_pp.ge(2.0)
    ].threshold.tolist()
    utility_positive = matched[
        matched.mi_gain_vs_probe_accuracy_pp.gt(0)
    ].threshold.tolist()
    risk_positive = matched[
        matched.mi_gain_vs_probe_targeted_wrong_rate_pp.gt(0)
    ].threshold.tolist()
    show_thresholds = lambda values: (
        ", ".join(f"{value:.2f}" for value in values) if values else "无"
    )
    legacy = matched[matched.threshold.eq(0.08)].iloc[0]
    indexed = calibrated.set_index("gate_variant")
    probe = indexed.loc["probe_only"]
    mi_only = indexed.loc["mi_only"]
    full = indexed.loc["probe_plus_mi"]
    gain_vs_probe = 100.0 * (
        full.unseen_final_accuracy - probe.unseen_final_accuracy
    )
    gain_vs_mi = 100.0 * (
        full.unseen_final_accuracy - mi_only.unseen_final_accuracy
    )
    net_gain_vs_probe = int(
        full.unseen_net_corrected - probe.unseen_net_corrected
    )
    net_gain_vs_mi = int(
        full.unseen_net_corrected - mi_only.unseen_net_corrected
    )
    all_positive = gain_vs_probe > 0 and gain_vs_mi > 0
    all_significant = bool(
        paired_tests.cluster_bootstrap_ci95_lower_pp.gt(0).all()
        and paired_tests.mcnemar_exact_p.lt(0.05).all()
    )
    utility_conclusion = (
        "完整Probe+MI Gate获得最高点估计"
        if all_positive
        else "完整Probe+MI Gate未在两个消融对照上同时获得最高点估计"
    )
    uncertainty_conclusion = (
        "配对结果达到预设显著性标准。"
        if all_significant
        else (
            "至少一项置信区间跨越0或配对检验未达到0.05显著水平，"
            "因此只能报告点估计，不能宣称MI增益已得到统计显著证明。"
        )
    )
    return f"""# {model_name} Gate阈值与MI消融联合分析

## 为什么必须联合分析

Gate阈值会改变Targeted、Generic与None的分配边界，因此MI消融不能只在单个任意阈值上判断。本文保留两种互补口径：

1. **同阈值消融**：三个分支使用相同阈值，回答MI在相同决策边界下改变了什么。
2. **独立校准消融**：每个分支只在{calibration_label}上选择自己的阈值，随后冻结并在独立未见集测试，回答各方法在公平校准后的端到端效用。

测试集不能参与阈值选择，否则会把测试集调参收益错误归因于MI。

## 公平校准后的主结果

{chr(10).join(calibrated_lines)}

Probe+MI相对Probe-only在未见集准确率上变化 **{gain_vs_probe:+.2f}个百分点**，净纠正变化 **{net_gain_vs_probe:+d}例**，每100次复核净纠正由 **{probe.unseen_net_per_100_reviews:.2f}** 变为 **{full.unseen_net_per_100_reviews:.2f}**。相对MI-only，准确率变化 **{gain_vs_mi:+.2f}个百分点**，净纠正变化 **{net_gain_vs_mi:+d}例**。

公平校准后Probe+MI的Targeted错题率为{full.unseen_targeted_wrong_rate:.2%}，Probe-only为{probe.unseen_targeted_wrong_rate:.2%}。是否支持“MI提供增量端到端效用”必须同时依据上表的准确率、净纠正和配对不确定性，不能只依据Targeted错题纯度。

## 配对不确定性

{chr(10).join(test_lines)}

{uncertainty_conclusion} {model_name}的最终确认应使用预先冻结规则的更大独立样本或跨模型汇总。

## 同阈值机制结果

- Probe+MI相对Probe-only提高最终准确率的阈值：{show_thresholds(utility_positive)}。
- Probe+MI相对Probe-only提高Targeted错题纯度的阈值：{show_thresholds(risk_positive)}。
- 同时满足“准确率提高”且“Targeted错题率至少提高2个百分点”的阈值：{show_thresholds(joint)}。
- 旧冻结阈值0.08恰好是MI不占优的工作点：准确率差为{legacy.mi_gain_vs_probe_accuracy_pp:.2f}个百分点，Targeted错题率差为{legacy.mi_gain_vs_probe_targeted_wrong_rate_pp:.2f}个百分点。它不能代表整条阈值曲线上的MI贡献。

## 论文中可使用的严谨表述

“Gate operating threshold modulates the observable contribution of MI features. After independently calibrating each ablation branch on a pre-specified calibration split, we froze the scalar threshold and evaluated it on held-out source questions.”

现阶段不要写成“MI在所有阈值均有效”。更准确的结论是：**MI的可见贡献具有工作点依赖性；公平校准后，{utility_conclusion}。**
"""


def validate_tables(
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    cases: pd.DataFrame,
) -> dict:
    failures = []
    for row in summary.itertuples(index=False):
        if row.targeted_n + row.generic_n + row.none_n != row.n_cases:
            failures.append(f"action count mismatch: {row.scope}/{row.gate_variant}/{row.threshold}")
        if row.final_correct_n != row.first_pass_correct_n + row.net_corrected:
            failures.append(f"net identity mismatch: {row.scope}/{row.gate_variant}/{row.threshold}")
        if row.review_n != row.targeted_n + row.generic_n:
            failures.append(f"review count mismatch: {row.scope}/{row.gate_variant}/{row.threshold}")
        action_wrong = (
            row.targeted_first_pass_wrong_n
            + row.generic_first_pass_wrong_n
            + row.none_first_pass_wrong_n
        )
        if action_wrong != row.first_pass_wrong_n:
            failures.append(f"wrong count mismatch: {row.scope}/{row.gate_variant}/{row.threshold}")
    for (_, _), group in summary.groupby(["scope", "gate_variant"]):
        ordered = group.sort_values("threshold")
        if not ordered.review_n.diff().fillna(0).le(0).all():
            failures.append(
                f"review count is not monotone: {ordered.scope.iloc[0]}/{ordered.gate_variant.iloc[0]}"
            )
    official = summary[
        summary.scope.eq("all_20pct")
        & summary.gate_variant.eq("probe_plus_mi")
        & summary.threshold.eq(0.08)
    ].iloc[0]
    expected_counts = cases.action.value_counts().to_dict()
    observed_counts = {
        "targeted": int(official.targeted_n),
        "generic": int(official.generic_n),
        "none": int(official.none_n),
    }
    if observed_counts != expected_counts:
        failures.append(
            f"0.08 policy reproduction mismatch: {observed_counts} != {expected_counts}"
        )
    expected_strata_rows = len(summary) * 3
    if len(strata) != expected_strata_rows:
        failures.append(
            f"strata row mismatch: {len(strata)} != {expected_strata_rows}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "passed": True,
        "summary_rows": len(summary),
        "strata_rows": len(strata),
        "thresholds_per_scope_variant": len(THRESHOLDS),
        "official_margin_008_action_counts_reproduced": observed_counts,
        "identities_checked": [
            "actions sum to cases",
            "reviews equal targeted plus generic",
            "action wrong counts sum to total wrong",
            "final correct equals first-pass correct plus net corrected",
            "review count is monotone non-increasing with threshold",
        ],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(CASE_RESULTS)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    discovery_q = set(split["discovery_q_idx"])
    unseen = cases[~cases.q_idx.isin(discovery_q)].copy()

    probability_columns = {
        "probe_only": [
            "p_none_probe_only", "p_targeted_probe_only", "p_generic_probe_only"
        ],
        "mi_only": [
            "p_none_mi_only", "p_targeted_mi_only", "p_generic_mi_only"
        ],
        "probe_plus_mi": [
            "p_none_probe_plus_mi", "p_targeted_probe_plus_mi",
            "p_generic_probe_plus_mi",
        ],
    }
    all_summaries = []
    all_strata = []
    scope_frames = {"all_20pct": cases, "primary_unseen": unseen}
    for scope, frame in scope_frames.items():
        for variant, columns in probability_columns.items():
            summary, strata = sweep(
                frame, frame[columns].to_numpy(float), scope, variant
            )
            all_summaries.append(summary)
            all_strata.append(strata)

    discovery, _, _ = load_discovery()
    discovery = discovery.rename(columns={"learned_challenger": "challenger"})
    feature_sets = gate_feature_sets(discovery)
    discovery_summaries = []
    discovery_strata = []
    for variant in ("probe_only", "mi_only", "probe_plus_mi"):
        probabilities = oof_probabilities(discovery, feature_sets[variant])
        summary, strata = sweep(
            discovery, probabilities, "discovery_grouped_oof", variant
        )
        discovery_summaries.append(summary)
        discovery_strata.append(strata)

    summary = pd.concat(all_summaries + discovery_summaries, ignore_index=True)
    strata = pd.concat(all_strata + discovery_strata, ignore_index=True)
    validation = validate_tables(summary, strata, cases)
    summary.to_csv(OUTPUT / "qwen_gate_threshold_sweep_full_metrics.csv", index=False)
    strata.to_csv(OUTPUT / "qwen_gate_threshold_action_strata.csv", index=False)

    full_gate = summary[summary.gate_variant.eq("probe_plus_mi")].copy()
    for scope in ("all_20pct", "primary_unseen", "discovery_grouped_oof"):
        full_gate[full_gate.scope.eq(scope)].to_csv(
            OUTPUT / f"qwen_gate_threshold_{scope}.csv", index=False
        )
    calibration = full_gate[full_gate.scope.eq("discovery_grouped_oof")].copy()
    eligible = calibration[
        calibration.risk_ordered
        & calibration.targeted_first_pass_wrong_rate.ge(0.58)
    ]
    selected = eligible.sort_values(
        ["final_accuracy", "net_per_100_reviews", "review_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    selected_threshold = float(selected.threshold)
    unseen_full = full_gate[full_gate.scope.eq("primary_unseen")].copy()
    all_full = full_gate[full_gate.scope.eq("all_20pct")].copy()
    adaptive = {
        "threshold_range": [float(THRESHOLDS.min()), float(THRESHOLDS.max())],
        "threshold_step": 0.01,
        "selection_scope": "discovery_grouped_oof",
        "selection_constraints": {
            "risk_ordered": True,
            "targeted_wrong_rate_min": 0.58,
        },
        "selection_order": [
            "maximize final_accuracy", "maximize net_per_100_reviews",
            "minimize review_rate",
        ],
        "selected_threshold": selected_threshold,
        "calibration_metrics": selected.to_dict(),
        "primary_unseen_metrics": unseen_full[
            unseen_full.threshold.eq(selected_threshold)
        ].iloc[0].to_dict(),
    }
    (OUTPUT / "adaptive_threshold_selection.json").write_text(
        json.dumps(adaptive, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_markdown(
        calibration, unseen_full, all_full, selected_threshold
    )
    (OUTPUT / "Qwen-20pct-Gate阈值全指标汇总.md").write_text(
        report, encoding="utf-8"
    )
    (OUTPUT / "Gate阈值指标字典.md").write_text(
        metric_dictionary(), encoding="utf-8"
    )

    calibrated_variants = select_variant_thresholds(summary)
    matched_ablation = matched_threshold_ablation(summary)
    paired_tests = paired_variant_tests(unseen, calibrated_variants)
    calibrated_variants.to_csv(
        OUTPUT / "qwen_gate_adaptive_variant_ablation.csv", index=False
    )
    matched_ablation.to_csv(
        OUTPUT / "qwen_gate_threshold_mi_matched_ablation.csv", index=False
    )
    paired_tests.to_csv(
        OUTPUT / "qwen_gate_adaptive_variant_paired_tests.csv", index=False
    )
    (OUTPUT / "Qwen-Gate阈值与MI消融联合分析.md").write_text(
        render_mi_ablation_report(
            calibrated_variants, matched_ablation, paired_tests
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "n_summary_rows": len(summary),
        "n_strata_rows": len(strata),
        "selected_threshold": selected_threshold,
        "mi_ablation_outputs": {
            "calibrated_variants": len(calibrated_variants),
            "matched_thresholds": len(matched_ablation),
            "paired_tests": len(paired_tests),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
