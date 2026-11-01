#!/usr/bin/env python3
"""Build full-run Targeted/None analysis reports for MI-SR models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "output_mi_sr_full_targeted_v5"
THRESHOLDS = np.round(np.arange(0.0, 0.201, 0.01), 2)
RANDOM_SEED = 20260722
RANDOM_REPS = 3000
BOOTSTRAP_REPS = 3000

MODEL_SPECS = {
    "qwen3_14b_reference": {
        "label": "Qwen3-14B",
        "short": "Qwen3-14B",
        "out": "output_qwen3_14b_full_binary_analysis_v2",
        "report": "Qwen3-14B-100pct-Targeted-None完整实验报告.md",
        "seed_q": 637,
        "added_name": "Added-50%",
        "include_generic": True,
    },
    "deepseek_r1_qwen14b": {
        "label": "DeepSeek-R1-Distill-Qwen-14B",
        "short": "DeepSeek-R1-Distill-Qwen-14B",
        "out": "output_deepseek_full_binary_analysis_v1",
        "report": "DeepSeek-R1-Distill-Qwen-14B-100pct-Targeted-None完整实验报告.md",
        "seed_q": 637,
        "added_name": "Added-50%",
        "include_generic": True,
    },
    "mistral7b_instruct_v03": {
        "label": "Mistral-7B-Instruct-v0.3",
        "short": "Mistral-7B-Instruct-v0.3",
        "out": "output_mistral7b_full_binary_analysis_v1",
        "report": "Mistral-7B-Instruct-v0.3-100pct-Targeted-None完整实验报告.md",
        "seed_q": 127,
        "added_name": "Added-90%",
        "include_generic": True,
    },
}


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1"})


def fmt(value, kind: str) -> str:
    if pd.isna(value):
        return "待补"
    if kind == "str":
        return str(value)
    if kind == "int":
        return str(int(value))
    if kind == "pct":
        return f"{float(value):.2%}"
    if kind == "pp":
        return f"{float(value):+.2f} pp"
    if kind == "float":
        return f"{float(value):.2f}"
    if kind == "sci":
        return f"{float(value):.3g}"
    return str(value)


def markdown_table(frame: pd.DataFrame, specs: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in specs) + " |"
    sep = "| " + " | ".join(["---"] * len(specs)) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        lines.append(
            "| " + " | ".join(fmt(row.get(col, np.nan), kind) for col, _, kind in specs) + " |"
        )
    return "\n".join(lines)


def exact_paired_p(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    a = as_bool(a)
    b = as_bool(b)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    p = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
    return a_only, b_only, p


def cluster_bootstrap_ci(frame: pd.DataFrame, first: str, second: str) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED + len(frame) + len(first))
    grouped = (
        frame.assign(
            diff=as_bool(frame[first]).astype(float) - as_bool(frame[second]).astype(float)
        )
        .groupby("q_idx", sort=False)
        ["diff"].mean()
        .to_numpy()
    )
    samples = rng.choice(grouped, size=(BOOTSTRAP_REPS, len(grouped)), replace=True)
    return tuple(np.quantile(samples.mean(axis=1) * 100, [0.025, 0.975]))


def review_answer(frame: pd.DataFrame, answer_col: str, complete_col: str) -> pd.Series:
    valid = as_bool(frame[complete_col]) & frame[answer_col].fillna("").astype(str).isin(tuple("ABCDE"))
    answer = frame["model_answer"].astype(str).copy()
    answer.loc[valid] = frame.loc[valid, answer_col].astype(str)
    return answer


def load_cases(model_key: str, include_generic: bool) -> tuple[pd.DataFrame, dict]:
    run_dir = RUN_ROOT / model_key / "all_biases"
    labels = pd.read_csv(run_dir / "first_pass_evaluation_labels.csv")
    policy = pd.read_csv(run_dir / "binary_deploy_policy.csv")
    targeted = pd.read_csv(run_dir / "targeted_all_reevaluation.csv")
    clean = pd.read_csv(run_dir / "clean_reference_outputs.csv")

    labels["baseline_correct"] = as_bool(labels["biased_correct"])
    labels["biased_complete_value"] = as_bool(labels["biased_complete"])
    clean["clean_correct_value"] = as_bool(clean["clean_correct"])
    clean["clean_complete_value"] = as_bool(clean["clean_complete"])
    targeted["targeted_complete_value"] = as_bool(targeted["reeval_complete"])

    cases = (
        policy.merge(
            labels[
                [
                    "case_id",
                    "q_idx",
                    "bias_type",
                    "gold",
                    "biased_answer",
                    "baseline_correct",
                    "biased_complete_value",
                ]
            ],
            on=["case_id", "q_idx", "bias_type"],
            validate="one_to_one",
        )
        .merge(
            targeted[["case_id", "reeval_answer", "targeted_complete_value"]],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            clean[["q_idx", "clean_correct_value", "clean_complete_value"]],
            on="q_idx",
            how="left",
            validate="many_to_one",
        )
    )
    if not cases["model_answer"].astype(str).eq(cases["biased_answer"].astype(str)).all():
        raise RuntimeError(f"{model_key}: policy answer and first-pass answer mismatch")

    # A newly evaluable first-pass case may still have no parseable review row.
    # The prespecified analysis treats both missing and incomplete reviews as
    # retain-first-answer fallbacks.
    cases["targeted_complete_value"] = as_bool(
        cases["targeted_complete_value"]
    )
    cases["targeted_answer"] = review_answer(cases, "reeval_answer", "targeted_complete_value")
    cases["targeted_correct"] = cases["targeted_answer"].eq(cases["gold"].astype(str))
    cases["challenger_is_gold"] = cases["challenger"].astype(str).eq(cases["gold"].astype(str))
    cases["review_score"] = cases[["p_targeted_correct", "p_generic_correct"]].max(axis=1) - cases["p_none_correct"]
    cases["selected"] = cases["review_score"].gt(0.0)
    cases["mi_sr_correct"] = cases["baseline_correct"].where(~cases["selected"], cases["targeted_correct"])
    cases["outcome_group"] = np.select(
        [
            cases["clean_correct_value"] & cases["baseline_correct"],
            cases["clean_correct_value"] & ~cases["baseline_correct"],
            ~cases["clean_correct_value"] & cases["baseline_correct"],
        ],
        ["Robust", "Harmed", "Recovered"],
        default="BaselineWrong",
    )

    generic_available = False
    if include_generic and (run_dir / "generic_all_reevaluation.csv").exists():
        generic = pd.read_csv(run_dir / "generic_all_reevaluation.csv")
        generic["generic_complete_value"] = as_bool(generic["reeval_complete"])
        cases = cases.merge(
            generic[["case_id", "reeval_answer", "generic_complete_value"]].rename(
                columns={"reeval_answer": "generic_answer"}
            ),
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        cases["generic_complete_value"] = as_bool(
            cases["generic_complete_value"]
        )
        cases["generic_answer_final"] = review_answer(
            cases.rename(columns={"generic_answer": "generic_tmp_answer"}),
            "generic_tmp_answer",
            "generic_complete_value",
        )
        cases["generic_correct"] = cases["generic_answer_final"].eq(cases["gold"].astype(str))
        generic_available = int(cases["generic_correct"].notna().sum()) == len(cases)
    else:
        cases["generic_correct"] = np.nan

    audit = {
        "planned_cases": int(len(labels)),
        "policy_evaluable_cases": int(len(cases)),
        "excluded_cases": int(len(labels) - len(cases)),
        "clean_questions": int(clean["q_idx"].nunique()),
        "clean_complete": int(clean["clean_complete_value"].sum()),
        "clean_correct": int(clean["clean_correct_value"].sum()),
        "clean_accuracy": float(clean["clean_correct_value"].mean()),
        "clean_accuracy_complete_only": float(clean.loc[clean["clean_complete_value"], "clean_correct_value"].mean()),
        "targeted_complete": int(cases["targeted_complete_value"].sum()),
        "targeted_incomplete": int((~cases["targeted_complete_value"]).sum()),
        "generic_available": generic_available,
    }
    return cases, audit


def strategy_row(scope: str, name: str, baseline: pd.Series, final: pd.Series, review_n: int) -> dict:
    baseline = as_bool(baseline)
    final = as_bool(final)
    corrected = int((~baseline & final).sum())
    damaged = int((baseline & ~final).sum())
    n = len(final)
    return {
        "scope": scope,
        "strategy": name,
        "n": n,
        "review_n": int(review_n),
        "review_rate": review_n / n if n else np.nan,
        "correct_n": int(final.sum()),
        "accuracy": float(final.mean()),
        "gain_pp": float((final.mean() - baseline.mean()) * 100),
        "corrected": corrected,
        "damaged": damaged,
        "net_corrected": corrected - damaged,
        "net_per_100_reviews": (corrected - damaged) / review_n * 100 if review_n else 0.0,
    }


def random_budget(frame: pd.DataFrame, selected: pd.Series, outcome_col: str) -> dict:
    if outcome_col not in frame or frame[outcome_col].isna().any():
        return {
            "review_n": int(selected.sum()),
            "observed_net_corrected": np.nan,
            "random_mean_net_corrected": np.nan,
            "random_net_low": np.nan,
            "random_net_high": np.nan,
            "random_mean_accuracy": np.nan,
            "random_accuracy_low": np.nan,
            "random_accuracy_high": np.nan,
            "monte_carlo_p": np.nan,
        }
    rng = np.random.default_rng(RANDOM_SEED + len(frame) + len(outcome_col))
    base = as_bool(frame["baseline_correct"])
    final = as_bool(frame[outcome_col])
    gains = final.astype(int) - base.astype(int)
    observed = int(gains[selected].sum())
    simulated = np.zeros(RANDOM_REPS, dtype=int)
    work = frame.assign(selected=selected.to_numpy(), gain=gains.to_numpy())
    for _, group in work.groupby("bias_type", sort=True):
        values = group["gain"].to_numpy()
        draw_n = int(group["selected"].sum())
        if draw_n:
            for idx in range(RANDOM_REPS):
                simulated[idx] += rng.choice(values, size=draw_n, replace=False).sum()
    random_acc = (int(base.sum()) + simulated) / len(frame)
    return {
        "review_n": int(selected.sum()),
        "observed_net_corrected": observed,
        "random_mean_net_corrected": float(simulated.mean()),
        "random_net_low": float(np.quantile(simulated, 0.025)),
        "random_net_high": float(np.quantile(simulated, 0.975)),
        "random_mean_accuracy": float(random_acc.mean()),
        "random_accuracy_low": float(np.quantile(random_acc, 0.025)),
        "random_accuracy_high": float(np.quantile(random_acc, 0.975)),
        "monte_carlo_p": float((1 + np.sum(simulated >= observed)) / (RANDOM_REPS + 1)),
    }


def gate_rows(frame: pd.DataFrame, scope: str) -> list[dict]:
    rows = []
    for action_name, selected in [("Targeted", frame["selected"]), ("None", ~frame["selected"])]:
        subset = frame[selected].copy()
        if subset.empty:
            continue
        wrong = ~frame["baseline_correct"]
        subset_wrong = ~subset["baseline_correct"]
        deployed = subset["targeted_correct"] if action_name == "Targeted" else subset["baseline_correct"]
        rows.append(
            {
                "scope": scope,
                "gate_action": action_name,
                "n": len(subset),
                "fraction": len(subset) / len(frame),
                "baseline_accuracy": float(subset["baseline_correct"].mean()),
                "first_pass_wrong_rate": float(subset_wrong.mean()),
                "wrong_case_recall": float((selected & wrong).sum() / wrong.sum()) if wrong.sum() else np.nan,
                "challenger_gold_rate_wrong": float(subset.loc[subset_wrong, "challenger_is_gold"].mean()) if subset_wrong.any() else np.nan,
                "targeted_counterfactual_accuracy": float(subset["targeted_correct"].mean()),
                "deployed_accuracy": float(deployed.mean()),
                "corrected": int((~subset["baseline_correct"] & deployed).sum()),
                "damaged": int((subset["baseline_correct"] & ~deployed).sum()),
            }
        )
    return rows


def challenger_rows(frame: pd.DataFrame, scope: str) -> list[dict]:
    rows = []
    groups = [
        ("Wrong", "Gold", ~frame["baseline_correct"] & frame["challenger_is_gold"]),
        ("Wrong", "Non-gold", ~frame["baseline_correct"] & ~frame["challenger_is_gold"]),
        ("Correct", "Non-gold", frame["baseline_correct"]),
    ]
    for first_state, challenger_state, mask in groups:
        subset = frame[mask]
        if subset.empty:
            continue
        rows.append(
            {
                "scope": scope,
                "first_pass_state": first_state,
                "challenger_state": challenger_state,
                "n": len(subset),
                "targeted_correct_n": int(subset["targeted_correct"].sum()),
                "targeted_accuracy": float(subset["targeted_correct"].mean()),
                "changed_answer_rate": float(
                    (~subset["targeted_answer"].astype(str).eq(subset["model_answer"].astype(str))).mean()
                ),
            }
        )
    return rows


def outcome_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby("outcome_group", sort=True):
        wrong = ~subset["baseline_correct"]
        rows.append(
            {
                "outcome_group": group,
                "n": len(subset),
                "baseline_accuracy": float(subset["baseline_correct"].mean()),
                "selected_n": int(subset["selected"].sum()),
                "selected_rate": float(subset["selected"].mean()),
                "challenger_gold_rate_wrong": float(subset.loc[wrong, "challenger_is_gold"].mean()) if wrong.any() else np.nan,
                "mi_sr_accuracy": float(subset["mi_sr_correct"].mean()),
                "all_targeted_accuracy": float(subset["targeted_correct"].mean()),
                "all_generic_accuracy": float(subset["generic_correct"].mean()) if subset["generic_correct"].notna().all() else np.nan,
                "mi_sr_corrected": int((~subset["baseline_correct"] & subset["mi_sr_correct"]).sum()),
                "mi_sr_damaged": int((subset["baseline_correct"] & ~subset["mi_sr_correct"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def bias_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bias, subset in frame.groupby("bias_type", sort=True):
        selected = subset["selected"]
        wrong = ~subset["baseline_correct"]
        rows.append(
            {
                "bias_type": bias,
                "n": len(subset),
                "baseline_accuracy": float(subset["baseline_correct"].mean()),
                "selected_n": int(selected.sum()),
                "selected_rate": float(selected.mean()),
                "selected_wrong_rate": float((~subset.loc[selected, "baseline_correct"]).mean()) if selected.any() else np.nan,
                "wrong_case_recall": float((selected & wrong).sum() / wrong.sum()) if wrong.sum() else np.nan,
                "challenger_gold_rate_wrong": float(subset.loc[wrong, "challenger_is_gold"].mean()) if wrong.any() else np.nan,
                "mi_sr_accuracy": float(subset["mi_sr_correct"].mean()),
                "gain_pp": float((subset["mi_sr_correct"].mean() - subset["baseline_correct"].mean()) * 100),
                "corrected": int((~subset["baseline_correct"] & subset["mi_sr_correct"]).sum()),
                "damaged": int((subset["baseline_correct"] & ~subset["mi_sr_correct"]).sum()),
                "all_generic_accuracy": float(subset["generic_correct"].mean()) if subset["generic_correct"].notna().all() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ablation_rows(frame: pd.DataFrame, scope: str) -> list[dict]:
    configs = [
        ("Probe-only Gate", "p_none_probe_only", "p_targeted_probe_only", "p_generic_probe_only"),
        ("MI-only Gate", "p_none_mi_only", "p_targeted_mi_only", "p_generic_mi_only"),
        ("Probe+MI Gate", "p_none_probe_plus_mi", "p_targeted_probe_plus_mi", "p_generic_probe_plus_mi"),
    ]
    rows = []
    for name, pn, pt, pg in configs:
        if not {pn, pt, pg}.issubset(frame.columns):
            continue
        selected = (frame[[pt, pg]].max(axis=1) - frame[pn]).gt(0.0)
        final = frame["baseline_correct"].where(~selected, frame["targeted_correct"])
        corrected = int((~frame["baseline_correct"] & final).sum())
        damaged = int((frame["baseline_correct"] & ~final).sum())
        wrong = ~frame["baseline_correct"]
        rows.append(
            {
                "scope": scope,
                "strategy": name,
                "review_n": int(selected.sum()),
                "review_rate": float(selected.mean()),
                "selected_wrong_rate": float((~frame.loc[selected, "baseline_correct"]).mean()) if selected.any() else np.nan,
                "wrong_case_recall": float((selected & wrong).sum() / wrong.sum()) if wrong.sum() else np.nan,
                "accuracy": float(final.mean()),
                "gain_pp": float((final.mean() - frame["baseline_correct"].mean()) * 100),
                "corrected": corrected,
                "damaged": damaged,
                "net_corrected": corrected - damaged,
            }
        )
    return rows


def threshold_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for tau in THRESHOLDS:
        selected = frame["review_score"].gt(tau)
        final = frame["baseline_correct"].where(~selected, frame["targeted_correct"])
        wrong = ~frame["baseline_correct"]
        corrected = int((~frame["baseline_correct"] & final).sum())
        damaged = int((frame["baseline_correct"] & ~final).sum())
        rows.append(
            {
                "scope": scope,
                "tau": float(tau),
                "review_n": int(selected.sum()),
                "review_rate": float(selected.mean()),
                "selected_wrong_rate": float((~frame.loc[selected, "baseline_correct"]).mean()) if selected.any() else np.nan,
                "wrong_case_recall": float((selected & wrong).sum() / wrong.sum()) if wrong.sum() else np.nan,
                "none_wrong_rate": float((~frame.loc[~selected, "baseline_correct"]).mean()) if (~selected).any() else np.nan,
                "selected_wrong_challenger_gold_rate": float(frame.loc[selected & wrong, "challenger_is_gold"].mean()) if (selected & wrong).any() else np.nan,
                "accuracy": float(final.mean()),
                "gain_pp": float((final.mean() - frame["baseline_correct"].mean()) * 100),
                "corrected": corrected,
                "damaged": damaged,
                "net_corrected": corrected - damaged,
                "net_per_100_reviews": (corrected - damaged) / selected.sum() * 100 if selected.sum() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def paired_rows(scopes: dict[str, pd.DataFrame], include_generic: bool) -> pd.DataFrame:
    rows = []
    for scope, frame in scopes.items():
        comparisons = [
            ("MI-SR_vs_First", "mi_sr_correct", "baseline_correct"),
            ("MI-SR_vs_AllTargeted", "mi_sr_correct", "targeted_correct"),
        ]
        if include_generic and frame["generic_correct"].notna().all():
            comparisons.extend(
                [
                    ("MI-SR_vs_AllGeneric", "mi_sr_correct", "generic_correct"),
                    ("AllTargeted_vs_AllGeneric", "targeted_correct", "generic_correct"),
                ]
            )
        for name, first, second in comparisons:
            first_only, second_only, p = exact_paired_p(frame[first], frame[second])
            low, high = cluster_bootstrap_ci(frame, first, second)
            rows.append(
                {
                    "scope": scope,
                    "comparison": name,
                    "difference_pp": float((as_bool(frame[first]).mean() - as_bool(frame[second]).mean()) * 100),
                    "first_only_correct": first_only,
                    "second_only_correct": second_only,
                    "mcnemar_exact_p": p,
                    "ci_low_pp": low,
                    "ci_high_pp": high,
                }
            )
    return pd.DataFrame(rows)


def analyze(model_key: str) -> Path:
    spec = MODEL_SPECS[model_key]
    out = ROOT / spec["out"]
    out.mkdir(parents=True, exist_ok=True)
    cases, audit = load_cases(model_key, bool(spec["include_generic"]))
    cases["scope"] = np.where(cases["q_idx"].lt(spec["seed_q"]), "Seed", spec["added_name"])
    scopes = {
        "Full-100%": cases,
        spec["added_name"]: cases[cases["q_idx"].ge(spec["seed_q"])],
    }

    strategy = []
    for scope, frame in scopes.items():
        selected = frame["selected"]
        strategy.extend(
            [
                strategy_row(scope, "First Pass", frame["baseline_correct"], frame["baseline_correct"], 0),
                strategy_row(
                    scope,
                    "MI-SR Binary Targeted/None",
                    frame["baseline_correct"],
                    frame["mi_sr_correct"],
                    int(selected.sum()),
                ),
                strategy_row(scope, "All Targeted", frame["baseline_correct"], frame["targeted_correct"], len(frame)),
            ]
        )
        if spec["include_generic"] and frame["generic_correct"].notna().all():
            same_budget_generic = frame["baseline_correct"].where(~selected, frame["generic_correct"])
            strategy.extend(
                [
                    strategy_row(scope, "MI-SR Gate + Generic Execution", frame["baseline_correct"], same_budget_generic, int(selected.sum())),
                    strategy_row(scope, "All Generic", frame["baseline_correct"], frame["generic_correct"], len(frame)),
                ]
            )
    strategy_df = pd.DataFrame(strategy)
    gate_df = pd.DataFrame([row for scope, frame in scopes.items() for row in gate_rows(frame, scope)])
    challenger_df = pd.DataFrame([row for scope, frame in scopes.items() for row in challenger_rows(frame, scope)])
    outcome_df = outcome_rows(cases)
    bias_df = bias_rows(cases)
    ablation_df = pd.DataFrame([row for scope, frame in scopes.items() for row in ablation_rows(frame, scope)])
    threshold_df = pd.concat([threshold_rows(frame, scope) for scope, frame in scopes.items()], ignore_index=True)
    paired_df = paired_rows(scopes, bool(spec["include_generic"]))

    random_rows = []
    for scope, frame in scopes.items():
        selected = frame["selected"]
        targeted_random = random_budget(frame, selected, "targeted_correct")
        targeted_random["scope"] = scope
        targeted_random["control"] = "Same-budget random Targeted"
        random_rows.append(targeted_random)
        if spec["include_generic"] and frame["generic_correct"].notna().all():
            generic_random = random_budget(frame, selected, "generic_correct")
            generic_random["scope"] = scope
            generic_random["control"] = "Same-budget random Generic"
            random_rows.append(generic_random)
    random_df = pd.DataFrame(random_rows)

    # Statistical headline values.
    full = cases
    selected = full["selected"]
    wrong = ~full["baseline_correct"]
    table = [
        [int((selected & wrong).sum()), int((selected & ~wrong).sum())],
        [int((~selected & wrong).sum()), int((~selected & ~wrong).sum())],
    ]
    gate_or, gate_p = fisher_exact(table)
    wrong_cases = full[wrong]
    challenger_test = binomtest(int(wrong_cases["challenger_is_gold"].sum()), len(wrong_cases), 0.25, alternative="greater")
    gold_wrong = wrong_cases[wrong_cases["challenger_is_gold"]]
    nongold_wrong = wrong_cases[~wrong_cases["challenger_is_gold"]]
    treat_or, treat_p = fisher_exact(
        [
            [int(gold_wrong["targeted_correct"].sum()), int((~gold_wrong["targeted_correct"]).sum())],
            [int(nongold_wrong["targeted_correct"].sum()), int((~nongold_wrong["targeted_correct"]).sum())],
        ]
    )
    harmed = full[full["outcome_group"].eq("Harmed")]
    baseline_wrong = full[full["outcome_group"].eq("BaselineWrong")]
    harmed_or, harmed_p = fisher_exact(
        [
            [int(harmed["challenger_is_gold"].sum()), int((~harmed["challenger_is_gold"]).sum())],
            [int(baseline_wrong["challenger_is_gold"].sum()), int((~baseline_wrong["challenger_is_gold"]).sum())],
        ]
    )
    stats = {
        "gate_wrong_enrichment_or": float(gate_or),
        "gate_wrong_enrichment_p": float(gate_p),
        "wrong_challenger_gold": f"{int(wrong_cases['challenger_is_gold'].sum())}/{len(wrong_cases)}",
        "wrong_challenger_gold_binom_p_gt_25": float(challenger_test.pvalue),
        "targeted_success_gold_vs_nongold_or": float(treat_or),
        "targeted_success_gold_vs_nongold_p": float(treat_p),
        "harmed_challenger_gold": f"{int(harmed['challenger_is_gold'].sum())}/{len(harmed)}",
        "baselinewrong_challenger_gold": f"{int(baseline_wrong['challenger_is_gold'].sum())}/{len(baseline_wrong)}",
        "harmed_vs_baselinewrong_or": float(harmed_or),
        "harmed_vs_baselinewrong_p": float(harmed_p),
    }

    cases.to_csv(out / f"{model_key}_full_case_level.csv", index=False)
    strategy_df.to_csv(out / "strategy_summary.csv", index=False)
    gate_df.to_csv(out / "gate_strata.csv", index=False)
    challenger_df.to_csv(out / "challenger_effects.csv", index=False)
    outcome_df.to_csv(out / "outcome_groups.csv", index=False)
    bias_df.to_csv(out / "bias_types.csv", index=False)
    ablation_df.to_csv(out / "gate_ablations.csv", index=False)
    threshold_df.to_csv(out / "threshold_sweep.csv", index=False)
    paired_df.to_csv(out / "paired_tests.csv", index=False)
    random_df.to_csv(out / "random_budget_tests.csv", index=False)
    (out / "statistical_tests.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    best_full = threshold_df[threshold_df["scope"].eq("Full-100%")].sort_values(["accuracy", "review_rate"], ascending=[False, True]).iloc[0]
    best_added = threshold_df[threshold_df["scope"].eq(spec["added_name"])].sort_values(["accuracy", "review_rate"], ascending=[False, True]).iloc[0]
    generic_note = (
        "Generic 对照已完整纳入本报告，包括 All Generic、同 Gate 预算 Generic execution 与同预算随机 Generic。"
        if spec["include_generic"] and audit["generic_available"]
        else "Generic 对照本版按用户要求留空；待 `generic_all_reevaluation.csv` 完整后可用同一脚本补齐。"
    )
    counterfactual_note = (
        "All Targeted 和 All Generic 都是反事实对照，不代表可部署策略；MI-SR 的部署成本由 Gate 选中的复核数量决定。"
        if spec["include_generic"] and audit["generic_available"]
        else "All Targeted 是反事实治疗上界，不代表可部署策略；All Generic 在本版报告中保留为待补对照。MI-SR 的部署成本由 Gate 选中的复核数量决定。"
    )

    report = f"""# {spec['label']} 100% Targeted/None 实验完整统计报告

## 1. 统计口径与完整性

- 数据目录：`output_mi_sr_full_targeted_v5/{model_key}/all_biases/`。
- 计划首答 case：`{audit['planned_cases']}`；可进入冻结 Challenger+Gate 的 case：`{audit['policy_evaluable_cases']}`；排除 `{audit['excluded_cases']}` 个首答或特征不完整 case。
- Clean reference：`{audit['clean_complete']}/{audit['clean_questions']}` 完整；全体口径准确率 `{audit['clean_accuracy']:.2%}`，仅完整输出口径 `{audit['clean_accuracy_complete_only']:.2%}`。
- Targeted 反事实：`{audit['targeted_complete']}/{audit['policy_evaluable_cases']}` 完整，`{audit['targeted_incomplete']}` 个不完整输出按预定义规则回退首答。
- 当前部署动作只有 **Targeted / None**：`max(p_T,p_G)-p_N > 0.00` 执行 Targeted，否则保留首答。Generic 不进入部署路径，只作为对照实验。
- {generic_note}

## 2. 端到端主结果

{markdown_table(strategy_df, [
    ('scope','数据范围','str'), ('strategy','策略','str'), ('n','n','int'),
    ('review_n','复核数','int'), ('review_rate','复核率','pct'),
    ('correct_n','正确数','int'), ('accuracy','准确率','pct'),
    ('gain_pp','较首答','pp'), ('corrected','纠正','int'),
    ('damaged','改错','int'), ('net_corrected','净纠正','int'),
    ('net_per_100_reviews','每100次复核净纠正','float')])}

{counterfactual_note}

## 3. Gate 是否识别高风险回答

{markdown_table(gate_df, [
    ('scope','数据范围','str'), ('gate_action','动作','str'), ('n','n','int'),
    ('fraction','占比','pct'), ('baseline_accuracy','首答准确率','pct'),
    ('first_pass_wrong_rate','首答错误率','pct'), ('wrong_case_recall','覆盖全部错题','pct'),
    ('challenger_gold_rate_wrong','错题中 challenger=gold','pct'),
    ('targeted_counterfactual_accuracy','Targeted反事实准确率','pct'),
    ('deployed_accuracy','执行后组内准确率','pct'), ('corrected','纠正','int'),
    ('damaged','改错','int')])}

Full-100% 上，Targeted 与 None 的首答错误富集 OR=`{gate_or:.2f}`，Fisher 精确检验 `p={gate_p:.3g}`。该指标说明 Gate 是否把更高风险样本送入定向复核。

## 4. Challenger 定位与 Targeted 治疗中介

{markdown_table(challenger_df, [
    ('scope','数据范围','str'), ('first_pass_state','首答状态','str'),
    ('challenger_state','Challenger','str'), ('n','n','int'),
    ('targeted_correct_n','Targeted后正确','int'),
    ('targeted_accuracy','Targeted准确率','pct'),
    ('changed_answer_rate','答案改变率','pct')])}

Full-100% 的首答错题中，challenger=gold 为 `{int(wrong_cases['challenger_is_gold'].sum())}/{len(wrong_cases)}`（`{wrong_cases['challenger_is_gold'].mean():.2%}`），高于 25% 随机候选基线，单侧二项检验 `p={challenger_test.pvalue:.3g}`。当 challenger=gold 时 Targeted 正确率为 `{gold_wrong['targeted_correct'].mean():.2%}`，非 gold 时为 `{nongold_wrong['targeted_correct'].mean():.2%}`，OR=`{treat_or:.2f}`，`p={treat_p:.3g}`。

## 5. 四类结果组

{markdown_table(outcome_df, [
    ('outcome_group','结果组','str'), ('n','n','int'),
    ('baseline_accuracy','首答准确率','pct'), ('selected_n','复核数','int'),
    ('selected_rate','复核率','pct'), ('challenger_gold_rate_wrong','错题 challenger=gold','pct'),
    ('mi_sr_accuracy','MI-SR准确率','pct'), ('all_targeted_accuracy','All Targeted准确率','pct'),
    ('all_generic_accuracy','All Generic准确率','pct'),
    ('mi_sr_corrected','纠正','int'), ('mi_sr_damaged','改错','int')])}

Harmed 是 clean 正确而 biased 错误的严格偏见致错组。Harmed 的 challenger 命中率为 `{harmed['challenger_is_gold'].mean():.2%}`，BaselineWrong 为 `{baseline_wrong['challenger_is_gold'].mean():.2%}`；两者 OR=`{harmed_or:.2f}`，`p={harmed_p:.3g}`。

## 6. 七类偏见

{markdown_table(bias_df, [
    ('bias_type','偏见类型','str'), ('n','n','int'),
    ('baseline_accuracy','首答准确率','pct'), ('selected_n','复核数','int'),
    ('selected_rate','复核率','pct'), ('selected_wrong_rate','复核组错误率','pct'),
    ('wrong_case_recall','错题召回率','pct'), ('challenger_gold_rate_wrong','错题定位命中率','pct'),
    ('mi_sr_accuracy','MI-SR准确率','pct'), ('gain_pp','增益','pp'),
    ('corrected','纠正','int'), ('damaged','改错','int'),
    ('all_generic_accuracy','All Generic准确率','pct')])}

## 7. Gate 消融（同一 tau=0.00、同一 Targeted 输出）

{markdown_table(ablation_df, [
    ('scope','数据范围','str'), ('strategy','Gate特征','str'),
    ('review_n','复核数','int'), ('review_rate','复核率','pct'),
    ('selected_wrong_rate','复核组错误率','pct'), ('wrong_case_recall','错题召回率','pct'),
    ('accuracy','准确率','pct'), ('gain_pp','较首答','pp'),
    ('corrected','纠正','int'), ('damaged','改错','int'),
    ('net_corrected','净纠正','int')])}

该消融只检验 MI 是否为 Gate 提供增量信息；Challenger 保持不变。三种消融均按当前二元执行口径计算。

## 8. 阈值灵敏度与统计检验

- Full-100% 事后最高点：`tau={best_full.tau:.2f}`，准确率 `{best_full.accuracy:.2%}`，复核率 `{best_full.review_rate:.2%}`。
- {spec['added_name']} 事后最高点：`tau={best_added.tau:.2f}`，准确率 `{best_added.accuracy:.2%}`，复核率 `{best_added.review_rate:.2%}`。
- 主结果仍采用 `tau=0.00`；完整阈值表见 `threshold_sweep.csv`。

### 8.1 配对检验

{markdown_table(paired_df, [
    ('scope','数据范围','str'), ('comparison','比较','str'),
    ('difference_pp','差值','pp'), ('first_only_correct','前者独有正确','int'),
    ('second_only_correct','后者独有正确','int'),
    ('mcnemar_exact_p','McNemar p','sci'), ('ci_low_pp','聚类CI下界','float'),
    ('ci_high_pp','聚类CI上界','float')])}

### 8.2 同预算随机复核

{markdown_table(random_df, [
    ('scope','数据范围','str'), ('control','对照','str'), ('review_n','复核数','int'),
    ('observed_net_corrected','观察净纠正','int'),
    ('random_mean_net_corrected','随机平均净纠正','float'),
    ('random_net_low','随机95%下界','float'), ('random_net_high','随机95%上界','float'),
    ('random_mean_accuracy','随机平均准确率','pct'),
    ('random_accuracy_low','随机准确率下界','pct'),
    ('random_accuracy_high','随机准确率上界','pct'),
    ('monte_carlo_p','单侧Monte Carlo p','sci')])}

## 9. 可直接用于论文的结论边界

1. 可以声称：该模型上 MI-SR 使用未知标准答案的单题决策路径，在 Targeted/None 部署下改变最终准确率。
2. 可以声称：challenger=gold 与 Targeted 成功强相关，因此 Challenger 是纠正性候选定位的核心操作变量。
3. 可以声称：Gate 的价值需要同时看首答错误富集、复核率、纠正数、改错数和同预算随机对照。
4. 不能声称：All Targeted 或 All Generic 是可部署策略；它们只是反事实对照。
5. 不能把端到端收益全部归因于 MI；MI 的贡献应结合 Gate 消融和可解释路径分别表述。
"""
    report_path = out / spec["report"]
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=MODEL_SPECS, required=True)
    args = parser.parse_args()
    print(analyze(args.model_key))


if __name__ == "__main__":
    main()
