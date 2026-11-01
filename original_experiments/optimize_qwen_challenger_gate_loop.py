"""Versioned Challenger and Gate development on frozen Qwen phase-10 caches.

The script has two deliberately separate stages:

1. ``discovery`` performs grouped out-of-fold model/threshold selection using
   only source questions assigned to Discovery.
2. ``holdout`` loads the frozen Discovery configuration and evaluates it once
   on source questions that were never used for model or threshold selection.

Gold and review outcomes are labels only.  They are excluded from every
deployment feature list.  Splits are grouped by ``q_idx`` so the seven bias
variants of one source question can never cross folds or partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output_mi_sr_full_v2/qwen3_14b_reference/all_biases"
OUTPUT = ROOT / "output_challenger_gate_loop_v20260719"
SEED = 20260719
LETTERS = tuple("ABCDE")
N_DISCOVERY_QUESTIONS = 95
MIN_GATE_REVIEW_RATE = 0.10
MAX_GATE_REVIEW_RATE = 0.50
TARGET_GATE_PRECISION = 0.60
EPSILON = 1e-9

OUTCOME_COLUMNS = {
    "gold",
    "baseline_correct",
    "targeted_correct",
    "generic_correct",
    "mi_sr_correct",
    "challenger_is_gold",
    "raw_challenger_is_gold",
    "target",
    "wrong",
    "outcome_group",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    parameter: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def split_manifest(case_results: pd.DataFrame) -> dict:
    path = OUTPUT / "split_manifest.json"
    questions = np.asarray(sorted(case_results.q_idx.astype(int).unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(questions)
    payload = {
        "seed": SEED,
        "group_column": "q_idx",
        "discovery_q_idx": sorted(
            int(value) for value in questions[:N_DISCOVERY_QUESTIONS]
        ),
        "holdout_q_idx": sorted(
            int(value) for value in questions[N_DISCOVERY_QUESTIONS:]
        ),
        "same_source_question_bias_variants_kept_together": True,
    }
    payload["split_hash"] = json_sha256(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Immutable split mismatch: {path}")
    else:
        write_json(path, payload)
    return payload


def within_case_z(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("case_id")[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std", ddof=0).fillna(0.0)
    return (frame[column] - mean) / (std + EPSILON)


def within_case_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("case_id")[column].rank(
        method="average", ascending=False
    )


def curve_statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"Invalid curve for {prefix}: {values.shape}")
    n = len(values)
    thirds = np.array_split(values, 3)
    late = values[max(0, int(0.75 * n)) :]
    prominence_floor = max(float(np.std(values)) * 0.75, 1e-6)
    peaks, properties = find_peaks(values, prominence=prominence_floor)
    prominences = properties.get("prominences", np.asarray([], dtype=float))
    x = np.linspace(0.0, 1.0, n)
    slope = float(np.polyfit(x, values, 1)[0]) if n > 1 else 0.0
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_auc": float(np.trapz(values, x)),
        f"{prefix}_slope": slope,
        f"{prefix}_early_mean": float(thirds[0].mean()),
        f"{prefix}_middle_mean": float(thirds[1].mean()),
        f"{prefix}_late_mean": float(thirds[2].mean()),
        f"{prefix}_last_quarter_mean": float(late.mean()),
        f"{prefix}_final": float(values[-1]),
        f"{prefix}_peak_count": int(len(peaks)),
        f"{prefix}_peak_prominence_max": (
            float(prominences.max()) if len(prominences) else 0.0
        ),
        f"{prefix}_positive_fraction": float(np.mean(values > 0)),
    }


def load_candidate_features(case_results: pd.DataFrame) -> pd.DataFrame:
    cache = OUTPUT / "candidate_feature_cache.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    peaks = pd.read_csv(SOURCE / "mi_sr_peak_features/candidate_peak_features.csv")
    probes = pd.read_csv(SOURCE / "checkpoint_probe_logits.csv")
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
    meta = case_results[
        [
            "case_id",
            "q_idx",
            "bias_type",
            "gold",
            "model_answer",
            "baseline_correct",
            "targeted_correct",
            "generic_correct",
        ]
    ].copy()
    frame = peaks.merge(
        probe_wide, on=["case_id", "candidate"], validate="one_to_one"
    ).merge(meta, on=["case_id", "q_idx", "bias_type", "model_answer"])
    frame = frame[frame.candidate.ne(frame.model_answer)].copy()

    with (SOURCE / "all_candidate_curve_store.pkl").open("rb") as handle:
        store = pickle.load(handle)
    curve_rows = []
    for row in frame[["case_id", "candidate", "model_answer"]].itertuples(
        index=False
    ):
        candidate_curve = np.asarray(
            store[row.case_id]["curves"][row.candidate], dtype=float
        )
        answer_curve = np.asarray(
            store[row.case_id]["curves"][row.model_answer], dtype=float
        )
        n = min(len(candidate_curve), len(answer_curve))
        candidate_curve = candidate_curve[:n]
        answer_curve = answer_curve[:n]
        item = {"case_id": row.case_id, "candidate": row.candidate}
        item.update(curve_statistics(candidate_curve, "mi_candidate"))
        item.update(curve_statistics(answer_curve, "mi_answer"))
        item.update(
            curve_statistics(candidate_curve - answer_curve, "mi_candidate_minus_answer")
        )
        curve_rows.append(item)
    frame = frame.merge(
        pd.DataFrame(curve_rows), on=["case_id", "candidate"], validate="one_to_one"
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
    frame["target"] = frame.candidate.eq(frame.gold).astype(int)
    frame["wrong"] = (~frame.baseline_correct.astype(bool)).astype(int)
    counts = frame.groupby("case_id").size()
    if not counts.eq(4).all():
        raise RuntimeError(f"Expected four alternatives: {counts.value_counts()}")
    frame.to_pickle(cache)
    return frame


def candidate_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [
        column
        for column in frame.columns
        if column not in OUTCOME_COLUMNS
        and column
        not in {
            "case_id",
            "q_idx",
            "bias_type",
            "candidate",
            "model_answer",
            "peak_span",
        }
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    probe = [
        column
        for column in numeric
        if column.startswith(
            (
                "prob_",
                "logit_",
                "entropy_",
                "log_prob_",
                "z_log_prob_",
                "rank_prob_",
                "probe_",
            )
        )
    ]
    mi = [column for column in numeric if column not in probe]
    for name, columns in {"probe": probe, "mi": mi}.items():
        leaked = OUTCOME_COLUMNS & set(columns)
        if leaked:
            raise RuntimeError(f"Outcome leakage in {name}: {sorted(leaked)}")
    return {
        "probe_only": sorted(probe),
        "mi_only": sorted(mi),
        "probe_plus_mi": sorted(probe + mi),
    }


def build_model(spec: ModelSpec):
    if spec.family == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=spec.parameter,
                class_weight="balanced",
                max_iter=4000,
                random_state=SEED,
            ),
        )
    if spec.family == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=int(spec.parameter),
                max_features=0.7,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
            ),
        )
    if spec.family == "hist_gradient":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=180,
                max_leaf_nodes=int(spec.parameter),
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=SEED,
            ),
        )
    raise ValueError(spec.family)


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("logistic_C0.01", "logistic", 0.01),
        ModelSpec("logistic_C0.1", "logistic", 0.1),
        ModelSpec("logistic_C1", "logistic", 1.0),
        ModelSpec("extra_trees_leaf5", "extra_trees", 5),
        ModelSpec("extra_trees_leaf10", "extra_trees", 10),
        ModelSpec("extra_trees_leaf20", "extra_trees", 20),
        ModelSpec("hist_gradient_leaf7", "hist_gradient", 7),
    ]


def question_folds(question_ids, n_splits: int = 5) -> list[set[int]]:
    questions = np.asarray(sorted({int(value) for value in question_ids}))
    rng = np.random.default_rng(SEED)
    rng.shuffle(questions)
    return [
        {int(value) for value in fold}
        for fold in np.array_split(questions, n_splits)
    ]


def grouped_oof(
    frame: pd.DataFrame,
    columns: list[str],
    target: str,
    spec: ModelSpec,
) -> np.ndarray:
    labels = frame[target].astype(int).to_numpy()
    scores = np.zeros(len(frame), dtype=float)
    for validation_q in question_folds(frame.q_idx):
        validation_mask = frame.q_idx.astype(int).isin(validation_q).to_numpy()
        train_index = np.flatnonzero(~validation_mask)
        validation_index = np.flatnonzero(validation_mask)
        model = build_model(spec)
        model.fit(frame.iloc[train_index][columns], labels[train_index])
        scores[validation_index] = model.predict_proba(
            frame.iloc[validation_index][columns]
        )[:, 1]
    return scores


def top_candidates(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    ranked = frame[
        ["case_id", "q_idx", "candidate", "gold", "model_answer", "target"]
    ].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(
        ["case_id", "score", "candidate"], ascending=[True, False, True]
    )
    ranked["score_rank"] = ranked.groupby("case_id").cumcount() + 1
    top = ranked[ranked.score_rank.eq(1)].copy()
    second = ranked[ranked.score_rank.eq(2)][["case_id", "score"]].rename(
        columns={"score": "second_score"}
    )
    top = top.merge(second, on="case_id", validate="one_to_one")
    top["score_margin"] = top.score - top.second_score
    top["hit"] = top.target.astype(bool)
    return top


def challenger_metrics(name: str, selected: pd.DataFrame) -> dict:
    hits = int(selected.hit.sum())
    total = len(selected)
    return {
        "method": name,
        "n_wrong": total,
        "challenger_gold": hits,
        "challenger_gold_rate": hits / total,
        "random_baseline": 0.25,
        "one_sided_exact_p_vs_25pct": float(
            binomtest(hits, total, 0.25, alternative="greater").pvalue
        ),
    }


def run_challenger_discovery(
    candidates: pd.DataFrame, discovery_q: set[int]
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    wrong = candidates[
        candidates.wrong.eq(1) & candidates.q_idx.isin(discovery_q)
    ].reset_index(drop=True)
    all_discovery = candidates[candidates.q_idx.isin(discovery_q)].reset_index(
        drop=True
    )
    feature_sets = candidate_feature_sets(wrong)
    rows = []
    prediction_store: dict[str, np.ndarray] = {}

    raw_scores = wrong.prob_pre_final.to_numpy(float)
    raw_selected = top_candidates(wrong, raw_scores)
    rows.append(challenger_metrics("v000_raw_prefinal", raw_selected))
    prediction_store["v000_raw_prefinal"] = raw_scores

    for feature_name, columns in feature_sets.items():
        for spec in model_specs():
            method = f"v001_{feature_name}_{spec.name}"
            scores = grouped_oof(wrong, columns, "target", spec)
            prediction_store[method] = scores
            item = challenger_metrics(method, top_candidates(wrong, scores))
            item.update(
                {
                    "feature_set": feature_name,
                    "model_name": spec.name,
                    "model_family": spec.family,
                    "model_parameter": spec.parameter,
                    "n_features": len(columns),
                }
            )
            rows.append(item)

    search = pd.DataFrame(rows).sort_values(
        ["challenger_gold_rate", "n_features", "method"],
        ascending=[False, True, True],
    )
    eligible = search[
        search.method.ne("v000_raw_prefinal")
        & search.one_sided_exact_p_vs_25pct.lt(0.05)
    ]
    if eligible.empty:
        eligible = search[search.method.ne("v000_raw_prefinal")]
    winner_row = eligible.iloc[0]
    winner = str(winner_row.method)
    spec = next(item for item in model_specs() if item.name == winner_row.model_name)
    columns = feature_sets[str(winner_row.feature_set)]
    # Refit within the same grouped folds selected above, but predict every
    # case in each held-out q_idx group.  Training remains restricted to
    # first-pass wrong cases, where exactly one alternative is gold.
    all_scores = np.full(len(all_discovery), np.nan, dtype=float)
    for validation_q in question_folds(discovery_q):
        train_index = wrong.index[
            ~wrong.q_idx.astype(int).isin(validation_q)
        ].to_numpy()
        output_index = all_discovery.index[
            all_discovery.q_idx.astype(int).isin(validation_q)
        ].to_numpy()
        model = build_model(spec)
        model.fit(wrong.iloc[train_index][columns], wrong.iloc[train_index].target)
        all_scores[output_index] = model.predict_proba(
            all_discovery.loc[output_index, columns]
        )[:, 1]
    if np.isnan(all_scores).any():
        raise RuntimeError("Cross-fitted Challenger did not score every Discovery case")
    all_selected = top_candidates(all_discovery, all_scores)
    winner_selected = all_selected[all_selected.case_id.isin(set(wrong.case_id))]
    config = {
        "version": "challenger_v001",
        "method": winner,
        "feature_set": str(winner_row.feature_set),
        "feature_columns": columns,
        "model": {
            "name": spec.name,
            "family": spec.family,
            "parameter": spec.parameter,
        },
        "selection": "highest candidate probability among four alternatives",
        "development_protocol": "5 deterministic source-question folds by q_idx",
        "discovery_metrics": challenger_metrics(winner, winner_selected),
        "holdout_labels_seen": False,
    }
    all_discovery = all_discovery.copy()
    all_discovery["winner_oof_score"] = all_scores
    return search, config, all_selected.merge(
        all_discovery.drop(columns=["gold", "target", "model_answer"]),
        on=["case_id", "q_idx", "candidate"],
        validate="one_to_one",
    )


def load_case_features(
    case_results: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    probes = pd.read_csv(SOURCE / "checkpoint_probe_logits.csv")
    rows = []
    for row in probes.itertuples(index=False):
        probabilities = np.asarray(
            [getattr(row, f"prob_{letter}") for letter in LETTERS], dtype=float
        )
        answer_index = LETTERS.index(str(row.model_answer))
        alternatives = np.delete(probabilities, answer_index)
        sorted_alternatives = np.sort(alternatives)
        rows.append(
            {
                "case_id": row.case_id,
                "checkpoint": row.checkpoint,
                "answer_probability": probabilities[answer_index],
                "best_alternative_probability": alternatives.max(),
                "second_alternative_probability": sorted_alternatives[-2],
                "alternative_gap": (
                    sorted_alternatives[-1] - sorted_alternatives[-2]
                ),
                "candidate_entropy": row.candidate_entropy_normalized,
                "answer_rank": 1 + int(np.sum(probabilities > probabilities[answer_index])),
                "best_alternative_minus_answer": (
                    alternatives.max() - probabilities[answer_index]
                ),
            }
        )
    probe = pd.DataFrame(rows).pivot(index="case_id", columns="checkpoint")
    probe.columns = [f"{name}_{checkpoint}" for name, checkpoint in probe.columns]
    probe = probe.reset_index()

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
        "score",
        "second_score",
        "score_margin",
        "winner_oof_score",
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
    selected_features = selected[selected_columns].rename(
        columns={
            column: f"challenger_{column}"
            for column in selected_columns
            if column not in {"case_id", "candidate"}
        }
    )
    selected_features = selected_features.rename(
        columns={"candidate": "learned_challenger"}
    )
    frame = (
        case_results.merge(probe, on="case_id", validate="one_to_one")
        .merge(aggregate, on="case_id", validate="one_to_one")
        .merge(selected_features, on="case_id", validate="one_to_one")
    )
    for base in (
        "answer_probability",
        "best_alternative_probability",
        "second_alternative_probability",
        "alternative_gap",
        "candidate_entropy",
        "answer_rank",
        "best_alternative_minus_answer",
    ):
        frame[f"{base}_change"] = (
            frame[f"{base}_pre_final"] - frame[f"{base}_prompt_end"]
        )
    frame["wrong"] = (~frame.baseline_correct.astype(bool)).astype(int)
    return frame


def gate_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [
        column
        for column in frame.columns
        if column not in OUTCOME_COLUMNS
        and column
        not in {
            "case_id",
            "q_idx",
            "bias_type",
            "model_answer",
            "raw_prefinal_challenger",
            "ratio_mi_challenger",
            "mi_top2_candidate",
            "selected_peak_span",
            "raw_peak_span",
            "action",
            "learned_challenger",
        }
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    probe = [
        column
        for column in numeric
        if any(
            token in column
            for token in (
                "answer_probability",
                "best_alternative_probability",
                "second_alternative_probability",
                "alternative_gap",
                "candidate_entropy",
                "answer_rank",
                "best_alternative_minus_answer",
                "challenger_score",
                "challenger_second_score",
                "challenger_score_margin",
                "winner_oof_score",
            )
        )
    ]
    mi = [column for column in numeric if column not in probe]
    return {
        "probe_only": sorted(probe),
        "mi_only": sorted(mi),
        "probe_plus_mi": sorted(probe + mi),
    }


def gate_threshold_table(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    thresholds = np.unique(scores)
    for threshold in thresholds:
        selected = scores >= threshold
        n = int(selected.sum())
        if not n:
            continue
        rate = n / len(labels)
        if rate < MIN_GATE_REVIEW_RATE or rate > MAX_GATE_REVIEW_RATE:
            continue
        selected_wrong = int(labels[selected].sum())
        rows.append(
            {
                "threshold": float(threshold),
                "selected_n": n,
                "review_rate": rate,
                "selected_wrong": selected_wrong,
                "selected_error_precision": selected_wrong / n,
                "error_recall": selected_wrong / max(int(labels.sum()), 1),
            }
        )
    return pd.DataFrame(rows)


def run_gate_discovery(
    cases: pd.DataFrame, discovery_q: set[int]
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    discovery = cases[cases.q_idx.isin(discovery_q)].reset_index(drop=True)
    feature_sets = gate_feature_sets(discovery)
    labels = discovery.wrong.astype(int).to_numpy()
    rows = []
    stores: dict[str, tuple[np.ndarray, pd.DataFrame, ModelSpec, list[str]]] = {}
    for feature_name, columns in feature_sets.items():
        for spec in model_specs():
            method = f"gate_v001_{feature_name}_{spec.name}"
            scores = grouped_oof(discovery, columns, "wrong", spec)
            table = gate_threshold_table(labels, scores)
            if table.empty:
                continue
            eligible = table[
                table.selected_error_precision.ge(TARGET_GATE_PRECISION)
            ]
            if eligible.empty:
                chosen = table.sort_values(
                    ["selected_error_precision", "error_recall"],
                    ascending=[False, False],
                ).iloc[0]
            else:
                chosen = eligible.sort_values(
                    ["error_recall", "review_rate"], ascending=[False, False]
                ).iloc[0]
            rows.append(
                {
                    "method": method,
                    "feature_set": feature_name,
                    "model_name": spec.name,
                    "model_family": spec.family,
                    "model_parameter": spec.parameter,
                    "n_features": len(columns),
                    "roc_auc": roc_auc_score(labels, scores),
                    "average_precision": average_precision_score(labels, scores),
                    **chosen.to_dict(),
                }
            )
            stores[method] = (scores, table, spec, columns)
    search = pd.DataFrame(rows).sort_values(
        ["selected_error_precision", "error_recall", "review_rate"],
        ascending=[False, False, False],
    )
    eligible = search[search.selected_error_precision.ge(TARGET_GATE_PRECISION)]
    if eligible.empty:
        winner_row = search.iloc[0]
    else:
        winner_row = eligible.sort_values(
            ["error_recall", "selected_error_precision", "n_features"],
            ascending=[False, False, True],
        ).iloc[0]
    winner = str(winner_row.method)
    scores, threshold_table, spec, columns = stores[winner]
    threshold = float(winner_row.threshold)
    selected = scores >= threshold
    scored = discovery[
        [
            "case_id",
            "q_idx",
            "baseline_correct",
            "targeted_correct",
            "generic_correct",
            "learned_challenger",
            "gold",
        ]
    ].copy()
    scored["oof_wrong_probability"] = scores
    scored["gate_selected"] = selected
    scored["challenger_is_gold"] = scored.learned_challenger.eq(scored.gold)
    scored["selected_action"] = np.where(selected, "targeted", "none")
    baseline = scored.baseline_correct.astype(bool)
    targeted = scored.targeted_correct.astype(bool)
    result = baseline.where(~selected, targeted)
    scored["diagnostic_result_with_cached_old_challenger_review"] = result
    corrected = int((~baseline & result).sum())
    damaged = int((baseline & ~result).sum())
    config = {
        "version": "gate_v001",
        "method": winner,
        "feature_set": str(winner_row.feature_set),
        "feature_columns": columns,
        "model": {
            "name": spec.name,
            "family": spec.family,
            "parameter": spec.parameter,
        },
        "absolute_probability_threshold": threshold,
        "threshold_selection": (
            "maximize error recall subject to >=60% selected error precision "
            "and 10%-50% discovery review rate"
        ),
        "discovery_metrics": {
            key: float(winner_row[key])
            for key in (
                "selected_n",
                "review_rate",
                "selected_wrong",
                "selected_error_precision",
                "error_recall",
                "roc_auc",
                "average_precision",
            )
        },
        "cached_review_diagnostic": {
            "warning": (
                "Targeted cache used the old Ratio-MI challenger; this outcome "
                "is not a valid end-to-end estimate for the learned challenger."
            ),
            "corrected": corrected,
            "damaged": damaged,
            "net": corrected - damaged,
        },
        "holdout_labels_seen": False,
    }
    threshold_table.to_csv(OUTPUT / "gate_v001_threshold_curve.csv", index=False)
    return search, config, scored


def failed_methods_markdown(
    challenger_search: pd.DataFrame, gate_search: pd.DataFrame
) -> str:
    challenger = challenger_search.iloc[1:].copy()
    gate = gate_search.iloc[1:].copy()
    lines = [
        "# 已验证但未冻结的方法",
        "",
        "以下配置均使用同一 Discovery grouped-OOF 口径。记录它们是为了避免后续重复搜索；Holdout 未用于这些方法的选择。",
        "",
        "## Challenger",
        "",
        "| 方法 | 命中/错题 | 命中率 | vs 25% 单侧 p | 未冻结原因 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in challenger.itertuples(index=False):
        reason = "低于本轮最佳配置"
        if row.challenger_gold_rate <= 0.25:
            reason = "未超过随机基线"
        elif row.one_sided_exact_p_vs_25pct >= 0.05:
            reason = "未显著超过随机基线"
        lines.append(
            f"| {row.method} | {int(row.challenger_gold)}/{int(row.n_wrong)} "
            f"| {row.challenger_gold_rate:.2%} "
            f"| {row.one_sided_exact_p_vs_25pct:.4g} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| 方法 | 复核率 | 选中错误率 | 错题召回率 | 未冻结原因 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in gate.itertuples(index=False):
        reason = "低于本轮最佳 Gate 的错误召回或精度"
        if row.selected_error_precision < TARGET_GATE_PRECISION:
            reason = "未达到 60% 错误富集目标"
        lines.append(
            f"| {row.method} | {row.review_rate:.2%} "
            f"| {row.selected_error_precision:.2%} "
            f"| {row.error_recall:.2%} | {reason} |"
        )
    return "\n".join(lines) + "\n"


def discovery_stage() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case_results = pd.read_csv(SOURCE / "analysis/case_level_results.csv")
    split = split_manifest(case_results)
    discovery_q = set(split["discovery_q_idx"])
    candidates = load_candidate_features(case_results)
    challenger_search, challenger_config, selected = run_challenger_discovery(
        candidates, discovery_q
    )
    case_features = load_case_features(case_results, candidates, selected)
    gate_search, gate_config, gate_scores = run_gate_discovery(
        case_features, discovery_q
    )

    challenger_search.to_csv(OUTPUT / "challenger_v001_search.csv", index=False)
    selected.to_csv(OUTPUT / "challenger_v001_discovery_oof.csv", index=False)
    gate_search.to_csv(OUTPUT / "gate_v001_search.csv", index=False)
    gate_scores.to_csv(OUTPUT / "gate_v001_discovery_oof.csv", index=False)
    write_json(OUTPUT / "frozen_challenger_v001.json", challenger_config)
    write_json(OUTPUT / "frozen_gate_v001.json", gate_config)
    (OUTPUT / "failed_methods.md").write_text(
        failed_methods_markdown(challenger_search, gate_search), encoding="utf-8"
    )
    manifest = {
        "stage": "discovery",
        "source": str(SOURCE),
        "source_hashes": {
            name: file_sha256(SOURCE / name)
            for name in (
                "analysis/case_level_results.csv",
                "checkpoint_probe_logits.csv",
                "mi_sr_peak_features/candidate_peak_features.csv",
                "all_candidate_curve_store.pkl",
            )
        },
        "split_hash": split["split_hash"],
        "script_hash": file_sha256(Path(__file__)),
        "holdout_evaluated": False,
        "challenger_winner": challenger_config["method"],
        "gate_winner": gate_config["method"],
    }
    write_json(OUTPUT / "discovery_manifest.json", manifest)
    print("\nChallenger Discovery winner")
    print(json.dumps(challenger_config, indent=2))
    print("\nGate Discovery winner")
    print(json.dumps(gate_config, indent=2))


def spec_from_config(config: dict) -> ModelSpec:
    model = config["model"]
    return ModelSpec(
        name=str(model["name"]),
        family=str(model["family"]),
        parameter=float(model["parameter"]),
    )


def holdout_stage() -> None:
    required = [
        OUTPUT / "frozen_challenger_v001.json",
        OUTPUT / "frozen_gate_v001.json",
        OUTPUT / "discovery_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Run discovery first; missing: {missing}")
    case_results = pd.read_csv(SOURCE / "analysis/case_level_results.csv")
    split = split_manifest(case_results)
    discovery_q = set(split["discovery_q_idx"])
    holdout_q = set(split["holdout_q_idx"])
    challenger_config = json.loads(required[0].read_text(encoding="utf-8"))
    gate_config = json.loads(required[1].read_text(encoding="utf-8"))
    candidates = load_candidate_features(case_results)
    feature_columns = challenger_config["feature_columns"]
    challenger_model = build_model(spec_from_config(challenger_config))
    train_candidates = candidates[
        candidates.wrong.eq(1) & candidates.q_idx.isin(discovery_q)
    ]
    test_candidates = candidates[
        candidates.wrong.eq(1) & candidates.q_idx.isin(holdout_q)
    ].copy()
    challenger_model.fit(train_candidates[feature_columns], train_candidates.target)
    test_scores = challenger_model.predict_proba(test_candidates[feature_columns])[:, 1]
    holdout_selected = top_candidates(test_candidates, test_scores)
    holdout_metrics = challenger_metrics(
        challenger_config["method"], holdout_selected
    )

    # Build selected candidate features for every case.  Wrong-case labels are
    # used only for Challenger evaluation; deployment selection covers all cases.
    all_holdout_candidates = candidates[candidates.q_idx.isin(holdout_q)].copy()
    challenger_model.fit(train_candidates[feature_columns], train_candidates.target)
    all_holdout_scores = challenger_model.predict_proba(
        all_holdout_candidates[feature_columns]
    )[:, 1]
    all_holdout_selected = top_candidates(all_holdout_candidates, all_holdout_scores)
    all_holdout_candidates["winner_oof_score"] = all_holdout_scores
    selected_features = all_holdout_selected.merge(
        all_holdout_candidates.drop(columns=["gold", "target", "model_answer"]),
        on=["case_id", "q_idx", "candidate"],
        validate="one_to_one",
    )
    holdout_cases = load_case_features(
        case_results[case_results.q_idx.isin(holdout_q)],
        candidates[candidates.q_idx.isin(holdout_q)],
        selected_features,
    )
    gate_columns = gate_config["feature_columns"]
    gate_spec = spec_from_config(gate_config)

    # Recreate Discovery OOF-selected candidate features for Gate fitting.
    discovery_selected = pd.read_csv(OUTPUT / "challenger_v001_discovery_oof.csv")
    discovery_cases = load_case_features(
        case_results[case_results.q_idx.isin(discovery_q)],
        candidates[candidates.q_idx.isin(discovery_q)],
        discovery_selected,
    )
    gate_model = build_model(gate_spec)
    gate_model.fit(discovery_cases[gate_columns], discovery_cases.wrong)
    gate_scores = gate_model.predict_proba(holdout_cases[gate_columns])[:, 1]
    selected_gate = gate_scores >= float(gate_config["absolute_probability_threshold"])
    wrong = holdout_cases.wrong.astype(bool).to_numpy()
    selected_wrong = int(np.sum(selected_gate & wrong))
    gate_metrics = {
        "n_cases": len(holdout_cases),
        "selected_n": int(selected_gate.sum()),
        "review_rate": float(selected_gate.mean()),
        "selected_wrong": selected_wrong,
        "selected_error_precision": (
            selected_wrong / max(int(selected_gate.sum()), 1)
        ),
        "error_recall": selected_wrong / max(int(wrong.sum()), 1),
        "wrong_prevalence": float(wrong.mean()),
    }
    output = holdout_cases[
        [
            "case_id",
            "q_idx",
            "gold",
            "model_answer",
            "learned_challenger",
            "baseline_correct",
            "targeted_correct",
            "generic_correct",
        ]
    ].copy()
    output["wrong_probability"] = gate_scores
    output["gate_selected"] = selected_gate
    output["challenger_is_gold"] = output.learned_challenger.eq(output.gold)
    output.to_csv(OUTPUT / "holdout_once_case_results.csv", index=False)
    holdout_selected.to_csv(OUTPUT / "holdout_once_challenger_results.csv", index=False)
    report = {
        "stage": "holdout_once",
        "split_hash": split["split_hash"],
        "challenger": holdout_metrics,
        "gate": gate_metrics,
        "success": {
            "challenger_significantly_above_25pct": bool(
                holdout_metrics["challenger_gold_rate"] > 0.25
                and holdout_metrics["one_sided_exact_p_vs_25pct"] < 0.05
            ),
            "gate_selected_error_precision_near_60pct": bool(
                gate_metrics["selected_error_precision"] >= 0.58
            ),
        },
        "holdout_used_once": True,
        "no_holdout_tuning_permitted": True,
    }
    write_json(OUTPUT / "holdout_once_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("discovery", "holdout"), default="discovery"
    )
    args = parser.parse_args()
    if args.stage == "discovery":
        discovery_stage()
    else:
        holdout_stage()


if __name__ == "__main__":
    main()
