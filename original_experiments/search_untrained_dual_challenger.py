"""Search a training-free three-stage/MI challenger selector on cached runs."""

from __future__ import annotations

import pickle
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
SOURCES = (
    ("offset00", "discovery", ROOT / "output_mi_aligned_confidence_leap_v1"),
    ("offset20", "discovery", ROOT / "output_mi_aligned_confidence_leap_holdout"),
    ("offset40", "discovery", ROOT / "output_mi_aligned_confidence_leap_holdout2"),
    ("offset60", "validation", ROOT / "output_mi_aligned_confidence_leap_holdout3"),
    ("offset80", "validation", ROOT / "output_mi_aligned_confidence_leap_holdout4"),
)
OUTPUT = ROOT / "output_untrained_dual_challenger_v1"
MODEL = Path(os.environ.get("MI_SR_QWEN_MODEL_PATH", ROOT / "models/Qwen3-14B"))
POS = re.compile(
    r"\b(likely|supports?|consistent|suggests?|indicated|plausible|best|classic|"
    r"hallmark|therefore|thus|appropriate)\b",
    re.I,
)
NEG = re.compile(
    r"\b(not|unlikely|less likely|does not|do not|cannot|no evidence|against|"
    r"rules? out|inconsistent|rather than)\b",
    re.I,
)
STOP = {
    "this", "that", "with", "from", "most", "appropriate", "treatment",
    "management", "patient", "disease", "syndrome", "option", "following",
}


def words(text: str) -> set[str]:
    return {
        word.lower() for word in re.findall(r"[A-Za-z]{4,}", str(text))
        if word.lower() not in STOP
    }


def option_anchor(span: str, candidate: str, option_text: str) -> bool:
    if re.search(rf"(?:\boption\s+{candidate}\b|\({candidate}\)|\b{candidate}\s*[\.:)])", span, re.I):
        return True
    option_words = words(option_text)
    overlap = option_words & words(span)
    return bool(option_words) and len(overlap) >= min(2, len(option_words))


def find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    for index in range(len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None


def zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("case_id")[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").fillna(0.0)
    return (frame[column] - mean) / (std + 1e-8)


def load_batch(batch: str, split: str, directory: Path, tokenizer) -> pd.DataFrame:
    labels = pd.read_csv(directory / "generation_labels.csv")
    labels = labels[labels.complete.astype(bool)].copy()
    features = pd.read_csv(directory / "candidate_chunk_features.csv")
    probes = pd.read_csv(directory / "chunk_checkpoint_probes.csv")
    plan = pd.read_csv(directory / "chunk_probe_plan.csv")
    with (directory / "generation_store.pkl").open("rb") as handle:
        store = pickle.load(handle)
    with (directory / "full_option_curve_store.pkl").open("rb") as handle:
        curves = pickle.load(handle)
    outputs = pd.read_csv(directory / "generation_outputs.csv").set_index("pilot_case_id")
    option_maps = {
        case_id: dict(re.findall(r"^([A-E])\.\s+(.+)$", str(row.prompt), re.M))
        for case_id, row in outputs.iterrows()
    }

    mi = features[features.branch.eq("mi")].copy()
    alternatives = mi[mi.candidate.ne(mi.model_answer)].copy()
    alternatives = alternatives.rename(columns={"pilot_case_id": "case_id"})
    prompt = probes[probes.step.eq(-1)].set_index("pilot_case_id")
    comparison_marker = tokenizer.encode("Comparison:", add_special_tokens=False)
    comparison = {}
    for case_id in alternatives.case_id.unique():
        generated = [int(value) for value in store[case_id]["gen_ids"]]
        marker = find_subsequence(generated, comparison_marker)
        common = plan[
            plan.pilot_case_id.eq(case_id) & plan.request_role.eq("common")
        ].step.astype(int)
        common = sorted(common[common.ge(0)].unique())
        target = (marker - 1) if marker is not None else common[len(common) // 2]
        eligible = [step for step in common if step <= target]
        comparison[case_id] = max(eligible) if eligible else min(common)
    probe_lookup = probes.set_index(["pilot_case_id", "step"])
    alternatives["probe_prompt"] = [
        float(prompt.loc[cid, f"prob_{candidate}"])
        for cid, candidate in zip(alternatives.case_id, alternatives.candidate)
    ]
    alternatives["probe_comparison"] = [
        float(probe_lookup.loc[(cid, comparison[cid]), f"prob_{candidate}"])
        for cid, candidate in zip(alternatives.case_id, alternatives.candidate)
    ]
    alternatives["probe_final"] = alternatives.support_final.astype(float)
    alternatives["mi_semantic"] = [
        bool(POS.search(str(span))) and not bool(NEG.search(str(span)))
        for span in alternatives.chunk_span
    ]
    alternatives["mi_anchor"] = [
        option_anchor(span, candidate, option_maps[cid].get(candidate, ""))
        for cid, candidate, span in zip(
            alternatives.case_id, alternatives.candidate, alternatives.chunk_span
        )
    ]
    pair_rows = []
    for row in alternatives.itertuples(index=False):
        candidate_curve = np.asarray(curves[row.case_id]["curves"][row.candidate], dtype=float)
        answer_curve = np.asarray(curves[row.case_id]["curves"][row.model_answer], dtype=float)
        length = min(len(candidate_curve), len(answer_curve))
        diff = candidate_curve[:length] - answer_curve[:length]
        start = min(10, max(0, length - 1))
        active = diff[start:]
        peak = int(np.argmax(active) + start)
        late = diff[max(start, int(length * 0.75)) :]
        gen_ids = store[row.case_id]["gen_ids"]
        token_step = int(round(peak * (len(gen_ids) - 1) / max(length - 1, 1)))
        span = tokenizer.decode(
            gen_ids[max(0, token_step - 10) : min(len(gen_ids), token_step + 11)],
            skip_special_tokens=True,
        )
        pair_rows.append(
            {
                "case_id": row.case_id,
                "candidate": row.candidate,
                "mi_diff_max": float(active.max()),
                "mi_diff_late": float(late.mean()) if len(late) else float(active[-1]),
                "mi_diff_auc": float(np.maximum(active, 0).mean()),
                "mi_diff_final": float(active[-1]),
                "mi_diff_anchor": option_anchor(
                    span, row.candidate, option_maps[row.case_id].get(row.candidate, "")
                ),
                "mi_diff_semantic": bool(POS.search(span)) and not bool(NEG.search(span)),
            }
        )
    alternatives = alternatives.merge(
        pd.DataFrame(pair_rows), on=["case_id", "candidate"], validate="one_to_one"
    )

    controls = features[features.branch.isin(["uniform", "random"])].pivot_table(
        index=["pilot_case_id", "candidate"],
        columns="branch",
        values="positive_persistent_leap",
        aggfunc="first",
    )
    alternatives = alternatives.merge(
        controls,
        left_on=["case_id", "candidate"],
        right_index=True,
        validate="one_to_one",
    )
    alternatives["mi_distinctive"] = alternatives.positive_persistent_leap.gt(
        alternatives[["uniform", "random"]].max(axis=1)
    )
    alternatives["batch"] = batch
    alternatives["split"] = split
    alternatives["q_idx"] = alternatives.case_id.map(
        labels.set_index("pilot_case_id").q_idx
    )
    alternatives["correct"] = alternatives.case_id.map(
        labels.set_index("pilot_case_id").correct
    ).astype(bool)
    for column in (
        "probe_prompt",
        "probe_comparison",
        "probe_final",
        "positive_persistent_leap",
        "peak_prominence",
        "confidence_leap",
        "persistent_support",
        "mi_diff_max",
        "mi_diff_late",
        "mi_diff_auc",
        "mi_diff_final",
    ):
        alternatives[f"z_{column}"] = zscore(alternatives, column)
    alternatives["probe_borda"] = sum(
        alternatives.groupby("case_id")[column].rank(pct=True)
        for column in ("probe_prompt", "probe_comparison", "probe_final")
    )
    alternatives["probe_votes"] = sum(
        alternatives[column].eq(alternatives.groupby("case_id")[column].transform("max"))
        for column in ("probe_prompt", "probe_comparison", "probe_final")
    )
    alternatives["probe_growth"] = (
        alternatives.probe_final - alternatives.probe_prompt
    )
    alternatives["mi_directional"] = (
        alternatives.z_positive_persistent_leap
        + 0.5 * alternatives.z_peak_prominence
        + alternatives.mi_semantic.astype(float)
        + 0.5 * alternatives.mi_distinctive.astype(float)
    )
    alternatives["mi_grounded"] = (
        alternatives.z_positive_persistent_leap
        + 0.5 * alternatives.z_peak_prominence
        + 1.5 * alternatives.mi_anchor.astype(float)
        + 0.5 * alternatives.mi_semantic.astype(float)
        + 0.5 * alternatives.mi_distinctive.astype(float)
    )
    alternatives["mi_pairwise"] = (
        alternatives.z_mi_diff_max
        + alternatives.z_mi_diff_late
        + alternatives.z_mi_diff_auc
        + 0.5 * alternatives.z_mi_diff_final
        + 0.5 * alternatives.mi_diff_anchor.astype(float)
        + 0.25 * alternatives.mi_diff_semantic.astype(float)
    )
    alternatives["local_case_id"] = alternatives.case_id
    alternatives["case_id"] = batch + ":" + alternatives.case_id.astype(str)
    return alternatives


def choose(frame: pd.DataFrame, score: pd.Series, method: str) -> pd.DataFrame:
    current = frame[["case_id", "batch", "split", "q_idx", "gold", "candidate"]].copy()
    current["score"] = score
    current = current.sort_values(
        ["case_id", "score", "candidate"], ascending=[True, False, True]
    ).drop_duplicates("case_id")
    current["method"] = method
    current["hit"] = current.candidate.eq(current.gold)
    return current


def score_methods(frame: pd.DataFrame) -> dict[str, pd.Series]:
    probe_borda = frame.probe_borda + 0.25 * frame.z_probe_final
    probe_stable = (
        frame.probe_votes + 0.5 * frame.probe_borda + 0.25 * zscore(frame, "probe_growth")
    )
    mi = frame.mi_directional
    grounded = frame.mi_grounded
    pairwise = frame.mi_pairwise
    final_top2 = frame.groupby("case_id").probe_final.rank(
        method="first", ascending=False
    ).le(2)
    borda_top2 = probe_borda.groupby(frame.case_id).rank(
        method="first", ascending=False
    ).le(2)
    return {
        "prefinal": frame.probe_final,
        "probe_borda": probe_borda,
        "probe_stable": probe_stable,
        "mi_only": mi,
        "mi_grounded": grounded,
        "mi_pairwise": pairwise,
        "dual_prefinal_top2_mi": grounded.where(final_top2, -1e9),
        "dual_borda_top2_mi": grounded.where(borda_top2, -1e9),
        "dual_025": zscore(frame.assign(_p=probe_stable), "_p") + 0.25 * mi,
        "dual_050": zscore(frame.assign(_p=probe_stable), "_p") + 0.50 * mi,
        "dual_100": zscore(frame.assign(_p=probe_stable), "_p") + mi,
        "dual_grounded_025": zscore(frame.assign(_p=probe_stable), "_p") + 0.25 * grounded,
        "dual_grounded_050": zscore(frame.assign(_p=probe_stable), "_p") + 0.50 * grounded,
        "dual_pairwise_010": zscore(frame.assign(_p=probe_stable), "_p") + 0.10 * pairwise,
        "dual_pairwise_025": zscore(frame.assign(_p=probe_stable), "_p") + 0.25 * pairwise,
        "dual_pairwise_050": zscore(frame.assign(_p=probe_stable), "_p") + 0.50 * pairwise,
    }


def consensus_selector(frame: pd.DataFrame, probe: pd.Series, mi: pd.Series) -> pd.DataFrame:
    probe_choice = choose(frame, probe, "probe").set_index("case_id")
    mi_choice = choose(frame, mi, "mi").set_index("case_id")
    rows = []
    for case_id, group in frame.groupby("case_id", sort=False):
        p = str(probe_choice.loc[case_id, "candidate"])
        m = str(mi_choice.loc[case_id, "candidate"])
        mi_row = group[group.candidate.eq(m)].iloc[0]
        probe_sorted = group.assign(_score=probe.loc[group.index]).sort_values(
            "_score", ascending=False
        )
        probe_margin = float(probe_sorted._score.iloc[0] - probe_sorted._score.iloc[1])
        reliable_mi = bool(
            mi_row.mi_anchor
            and mi_row.mi_semantic
            and mi_row.mi_distinctive
            and mi_row.confidence_leap > 0
        )
        candidate = p if p == m or not (reliable_mi and probe_margin < 0.35) else m
        source = "agreement" if p == m else ("mi_override" if candidate == m else "probe_fallback")
        row = group[group.candidate.eq(candidate)].iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "batch": row.batch,
                "split": row.split,
                "q_idx": row.q_idx,
                "gold": row.gold,
                "candidate": candidate,
                "method": "dual_consensus",
                "source": source,
                "hit": candidate == row.gold,
            }
        )
    return pd.DataFrame(rows)


def summarize(selections: pd.DataFrame) -> pd.DataFrame:
    return selections.groupby(["split", "method"], as_index=False).agg(
        n=("case_id", "size"), gold=("hit", "sum"), accuracy=("hit", "mean")
    ).sort_values(["split", "accuracy"], ascending=[True, False])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    features = pd.concat(
        [load_batch(batch, split, directory, tokenizer) for batch, split, directory in SOURCES],
        ignore_index=True,
    )
    features = features[~features.correct.astype(bool)].copy()
    scores = score_methods(features)
    selections = [choose(features, score, method) for method, score in scores.items()]
    selections.append(
        consensus_selector(features, scores["probe_stable"], scores["mi_grounded"])
    )
    selected = pd.concat(selections, ignore_index=True)
    expected = features.groupby("split").case_id.nunique().to_dict()
    coverage = selected.groupby(["split", "method"]).case_id.nunique()
    for (split, method), count in coverage.items():
        if int(count) != int(expected[split]):
            raise RuntimeError(
                f"Incomplete selector coverage {split}/{method}: {count}/{expected[split]}"
            )
    summary = summarize(selected)

    development = summary[summary.split.eq("discovery")]
    mi_methods = development[development.method.str.startswith(("dual_", "dual_consensus"))]
    winner = str(mi_methods.sort_values(["accuracy", "method"], ascending=[False, True]).iloc[0].method)
    validation = selected[selected.split.eq("validation")]
    wide = validation.pivot(index="case_id", columns="method", values="hit")
    comparisons = []
    for comparator in ("prefinal", "probe_borda", "probe_stable", "mi_only"):
        a = wide[winner].astype(bool)
        b = wide[comparator].astype(bool)
        a_only = int((a & ~b).sum())
        b_only = int((~a & b).sum())
        comparisons.append(
            {
                "winner": winner,
                "comparator": comparator,
                "winner_only": a_only,
                "comparator_only": b_only,
                "p": float(binomtest(a_only, a_only + b_only, 0.5).pvalue)
                if a_only + b_only
                else 1.0,
            }
        )
    features.to_csv(OUTPUT / "candidate_features.csv", index=False)
    selected.to_csv(OUTPUT / "challenger_selections.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "validation_tests.csv", index=False)
    (OUTPUT / "frozen_winner.txt").write_text(winner + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("frozen winner:", winner)
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
