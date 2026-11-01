"""Fit and serialize the already selected Qwen Challenger/Gate components."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from develop_three_way_gate_v2 import TARGETS
from optimize_qwen_challenger_gate_loop import (
    OUTPUT,
    SOURCE,
    ModelSpec,
    build_model,
    file_sha256,
    load_candidate_features,
    load_case_features,
    split_manifest,
    write_json,
)


def model_spec(config: dict) -> ModelSpec:
    model = config["model"]
    return ModelSpec(
        name=str(model["name"]),
        family=str(model["family"]),
        parameter=float(model["parameter"]),
    )


def main() -> None:
    cases = pd.read_csv(SOURCE / "analysis/case_level_results.csv")
    split = split_manifest(cases)
    discovery_q = set(split["discovery_q_idx"])
    candidates = load_candidate_features(cases)

    challenger_config_path = OUTPUT / "frozen_challenger_v001.json"
    challenger_config = json.loads(
        challenger_config_path.read_text(encoding="utf-8")
    )
    challenger_columns = challenger_config["feature_columns"]
    challenger_train = candidates[
        candidates.q_idx.isin(discovery_q) & candidates.wrong.eq(1)
    ]
    challenger = build_model(model_spec(challenger_config))
    challenger.fit(
        challenger_train[challenger_columns], challenger_train.target
    )
    challenger_path = OUTPUT / "frozen_challenger_v001.joblib"
    joblib.dump(challenger, challenger_path)

    selected = pd.read_csv(OUTPUT / "challenger_v001_discovery_oof.csv")
    discovery_cases = load_case_features(
        cases[cases.q_idx.isin(discovery_q)],
        candidates[candidates.q_idx.isin(discovery_q)],
        selected,
    ).reset_index(drop=True)
    gate_config_path = OUTPUT / "frozen_gate_v002.json"
    gate_config = json.loads(gate_config_path.read_text(encoding="utf-8"))
    gate_columns = gate_config["feature_columns"]
    gate_models = {}
    for target in TARGETS:
        model = build_model(model_spec(gate_config))
        model.fit(discovery_cases[gate_columns], discovery_cases[target])
        gate_models[target] = model
    gate_path = OUTPUT / "frozen_gate_v002_models.joblib"
    joblib.dump(gate_models, gate_path)

    manifest = {
        "split_hash": split["split_hash"],
        "challenger": {
            "config": challenger_config_path.name,
            "config_sha256": file_sha256(challenger_config_path),
            "artifact": challenger_path.name,
            "artifact_sha256": file_sha256(challenger_path),
            "training_wrong_cases": int(challenger_train.case_id.nunique()),
            "feature_count": len(challenger_columns),
        },
        "gate": {
            "config": gate_config_path.name,
            "config_sha256": file_sha256(gate_config_path),
            "artifact": gate_path.name,
            "artifact_sha256": file_sha256(gate_path),
            "training_cases": len(discovery_cases),
            "targets": list(TARGETS),
            "feature_count": len(gate_columns),
        },
        "gold_required_at_inference": False,
        "outcomes_used_only_for_discovery_fitting": True,
    }
    write_json(OUTPUT / "frozen_artifact_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
