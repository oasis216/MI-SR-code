"""Freeze Probe-only and MI-only Gate v2 ablations from Qwen Discovery."""

from __future__ import annotations

import json

import joblib

from develop_three_way_gate_v2 import MODEL_SPEC, TARGETS, load_discovery
from optimize_qwen_challenger_gate_loop import (
    OUTPUT,
    build_model,
    file_sha256,
    gate_feature_sets,
    write_json,
)


def main() -> None:
    discovery, _, split = load_discovery()
    feature_sets = gate_feature_sets(discovery)
    frozen = {}
    manifest = {
        "version": "gate_v002_minimal_feature_ablation",
        "split_hash": split["split_hash"],
        "model": {
            "name": MODEL_SPEC.name,
            "family": MODEL_SPEC.family,
            "parameter": MODEL_SPEC.parameter,
        },
        "targets": list(TARGETS),
        "absolute_action_margin": 0.08,
        "feature_sets": {},
    }
    for feature_set in ("probe_only", "mi_only", "probe_plus_mi"):
        columns = feature_sets[feature_set]
        frozen[feature_set] = {}
        for target in TARGETS:
            model = build_model(MODEL_SPEC)
            model.fit(discovery[columns], discovery[target])
            frozen[feature_set][target] = model
        manifest["feature_sets"][feature_set] = columns
    artifact = OUTPUT / "frozen_gate_v002_ablation_models.joblib"
    joblib.dump(frozen, artifact)
    manifest["artifact"] = artifact.name
    manifest["artifact_sha256"] = file_sha256(artifact)
    manifest["training_cases"] = len(discovery)
    write_json(OUTPUT / "frozen_gate_v002_ablation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
