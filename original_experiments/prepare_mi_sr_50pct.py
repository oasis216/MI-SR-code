#!/usr/bin/env python3
"""Seed isolated 50% Qwen/DeepSeek runs from completed 20% caches."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "output_mi_sr_20pct_v3"
TARGET_ROOT = ROOT / "output_mi_sr_50pct_v4"
MODELS = ("qwen3_14b_reference", "deepseek_r1_qwen14b")
STALE_NAMES = (
    "analysis_v3",
    "run_manifest_v3.json",
    "deploy_policy_manifest_v3.json",
    "stage_log_v3.jsonl",
    "monitor_snapshot_v3.json",
)


def prepare(model_key: str) -> dict:
    source = SOURCE_ROOT / model_key / "all_biases"
    target = TARGET_ROOT / model_key / "all_biases"
    if not source.exists():
        raise FileNotFoundError(source)
    if not target.exists():
        shutil.copytree(source, target)
        state = "copied_20pct_cache"
    else:
        state = "existing_50pct_cache_preserved"
    removed = []
    for name in STALE_NAMES:
        path = target / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(name)
        elif path.exists():
            path.unlink()
            removed.append(name)
    manifest = {
        "model_key": model_key,
        "source": str(source),
        "target": str(target),
        "state": state,
        "stale_analysis_removed": removed,
        "old_raw_feature_and_review_caches_preserved": True,
    }
    (target / "cache_seed_50pct_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=MODELS)
    args = parser.parse_args()
    selected = (args.model_key,) if args.model_key else MODELS
    for model_key in selected:
        print(json.dumps(prepare(model_key), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
