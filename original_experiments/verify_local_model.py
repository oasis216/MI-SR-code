"""Verify that a local Hugging Face model snapshot contains all weight shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(model_dir: Path, min_weight_gib: float) -> dict:
    result = {
        "model_dir": str(model_dir),
        "config_present": (model_dir / "config.json").is_file(),
        "tokenizer_present": any(
            (model_dir / name).is_file()
            for name in ("tokenizer.json", "tokenizer.model", "spiece.model")
        ),
        "index_file": None,
        "expected_weight_files": [],
        "missing_weight_files": [],
        "weight_bytes": 0,
        "min_weight_gib": min_weight_gib,
        "complete": False,
    }
    index_paths = [
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    ]
    index_path = next((path for path in index_paths if path.is_file()), None)
    if index_path is not None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        expected = sorted(set(payload.get("weight_map", {}).values()))
        result["index_file"] = index_path.name
    else:
        expected = [
            path.name
            for path in sorted(model_dir.glob("*.safetensors"))
            + sorted(model_dir.glob("pytorch_model*.bin"))
        ]
    result["expected_weight_files"] = expected
    missing = [
        name
        for name in expected
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
    ]
    result["missing_weight_files"] = missing
    result["weight_bytes"] = sum(
        (model_dir / name).stat().st_size
        for name in expected
        if (model_dir / name).is_file()
    )
    enough_bytes = result["weight_bytes"] >= min_weight_gib * 1024**3
    result["complete"] = bool(
        result["config_present"]
        and result["tokenizer_present"]
        and expected
        and not missing
        and enough_bytes
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--min-weight-gib", type=float, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = verify(args.model_dir, args.min_weight_gib)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
