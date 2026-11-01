"""Read-only checks before the U1 rerun. usage: u1_preflight.py --model qwen3_14b_reference"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("U1_ROOT", str(Path(__file__).resolve().parents[2] / "original_experiments"))).resolve()
EXPECTED_BYTES = {"qwen3_14b_reference": 29536665640, "deepseek_r1_qwen14b": 29540133872, "mistral7b_instruct_v03": 14496080928}
PATHS = {"qwen3_14b_reference": os.environ.get("MI_SR_QWEN_MODEL_PATH", "models/Qwen3-14B"), "deepseek_r1_qwen14b": os.environ.get("MI_SR_MODEL_ROOT", "models") + "/DeepSeek-R1-Distill-Qwen-14B", "mistral7b_instruct_v03": os.environ.get("MI_SR_MODEL_ROOT", "models") + "/Mistral-7B-Instruct-v0.3"}
p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--out", default="work/u1_numeric_rerun"); a = p.parse_args()
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


src = ROOT / "output_mi_sr_full_targeted_v5" / a.model / "all_biases"
for name in ["first_pass_biased_generation_store.pkl", "all_candidate_curve_store.pkl", "first_pass_evaluation_labels.csv", "checkpoint_probe_logits.csv", "experiment_plan.csv", "first_pass_prompts_outputs.csv", "binary_deploy_policy.csv", "targeted_all_reevaluation.csv", "deploy_gate_features_before_unblinding.pkl", "run_manifest_full_targeted.json", "mi_sr_peak_features/chunk_checkpoint_probes.csv", "mi_sr_peak_features/peak_probe_plan.csv"]:
    check(f"file {name}", (src / name).exists())
check("question data", (ROOT.parent / "data_clean/questions/US/test_with_bias_irrel.jsonl").exists())
if (src / "run_manifest_full_targeted.json").exists():
    hashes = json.load(open(src / "run_manifest_full_targeted.json"))["frozen_artifact_hashes"]
    for name, want in hashes.items():
        f = ROOT / "frozen_artifacts" / name
        check(f"frozen hash {name}", f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == want)
if (src / "all_candidate_curve_store.pkl").exists():
    n = len(pickle.load(open(src / "all_candidate_curve_store.pkl", "rb")))
    check("stored trajectories", n > 8000, str(n))
weights = Path(PATHS[a.model])
files = list(weights.glob("model-*.safetensors"))
size = sum(f.stat().st_size for f in files)
check("weight bytes match the run manifest", size == EXPECTED_BYTES[a.model], f"{size} at {weights}")
check("config and tokenizer files", (weights / "config.json").exists() and any((weights / n).exists() for n in ("tokenizer.json", "tokenizer.model")))
free = shutil.disk_usage(Path(a.out).parent if not Path(a.out).exists() else a.out).free / 2**30
check("output disk has at least 5 GB free", free >= 5, f"{free:.1f} GB")
try:
    import torch
    import transformers
    check("GPU available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    if torch.cuda.is_available():
        check("GPU memory at least 30 GB (weights take about 27.5 GiB)", torch.cuda.get_device_properties(0).total_memory / 2**30 >= 30, f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
        x = torch.rand(4, 8, 8, dtype=torch.float64, device="cuda") + torch.eye(8, dtype=torch.float64, device="cuda")
        torch.linalg.solve(x, torch.eye(8, dtype=torch.float64, device="cuda").expand(4, -1, -1))
        check("GPU float64 batched solve", True)
    print("INFO versions", "torch", torch.__version__, "transformers", transformers.__version__, flush=True)
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    import run_frozen_holdout  # noqa: F401
    import mi_sr_deploy_v3  # noqa: F401
    import mi_sr_peak_features  # noqa: F401
    check("pipeline modules import", True)
except Exception as e:
    check("environment", False, repr(e)[:300])
print("PREFLIGHT " + ("OK" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
