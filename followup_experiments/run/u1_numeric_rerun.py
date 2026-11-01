"""U1 / R1+R4: recompute trajectories from the stored first-pass sequences, without regenerating anything.

Variants written side by side under <out>/<variant>/<model>/all_biases:
  f32_rerun  the submitted float32 K@H arithmetic on this machine (environment baseline)
  f64_hkh    float64 HKH, the paper's formula
  f32_hkh    float32 HKH, a cheap alternative

Then each variant goes through the submission's own peak-feature and frozen-policy code. The original outputs
are read only. Challenger and gate artifacts stay frozen. usage:
  u1_numeric_rerun.py --model qwen3_14b_reference --out <dir> [--limit N] [--stage curves|peaks|policy|all]
"""
import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("U1_ROOT", str(Path(__file__).resolve().parents[2] / "original_experiments"))).resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import run_frozen_holdout as runner  # noqa: E402
from cross_model_runtime import load_model_and_tokenizer  # noqa: E402
from mi_sr_deploy_v3 import build_deploy_policy  # noqa: E402
from mi_sr_peak_features import extract_peak_features  # noqa: E402

mi_exp = runner.mi_exp
VARIANTS = ("f32_rerun", "f64_hkh", "f32_hkh")
MAX_STEPS, MAX_DIM, SIGMA, EPS = 256, 128, 50.0, 1e-5


def kernel(values, dtype):
    values = values.to(dtype)
    squared = (values * values).unsqueeze(-1)
    distances = torch.abs(squared - 2 * values.unsqueeze(-1) * values.unsqueeze(-2) + squared.transpose(-1, -2))
    return torch.exp(-distances / (2 * SIGMA * SIGMA))


def regularized(values, dtype, center_mode):
    """R(K) for a batch of vectors (n, m). center_mode 'kh' is the submitted form, 'hkh' the paper's."""
    m = values.shape[-1]
    device = values.device
    center = torch.eye(m, dtype=dtype, device=device) - torch.ones(m, m, dtype=dtype, device=device) / m
    ident = torch.eye(m, dtype=dtype, device=device)
    k = kernel(values, dtype).to(dtype)
    centered = k @ center if center_mode == "kh" else center @ k @ center
    return centered @ torch.linalg.solve(centered + EPS * m * ident.expand(len(values), -1, -1), ident.expand(len(values), -1, -1))


def score_all(x, ys, dtype, center_mode, chunk):
    """Trajectory per candidate; the step side is solved once and shared."""
    ry = regularized(ys, dtype, center_mode)
    out = {i: [] for i in range(len(ys))}
    for part in x.split(chunk):
        rx = regularized(part, dtype, center_mode)
        for i in range(len(ys)):
            out[i].append((rx * ry[i].T).sum(dim=(-2, -1)))
    return {i: torch.cat(v) for i, v in out.items()}


def reduce_dim(hidden, rep):
    d = min(hidden.shape[1], rep.shape[0])
    x, y = hidden[:, :d], rep[:d]
    if d > MAX_DIM:
        idx = np.linspace(0, d - 1, MAX_DIM, dtype=int)
        x, y = x[:, idx], y[idx]
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y, dtype=np.float32)


def shadow_dir(out, variant, model_key):
    return Path(out) / variant / model_key / "all_biases"


def prepare_dirs(out, source, model_key):
    shared = Path(out) / "_shared_probes" / model_key
    shared.mkdir(parents=True, exist_ok=True)
    probes = shared / "chunk_checkpoint_probes.csv"
    if not probes.exists():
        shutil.copy(source / "mi_sr_peak_features/chunk_checkpoint_probes.csv", probes)
    for variant in VARIANTS:
        d = shadow_dir(out, variant, model_key)
        (d / "mi_sr_peak_features").mkdir(parents=True, exist_ok=True)
        for name in ("first_pass_evaluation_labels.csv", "checkpoint_probe_logits.csv", "first_pass_biased_generation_store.pkl", "experiment_plan.csv"):
            if not (d / name).exists():
                (d / name).symlink_to(source / name)
        link = d / "mi_sr_peak_features/chunk_checkpoint_probes.csv"
        if not link.exists():
            link.symlink_to(probes)


def stage_curves(model, tokenizer, source, out, model_key, limit):
    plan = pd.read_csv(source / "experiment_plan.csv").set_index("case_id")
    gen = pickle.load(open(source / "first_pass_biased_generation_store.pkl", "rb"))
    stored = pickle.load(open(source / "all_candidate_curve_store.pkl", "rb"))
    ids = list(stored)
    if limit:
        rng = np.random.default_rng(20261031)
        ids = sorted(rng.choice(ids, limit, replace=False).tolist())
        if model_key == "qwen3_14b_reference" and "q0170_b6" in stored and "q0170_b6" not in ids:
            ids[0] = "q0170_b6"
    stores = {}
    for v in VARIANTS:
        path = shadow_dir(out, v, model_key) / "all_candidate_curve_store.pkl"
        stores[v] = pickle.load(open(path, "rb")) if path.exists() else {}
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    cache = mi_exp.AnswerRepCache(model, tokenizer, "full_option")
    device = next(model.parameters()).device
    t0 = time.time(); checked = 0
    pending = [c for c in ids if c not in stores["f32_rerun"]]
    for n, case_id in enumerate(pending, 1):
        record = runner.record_for_plan(questions, plan.loc[case_id].rename(case_id))
        hidden = mi_exp.hidden_seq(model, gen[case_id]["seq"], gen[case_id]["prompt_len"], MAX_STEPS)
        letters = [k for k in stored[case_id]["curves"]]
        reps = [cache.get(mi_exp.target_text(record, k, "full_option")) for k in letters]
        pairs = [reduce_dim(hidden, r) for r in reps]
        x = torch.from_numpy(pairs[0][0])
        ys = torch.from_numpy(np.stack([p[1] for p in pairs]))
        with torch.no_grad():
            a = score_all(x, ys, torch.float32, "kh", 64)
            b = score_all(x.to(device), ys.to(device), torch.float64, "hkh", 512)
            c = score_all(x.to(device), ys.to(device), torch.float32, "hkh", 512)
        if checked < 5:
            # The reused step side must match the submission's own function exactly.
            for i, k in enumerate(letters):
                own = mi_exp.estimate_mi_curve(hidden, reps[i], MAX_DIM, SIGMA)
                assert np.array_equal(own, a[i].numpy()), (case_id, k, np.abs(own - a[i].numpy()).max())
            checked += 1
        for v, res in (("f32_rerun", a), ("f64_hkh", b), ("f32_hkh", c)):
            curves = {k: res[i].detach().cpu().numpy() for i, k in enumerate(letters)}
            stores[v][case_id] = {"curves": curves, "support": runner.candidate_support_scores(curves), "model_answer": gen[case_id]["model_answer"]}
        if n % 500 == 0 or n == len(pending):
            for v in VARIANTS:
                runner.save_pickle(stores[v], shadow_dir(out, v, model_key) / "all_candidate_curve_store.pkl")
            print(f"curves {n}/{len(pending)} {time.time() - t0:.0f}s", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    print("curves done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stage", default="all")
    a = p.parse_args()
    source = ROOT / "output_mi_sr_full_targeted_v5" / a.model / "all_biases"
    prepare_dirs(a.out, source, a.model)
    torch.set_num_threads(int(os.environ.get("U1_THREADS", "20")))
    model = tokenizer = None
    if a.stage in ("all", "curves", "peaks"):
        model, tokenizer, meta = load_model_and_tokenizer(a.model)
        print(json.dumps({k: meta[k] for k in ("dtype", "attention_backend")}), flush=True)
    if a.stage in ("all", "curves"):
        stage_curves(model, tokenizer, source, a.out, a.model, a.limit)
    if a.stage in ("all", "peaks"):
        for v in VARIANTS:
            t0 = time.time()
            extract_peak_features(model, tokenizer, shadow_dir(a.out, v, a.model), force=True, preserve_existing_probes=True)
            print(f"peaks {v} {time.time() - t0:.0f}s", flush=True)
    if a.stage in ("all", "policy"):
        for v in VARIANTS:
            policy, _ = build_deploy_policy(shadow_dir(a.out, v, a.model))
            print(f"policy {v} {len(policy)}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
