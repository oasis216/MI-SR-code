"""D-6 / R2: controlled per-question timing on one GPU, serial, with no cache carried between questions.

Runs each stage of the submitted implementation on a fixed sample and records wall-clock time, tokens and
peak memory. Model loading and warm-up questions are excluded. Original outputs are read only.
usage: u1_timing_benchmark.py --model qwen3_14b_reference --out <dir> [--n 100]
"""
import argparse
import json
import os
import pickle
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("U1_ROOT", str(Path(__file__).resolve().parents[2] / "original_experiments"))).resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import joblib  # noqa: E402
import run_frozen_holdout as runner  # noqa: E402
from cross_model_runtime import load_model_and_tokenizer  # noqa: E402
from final_answer_logits import build_checkpoint_prefixes  # noqa: E402

mi_exp = runner.mi_exp
MAX_NEW = {"qwen3_14b_reference": 384, "deepseek_r1_qwen14b": 640, "mistral7b_instruct_v03": 384}
PROBE_TEXT = "\nProvisional Answer:"
WARMUP = 3


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return value, time.perf_counter() - start


def forward_probes(model, prefixes, probe_ids, pad, batch_size=4):
    """Same left-padded, cache-free probe forward as the submitted peak and endpoint stages."""
    device = next(model.parameters()).device
    tokens = 0
    for start in range(0, len(prefixes), batch_size):
        current = [torch.cat([p, probe_ids]) for p in prefixes[start:start + batch_size]]
        width = max(len(s) for s in current)
        ids = torch.full((len(current), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(current):
            ids[i, -len(s):] = s.to(device)
            mask[i, -len(s):] = 1
            tokens += len(s)
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits[:, -1].float()
    return tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    a = p.parse_args()
    src = ROOT / "output_mi_sr_full_targeted_v5" / a.model / "all_biases"
    art = ROOT / "frozen_artifacts"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("U1_THREADS", "12")))

    gen = pickle.load(open(src / "first_pass_biased_generation_store.pkl", "rb"))
    prompts = pd.read_csv(src / "first_pass_prompts_outputs.csv", encoding="utf-8-sig")
    prompts = prompts[prompts.condition == "biased"].set_index("case_id")
    plan = pd.read_csv(src / "experiment_plan.csv").set_index("case_id")
    policy = pd.read_csv(src / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    review = pd.read_csv(src / "targeted_all_reevaluation.csv", encoding="utf-8-sig").set_index("case_id")
    peaks = pd.read_csv(src / "mi_sr_peak_features/peak_probe_plan.csv", encoding="utf-8-sig")
    features = pickle.load(open(src / "deploy_gate_features_before_unblinding.pkl", "rb")).set_index("case_id")
    gate_cols = json.load(open(art / "frozen_gate_v002.json"))["feature_columns"]
    gate = joblib.load(art / "frozen_gate_v002_models.joblib")
    ids = sorted(set(policy.index) & set(review.index) & set(prompts.index) & set(gen))
    rng = np.random.default_rng(20261031)
    sample = rng.choice(ids, min(a.n + WARMUP, len(ids)), replace=False).tolist()

    model, tokenizer, meta = load_model_and_tokenizer(a.model)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    probe_ids = torch.tensor(tokenizer.encode(PROBE_TEXT, add_special_tokens=False), dtype=torch.long)
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for n, case_id in enumerate(sample):
        g = gen[case_id]
        row = {"case_id": case_id, "selected": bool(policy.at[case_id, "review_score"] > 0)}
        output, row["first_pass_s"] = timed(lambda: mi_exp.generate_once(model, tokenizer, prompts.at[case_id, "prompt"], MAX_NEW[a.model]))
        row["first_pass_new_tokens"] = len(output[2])
        row["first_pass_stored_tokens"] = len(g["gen_ids"])
        del output
        hidden, row["teacher_forced_s"] = timed(lambda: mi_exp.hidden_seq(model, g["seq"], g["prompt_len"], 256))
        row["teacher_forced_tokens"] = int(g["seq"].numel())
        record = runner.record_for_plan(questions, plan.loc[case_id].rename(case_id))
        cache = mi_exp.AnswerRepCache(model, tokenizer, "full_option")  # fresh per question: uncached cost
        letters = [k for k in "ABCDE" if k in record["options"]]
        reps, row["option_encoding_s"] = timed(lambda: [cache.get(mi_exp.target_text(record, k, "full_option")) for k in letters])
        _, row["trajectories_s"] = timed(lambda: [mi_exp.estimate_mi_curve(hidden, r, 128, 50.0) for r in reps])
        del hidden, reps
        prefixes = build_checkpoint_prefixes(g, tokenizer)
        sequence = g["seq"].detach().cpu().reshape(-1)
        ends = [prefixes["prompt_end"], prefixes.get("pre_final", sequence)]
        row["endpoint_tokens"], row["endpoint_s"] = timed(lambda: forward_probes(model, ends, probe_ids, pad))
        steps = sorted(set(peaks[peaks.case_id == case_id][["before_step", "after_step", "next_step"]].to_numpy().ravel().tolist()))
        peak_prefixes = [sequence[:int(g["prompt_len"]) if s < 0 else min(sequence.numel(), int(g["prompt_len"]) + int(s) + 1)] for s in steps]
        row["peak_prefixes"] = len(peak_prefixes)
        row["peak_tokens"], row["peak_probe_s"] = timed(lambda: forward_probes(model, peak_prefixes, probe_ids, pad) if peak_prefixes else 0)
        x = features.loc[[case_id], gate_cols]
        _, row["gate_decision_s"] = timed(lambda: [gate[t].predict_proba(x) for t in ("baseline_correct", "targeted_correct", "generic_correct")])
        output, row["review_s"] = timed(lambda: mi_exp.generate_once(model, tokenizer, review.at[case_id, "prompt"], MAX_NEW[a.model]))
        row["review_new_tokens"] = len(output[2])
        del output
        if n >= WARMUP:
            rows.append(row)
        print(f"timing {n + 1}/{len(sample)} {case_id} first={row['first_pass_s']:.1f}s review={row['review_s']:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / f"timing_rows_{a.model}.csv", index=False)

    d = pd.DataFrame(rows)
    stages = ["first_pass_s", "teacher_forced_s", "option_encoding_s", "trajectories_s", "endpoint_s", "peak_probe_s", "gate_decision_s", "review_s"]
    analysis = ["teacher_forced_s", "option_encoding_s", "trajectories_s", "peak_probe_s", "gate_decision_s"]
    d["analysis_all_s"] = d[analysis + ["endpoint_s"]].sum(axis=1)
    d["mi_sr_total_s"] = d.first_pass_s + d.analysis_all_s + d.selected * d.review_s
    d["all_targeted_total_s"] = d.first_pass_s + d.endpoint_s + d.review_s
    d["first_pass_only_s"] = d.first_pass_s
    full_rate = float((policy.review_score > 0).mean())
    columns = stages + ["analysis_all_s", "mi_sr_total_s", "all_targeted_total_s"]
    result = {
        "model": a.model, "n": len(d), "warmup_excluded": WARMUP, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor(), "threads": torch.get_num_threads(), "torch": torch.__version__, "dtype": meta["dtype"],
        "serial_batch_size_generation": 1, "note": "Serial run; each question re-encodes its options; model loading excluded; other GPU work must be absent.",
        "seconds": {c: dict(mean=float(d[c].mean()), p50=float(d[c].median()), p95=float(d[c].quantile(.95))) for c in columns},
        "mi_sr_expected_at_full_review_rate_s": float(d.first_pass_s.mean() + d.analysis_all_s.mean() + full_rate * d.review_s.mean()),
        "full_review_rate": full_rate, "sample_review_rate": float(d.selected.mean()),
        "tokens_per_question": {c: float(d[c].mean()) for c in ["first_pass_new_tokens", "teacher_forced_tokens", "endpoint_tokens", "peak_tokens", "review_new_tokens"]},
        "first_pass_token_match_pct": float((d.first_pass_new_tokens == d.first_pass_stored_tokens).mean() * 100),
        "peak_gpu_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None,
        "peak_gpu_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30) if torch.cuda.is_available() else None,
        "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    path = out / f"timing_summary_{a.model}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["seconds"]["mi_sr_total_s"]), "WROTE", path, flush=True)


if __name__ == "__main__":
    main()
