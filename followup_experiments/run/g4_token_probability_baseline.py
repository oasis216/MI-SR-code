"""G4 / R5: token-probability gates versus MI-SR at the same review count, using the stored targeted reviews.
One forward pass per stored sequence, no new generation. Excludes the 95 development questions from the comparison.
usage: g4_token_probability_baseline.py --model qwen3_14b_reference --out <dir>; resumes after interruption."""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("U1_ROOT", str(Path(__file__).resolve().parents[2] / "original_experiments"))).resolve()
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from cross_model_runtime import load_model_and_tokenizer  # noqa: E402

BUDGETS = [("5%", .05), ("10%", .1), ("20%", .2), ("30%", .3), ("40%", .4), ("50%", .5), ("70%", .7), ("90%", .9)]


def features(model, tokenizer, stored):
    seq = stored["seq"].reshape(1, -1).to(next(model.parameters()).device)
    start = int(stored["prompt_len"])
    with torch.inference_mode():
        logits = model(input_ids=seq, use_cache=False, return_dict=True).logits[0, start - 1:-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    tokens = seq[0, start:]
    chosen = logp[torch.arange(len(tokens)), tokens]
    top = logp.max(dim=-1).values
    entropy = -(logp.exp() * logp).sum(-1)
    answer = str(stored["model_answer"])
    answer_lp = np.nan
    for i in range(len(tokens) - 1, -1, -1):
        if tokenizer.decode([int(tokens[i])], skip_special_tokens=True).strip() == answer:
            answer_lp = float(chosen[i])
            break
    return dict(mean_logp=float(chosen.mean()), min_logp=float(chosen.min()), mean_top1=float(top.exp().mean()), mean_entropy=float(entropy.mean()), answer_token_logp=answer_lp, n_tokens=int(len(tokens)))


def analyze(rows, src, model_key):
    disc = set(json.load(open(HERE / "discovery_q_idx.json")))
    pol = pd.read_csv(src / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    lab = pd.read_csv(src / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id")
    rv = pd.read_csv(src / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id")
    ids = rows.index.intersection(pol.index)
    ids = ids[~lab.loc[ids, "q_idx"].isin(disc).to_numpy()]
    x = rows.loc[ids]
    gold = lab.loc[ids, "gold"].astype(str).to_numpy(); base = lab.loc[ids, "biased_correct"].astype(bool).to_numpy()
    r = rv.reindex(ids)
    rev = np.where(r.reeval_complete.fillna(False).astype(bool).to_numpy(), r.reeval_answer.astype(str).to_numpy() == gold, base)
    gain = rev.astype(int) - base.astype(int)
    q = lab.loc[ids, "q_idx"].to_numpy(); uq, inv = np.unique(q, return_inverse=True); counts = np.bincount(inv)
    score0 = pol.loc[ids, "review_score"].to_numpy()
    gates = {"low_mean_token_prob": -x.mean_logp, "low_min_token_prob": -x.min_logp, "low_mean_top1": -x.mean_top1, "high_token_entropy": x.mean_entropy, "low_answer_token_prob": -x.answer_token_logp.fillna(x.answer_token_logp.median())}
    rng = np.random.default_rng(20261031)
    n = len(ids)
    out = []

    def top(s, k):
        m = np.zeros(n, bool); m[np.argsort(-np.asarray(s), kind="stable")[:k]] = True
        return m

    for label, k in BUDGETS + [("published", int((score0 > 0).sum()) / n)]:
        k = int(round(k * n)); a = top(score0, k)
        for name, s in gates.items():
            b = top(s, k); d = np.bincount(inv, (a.astype(int) - b.astype(int)) * gain, len(uq))
            draws = rng.integers(0, len(uq), (4000, len(uq))); boot = d[draws].sum(1) / counts[draws].sum(1) * 100
            out.append(dict(model=model_key, budget=label, k=k, gate=name, misr_acc=float((base.sum() + gain[a].sum()) / n * 100), gate_acc=float((base.sum() + gain[b].sum()) / n * 100), lead_pp=float((gain[a].sum() - gain[b].sum()) / n * 100), ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
                            misr_damaged=int((a & ~rev & base).sum()), gate_damaged=int((b & ~rev & base).sum()), misr_corrected=int((a & rev & ~base).sum()), gate_corrected=int((b & rev & ~base).sum())))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--out", required=True); a = p.parse_args()
    src = ROOT / "output_mi_sr_full_targeted_v5" / a.model / "all_biases"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"token_features_{a.model}.csv"
    done = pd.read_csv(path).to_dict("records") if path.exists() else []
    seen = {r["case_id"] for r in done}
    gen = pickle.load(open(src / "first_pass_biased_generation_store.pkl", "rb"))
    ids = [c for c in pd.read_csv(src / "binary_deploy_policy.csv", encoding="utf-8-sig", usecols=["case_id"]).case_id if c not in seen]
    if ids:
        model, tokenizer, _ = load_model_and_tokenizer(a.model)
        for i, c in enumerate(ids, 1):
            done.append(dict(case_id=c, **features(model, tokenizer, gen[c])))
            if i % 250 == 0 or i == len(ids):
                pd.DataFrame(done).to_csv(path, index=False)
                print(f"token features {i}/{len(ids)}", flush=True)
    rows = pd.DataFrame(done).set_index("case_id")
    res = analyze(rows, src, a.model)
    res.to_csv(out / f"token_gate_comparison_{a.model}.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(res[res.budget == "published"].round(2).to_string())
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
