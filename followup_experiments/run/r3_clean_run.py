"""R3: run the frozen MI-SR pipeline on clean questions, using the stored clean first-pass outputs.

Clean first-pass text is re-tokenized into a generation store, then trajectories, endpoint probes, peak probes and the
frozen policy run unchanged. Targeted reviews are generated for selected questions first, then for the rest so that
All Targeted can be compared on the same questions. Nothing in the original outputs is changed; resumable.
usage: r3_clean_run.py --model qwen3_14b_reference --out <dir>
"""
import argparse
import gc
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("U1_ROOT", str(Path(__file__).resolve().parents[2] / "original_experiments"))).resolve()
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import run_frozen_holdout as runner  # noqa: E402
from cross_model_runtime import load_model_and_tokenizer  # noqa: E402
from final_answer_logits import extract_checkpoint_probe_logits  # noqa: E402
from mi_sr_deploy_v3 import build_deploy_policy  # noqa: E402
from mi_sr_peak_features import extract_peak_features  # noqa: E402

mi_exp = runner.mi_exp
LIMITS = {"qwen3_14b_reference": (384, 512), "deepseek_r1_qwen14b": (640, 768), "mistral7b_instruct_v03": (384, 512)}
RANDOM_SEED = 20260810


def clean_record(questions, q_idx):
    source = SimpleNamespace(q_idx=q_idx, bias_type=mi_exp.BIAS_TYPES[0], record_seed=RANDOM_SEED + 20_000_000 + q_idx * 31)
    record = runner.record_for_plan(questions, source)
    record["biased_question"] = record["question"]
    return record


def build_inputs(tokenizer, src, run):
    clean = pd.read_csv(src / "clean_reference_outputs.csv", encoding="utf-8-sig")
    clean = clean[clean.clean_complete.fillna(False).astype(bool) & clean.clean_answer.isin(list("ABCDE"))]
    store, labels, plan = {}, [], []
    for r in clean.itertuples(index=False):
        case_id = f"q{int(r.q_idx):04d}_clean"
        p = tokenizer(r.prompt, return_tensors="pt")["input_ids"][0]
        g = torch.tensor(tokenizer.encode(str(r.gen_text), add_special_tokens=False), dtype=torch.long)
        store[case_id] = {"seq": torch.cat([p, g]).unsqueeze(0), "prompt_len": len(p), "gen_ids": g.tolist(), "gen_text": str(r.gen_text), "model_answer": r.clean_answer, "complete": True}
        labels.append(dict(case_id=case_id, split="clean", q_idx=int(r.q_idx), bias_type="clean", gold=r.gold, clean_answer=r.clean_answer, biased_answer=r.clean_answer, clean_correct=bool(r.clean_correct), biased_correct=bool(r.clean_correct), clean_complete=True, biased_complete=True, outcome_group="clean"))
        plan.append(dict(case_id=case_id, split="clean", split_position=len(plan), q_idx=int(r.q_idx), bias_type="clean", record_seed=0))
    runner.save_pickle(store, run / "first_pass_biased_generation_store.pkl")
    pd.DataFrame(labels).to_csv(run / "first_pass_evaluation_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(plan).to_csv(run / "experiment_plan.csv", index=False, encoding="utf-8-sig")
    return store


def stage_curves(model, tokenizer, store, run, questions):
    path = run / "all_candidate_curve_store.pkl"
    curves = pickle.load(open(path, "rb")) if path.exists() else {}
    cache = mi_exp.AnswerRepCache(model, tokenizer, "full_option")
    args = SimpleNamespace(target_mode="full_option", mi_max_dim=128, mi_sigma=50.0)
    pending = [c for c in store if c not in curves]
    for n, c in enumerate(pending, 1):
        q = int(c[1:5])
        record = clean_record(questions, q)
        hidden = mi_exp.hidden_seq(model, store[c]["seq"], store[c]["prompt_len"], 256)
        cs = mi_exp.candidate_curves(record, hidden, cache, args)
        curves[c] = {"curves": cs, "support": runner.candidate_support_scores(cs), "model_answer": store[c]["model_answer"]}
        if n % 100 == 0 or n == len(pending):
            runner.save_pickle(curves, path)
            print(f"clean curves {n}/{len(pending)}", flush=True)
            gc.collect(); torch.cuda.empty_cache()


def stage_probes(model, tokenizer, store, run):
    path = run / "checkpoint_probe_logits.csv"
    if path.exists():
        return
    frame = extract_checkpoint_probe_logits(model, tokenizer, store, list(store), batch_size=8)
    labels = pd.read_csv(run / "first_pass_evaluation_labels.csv", encoding="utf-8-sig")
    meta = labels[["case_id", "split", "q_idx", "bias_type", "outcome_group", "biased_answer", "gold"]].rename(columns={"biased_answer": "model_answer"})
    frame.merge(meta, on="case_id", how="left").to_csv(path, index=False, encoding="utf-8-sig")
    print("clean probes done", len(frame), flush=True)


def stage_reviews(model, tokenizer, run, questions, order, limits, name):
    path = run / "targeted_all_reevaluation.csv"
    rows = pd.read_csv(path, encoding="utf-8-sig").to_dict("records") if path.exists() else []
    done = {r["case_id"] for r in rows}
    pending = [r for r in order.itertuples(index=False) if r.case_id not in done]
    for n, r in enumerate(pending, 1):
        record = clean_record(questions, int(r.case_id[1:5]))
        prompt = runner.build_targeted_review_prompt(tokenizer, record, r.model_answer, r.challenger)
        out = mi_exp.generate_complete(model, tokenizer, prompt, limits[0], limits[1])
        rows.append(dict(case_id=r.case_id, review_type="targeted", model_answer=r.model_answer, challenger=r.challenger, reeval_answer=out["answer"], reeval_complete=out["complete"], prompt=prompt, gen_text=out["gen_text"]))
        del out
        if n % 25 == 0 or n == len(pending):
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"clean reviews {name} {n}/{len(pending)}", flush=True)
            gc.collect(); torch.cuda.empty_cache()


def summarize(run, out_path, model_key):
    disc = set(json.load(open(HERE / "discovery_q_idx.json")))
    pol = pd.read_csv(run / "frozen_mi_sr_policy_v3.csv", encoding="utf-8-sig").set_index("case_id")
    lab = pd.read_csv(run / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id").loc[pol.index]
    rv = pd.read_csv(run / "targeted_all_reevaluation.csv", encoding="utf-8-sig").set_index("case_id").reindex(pol.index)
    gold = lab.gold.astype(str).to_numpy(); base = lab.biased_correct.astype(bool).to_numpy(); q = lab.q_idx.to_numpy()
    have = rv.reeval_answer.notna().to_numpy()
    rev = np.where(rv.reeval_complete.fillna(False).astype(bool).to_numpy(), rv.reeval_answer.astype(str).to_numpy() == gold, base)
    t = (pol[["p_targeted_correct", "p_generic_correct"]].max(axis=1) - pol.p_none_correct).to_numpy() > 0
    gain = rev.astype(int) - base.astype(int)

    def block(mask):
        tt = t & mask
        uq, inv = np.unique(q[mask], return_inverse=True); c = np.bincount(inv)
        rng = np.random.default_rng(1)

        def ci(sel):
            d = np.bincount(inv, (gain * sel)[mask], len(uq)); dr = rng.integers(0, len(uq), (4000, len(uq))); b = d[dr].sum(1) / c[dr].sum(1) * 100
            return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]

        complete_at = bool(have[mask].all())
        res = dict(n=int(mask.sum()), clean_first=float(base[mask].mean() * 100), review_rate=float(tt.sum() / mask.sum() * 100), misr=float((base[mask].sum() + gain[tt].sum()) / mask.sum() * 100), misr_gain=float(gain[tt].sum() / mask.sum() * 100), misr_ci=ci(t.astype(int)),
                   corrected=int((tt & rev & ~base).sum()), damaged=int((tt & ~rev & base).sum()), damage_rate_of_correct=float((tt & ~rev & base).sum() / max((base & mask).sum(), 1) * 100), all_targeted_complete=complete_at)
        if complete_at:
            res.update(all_targeted=float(rev[mask].mean() * 100), at_gain=float(gain[mask].mean() * 100), at_ci=ci(np.ones(len(t), int)), at_corrected=int((mask & rev & ~base).sum()), at_damaged=int((mask & ~rev & base).sum()), at_damage_rate_of_correct=float((mask & ~rev & base).sum() / max((base & mask).sum(), 1) * 100))
        return res

    keep = ~np.isin(q, list(disc))
    result = dict(model=model_key, all=block(np.ones(len(t), bool)), excl95=block(keep))
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["excl95"], ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--limit", type=int, default=0); p.add_argument("--max-new", type=int, default=0)
    a = p.parse_args()
    src = ROOT / "output_mi_sr_full_targeted_v5" / a.model / "all_biases"
    run = Path(a.out) / a.model / "all_biases"
    (run / "mi_sr_peak_features").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("U1_THREADS", "12")))
    model, tokenizer, _ = load_model_and_tokenizer(a.model)
    limits = (a.max_new, a.max_new) if a.max_new else LIMITS[a.model]
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    store_path = run / "first_pass_biased_generation_store.pkl"
    if store_path.exists():
        store = pickle.load(open(store_path, "rb"))
    else:
        store = build_inputs(tokenizer, src, run)
    if a.limit:
        store = dict(list(store.items())[:a.limit])
        runner.save_pickle(store, store_path)
        lab = pd.read_csv(run / "first_pass_evaluation_labels.csv", encoding="utf-8-sig")
        lab[lab.case_id.isin(store)].to_csv(run / "first_pass_evaluation_labels.csv", index=False, encoding="utf-8-sig")
    print("clean cases", len(store), flush=True)
    stage_curves(model, tokenizer, store, run, questions)
    stage_probes(model, tokenizer, store, run)
    if not (run / "frozen_mi_sr_policy_v3.csv").exists():
        extract_peak_features(model, tokenizer, run, force=True, preserve_existing_probes=True)
        build_deploy_policy(run)
    pol = pd.read_csv(run / "frozen_mi_sr_policy_v3.csv", encoding="utf-8-sig")
    pol["score"] = pol[["p_targeted_correct", "p_generic_correct"]].max(axis=1) - pol.p_none_correct
    stage_reviews(model, tokenizer, run, questions, pol[pol.score > 0], limits, "selected")
    summarize(run, Path(a.out) / f"r3_summary_{a.model}.json", a.model)
    stage_reviews(model, tokenizer, run, questions, pol[~(pol.score > 0)], limits, "rest")
    summarize(run, Path(a.out) / f"r3_summary_{a.model}.json", a.model)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
