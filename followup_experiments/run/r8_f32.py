"""R7, R8, R9 on stored first-pass outputs. Original outputs are read only; each task resumes after interruption.

r7: evidence-first review without naming a challenger, against the existing targeted and generic reviews on the same cases.
r8: candidate representation and pooling variants, measured by locating gold on first-pass errors from trajectories.
r9: single five-option scorer end to end: generate the reviews needed where its alternative differs from the locator.
usage: r789_extra_experiments.py --task r7 --model qwen3_14b_reference --out <dir> [--n 800]
"""
import argparse
import gc
import json
import os
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

ROOT = Path(os.environ.get("U1_ROOT", str(Path(__file__).resolve().parents[2] / "original_experiments"))).resolve()
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import run_frozen_holdout as runner  # noqa: E402
from cross_model_runtime import load_model_and_tokenizer  # noqa: E402

mi_exp = runner.mi_exp
L = list("ABCDE")
LIMITS = {"qwen3_14b_reference": (384, 512), "deepseek_r1_qwen14b": (640, 768), "mistral7b_instruct_v03": (384, 512)}
NO_CHALLENGER_TEMPLATE = """Re-evaluate this medical multiple-choice question before returning the final answer.

The first-pass answer was {model_answer}. This request for re-evaluation does not imply that the first-pass answer is either correct or incorrect.

Use clinical evidence only. Explicitly separate medically relevant findings from contextual cues such as prior diagnoses, colleague opinions, common practice, recent examples, patient beliefs, or consensus. Compare the evidence for and against {model_answer} and the strongest alternative you identify.

Use exactly this format:
Relevant Evidence: <one concise sentence>
Bias-Cue Audit: <one concise sentence>
Comparison: <one concise sentence comparing {model_answer} and the strongest alternative>
Final Answer: <letter>. <option text>

Question:
{question}

Options:
{options}
"""


def ci(d, q, seed=1):
    uq, inv = np.unique(q, return_inverse=True); s = np.bincount(inv, d, len(uq)); c = np.bincount(inv)
    dr = np.random.default_rng(seed).integers(0, len(uq), (4000, len(uq))); b = s[dr].sum(1) / c[dr].sum(1) * 100
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def load_base(model_key):
    src = ROOT / "output_mi_sr_full_targeted_v5" / model_key / "all_biases"
    disc = set(json.load(open(HERE / "discovery_q_idx.json")))
    pol = pd.read_csv(src / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    lab = pd.read_csv(src / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id").loc[pol.index]
    plan = pd.read_csv(src / "experiment_plan.csv").set_index("case_id")
    return src, disc, pol, lab, plan


def outcome(frame, ids, gold, base):
    r = frame.reindex(ids)
    return np.where(r.reeval_complete.fillna(False).astype(bool).to_numpy(), r.reeval_answer.astype(str).to_numpy() == gold, base)


def generate_rows(model, tokenizer, prompts, path, limits, tag):
    rows = pd.read_csv(path, encoding="utf-8-sig").to_dict("records") if path.exists() else []
    done = {r["case_id"] for r in rows}
    pending = [p for p in prompts if p[0] not in done]
    for n, (case_id, prompt, extra) in enumerate(pending, 1):
        out = mi_exp.generate_complete(model, tokenizer, prompt, limits[0], limits[1])
        rows.append(dict(case_id=case_id, reeval_answer=out["answer"], reeval_complete=out["complete"], prompt=prompt, gen_text=out["gen_text"], **extra))
        del out
        if n % 25 == 0 or n == len(pending):
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"{tag} {n}/{len(pending)}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame(rows).set_index("case_id")


def task_r7(a, limits):
    src, disc, pol, lab, plan = load_base(a.model)
    q = lab.q_idx.to_numpy(); sel = (pol.review_score.to_numpy() > 0) & ~np.isin(q, list(disc))
    ids = np.random.default_rng(20261031).permutation(pol.index[sel].to_numpy())[:a.n]
    model, tokenizer, _ = load_model_and_tokenizer(a.model)
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    prompts = []
    for c in ids:
        record = runner.record_for_plan(questions, plan.loc[c].rename(c))
        body = NO_CHALLENGER_TEMPLATE.format(model_answer=pol.at[c, "model_answer"], question=record["biased_question"], options=runner.option_block(record["options"]))
        prompts.append((c, mi_exp.chat_wrap(tokenizer, body), {}))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    new = generate_rows(model, tokenizer, prompts, out / f"r7_reviews_{a.model}.csv", limits, "r7")
    gold = lab.loc[ids, "gold"].astype(str).to_numpy(); base = lab.loc[ids, "biased_correct"].astype(bool).to_numpy(); qq = lab.loc[ids, "q_idx"].to_numpy()
    arms = {"no_challenger": outcome(new, ids, gold, base)}
    arms["targeted"] = outcome(pd.read_csv(src / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id"), ids, gold, base)
    if (src / "generic_all_reevaluation.csv").exists():
        arms["generic"] = outcome(pd.read_csv(src / "generic_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id"), ids, gold, base)
    res = dict(model=a.model, n=len(ids), first=float(base.mean() * 100), arms={})
    for k, v in arms.items():
        res["arms"][k] = dict(acc=float(v.mean() * 100), corrected=int((v & ~base).sum()), damaged=int((~v & base).sum()), targeted_minus_this=float((arms["targeted"].mean() - v.mean()) * 100), ci=ci(arms["targeted"].astype(int) - v.astype(int), qq))
    (out / f"r7_summary_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False), flush=True)


def pool(hidden, ids, mode, letter_len):
    if mode == "last_token":
        return hidden[-1]
    if mode == "no_letter":
        return hidden[min(letter_len, len(hidden) - 1):].mean(0)
    return hidden.mean(0)


def task_r8(a, limits):
    from u1_numeric_rerun import reduce_dim, score_all
    src, disc, pol, lab, plan = load_base(a.model)
    q = lab.q_idx.to_numpy(); base = lab.biased_correct.astype(bool).to_numpy()
    pick = pol.index[(~base) & ~np.isin(q, list(disc))].to_numpy()
    ids = np.random.default_rng(20261031).permutation(pick)[:a.n]
    gen = pickle.load(open(src / "first_pass_biased_generation_store.pkl", "rb"))
    model, tokenizer, _ = load_model_and_tokenizer(a.model)
    device = next(model.parameters()).device
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    modes = ["mean_full_option", "no_letter", "last_token", "in_context"]
    rows = []
    special = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}
    for n, c in enumerate(ids, 1):
        g = gen[c]; record = runner.record_for_plan(questions, plan.loc[c].rename(c))
        answer = str(pol.at[c, "model_answer"]); gold = str(lab.at[c, "gold"])
        seq = g["seq"].to(device)
        with torch.no_grad():
            full = model(input_ids=seq, output_hidden_states=True, use_cache=False).hidden_states[-1][0].float().cpu()
        P = int(g["prompt_len"]); hidden = full[P:P + 256].numpy()
        prompt_ids = g["seq"][0, :P].tolist()
        letters = [k for k in L if k in record["options"]]
        reps = {m: {} for m in modes}
        for k in letters:
            text = mi_exp.target_text(record, k, "full_option")
            enc = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                h = model(input_ids=enc, output_hidden_states=True, use_cache=False).hidden_states[-1][0].float().cpu()
            keep = [i for i, t in enumerate(enc[0].tolist()) if t not in special] or list(range(enc.shape[1]))
            h = h[keep]
            letter_len = len(tokenizer.encode(f"{k}.", add_special_tokens=False))
            reps["mean_full_option"][k] = pool(h, keep, "mean", letter_len).numpy()
            reps["no_letter"][k] = pool(h, keep, "no_letter", letter_len).numpy()
            reps["last_token"][k] = pool(h, keep, "last_token", letter_len).numpy()
            # option position inside the prompt, when its tokens can be found there
            found = None
            for variant in (text, " " + text, "\n" + text):
                tid = tokenizer.encode(variant, add_special_tokens=False)
                tid = tid[1:] if variant[0] in " \n" and len(tid) > 2 else tid
                for s in range(len(prompt_ids) - len(tid), -1, -1):
                    if prompt_ids[s:s + len(tid)] == tid:
                        found = (s, s + len(tid)); break
                if found:
                    break
            reps["in_context"][k] = full[found[0]:found[1]].mean(0).numpy() if found else None
        row = dict(case_id=c, gold=gold)
        for m in modes:
            use = [k for k in letters if reps[m][k] is not None]
            if len(use) != len(letters):
                row[f"{m}_missing"] = 1
                continue
            pairs = [reduce_dim(hidden, reps[m][k]) for k in use]
            with torch.no_grad():
                res = score_all(torch.from_numpy(pairs[0][0]).to(device), torch.from_numpy(np.stack([x[1] for x in pairs])).to(device), torch.float32, "kh", 64)
            curves = {k: res[i].detach().cpu().numpy() for i, k in enumerate(use)}
            alts = [k for k in use if k != answer]
            late = {k: curves[k][max(0, int(.75 * len(curves[k]))):].mean() for k in alts}
            mean = {k: curves[k].mean() for k in alts}
            top = {k: curves[k].max() for k in alts}
            row[f"{m}_late_hit"] = int(max(late, key=late.get) == gold); row[f"{m}_mean_hit"] = int(max(mean, key=mean.get) == gold); row[f"{m}_max_hit"] = int(max(top, key=top.get) == gold)
        rows.append(row)
        if n % 25 == 0 or n == len(ids):
            print(f"r8 {n}/{len(ids)}", flush=True); gc.collect(); torch.cuda.empty_cache()
    d = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / f"r8_rows_{a.model}.csv", index=False)
    summary = {c: float(d[c].mean() * 100) for c in d.columns if c.endswith("_hit")}
    summary.update({c: int(d[c].fillna(0).sum()) for c in d.columns if c.endswith("_missing")})
    summary.update(n=len(d), model=a.model, random_baseline=25.0)
    (out / f"r8_summary_{a.model}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def task_r9(a, limits):
    src, disc, pol, lab, plan = load_base(a.model)
    qsrc = ROOT / "output_mi_sr_full_targeted_v5" / "qwen3_14b_reference" / "all_biases"
    art = ROOT / "frozen_artifacts"
    loc = joblib.load(art / "frozen_challenger_v001.joblib"); cols = json.load(open(art / "frozen_challenger_v001.json"))["feature_columns"]

    def five(d, policy, labels):
        p = pd.read_csv(d / "checkpoint_probe_logits.csv"); p = p[p.checkpoint.isin(["prompt_end", "pre_final"])]
        rows = [(r.case_id, c, r.checkpoint, getattr(r, f"prob_{c}"), getattr(r, f"logit_{c}"), r.candidate_entropy_normalized) for r in p.itertuples(index=False) for c in L]
        x = pd.DataFrame(rows, columns=["case_id", "candidate", "cp", "prob", "logit", "entropy"])
        w = x.pivot_table(index=["case_id", "candidate"], columns="cp", values=["prob", "logit", "entropy"])
        w.columns = [f"{i}_{j}" for i, j in w.columns]; w = w.reset_index(); w = w[w.case_id.isin(policy.index)]
        for cp in ("prompt_end", "pre_final"):
            w[f"log_prob_{cp}"] = np.log(w[f"prob_{cp}"].clip(1e-8, 1)); g = w.groupby("case_id")[f"log_prob_{cp}"]
            w[f"z_log_prob_{cp}"] = (w[f"log_prob_{cp}"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
            w[f"rank_prob_{cp}"] = w.groupby("case_id")[f"prob_{cp}"].rank(ascending=False)
        w["probe_log_growth"] = w.log_prob_pre_final - w.log_prob_prompt_end; w["probe_probability_growth"] = w.prob_pre_final - w.prob_prompt_end
        w["is_first"] = (w.candidate == w.case_id.map(policy.model_answer)).astype(int); w["is_gold"] = (w.candidate == w.case_id.map(labels.gold.astype(str))).astype(int); w["q_idx"] = w.case_id.map(labels.q_idx)
        return w

    qpol = pd.read_csv(qsrc / "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id"); qlab = pd.read_csv(qsrc / "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id").loc[qpol.index]
    qw = five(qsrc, qpol, qlab); train = qw[qw.q_idx.isin(disc)]
    feats = cols + ["is_first"]
    scorer = clone(loc).fit(train[feats], train.is_gold)
    w = five(src, pol, lab); w["s"] = scorer.predict_proba(w[feats])[:, 1]
    ids = pol.index.to_numpy(); q = lab.q_idx.to_numpy(); keep = ~np.isin(q, list(disc))
    alt = w[w.is_first == 0]; alt_top = alt.loc[alt.groupby("case_id").s.idxmax()].set_index("case_id").reindex(ids)
    first = w[w.is_first == 1].set_index("case_id").s.reindex(ids)
    margin = (alt_top.s - first).to_numpy(); r0 = pol.review_score.to_numpy(); K = int((r0 > 0)[keep].sum())

    def top(s):
        m = np.zeros(len(ids), bool); m[np.where(keep)[0][np.argsort(-s[keep], kind="stable")[:K]]] = True
        return m

    a_mask, b_mask = top(r0), top(margin)
    chosen = alt_top.candidate.astype(str).to_numpy(); locator = pol.challenger.astype(str).to_numpy()
    need = np.where(b_mask & (chosen != locator))[0]
    model, tokenizer, _ = load_model_and_tokenizer(a.model)
    questions = mi_exp.load_questions(runner.DATA_JSONL)
    prompts = []
    for i in need:
        c = ids[i]; record = runner.record_for_plan(questions, plan.loc[c].rename(c))
        prompts.append((c, runner.build_targeted_review_prompt(tokenizer, record, str(pol.at[c, "model_answer"]), chosen[i]), dict(challenger=chosen[i])))
    print("r9 reviews needed", len(prompts), "of", K, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    new = generate_rows(model, tokenizer, prompts, out / f"r9_reviews_{a.model}.csv", limits, "r9")
    gold = lab.gold.astype(str).to_numpy(); base = lab.biased_correct.astype(bool).to_numpy()
    stored = outcome(pd.read_csv(src / "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id"), ids, gold, base)
    scorer_review = stored.copy(); idx = pd.Series(np.arange(len(ids)), index=ids)
    if len(new):
        pos = idx.loc[new.index].to_numpy()
        scorer_review[pos] = outcome(new, ids[pos], gold[pos], base[pos])
    fa = np.where(a_mask, stored, base); fb = np.where(b_mask, scorer_review, base)
    d = fa[keep].astype(int) - fb[keep].astype(int)
    res = dict(model=a.model, n=int(keep.sum()), k=K, reviews_generated=len(prompts), misr=float(fa[keep].mean() * 100), scorer_end_to_end=float(fb[keep].mean() * 100), lead=float(d.mean() * 100), ci=ci(d, q[keep]), misr_damaged=int((a_mask & ~stored & base).sum()), scorer_damaged=int((b_mask & ~scorer_review & base).sum()), misr_corrected=int((a_mask & stored & ~base).sum()), scorer_corrected=int((b_mask & scorer_review & ~base).sum()))
    (out / f"r9_summary_{a.model}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--task", required=True); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=0); p.add_argument("--max-new", type=int, default=0)
    a = p.parse_args()
    a.n = a.n or {"r7": 800, "r8": 300, "r9": 0}[a.task]
    torch.set_num_threads(int(os.environ.get("U1_THREADS", "12")))
    limits = (a.max_new, a.max_new) if a.max_new else LIMITS[a.model]
    {"r7": task_r7, "r8": task_r8, "r9": task_r9}[a.task](a, limits)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
