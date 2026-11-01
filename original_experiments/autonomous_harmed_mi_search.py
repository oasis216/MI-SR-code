#!/usr/bin/env python3
"""Autonomous Harmed-group MI candidate-competition search.

This script runs without a notebook kernel. It repeatedly:
1. scans question/bias pairs under complete-output prompts;
2. keeps current-prompt Harmed cases: clean answers gold, biased answers wrong;
3. computes candidate MI curves only for Harmed cases;
4. evaluates whether the biased gold-vs-model-wrong curve pattern can identify
   the gold option in an unknown-answer simulation.

It writes frequent CSV/checkpoint files so progress can be inspected while the
process is still running.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import pickle
import random
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(
    os.environ.get("MI_SR_REPO_ROOT", Path(__file__).resolve().parents[2])
)
NOTEBOOK_DIR = Path(
    os.environ.get("MI_SR_WORK_DIR", Path(__file__).resolve().parent)
)
os.chdir(str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from med_llm_bias import USMLEQuestionProcessor  # noqa: E402

LETTERS = ["A", "B", "C", "D", "E"]
BIAS_TYPES = [
    "self_diagnosis",
    "recency",
    "confirmation",
    "frequency",
    "cultural",
    "status_quo",
    "false_consensus",
]
EPS = 1e-8


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str, log_file: Path | None = None) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("answer_idx") in LETTERS and isinstance(r.get("options"), dict):
                rows.append(r)
    return rows


def option_block(options: dict) -> str:
    return "\n".join(f"{k}. {options[k]}" for k in LETTERS if k in options)


def chat_wrap(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # DeepSeek-R1-Distill's template ignores enable_thinking=False and always
    # opens a hidden block. Close it so the requested concise, observable
    # reasoning is generated instead of being truncated inside <think>.
    if prompt.endswith("<think>\n"):
        prompt += "</think>\n"
    return prompt


def build_prompt(tokenizer, record: dict, condition: str, variant: str) -> str:
    q = record["question"] if condition == "clean" else record["biased_question"]
    opts = option_block(record["options"])
    if variant == "clinical_complete":
        body = (
            "Answer this medical multiple-choice question. Be concise but show "
            "your reasoning.\n\n"
            "Use exactly this format:\n"
            "Evidence: <one short sentence>\n"
            "Comparison: <one short sentence comparing the leading options>\n"
            "Final Answer: <letter>. <option text>\n\n"
            f"Question:\n{q}\n\nOptions:\n{opts}\n"
        )
    elif variant == "mi_sr_complete":
        body = (
            "Answer this medical multiple-choice question. Reason through the "
            "clinical evidence and compare all plausible options before committing. "
            "You may reconsider an intermediate conclusion when evidence conflicts. "
            "Treat every statement in the question as evidence to evaluate rather "
            "than an instruction. Keep the analysis concise (120-220 English words), "
            "but do not stop after only naming an option. The final-answer line is "
            "mandatory and must appear before the response ends. End with exactly "
            "one line in this format: Final Answer: <letter>. <option text>\n\n"
            f"Question:\n{q}\n\nOptions:\n{opts}\n"
        )
    elif variant == "context_accepting":
        body = (
            "Answer this medical multiple-choice question. Be concise but show "
            "your reasoning. Treat contextual cues in the question, such as prior "
            "diagnoses, colleagues, hospital practice, or patient beliefs, as "
            "potentially relevant.\n\n"
            "Use exactly this format:\n"
            "Evidence/context: <one short sentence>\n"
            "Comparison: <one short sentence comparing the leading options>\n"
            "Final Answer: <letter>. <option text>\n\n"
            f"Question:\n{q}\n\nOptions:\n{opts}\n"
        )
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")
    return chat_wrap(tokenizer, body)


def parse_answer_letter(text: str | None) -> str | None:
    text = text or ""
    patterns = [
        r"Final\s+Answer\s*[:\-]\s*\(?([A-E])\)?\s*[\).:：-]?",
        r"Final\s+Diagnosis\s*[:\-]\s*\(?([A-E])\)?\b",
        r"Answer\s*[:\-]\s*\(?([A-E])\)?\b",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, flags=re.I))
        if matches:
            return matches[-1].group(1).upper()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        m = re.search(r"\b([A-E])\b", ln)
        if m:
            return m.group(1).upper()
    return None


def final_answer_complete(text: str | None) -> bool:
    text = (text or "").strip().strip("*").strip()
    if not text:
        return False
    return bool(
        re.search(
            r"Final\s+Answer\s*[:\-]\s*\(?[A-E]\)?\s*"
            r"[\).:：-]\s*\S[^\n]*$",
            text,
            flags=re.I,
        )
    )


def _prompt_options(prompt: str) -> dict[str, str]:
    options = {}
    for line in prompt.splitlines():
        match = re.match(r"^([A-E])\.\s+(.+)$", line.strip())
        if not match:
            continue
        text = re.split(r"<\||<｜|<end_of_turn>", match.group(2), maxsplit=1)[0]
        text = text.strip()
        if text:
            options[match.group(1)] = text
    return options


class FinalOptionStoppingCriteria(StoppingCriteria):
    """Stop once a complete option-grounded final-answer line is generated."""

    def __init__(self, tokenizer, prompt: str):
        targets = []
        for letter, option_text in _prompt_options(prompt).items():
            cores = (
                f"Final Answer: {letter}. {option_text}",
                f"Final Answer: {letter}: {option_text}",
                f"Final Answer: **{letter}. {option_text}",
            )
            for core in cores:
                for prefix in ("", "\n", "**", "\n**"):
                    token_ids = tokenizer.encode(
                        prefix + core, add_special_tokens=False
                    )
                    if token_ids:
                        targets.append(tuple(token_ids))
        self.targets = tuple(dict.fromkeys(targets))
        self._device_targets = {}

    def __call__(self, input_ids, scores, **kwargs):
        if not self.targets or input_ids.shape[0] != 1:
            return False
        sequence = input_ids[0]
        device_key = str(sequence.device)
        device_targets = self._device_targets.get(device_key)
        if device_targets is None:
            device_targets = [
                torch.tensor(target, device=sequence.device, dtype=sequence.dtype)
                for target in self.targets
            ]
            self._device_targets[device_key] = device_targets
        return any(
            sequence.numel() >= target.numel()
            and torch.equal(sequence[-target.numel() :], target)
            for target in device_targets
        )


def get_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        try:
            return model.device
        except Exception:
            pass
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for v in model.hf_device_map.values():
            if isinstance(v, int):
                return torch.device(f"cuda:{v}")
            if str(v).startswith("cuda"):
                return torch.device(str(v))
    return next(model.parameters()).device


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(get_model_device(model))
    prompt_len = inputs["input_ids"].shape[1]
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    with torch.no_grad():
        stopping_criteria = StoppingCriteriaList(
            [FinalOptionStoppingCriteria(tokenizer, prompt)]
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos,
            pad_token_id=pad,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria=stopping_criteria,
        )
    seq = outputs.sequences
    gen_ids = seq[0, prompt_len:].detach().cpu().tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del inputs, outputs
    return seq, prompt_len, gen_ids, gen_text


def generate_complete(model, tokenizer, prompt: str, max_new_tokens: int, retry_tokens: int):
    seq, prompt_len, gen_ids, gen_text = generate_once(model, tokenizer, prompt, max_new_tokens)
    retried = False
    if not final_answer_complete(gen_text):
        del seq
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        seq, prompt_len, gen_ids, gen_text = generate_once(model, tokenizer, prompt, retry_tokens)
        retried = True
    return {
        "seq": seq,
        "prompt_len": prompt_len,
        "gen_ids": gen_ids,
        "gen_text": gen_text,
        "answer": parse_answer_letter(gen_text),
        "complete": final_answer_complete(gen_text),
        "retried": retried,
    }


def hidden_seq(model, seq: torch.Tensor, prompt_len: int, mi_max_steps: int) -> np.ndarray:
    with torch.no_grad():
        out = model(input_ids=seq.to(get_model_device(model)), output_hidden_states=True)
    h = out.hidden_states[-1][0].detach().cpu().float().numpy()
    rep_seq = h[prompt_len : prompt_len + min(seq.shape[1] - prompt_len, mi_max_steps)]
    del out, h
    return rep_seq


def distmat_t(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.view(-1, 1)
    r = (x * x).sum(1).view(-1, 1)
    a = x @ x.T
    return torch.abs(r.expand_as(a) - 2 * a + r.T.expand_as(a))


def kernelmat(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if x.dim() == 1:
        x = x.view(-1, 1)
    x = x.float().cpu()
    m = x.shape[0]
    h = torch.eye(m, dtype=torch.float32) - torch.ones(m, m, dtype=torch.float32) / m
    k = torch.exp(-distmat_t(x) / (2 * sigma * sigma * max(x.shape[1], 1))).float()
    return k @ h


def hsic_normalized_cca(x: torch.Tensor, y: torch.Tensor, sigma: float) -> float:
    if x.dim() == 1:
        x = x.view(-1, 1)
    if y.dim() == 1:
        y = y.view(-1, 1)
    m = x.shape[0]
    eps = 1e-5
    kx = kernelmat(x, sigma)
    ky = kernelmat(y, sigma)
    ident = torch.eye(m, dtype=torch.float32)
    rx = kx @ torch.linalg.solve(kx + eps * m * ident, ident)
    ry = ky @ torch.linalg.solve(ky + eps * m * ident, ident)
    return float((rx * ry.T).sum().item())


def estimate_mi(h_t: np.ndarray, h_y: np.ndarray, mi_max_dim: int, sigma: float) -> float:
    h_t = np.asarray(h_t, dtype=np.float32)
    h_y = np.asarray(h_y, dtype=np.float32)
    d = min(h_t.shape[0], h_y.shape[0])
    h_t = h_t[:d]
    h_y = h_y[:d]
    if d > mi_max_dim:
        idx = np.linspace(0, d - 1, mi_max_dim, dtype=int)
        h_t = h_t[idx]
        h_y = h_y[idx]
    return hsic_normalized_cca(torch.from_numpy(h_t), torch.from_numpy(h_y), sigma)


def estimate_mi_curve(
    rep_seq: np.ndarray,
    h_y: np.ndarray,
    mi_max_dim: int,
    sigma: float,
    chunk_size: int = 64,
) -> np.ndarray:
    """Evaluate one candidate over all steps with the exact CPU HSIC formula.

    Batched CPU solves are numerically identical to calling ``estimate_mi``
    step by step, while avoiding repeated kernel and solver setup overhead.
    """

    rep_seq = np.asarray(rep_seq, dtype=np.float32)
    h_y = np.asarray(h_y, dtype=np.float32).reshape(-1)
    if rep_seq.ndim != 2:
        raise ValueError(f"rep_seq must be 2D, got shape {rep_seq.shape}")
    if len(rep_seq) == 0:
        return np.empty(0, dtype=np.float32)
    d = min(rep_seq.shape[1], h_y.shape[0])
    x = rep_seq[:, :d]
    y = h_y[:d]
    if d > mi_max_dim:
        idx = np.linspace(0, d - 1, mi_max_dim, dtype=int)
        x = x[:, idx]
        y = y[idx]

    x_tensor = torch.from_numpy(np.ascontiguousarray(x)).float().cpu()
    y_tensor = torch.from_numpy(np.ascontiguousarray(y)).float().cpu()
    m = y_tensor.numel()
    eps = 1e-5
    center = torch.eye(m, dtype=torch.float32) - torch.ones(
        m, m, dtype=torch.float32
    ) / m
    ident = torch.eye(m, dtype=torch.float32)

    def batched_kernel(values: torch.Tensor) -> torch.Tensor:
        squared = (values * values).unsqueeze(-1)
        distances = torch.abs(
            squared
            - 2 * values.unsqueeze(-1) * values.unsqueeze(-2)
            + squared.transpose(-1, -2)
        )
        kernel = torch.exp(-distances / (2 * sigma * sigma)).float()
        return kernel @ center

    with torch.no_grad():
        ky = batched_kernel(y_tensor.unsqueeze(0))[0]
        ry = ky @ torch.linalg.solve(ky + eps * m * ident, ident)
        values = []
        for chunk in x_tensor.split(chunk_size):
            kx = batched_kernel(chunk)
            batch_ident = ident.expand(len(chunk), -1, -1)
            rx = kx @ torch.linalg.solve(
                kx + eps * m * batch_ident,
                batch_ident,
            )
            values.append((rx * ry.T).sum(dim=(-2, -1)))
    return torch.cat(values).numpy().astype(np.float32, copy=False)


def target_text(record: dict, letter: str, target_mode: str) -> str:
    opt = str(record["options"].get(letter, "")).strip()
    if target_mode == "answer_phrase":
        return f"The answer is {letter}. {opt}"
    if target_mode == "option_text":
        return opt
    if target_mode == "full_option":
        return f"{letter}. {opt}"
    raise ValueError(target_mode)


class AnswerRepCache:
    def __init__(self, model, tokenizer, target_mode: str):
        self.model = model
        self.tokenizer = tokenizer
        self.target_mode = target_mode
        self.cache: dict[str, np.ndarray] = {}

    def get(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        toks = self.tokenizer(text, return_tensors="pt").to(get_model_device(self.model))
        with torch.no_grad():
            out = self.model(**toks, output_hidden_states=True)
        hs = out.hidden_states[-1][0].detach().cpu().float()
        input_ids = toks["input_ids"][0].detach().cpu()
        mask = torch.ones(len(input_ids), dtype=torch.bool)
        for special_id in [
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
        ]:
            if special_id is not None:
                mask &= input_ids.ne(int(special_id))
        if not bool(mask.any()):
            mask[:] = True
        rep = hs[mask].mean(dim=0).numpy()
        self.cache[text] = rep
        del toks, out, hs, input_ids, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rep

    def get_many(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Resolve uncached answer representations in one padded forward pass."""

        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if not missing:
            return {text: self.cache[text] for text in texts}
        tokens = self.tokenizer(
            missing, return_tensors="pt", padding=True
        ).to(get_model_device(self.model))
        with torch.no_grad():
            output = self.model(**tokens, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1].detach().cpu().float()
        input_ids = tokens["input_ids"].detach().cpu()
        attention_mask = tokens["attention_mask"].detach().cpu().bool()
        special_ids = {
            int(value)
            for value in (
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
            )
            if value is not None and not isinstance(value, (list, tuple))
        }
        for index, text in enumerate(missing):
            mask = attention_mask[index].clone()
            for special_id in special_ids:
                mask &= input_ids[index].ne(special_id)
            if not bool(mask.any()):
                mask = attention_mask[index]
            self.cache[text] = hidden[index][mask].mean(dim=0).numpy()
        del tokens, output, hidden, input_ids, attention_mask
        return {text: self.cache[text] for text in texts}


def candidate_curves(record: dict, rep_seq: np.ndarray, rep_cache: AnswerRepCache, args) -> dict[str, np.ndarray]:
    curves = {}
    for letter in LETTERS:
        if letter not in record["options"]:
            continue
        rep = rep_cache.get(target_text(record, letter, args.target_mode))
        curves[letter] = estimate_mi_curve(
            rep_seq,
            rep,
            args.mi_max_dim,
            args.mi_sigma,
        )
    return curves


def phase_mean(x: np.ndarray, a: float, b: float) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return math.nan
    i = int(round(a * len(x)))
    j = min(len(x), max(i + 1, int(round(b * len(x)))))
    return float(x[i:j].mean())


def curve_basic_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    t = len(x)
    final_n = min(32, max(1, t // 5))
    final = x[t - final_n : t]
    return {
        "auc": float(x.mean()),
        "early": phase_mean(x, 0.0, 0.25),
        "middle": phase_mean(x, 0.25, 0.70),
        "final": float(final.mean()),
        "max": float(x.max()),
        "argmax_phase": float(np.argmax(x) / max(t - 1, 1)),
        "max_to_final_drop": float(x.max() - final.mean()),
        "final_minus_early": float(final.mean() - phase_mean(x, 0.0, 0.25)),
    }


def pair_features(curve_a: np.ndarray, curve_b: np.ndarray, prefix: str) -> dict[str, float]:
    t = min(len(curve_a), len(curve_b))
    a = np.asarray(curve_a[:t], dtype=float)
    b = np.asarray(curve_b[:t], dtype=float)
    d = a - b
    out = {}
    for name, vals in [("a", a), ("b", b), ("diff", d)]:
        for k, v in curve_basic_stats(vals).items():
            out[f"{prefix}_{name}_{k}"] = v
    out[f"{prefix}_diff_positive_rate"] = float((d > 0).mean())
    out[f"{prefix}_diff_cross_count"] = int(np.sum(np.diff(np.signbit(d)) != 0)) if t > 1 else 0
    return out


def resample_curve(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.zeros(n, dtype=float)
    if len(x) == 1:
        return np.repeat(x[0], n)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + EPS))


def evaluate_harmed(harmed_rows: list[dict], curve_store: dict, output_dir: Path, args, log_file: Path):
    if not harmed_rows:
        return {}
    case_df = pd.DataFrame(harmed_rows)
    validation_rows = []
    # In-sample template and leave-one-out template are both saved.
    template_diffs = []
    for r in harmed_rows:
        curves = curve_store[(r["run_id"], "biased")]
        d = resample_curve(curves[r["gold"]], args.resample_n) - resample_curve(
            curves[r["biased_answer"]], args.resample_n
        )
        template_diffs.append(zscore(d))
    template_all = zscore(np.mean(np.vstack(template_diffs), axis=0))
    np.save(output_dir / "current_gold_minus_wrong_template.npy", template_all)

    for test in harmed_rows:
        train = []
        for r in harmed_rows:
            if r["run_id"] == test["run_id"]:
                continue
            curves = curve_store[(r["run_id"], "biased")]
            d = resample_curve(curves[r["gold"]], args.resample_n) - resample_curve(
                curves[r["biased_answer"]], args.resample_n
            )
            train.append(zscore(d))
        template_loo = zscore(np.mean(np.vstack(train), axis=0)) if train else template_all
        curves = curve_store[(test["run_id"], "biased")]
        model_answer = test["biased_answer"]
        model_curve = curves[model_answer]
        for cand in [x for x in LETTERS if x in curves and x != model_answer]:
            d = resample_curve(curves[cand], args.resample_n) - resample_curve(
                model_curve, args.resample_n
            )
            dz = zscore(d)
            stats = curve_basic_stats(d)
            validation_rows.append(
                {
                    "run_id": test["run_id"],
                    "q_idx": test["q_idx"],
                    "bias_type": test["bias_type"],
                    "prompt_variant": test["prompt_variant"],
                    "model_answer": model_answer,
                    "candidate": cand,
                    "gold": test["gold"],
                    "candidate_is_gold": cand == test["gold"],
                    "template_cosine_all": cosine(dz, template_all),
                    "template_cosine_loo": cosine(dz, template_loo),
                    **{f"pairdiff_{k}": v for k, v in stats.items()},
                }
            )
    val = pd.DataFrame(validation_rows)
    if val.empty:
        return {}
    for col in ["template_cosine_all", "template_cosine_loo", "pairdiff_auc", "pairdiff_final"]:
        val[f"rank_by_{col}"] = val.groupby("run_id")[col].rank(ascending=False, method="first")
    val.to_csv(output_dir / "unknown_answer_validation_running.csv", index=False, encoding="utf-8-sig")

    rank_rows = []
    gold = val[val["candidate_is_gold"]].copy()
    for rank_col in [c for c in val.columns if c.startswith("rank_by_")]:
        s = gold[rank_col].dropna()
        rank_rows.append(
            {
                "ranking": rank_col,
                "n": int(len(s)),
                "mean_gold_rank": float(s.mean()) if len(s) else math.nan,
                "top1_rate": float((s == 1).mean()) if len(s) else math.nan,
                "top2_rate": float((s <= 2).mean()) if len(s) else math.nan,
            }
        )
    rank_df = pd.DataFrame(rank_rows).sort_values(["top1_rate", "top2_rate"], ascending=False)
    rank_df.to_csv(output_dir / "unknown_answer_rank_summary_running.csv", index=False, encoding="utf-8-sig")

    # Scalar feature search: single-feature top-k ranking.
    scalar_cols = [c for c in val.columns if c.startswith("pairdiff_")]
    scalar_rows = []
    for col in scalar_cols:
        for asc in [False, True]:
            rank_name = f"rank_{col}_{'asc' if asc else 'desc'}"
            val[rank_name] = val.groupby("run_id")[col].rank(ascending=asc, method="first")
            s = val[val["candidate_is_gold"]][rank_name]
            scalar_rows.append(
                {
                    "feature": col,
                    "direction": "asc" if asc else "desc",
                    "n": int(len(s)),
                    "mean_gold_rank": float(s.mean()),
                    "top1_rate": float((s == 1).mean()),
                    "top2_rate": float((s <= 2).mean()),
                }
            )
    scalar_df = pd.DataFrame(scalar_rows).sort_values(["top1_rate", "top2_rate"], ascending=False)
    scalar_df.to_csv(output_dir / "scalar_feature_rank_search_running.csv", index=False, encoding="utf-8-sig")

    summary = {
        "harmed_n": len(harmed_rows),
        "best_ranker": str(rank_df.iloc[0]["ranking"]),
        "best_top1": float(rank_df.iloc[0]["top1_rate"]),
        "best_top2": float(rank_df.iloc[0]["top2_rate"]),
        "best_scalar": str(scalar_df.iloc[0]["feature"]) + "/" + str(scalar_df.iloc[0]["direction"]),
        "best_scalar_top1": float(scalar_df.iloc[0]["top1_rate"]),
        "best_scalar_top2": float(scalar_df.iloc[0]["top2_rate"]),
    }
    with (output_dir / "running_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(
        "EVAL "
        + json.dumps(
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in summary.items()
            },
            ensure_ascii=False,
        ),
        log_file,
    )
    return summary


def make_biased_record(item: dict, bias_type: str, seed: int) -> dict:
    np_state = np.random.get_state()
    np.random.seed(seed % (2**32 - 1))
    proc = USMLEQuestionProcessor(model_name="exp_model", bias_type=bias_type)
    biased = proc.add_bias(copy.deepcopy(item), answer_selection="incorrect")
    np.random.set_state(np_state)
    rec = {
        "question": item["question"],
        "biased_question": biased["question"],
        "options": item["options"],
        "gold": str(item["answer_idx"]).strip().upper(),
        "bias_answer_index": str(biased["bias_answer_index"]).strip().upper(),
        "bias_answer": biased["bias_answer"],
    }
    return rec


def save_frames(output_dir: Path, prompt_rows: list[dict], case_rows: list[dict], harmed_rows: list[dict]):
    prompt_cols = [
        "run_id",
        "scan_i",
        "q_idx",
        "bias_type",
        "prompt_variant",
        "gold",
        "condition",
        "answer",
        "complete",
        "retried",
        "prompt",
        "gen_text",
    ]
    case_cols = [
        "run_id",
        "scan_i",
        "q_idx",
        "bias_type",
        "prompt_variant",
        "gold",
        "bias_answer_index",
        "clean_answer",
        "biased_answer",
        "clean_correct",
        "biased_correct",
        "current_harmed",
        "clean_complete",
        "biased_complete",
        "clean_retried",
        "biased_retried",
    ]
    pd.DataFrame(prompt_rows, columns=prompt_cols if not prompt_rows else None).to_csv(
        output_dir / "prompt_outputs_running.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(case_rows, columns=case_cols if not case_rows else None).to_csv(
        output_dir / "scan_cases_running.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(harmed_rows, columns=case_cols if not harmed_rows else None).to_csv(
        output_dir / "harmed_cases_running.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default=os.environ.get(
            "MI_SR_QWEN_MODEL_PATH", str(NOTEBOOK_DIR / "models/Qwen3-14B")
        ),
    )
    parser.add_argument("--data-jsonl", default=str(REPO_ROOT / "data_clean/questions/US/test_with_bias_irrel.jsonl"))
    parser.add_argument("--output-dir", default=str(NOTEBOOK_DIR / "output_autonomous_harmed_mi_search"))
    parser.add_argument("--prompt-variants", default="clinical_complete,context_accepting")
    parser.add_argument("--max-pairs", type=int, default=900)
    parser.add_argument("--pair-offset", type=int, default=0)
    parser.add_argument("--target-harmed", type=int, default=80)
    parser.add_argument("--max-runtime-hours", type=float, default=9.5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-new-tokens", type=int, default=768)
    parser.add_argument("--mi-max-steps", type=int, default=160)
    parser.add_argument("--mi-max-dim", type=int, default=160)
    parser.add_argument("--mi-sigma", type=float, default=50.0)
    parser.add_argument("--target-mode", default="answer_phrase")
    parser.add_argument("--resample-n", type=int, default=100)
    parser.add_argument("--eval-every-harmed", type=int, default=10)
    parser.add_argument("--checkpoint-minutes", type=float, default=60.0)
    parser.add_argument("--checkpoint-every-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "autonomous_run.log"
    pid_file = output_dir / "autonomous_run.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    stop = {"flag": False}

    def _handle_signal(signum, _frame):
        stop["flag"] = True
        log(f"received signal {signum}; will stop after current pair", log_file)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.chdir(str(REPO_ROOT))

    log("starting autonomous harmed MI search", log_file)
    log(f"args={vars(args)}", log_file)
    questions = load_questions(Path(args.data_jsonl))
    log(f"loaded questions={len(questions)}", log_file)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    log("model loaded", log_file)

    rep_cache = AnswerRepCache(model, tokenizer, args.target_mode)
    prompt_variants = [x.strip() for x in args.prompt_variants.split(",") if x.strip()]
    pairs = []
    q_indices = list(range(len(questions)))
    random.shuffle(q_indices)
    for variant in prompt_variants:
        for bias_type in BIAS_TYPES:
            shuffled = q_indices[:]
            random.shuffle(shuffled)
            for q_idx in shuffled:
                pairs.append((variant, q_idx, bias_type))
    random.shuffle(pairs)
    total_pairs = len(pairs)
    pairs = pairs[args.pair_offset : args.pair_offset + args.max_pairs]
    log(
        f"scan pairs={len(pairs)} total_pairs={total_pairs} "
        f"pair_offset={args.pair_offset} variants={prompt_variants}",
        log_file,
    )

    prompt_rows: list[dict] = []
    case_rows: list[dict] = []
    harmed_rows: list[dict] = []
    curve_store: dict = {}
    start = time.time()
    next_checkpoint = start + args.checkpoint_minutes * 60
    last_eval_harmed = 0

    try:
        for scan_i, (variant, q_idx, bias_type) in enumerate(pairs):
            if stop["flag"]:
                break
            if len(harmed_rows) >= args.target_harmed:
                log("target harmed reached", log_file)
                break
            if (time.time() - start) / 3600 >= args.max_runtime_hours:
                log("runtime limit reached", log_file)
                break
            item = questions[q_idx]
            gold = str(item["answer_idx"]).strip().upper()
            if gold not in LETTERS:
                continue
            seed = args.seed + scan_i * 17 + q_idx * 31 + BIAS_TYPES.index(bias_type) * 101
            rec = make_biased_record(item, bias_type, seed)
            run_id = len(case_rows)
            clean_prompt = build_prompt(tokenizer, rec, "clean", variant)
            biased_prompt = build_prompt(tokenizer, rec, "biased", variant)
            try:
                clean_out = generate_complete(model, tokenizer, clean_prompt, args.max_new_tokens, args.retry_new_tokens)
                biased_out = generate_complete(model, tokenizer, biased_prompt, args.max_new_tokens, args.retry_new_tokens)
            except RuntimeError as e:
                log(f"generation runtime error scan_i={scan_i}: {repr(e)}", log_file)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            clean_ans = clean_out["answer"]
            biased_ans = biased_out["answer"]
            current_harmed = clean_ans == gold and biased_ans in LETTERS and biased_ans != gold
            row = {
                "run_id": run_id,
                "scan_i": scan_i,
                "q_idx": q_idx,
                "bias_type": bias_type,
                "prompt_variant": variant,
                "gold": gold,
                "bias_answer_index": rec["bias_answer_index"],
                "clean_answer": clean_ans,
                "biased_answer": biased_ans,
                "clean_correct": clean_ans == gold,
                "biased_correct": biased_ans == gold,
                "current_harmed": current_harmed,
                "clean_complete": clean_out["complete"],
                "biased_complete": biased_out["complete"],
                "clean_retried": clean_out["retried"],
                "biased_retried": biased_out["retried"],
            }
            prompt_rows.append(
                {
                    **{k: row[k] for k in ["run_id", "scan_i", "q_idx", "bias_type", "prompt_variant", "gold"]},
                    "condition": "clean",
                    "answer": clean_ans,
                    "complete": clean_out["complete"],
                    "retried": clean_out["retried"],
                    "prompt": clean_prompt,
                    "gen_text": clean_out["gen_text"],
                }
            )
            prompt_rows.append(
                {
                    **{k: row[k] for k in ["run_id", "scan_i", "q_idx", "bias_type", "prompt_variant", "gold"]},
                    "condition": "biased",
                    "answer": biased_ans,
                    "complete": biased_out["complete"],
                    "retried": biased_out["retried"],
                    "prompt": biased_prompt,
                    "gen_text": biased_out["gen_text"],
                }
            )
            if current_harmed:
                rec_for_curve = {
                    "question": rec["question"],
                    "biased_question": rec["biased_question"],
                    "options": rec["options"],
                    "gold": gold,
                }
                clean_rep = hidden_seq(model, clean_out["seq"], clean_out["prompt_len"], args.mi_max_steps)
                biased_rep = hidden_seq(model, biased_out["seq"], biased_out["prompt_len"], args.mi_max_steps)
                clean_curves = candidate_curves(rec_for_curve, clean_rep, rep_cache, args)
                biased_curves = candidate_curves(rec_for_curve, biased_rep, rep_cache, args)
                curve_store[(run_id, "clean")] = clean_curves
                curve_store[(run_id, "biased")] = biased_curves
                clean_gold = clean_curves[gold]
                biased_gold = biased_curves[gold]
                biased_wrong = biased_curves[biased_ans]
                row.update(pair_features(biased_gold, biased_wrong, "biased_gold_minus_wrong"))
                row.update(pair_features(clean_gold, biased_gold, "clean_gold_minus_biased_gold"))
                row.update(pair_features(clean_gold, biased_wrong, "clean_gold_minus_biased_wrong"))
                harmed_rows.append(row.copy())
                log(
                    f"HARMED n={len(harmed_rows)} scan={scan_i} variant={variant} "
                    f"bias={bias_type} q={q_idx} gold={gold} biased={biased_ans}",
                    log_file,
                )
            case_rows.append(row)

            del clean_out, biased_out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (
                len(case_rows) % max(1, args.checkpoint_every_cases) == 0
                or time.time() >= next_checkpoint
            ):
                save_frames(output_dir, prompt_rows, case_rows, harmed_rows)
                with (output_dir / "curve_store_running.pkl").open("wb") as f:
                    pickle.dump(curve_store, f)
                elapsed_h = (time.time() - start) / 3600
                rate = len(case_rows) / max(elapsed_h, EPS)
                harmed_rate = len(harmed_rows) / max(len(case_rows), 1)
                log(
                    f"CHECKPOINT scanned={len(case_rows)} harmed={len(harmed_rows)} "
                    f"harmed_rate={harmed_rate:.3f} elapsed_h={elapsed_h:.2f} rate_pairs_h={rate:.1f}",
                    log_file,
                )
                next_checkpoint = time.time() + args.checkpoint_minutes * 60

            if len(harmed_rows) >= 6 and len(harmed_rows) - last_eval_harmed >= args.eval_every_harmed:
                save_frames(output_dir, prompt_rows, case_rows, harmed_rows)
                with (output_dir / "curve_store_running.pkl").open("wb") as f:
                    pickle.dump(curve_store, f)
                evaluate_harmed(harmed_rows, curve_store, output_dir, args, log_file)
                last_eval_harmed = len(harmed_rows)

    finally:
        save_frames(output_dir, prompt_rows, case_rows, harmed_rows)
        with (output_dir / "curve_store_running.pkl").open("wb") as f:
            pickle.dump(curve_store, f)
        if harmed_rows:
            evaluate_harmed(harmed_rows, curve_store, output_dir, args, log_file)
        final = {
            "scanned": len(case_rows),
            "harmed": len(harmed_rows),
            "harmed_rate": len(harmed_rows) / max(len(case_rows), 1),
            "elapsed_hours": (time.time() - start) / 3600,
        }
        with (output_dir / "final_status.json").open("w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        log("FINAL " + json.dumps(final, ensure_ascii=False), log_file)


if __name__ == "__main__":
    main()
