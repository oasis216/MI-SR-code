"""Extract candidate-letter logits at the model's final-answer commitment step.

The functions in this module reuse frozen generation sequences. They do not
regenerate answers or modify the existing MI traces.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import pandas as pd
import torch


LETTERS = ("A", "B", "C", "D", "E")


def _as_token_list(token_ids) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        return token_ids.detach().cpu().reshape(-1).tolist()
    return [int(token_id) for token_id in token_ids]


def find_final_answer_token_index(gen_ids, tokenizer) -> int:
    """Return the generated-token index containing the final answer letter."""
    ids = _as_token_list(gen_ids)
    marker = tokenizer.encode("Final Answer:", add_special_tokens=False)

    for start in range(len(ids) - len(marker) - 1, -1, -1):
        if ids[start : start + len(marker)] == marker:
            answer_index = start + len(marker)
            if answer_index < len(ids):
                token_text = tokenizer.decode([ids[answer_index]])
                if re.search(r"[A-E]", token_text):
                    return answer_index

    # Tokenization can differ at a newline boundary. The fallback is slower but
    # is used only if the exact token subsequence cannot be found.
    for answer_index in range(len(ids) - 1, -1, -1):
        prefix = tokenizer.decode(ids[: answer_index + 1])
        if re.search(r"Final\s+Answer\s*:\s*[A-E]", prefix, flags=re.IGNORECASE):
            return answer_index
    raise ValueError("Could not locate a generated 'Final Answer: X' token")


def get_candidate_token_ids(tokenizer) -> dict[str, int]:
    """Resolve the one-token candidate IDs used after ``Final Answer:``."""
    result = {}
    for letter in LETTERS:
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Candidate {letter!r} is not one token: {ids}")
        result[letter] = ids[0]
    return result


def _build_prefix(record, tokenizer) -> torch.Tensor:
    answer_index = find_final_answer_token_index(record["gen_ids"], tokenizer)
    sequence = record["seq"]
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.as_tensor(sequence, dtype=torch.long)
    sequence = sequence.detach().cpu().reshape(-1)
    prefix_length = int(record["prompt_len"]) + answer_index
    if prefix_length <= 0 or prefix_length > sequence.numel():
        raise ValueError(
            f"Invalid final-answer prefix length {prefix_length}/{sequence.numel()}"
        )
    return sequence[:prefix_length]


def _find_subsequence(ids: list[int], pattern: list[int]) -> int | None:
    for start in range(len(ids) - len(pattern) + 1):
        if ids[start : start + len(pattern)] == pattern:
            return start
    return None


def _find_marker_start(
    generated: list[int], tokenizer, marker_text: str
) -> int | None:
    """Locate a marker across equivalent whitespace/Markdown tokenizations."""

    for prefix in ("", " ", "\n", "**", "\n**", " **"):
        marker = tokenizer.encode(prefix + marker_text, add_special_tokens=False)
        marker_start = _find_subsequence(generated, marker)
        if marker_start is not None:
            return marker_start
    return None


def _find_marker_start_from_text(record, tokenizer, marker_text: str) -> int | None:
    """Fallback for tokenizers whose decoded marker spans do not re-encode exactly."""

    generated_text = record.get("gen_text")
    if not generated_text:
        return None
    match = list(re.finditer(re.escape(marker_text), generated_text, re.IGNORECASE))
    if not match:
        return None
    prefix_text = generated_text[: match[-1].start()]
    token_count = len(tokenizer.encode(prefix_text, add_special_tokens=False))
    generated = _as_token_list(record["gen_ids"])
    if 0 <= token_count <= len(generated):
        return token_count
    return None


def build_checkpoint_prefixes(record, tokenizer) -> dict[str, torch.Tensor]:
    """Build frozen prefixes before evidence comparison and final commitment."""
    sequence = record["seq"]
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.as_tensor(sequence, dtype=torch.long)
    sequence = sequence.detach().cpu().reshape(-1)
    prompt_length = int(record["prompt_len"])
    generated = _as_token_list(record["gen_ids"])

    checkpoints = {"prompt_end": sequence[:prompt_length]}
    for name, marker_text in (
        ("pre_comparison", "Comparison:"),
        ("pre_final", "Final Answer:"),
    ):
        marker_start = _find_marker_start(generated, tokenizer, marker_text)
        if marker_start is None:
            marker_start = _find_marker_start_from_text(record, tokenizer, marker_text)
        if marker_start is not None:
            checkpoints[name] = sequence[: prompt_length + marker_start]
    return checkpoints


def extract_checkpoint_probe_logits(
    model,
    tokenizer,
    first_pass_store: dict,
    case_ids: Iterable[str],
    batch_size: int = 4,
    probe_text: str = "\nProvisional Answer:",
) -> pd.DataFrame:
    """Probe A-E support at fixed stages of an already generated rationale.

    The same short probe is appended to each frozen prefix. No sampled tokens are
    generated, so the original answer and trajectory remain unchanged.
    """
    token_ids = get_candidate_token_ids(tokenizer)
    probe_ids = torch.tensor(
        tokenizer.encode(probe_text, add_special_tokens=False), dtype=torch.long
    )
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    items = []
    errors = {}
    for case_id in case_ids:
        try:
            checkpoints = build_checkpoint_prefixes(
                first_pass_store[case_id], tokenizer
            )
            for checkpoint, prefix in checkpoints.items():
                items.append(
                    (case_id, checkpoint, torch.cat([prefix, probe_ids], dim=0))
                )
        except Exception as exc:
            errors[case_id] = str(exc)

    rows = []
    model.eval()
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]
        max_length = max(prefix.numel() for _, _, prefix in batch)
        input_ids = torch.full(
            (len(batch), max_length), pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, (_, _, prefix) in enumerate(batch):
            prefix = prefix.to(device)
            input_ids[row_index, -prefix.numel() :] = prefix
            attention_mask[row_index, -prefix.numel() :] = 1

        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            next_logits = output.logits[:, -1, :].float()
        selected = torch.stack(
            [next_logits[:, token_ids[letter]] for letter in LETTERS], dim=1
        )
        probabilities = torch.softmax(selected, dim=1).cpu()
        log_probabilities = torch.log_softmax(selected, dim=1).cpu()

        for row_index, (case_id, checkpoint, prefix) in enumerate(batch):
            probability = probabilities[row_index]
            entropy = float(-(probability * log_probabilities[row_index]).sum())
            row = {
                "case_id": case_id,
                "checkpoint": checkpoint,
                "prefix_tokens_with_probe": int(prefix.numel()),
                "candidate_entropy": entropy,
                "candidate_entropy_normalized": entropy / math.log(len(LETTERS)),
            }
            for letter_index, letter in enumerate(LETTERS):
                row[f"logit_{letter}"] = float(selected[row_index, letter_index].cpu())
                row[f"prob_{letter}"] = float(probability[letter_index])
            rows.append(row)
        del input_ids, attention_mask, output, next_logits, selected

    result = pd.DataFrame(rows)
    result.attrs["errors"] = errors
    return result


def extract_final_answer_logits(
    model,
    tokenizer,
    first_pass_store: dict,
    case_ids: Iterable[str],
    batch_size: int = 4,
) -> pd.DataFrame:
    """Compute A-E logits from frozen sequences at final-answer commitment."""
    case_ids = list(case_ids)
    token_ids = get_candidate_token_ids(tokenizer)
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    prefixes = []
    errors = {}
    for case_id in case_ids:
        try:
            prefixes.append((case_id, _build_prefix(first_pass_store[case_id], tokenizer)))
        except Exception as exc:  # Preserve per-case failures for auditability.
            errors[case_id] = str(exc)

    rows = []
    model.eval()
    for batch_start in range(0, len(prefixes), batch_size):
        batch = prefixes[batch_start : batch_start + batch_size]
        max_length = max(prefix.numel() for _, prefix in batch)
        input_ids = torch.full(
            (len(batch), max_length), pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, (_, prefix) in enumerate(batch):
            prefix = prefix.to(device)
            input_ids[row_index, -prefix.numel() :] = prefix
            attention_mask[row_index, -prefix.numel() :] = 1

        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            next_logits = output.logits[:, -1, :].float()

        selected = torch.stack(
            [next_logits[:, token_ids[letter]] for letter in LETTERS], dim=1
        )
        probabilities = torch.softmax(selected, dim=1).cpu()
        log_probabilities = torch.log_softmax(selected, dim=1).cpu()

        for row_index, (case_id, prefix) in enumerate(batch):
            probability = probabilities[row_index]
            entropy = float(-(probability * log_probabilities[row_index]).sum())
            normalized_entropy = entropy / math.log(len(LETTERS))
            row = {
                "case_id": case_id,
                "prefix_tokens": int(prefix.numel()),
                "candidate_entropy": entropy,
                "candidate_entropy_normalized": normalized_entropy,
            }
            for letter_index, letter in enumerate(LETTERS):
                row[f"logit_{letter}"] = float(selected[row_index, letter_index].cpu())
                row[f"prob_{letter}"] = float(probability[letter_index])
            rows.append(row)

        del input_ids, attention_mask, output, next_logits, selected

    result = pd.DataFrame(rows)
    result.attrs["errors"] = errors
    return result
