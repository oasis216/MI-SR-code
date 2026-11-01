"""Shared model registry and loading helpers for cross-model MI-SR runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verify_local_model import verify


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("MI_SR_MODEL_ROOT", ROOT / "models"))

MODEL_SPECS = {
    "qwen3_14b_reference": {
        "display_name": "Qwen3-14B-reference",
        "path": Path(
            os.environ.get("MI_SR_QWEN_MODEL_PATH", MODEL_ROOT / "Qwen3-14B")
        ),
        "min_weight_gib": 25.0,
        "model_family": "causal_lm",
    },
    "phi4": {
        "display_name": "Phi-4",
        "path": MODEL_ROOT / "Phi-4",
        "min_weight_gib": 25.0,
        "model_family": "causal_lm",
    },
    "gemma3_12b": {
        "display_name": "Gemma-3-12B-it",
        "path": MODEL_ROOT / "Gemma-3-12B-it",
        "min_weight_gib": 20.0,
        "model_family": "image_text_to_text",
    },
    "deepseek_r1_qwen14b": {
        "display_name": "DeepSeek-R1-Distill-Qwen-14B",
        "path": MODEL_ROOT / "DeepSeek-R1-Distill-Qwen-14B",
        "min_weight_gib": 25.0,
        "model_family": "causal_lm",
    },
    "llama31_8b_instruct": {
        "display_name": "Llama-3.1-8B-Instruct",
        "path": MODEL_ROOT / "Llama-3.1-8B-Instruct",
        "min_weight_gib": 14.0,
        "model_family": "causal_lm",
    },
    "mistral7b_instruct_v03": {
        "display_name": "Mistral-7B-Instruct-v0.3",
        "path": MODEL_ROOT / "Mistral-7B-Instruct-v0.3",
        "min_weight_gib": 13.0,
        "model_family": "causal_lm",
    },
    "glm4_9b_chat": {
        "display_name": "GLM-4-9B-Chat",
        "path": MODEL_ROOT / "GLM-4-9B-Chat",
        "min_weight_gib": 17.0,
        "model_family": "causal_lm",
    },
    "biomistral7b": {
        "display_name": "BioMistral-7B",
        "path": MODEL_ROOT / "BioMistral-7B",
        "min_weight_gib": 13.0,
        "model_family": "causal_lm",
    },
    "biollama_openbiollm_8b": {
        "display_name": "Llama3-OpenBioLLM-8B",
        "path": MODEL_ROOT / "BioLlama-OpenBioLLM-8B",
        "min_weight_gib": 13.0,
        "model_family": "causal_lm",
    },
}


def model_spec(model_key: str) -> dict:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key {model_key!r}; choose from {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[model_key]


def verify_model(model_key: str) -> dict:
    spec = model_spec(model_key)
    return verify(Path(spec["path"]), float(spec["min_weight_gib"]))


def load_model_and_tokenizer(model_key: str, gpu_id: int = 0):
    spec = model_spec(model_key)
    verification = verify_model(model_key)
    if not verification["complete"]:
        raise RuntimeError(
            f"Model snapshot is incomplete: {json.dumps(verification, indent=2)}"
        )
    model_path = Path(spec["path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if spec["model_family"] == "causal_lm" and tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|begin_of_text|>' if loop.first else '' }}"
            "{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{% endif %}"
        )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "pretrained_model_name_or_path": model_path,
        "dtype": dtype,
        "device_map": {"": gpu_id} if torch.cuda.is_available() else "cpu",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if spec["model_family"] == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    else:
        model_class = AutoModelForCausalLM
    try:
        model = model_class.from_pretrained(attn_implementation="sdpa", **kwargs)
        attention_backend = "sdpa"
    except (TypeError, ValueError, ImportError):
        model = model_class.from_pretrained(**kwargs)
        attention_backend = "model_default"
    model.eval()
    metadata = {
        "model_key": model_key,
        "display_name": spec["display_name"],
        "model_path": str(model_path),
        "model_family": spec["model_family"],
        "gpu_id": gpu_id if torch.cuda.is_available() else None,
        "dtype": str(dtype),
        "attention_backend": attention_backend,
        "verification": verification,
        "architectures": list(getattr(model.config, "architectures", []) or []),
        "model_type": getattr(model.config, "model_type", None),
    }
    return model, tokenizer, metadata


def output_root(model_key: str) -> Path:
    return ROOT / f"output_cross_model_{model_key}_v1"
