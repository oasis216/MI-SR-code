"""Integration smoke with a tiny random-weight Qwen3 and a stub tokenizer; proves plumbing only, numbers are meaningless."""
import os, sys, re
import torch
from transformers import BatchEncoding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
sys.path.insert(0, sys.argv[1])
import u1_numeric_rerun as rr

class Tok:
    pad_token_id = 0; eos_token_id = 1; bos_token_id = 2; chat_template = "x"
    def __call__(self, text, return_tensors="pt", padding=False):
        texts = [text] if isinstance(text, str) else text
        ids = [[3 + (ord(c) % 5000) for c in t][:64] or [3] for t in texts]
        L = max(map(len, ids)); pad = [i + [0] * (L - len(i)) for i in ids]; mask = [[1]*len(i) + [0]*(L-len(i)) for i in ids]
        return BatchEncoding({"input_ids": torch.tensor(pad), "attention_mask": torch.tensor(mask)})
    def encode(self, text, add_special_tokens=False):
        return [3 + (sum(map(ord, w)) % 5000) for w in text.split()] or [3]
    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int): ids = [ids]
        return " ".join(f"w{int(i)}" + ("." if int(i) % 11 == 0 else "") for i in ids)
    def convert_tokens_to_ids(self, t): return 3 + (ord(t[-1]) % 5000)

def fake(key, gpu_id=0):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=200000, hidden_size=256, intermediate_size=256, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=4096)
    m = Qwen3ForCausalLM(cfg).to(torch.bfloat16).to("cuda:0").eval()
    return m, Tok(), {"dtype": "bf16", "attention_backend": "sdpa"}

rr.load_model_and_tokenizer = fake
sys.argv = ["x", "--model", sys.argv[2], "--out", sys.argv[3], "--limit", "40"]
rr.main()
