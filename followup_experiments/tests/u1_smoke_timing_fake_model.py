"""Plumbing smoke of the timing script with a tiny random model and stub tokenizer. Timings are meaningless."""
import sys
import torch
from transformers import BatchEncoding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
sys.path.insert(0, sys.argv[1])
import u1_timing_benchmark as tb

class Tok:
    pad_token_id = 0; eos_token_id = 1; bos_token_id = 2; chat_template = "x"
    def __call__(self, text, return_tensors="pt", padding=False, **kw):
        texts = [text] if isinstance(text, str) else text
        ids = [[3 + (ord(c) % 5000) for c in t][:300] or [3] for t in texts]
        L = max(map(len, ids)); pad = [i + [0] * (L - len(i)) for i in ids]; mask = [[1]*len(i) + [0]*(L-len(i)) for i in ids]
        return BatchEncoding({"input_ids": torch.tensor(pad), "attention_mask": torch.tensor(mask)})
    def encode(self, text, add_special_tokens=False):
        return [3 + (sum(map(ord, w)) % 5000) for w in text.split()] or [3]
    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int): ids = [ids]
        return " ".join(f"w{int(i)}" + ("." if int(i) % 11 == 0 else "") for i in ids)

def fake(key, gpu_id=0):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=200000, hidden_size=256, intermediate_size=256, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=4096)
    return Qwen3ForCausalLM(cfg).to(torch.bfloat16).to("cuda:0").eval(), Tok(), {"dtype": "bf16", "attention_backend": "sdpa"}

tb.load_model_and_tokenizer = fake
tb.MAX_NEW[sys.argv[2]] = 12
sys.argv = ["x", "--model", sys.argv[2], "--out", sys.argv[3], "--n", "4"]
tb.main()
