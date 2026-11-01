"""Plumbing smoke for R3/R7/R8/R9 with a tiny random Qwen3 and stub tokenizer. Numbers are meaningless."""
import sys
import torch
from transformers import BatchEncoding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
task, scripts, model_key, out = sys.argv[1:5]
sys.path.insert(0, scripts)

class Tok:
    pad_token_id = 0; eos_token_id = 1; bos_token_id = 2; chat_template = "x"
    def _ids(self, t): return [3 + (sum(map(ord, w)) % 5000) for w in t.split()] or [3]
    def __call__(self, text, return_tensors="pt", padding=False, **kw):
        texts = [text] if isinstance(text, str) else text
        ids = [self._ids(t)[:300] for t in texts]
        L = max(map(len, ids)); pad = [i + [0] * (L - len(i)) for i in ids]; mask = [[1]*len(i) + [0]*(L-len(i)) for i in ids]
        return BatchEncoding({"input_ids": torch.tensor(pad), "attention_mask": torch.tensor(mask)})
    def encode(self, text, add_special_tokens=False): return self._ids(text)
    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int): ids = [ids]
        return " ".join(f"w{int(i)}" + ("." if int(i) % 11 == 0 else "") for i in ids) + "\nFinal Answer: A. x"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw): return messages[0]["content"]

def fake(key, gpu_id=0):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=200000, hidden_size=256, intermediate_size=256, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=4096)
    return Qwen3ForCausalLM(cfg).to(torch.bfloat16).to("cuda:0").eval(), Tok(), {"dtype": "bf16", "attention_backend": "sdpa"}

if task == "r3":
    import r3_clean_run as m
    m.load_model_and_tokenizer = fake
    sys.argv = ["x", "--model", model_key, "--out", out, "--limit", "25", "--max-new", "6"]
else:
    import r789_extra_experiments as m
    m.load_model_and_tokenizer = fake
    sys.argv = ["x", "--task", task, "--model", model_key, "--out", out, "--n", "16", "--max-new", "4"]
m.main()
