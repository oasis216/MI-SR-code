"""Plumbing smoke of the R5 script with a tiny random model; features and leads are meaningless."""
import sys
import pandas as pd
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
sys.path.insert(0, sys.argv[1])
import g4_token_probability_baseline as tb

class Tok:
    def decode(self, ids, skip_special_tokens=True): return "A" if int(ids[0]) % 7 == 0 else "w"

def fake(key, gpu_id=0):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=200000, hidden_size=64, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    return Qwen3ForCausalLM(cfg).to(torch.bfloat16).to("cuda:0").eval(), Tok(), {}

tb.load_model_and_tokenizer = fake
real = pd.read_csv
def limited(path, *a, **k):
    x = real(path, *a, **k)
    return x.head(240) if str(path).endswith("binary_deploy_policy.csv") else x
tb.pd.read_csv = limited
sys.argv = ["x", "--model", sys.argv[2], "--out", sys.argv[3]]
tb.main()
