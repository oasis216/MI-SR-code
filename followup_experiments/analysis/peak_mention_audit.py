"""Peak span vs explicit option mention audit (CPU only, existing outputs)."""
import re
import sys
import json
import numpy as np
import pandas as pd

base = sys.argv[1]
out = {}
rng = np.random.default_rng(0)
for model in ["qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"]:
    d = f"{base}/{model}/all_biases"
    fp = pd.read_csv(f"{d}/first_pass_prompts_outputs.csv", encoding="utf-8")
    fp = fp[fp.condition == "biased"].set_index("case_id")
    pk = pd.read_csv(f"{d}/mi_sr_peak_features/candidate_peak_features.csv", encoding="utf-8")
    pk = pk[pk.case_id.isin(fp.index)]
    opts = {}
    for cid, prompt in fp.prompt.items():
        seg = prompt.split("Options:")[-1]
        o = dict(re.findall(r"^([A-E])\.\s*(.+?)\s*$", seg, flags=re.M))
        opts[cid] = o

    def mentions(text, cid, letter):
        if not isinstance(text, str):
            return False
        t = text.lower()
        o = opts[cid].get(letter, "")
        if re.search(rf"(?<![A-Za-z]){letter}(?=[\.\):,]|\s*:)", text) or \
           re.search(rf"\b(option|answer|choice)\s*\(?{letter}\b", text, flags=re.I):
            return True
        o = o.lower().strip()
        return len(o) >= 4 and o in t

    own, other, rnd, same_span = [], [], [], []
    for cid, g in pk.groupby("case_id"):
        if cid not in opts or len(opts[cid]) < 5:
            continue
        sents = [s for s in re.split(r"(?<=[\.\!\?\n])", str(fp.at[cid, "gen_text"]).split("Final Answer")[0]) if len(s.strip()) > 10]
        spans = g.peak_span.fillna("").tolist()
        same_span.append(len(set(spans)) < len(spans))
        for _, r in g.iterrows():
            c = r.candidate
            own.append(mentions(r.peak_span, cid, c))
            for _, r2 in g.iterrows():
                if r2.candidate != c:
                    other.append(mentions(r2.peak_span, cid, c))
            if sents:
                rnd.append(mentions(sents[rng.integers(len(sents))], cid, c))
    out[model] = dict(cases=len(same_span), candidate_rows=len(own),
                      own_peak_mentions_c=float(np.mean(own)),
                      other_candidates_peak_mentions_c=float(np.mean(other)),
                      random_sentence_mentions_c=float(np.mean(rnd)),
                      cases_with_shared_peak_span=float(np.mean(same_span)))
    print(model, out[model], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=2)
