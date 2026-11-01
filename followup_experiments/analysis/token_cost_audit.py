"""C-3: token-level compute accounting for MI-SR vs All Targeted (CPU, existing outputs).

Exact token counts: first-pass prompt/generation (generation store), endpoint probe prefixes
(checkpoint_probe_logits), peak-probe prefixes (probe_requests steps + prompt length).
Estimated: review prompt/generation and option strings, via per-model chars-per-token ratios
measured on the first pass (tokenizers are not available locally).
"""
import json
import pickle
import re
import sys
import pandas as pd

base, out = sys.argv[1], sys.argv[2]
res = {}
for m in sys.argv[3].split(","):
    d = f"{base}/{m}/all_biases"
    pol = pd.read_csv(f"{d}/binary_deploy_policy.csv", encoding="utf-8-sig")
    cases = set(pol.case_id)
    with open(f"{d}/first_pass_biased_generation_store.pkl", "rb") as f:
        store = pickle.load(f)
    fp = pd.read_csv(f"{d}/first_pass_prompts_outputs.csv", encoding="utf-8-sig")
    fp = fp[(fp.condition == "biased") & fp.case_id.isin(cases)].set_index("case_id")
    P = {c: store[c]["prompt_len"] for c in cases}
    G = {c: len(store[c]["gen_ids"]) for c in cases}
    # chars-per-token ratios from the first pass
    r_prompt = sum(len(fp.at[c, "prompt"]) for c in fp.index) / sum(P[c] for c in fp.index)
    r_gen = sum(len(str(store[c]["gen_text"])) for c in cases) / sum(G[c] for c in cases)

    ck = pd.read_csv(f"{d}/checkpoint_probe_logits.csv", encoding="utf-8-sig")
    ck = ck[ck.case_id.isin(cases)]
    endpoint = ck.groupby("case_id").prefix_tokens_with_probe.sum()
    pe = ck[ck.checkpoint == "prompt_end"].set_index("case_id").prefix_tokens_with_probe
    suffix = float((pe - pd.Series(P)).median())
    req = pd.read_csv(f"{d}/mi_sr_peak_features/probe_requests.csv", encoding="utf-8-sig")
    req = req[req.pilot_case_id.isin(cases)].drop_duplicates(["pilot_case_id", "step"])
    req["tok"] = req.pilot_case_id.map(P) + req.step + suffix
    peak = req.groupby("pilot_case_id").tok.sum()

    def opt_tokens(prompt):
        seg = prompt.split("Options:")[-1]
        opts = re.findall(r"^([A-E]\.\s*.+?)\s*$", seg, flags=re.M)[:5]
        return sum(len(o) / r_gen + 1 for o in opts)
    opt = {c: opt_tokens(fp.at[c, "prompt"]) for c in fp.index}

    tg = pd.read_csv(f"{d}/targeted_all_reevaluation.csv", encoding="utf-8-sig")
    tg = tg[tg.case_id.isin(cases)].set_index("case_id")
    tg["p_tok"] = tg.prompt.str.len() / r_prompt
    tg["g_tok"] = tg.gen_text.fillna("").str.len() / r_gen
    sel = set(pol.loc[pol.action == "targeted", "case_id"])

    n = len(cases)
    first_prefill = sum(P.values()); first_decode = sum(G.values())
    teacher = sum(P[c] + G[c] for c in cases)
    extra_prefill = teacher + endpoint.sum() + peak.sum() + sum(opt.values())
    misr_rev_pre = tg.loc[tg.index.isin(sel), "p_tok"].sum()
    misr_rev_dec = tg.loc[tg.index.isin(sel), "g_tok"].sum()
    all_rev_pre, all_rev_dec = tg.p_tok.sum(), tg.g_tok.sum()
    res[m] = dict(
        n=n, reviews_misr=len(sel), reviews_all=len(tg),
        chars_per_token_prompt=round(r_prompt, 3), chars_per_token_gen=round(r_gen, 3),
        probe_suffix_tokens=suffix,
        per_question=dict(
            first_pass_prefill=first_prefill / n, first_pass_decode=first_decode / n,
            misr_extra_prefill_total=extra_prefill / n,
            misr_extra_teacher_forced=teacher / n, misr_extra_endpoint_probes=endpoint.sum() / n,
            misr_extra_peak_probes=peak.sum() / n, misr_extra_option_encoding=sum(opt.values()) / n,
            peak_probe_prefixes=len(req) / n,
            misr_review_prefill=misr_rev_pre / n, misr_review_decode=misr_rev_dec / n,
            all_targeted_review_prefill=all_rev_pre / n, all_targeted_review_decode=all_rev_dec / n),
        decode_tokens_saved_per_question=(all_rev_dec - misr_rev_dec) / n,
        review_prefill_saved_per_question=(all_rev_pre - misr_rev_pre) / n,
    )
    q = res[m]["per_question"]
    res[m]["misr_total_decode"] = q["first_pass_decode"] + q["misr_review_decode"]
    res[m]["all_targeted_total_decode"] = q["first_pass_decode"] + q["all_targeted_review_decode"]
    res[m]["misr_total_prefill"] = q["first_pass_prefill"] + q["misr_extra_prefill_total"] + q["misr_review_prefill"]
    res[m]["all_targeted_total_prefill"] = q["first_pass_prefill"] + q["all_targeted_review_prefill"]
    print(m, json.dumps(res[m], indent=1), flush=True)
json.dump(res, open(out, "w"), indent=2)
