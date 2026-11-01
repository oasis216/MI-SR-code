# Why pG is kept as a risk signal. Reads archived results only; CPU only.
import os, json
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
A = os.environ.get("MI_SR_WORK_ROOT", "work").rstrip("/") + "/"
disc = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "discovery_q_idx.json"))))
for m in ("qwen3_14b_reference", "deepseek_r1_qwen14b", "mistral7b_instruct_v03"):
    d = os.environ["U1_ROOT"].rstrip("/") + f"/output_mi_sr_full_targeted_v5/{m}/all_biases/"
    pol = pd.read_csv(d + "binary_deploy_policy.csv", encoding="utf-8-sig").set_index("case_id")
    lab = pd.read_csv(d + "first_pass_evaluation_labels.csv", encoding="utf-8-sig").set_index("case_id").loc[pol.index]
    rv = pd.read_csv(d + "targeted_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id").reindex(pol.index)
    gv = pd.read_csv(d + "generic_all_reevaluation.csv", encoding="utf-8-sig", usecols=["case_id", "reeval_answer", "reeval_complete"]).set_index("case_id").reindex(pol.index)
    base = lab.biased_correct.astype(bool).to_numpy(); gold = lab.gold.astype(str).to_numpy(); q = lab.q_idx.to_numpy()
    T = np.where(rv.reeval_complete.fillna(False).astype(bool).to_numpy(), rv.reeval_answer.astype(str).to_numpy() == gold, base)
    G = np.where(gv.reeval_complete.fillna(False).astype(bool).to_numpy(), gv.reeval_answer.astype(str).to_numpy() == gold, base)
    pN, pT, pG = [pol[c].to_numpy() for c in ("p_none_correct", "p_targeted_correct", "p_generic_correct")]
    keep = ~np.isin(q, list(disc))
    base, T, G, pN, pT, pG, q = base[keep], T[keep], G[keep], pN[keep], pT[keep], pG[keep], q[keep]
    print("==", m, "n", len(base))
    print("head AUC  pN->first %.3f  pT->T %.3f  pG->G %.3f" % (roc_auc_score(base, pN), roc_auc_score(T, pT), roc_auc_score(G, pG)))
    print("pG->T %.3f  pT->G %.3f" % (roc_auc_score(T, pG), roc_auc_score(G, pT)))
    print("first wrong AUC  -pT %.3f  -pG %.3f  -pN %.3f" % (roc_auc_score(~base, -pT), roc_auc_score(~base, -pG), roc_auc_score(~base, -pN)))
    gain = T.astype(int) - base.astype(int); ch = gain != 0
    sT, sG, sM = pT - pN, pG - pN, np.maximum(pT, pG) - pN
    print("fix-vs-damage AUC (changed n=%d)  pT-pN %.3f  pG-pN %.3f  max-pN %.3f" % (ch.sum(), roc_auc_score(gain[ch] > 0, sT[ch]), roc_auc_score(gain[ch] > 0, sG[ch]), roc_auc_score(gain[ch] > 0, sM[ch])))
    k = int((sM > 0).sum())
    rng = np.random.default_rng(0)
    for nm, s in (("pT-pN", sT), ("pG-pN", sG), ("max-pN", sM)):
        o = np.argsort(-(s + rng.normal(0, 1e-9, len(s))))[:k]
        print("  top k=%d by %-6s acc %.2f fix %d dmg %d net %d" % (k, nm, (base.sum() + gain[o].sum()) / len(base) * 100, (gain[o] > 0).sum(), (gain[o] < 0).sum(), gain[o].sum()))
    # Fixed-threshold rules and their coverage
    for nm, mask in (("pT>pN", sT > 0), ("pG>pN", sG > 0), ("max>pN", sM > 0), ("pG only", (sG > 0) & (sT <= 0))):
        print("  rule %-7s n %d rate %.1f first_wrong_in %d recall %.1f net %d acc %.2f" % (nm, mask.sum(), mask.mean() * 100, (~base & mask).sum(), (~base & mask).sum() / (~base).sum() * 100, gain[mask].sum(), (base.sum() + gain[mask].sum()) / len(base) * 100))
