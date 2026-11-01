import os
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
import numpy as np, pandas as pd, glob
A=os.environ.get("MI_SR_WORK_ROOT", "work").rstrip("/") + "/"; src=os.environ["U1_ROOT"].rstrip("/")+"/output_mi_sr_full_targeted_v5/qwen3_14b_reference/all_biases/"
lab=pd.read_csv(src+"first_pass_evaluation_labels.csv",encoding="utf-8-sig").set_index("case_id")
rt=pd.read_csv(src+"targeted_all_reevaluation.csv",encoding="utf-8-sig",usecols=["case_id","challenger","reeval_answer","reeval_complete"]).set_index("case_id")
r7=pd.read_csv(A+"u1_extra/r7_reviews_qwen3_14b_reference.csv",encoding="utf-8-sig"); print(r7.columns.tolist()[:10],len(r7))
r7=r7.drop_duplicates("case_id").set_index("case_id"); ids=r7.index
L=lab.loc[ids]; gold=L.gold.astype(str).to_numpy(); base=L.biased_correct.astype(bool).to_numpy(); cr=pd.read_csv(src+"clean_reference_outputs.csv",encoding="utf-8-sig").drop_duplicates("q_idx").set_index("q_idx"); cc=cr.clean_correct.reindex(L.q_idx).astype(str).str.lower().eq("true").to_numpy(); print("clean_correct among sample %.1f"%(cc.mean()*100)); q=L.q_idx.to_numpy()
T=rt.loc[ids]
def res(fr):
    c=fr.reeval_complete.fillna(False).astype(bool).to_numpy()
    return np.where(c,fr.reeval_answer.astype(str).to_numpy()==gold,base),c
vt,ct=res(T); vn,cn=res(r7)
print("complete targeted %.3f noch %.3f"%(ct.mean(),cn.mean()))
hit=T.challenger.astype(str).to_numpy()==gold
na=r7.reeval_answer.astype(str).to_numpy(); fa=L.biased_answer.astype(str).to_numpy(); ta=T.reeval_answer.astype(str).to_numpy(); ch=T.challenger.astype(str).to_numpy()
print("changed answer: targeted %.1f noch %.1f"%((ta!=fa).mean()*100,(na!=fa).mean()*100))
print("moved to challenger: targeted %.1f noch %.1f"%((ta==ch).mean()*100,(na==ch).mean()*100))
rng=np.random.default_rng(1)
def boot(d):
    uq,inv=np.unique(q[m],return_inverse=True); s=np.bincount(inv,d,len(uq)); c=np.bincount(inv,minlength=len(uq)); dr=rng.integers(0,len(uq),(4000,len(uq))); b=s[dr].sum(1)/c[dr].sum(1)*100
    return np.quantile(b,[.025,.975])
groups={"bias-induced err & ch=gold":(cc&~base&hit),"bias-induced err & ch!=gold":(cc&~base&~hit),"no-cue err & ch=gold":(~cc&~base&hit),"no-cue err & ch!=gold":(~cc&~base&~hit),"all first-wrong":~base,"first-correct":base}
for name,m in groups.items():
    if m.sum()==0: continue
    d=vt[m].astype(int)-vn[m].astype(int); lo,hi=boot(d)
    print(name,"n",m.sum(),"correct-after targeted %.1f noch %.1f diff %.1f [%.1f,%.1f]"%(vt[m].mean()*100,vn[m].mean()*100,d.mean()*100,lo,hi))
