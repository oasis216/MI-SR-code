import os, json
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd
A=os.environ.get("MI_SR_WORK_ROOT", "work").rstrip("/") + "/"; run=A+"u1_clean/qwen3_14b_reference/all_biases/"
src=os.environ["U1_ROOT"].rstrip("/")+"/output_mi_sr_full_targeted_v5/qwen3_14b_reference/all_biases/"
print(open(run+"deploy_policy_manifest_v3.json").read()[:800])
print(sorted(os.listdir(src))[:40])
pol=pd.read_csv(run+"frozen_mi_sr_policy_v3.csv",encoding="utf-8-sig").set_index("case_id")
lab=pd.read_csv(run+"first_pass_evaluation_labels.csv",encoding="utf-8-sig").set_index("case_id").loc[pol.index]
rv=pd.read_csv(run+"targeted_all_reevaluation.csv",encoding="utf-8-sig").set_index("case_id").reindex(pol.index)
gold=lab.gold.astype(str).to_numpy(); base=lab.biased_correct.astype(bool).to_numpy(); q=lab.q_idx.to_numpy()
comp=rv.reeval_complete.fillna(False).astype(bool).to_numpy()
rev=np.where(comp,rv.reeval_answer.astype(str).to_numpy()==gold,base)
score=pol[["p_targeted_correct","p_generic_correct"]].max(axis=1).to_numpy()-pol.p_none_correct.to_numpy()
t=score>0
print("n",len(t),"complete",comp.mean(),"p_none mean",pol.p_none_correct.mean(),"pT",pol.p_targeted_correct.mean(),"pG",pol.p_generic_correct.mean())
print("selected",t.mean(),"score q",np.quantile(score,[.1,.25,.5,.75,.9]).round(3))
gain=rev.astype(int)-base.astype(int)
disc=set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "discovery_q_idx.json")))) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "discovery_q_idx.json")) else set()
print("disc",len(disc))
rng=np.random.default_rng(1)
def pair(mask,a,b,name):
    d=(gain*a-gain*b)[mask]; n=len(d); idx=rng.integers(0,n,(4000,n)); bs=d[idx].mean(1)*100
    print(name,"n",n,"diff %.2f [%.2f,%.2f]"%(d.mean()*100,*np.quantile(bs,[.025,.975])))
allm=np.ones(len(t),bool); keep=~np.isin(q,list(disc))
for m,nm in ((allm,"all"),(keep,"excl95")):
    pair(m,t.astype(int),np.ones(len(t),int),"MISR-AT "+nm)
# review-only properties
print("selected flip rate",(rv.reeval_answer.astype(str).to_numpy()!=lab.biased_answer.astype(str).to_numpy())[t].mean(),"unselected flip",(rv.reeval_answer.astype(str).to_numpy()!=lab.biased_answer.astype(str).to_numpy())[~t].mean())
print("acc first selected",base[t].mean(),"unselected",base[~t].mean())
print("gain selected",gain[t].mean()*100,"unselected",gain[~t].mean()*100)
# same-count random
k=t.sum(); dr=[]
for i in range(4000):
    s=rng.choice(len(t),k,replace=False); dr.append(gain[s].sum()/len(t)*100)
print("MISR %.2f random same count %.2f [%.2f,%.2f]"%(gain[t].sum()/len(t)*100,np.mean(dr),*np.quantile(dr,[.025,.975])))
# score-ranked budgets
order=np.argsort(-score)
for f in (.1,.2,.3,.5,.81,1):
    c=int(f*len(t)); print("top%.0f%% gain %.2f"%(f*100,gain[order[:c]].sum()/len(t)*100),"random %.2f"%(gain.mean()*f*100))
# robust vs fragile using biased labels
bl=pd.read_csv(src+"first_pass_evaluation_labels.csv",encoding="utf-8-sig")
print(bl.columns.tolist()[:20], len(bl))
g=bl.groupby("q_idx").agg(flip=("biased_correct",lambda s:(~s.astype(bool)).mean()),cc=("clean_correct","first"))
fr=(g.flip.reindex(q).to_numpy())
for name,m in (("clean-correct robust(no cue flip)",base&(fr==0)),("clean-correct fragile(any wrong)",base&(fr>0)),("clean-wrong",~base)):
    print(name,"n",m.sum(),"review_rate %.1f"%(t[m].mean()*100),"damage|sel %.1f"%((~rev&base)[m&t].mean()*100),"fix|sel %.1f"%((rev&~base)[m&t].mean()*100),"AT damage %.1f AT fix %.1f"%((~rev&base)[m].mean()*100,(rev&~base)[m].mean()*100))
