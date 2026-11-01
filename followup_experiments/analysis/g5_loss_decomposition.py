"""Why the clean run loses accuracy: repair and damage among reviewed answers, with and without the cue (Qwen, excluding the 95 questions)."""
import os, json
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
A=os.environ.get("MI_SR_WORK_ROOT", "work").rstrip("/") + "/"; run=A+"u1_clean/qwen3_14b_reference/all_biases/"
src=os.environ["U1_ROOT"].rstrip("/")+"/output_mi_sr_full_targeted_v5/qwen3_14b_reference/all_biases/"
disc=set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "discovery_q_idx.json"))))
def auc(y,s):
    r=pd.Series(s).rank().to_numpy(); n1=y.sum(); n0=(~y).sum(); return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
def load(d,pf,lab_f,cols):
    p=pd.read_csv(d+pf,encoding="utf-8-sig").set_index("case_id"); l=pd.read_csv(d+lab_f,encoding="utf-8-sig").set_index("case_id").loc[p.index]
    r=pd.read_csv(d+"targeted_all_reevaluation.csv",encoding="utf-8-sig",usecols=["case_id","reeval_answer","reeval_complete"]).set_index("case_id").reindex(p.index)
    k=~l.q_idx.isin(disc).to_numpy(); p,l,r=p[k],l[k],r[k]
    g=l.gold.astype(str).to_numpy(); b=l.biased_correct.astype(bool).to_numpy(); ans=r.reeval_answer.astype(str).to_numpy()
    c=r.reeval_complete.fillna(False).astype(bool).to_numpy(); v=np.where(c,ans==g,b)
    sc=(p[["p_targeted_correct","p_generic_correct"]].max(axis=1)-p.p_none_correct).to_numpy(); return p,b,v,sc,c&(ans!=p.model_answer.astype(str).to_numpy())
for name,args in [("clean",(run,"frozen_mi_sr_policy_v3.csv","first_pass_evaluation_labels.csv",None)),("biased",(src,"binary_deploy_policy.csv","first_pass_evaluation_labels.csv",None))]:
    p,b,v,sc,chg=load(*args); t=sc>0; n=len(b)
    print("==",name,"n",n,"first acc %.2f"%(b.mean()*100),"correct",b.sum(),"wrong",(~b).sum())
    print("review rate overall %.1f; among correct %.1f; among wrong %.1f"%(t.mean()*100,t[b].mean()*100,t[~b].mean()*100))
    print("first acc selected %.1f unselected %.1f"%(b[t].mean()*100,b[~t].mean()*100))
    print("reviewed correct n",(t&b).sum(),"changed to wrong",(t&b&~v).sum(),"rate %.1f"%((~v)[t&b].mean()*100))
    print("reviewed wrong n",(t&~b).sum(),"changed to right",(t&~b&v).sum(),"rate %.1f"%(v[t&~b].mean()*100))
    print("answer changed among reviewed correct %.1f among reviewed wrong %.1f"%(chg[t&b].mean()*100,chg[t&~b].mean()*100))
    print("AUC first-correct: p_none %.3f pT %.3f pG %.3f score r %.3f"%(auc(b,p.p_none_correct.to_numpy()),auc(b,p.p_targeted_correct.to_numpy()),auc(b,p.p_generic_correct.to_numpy()),auc(b,sc)))
    print("mean p_none %.3f pT %.3f pG %.3f"%(p.p_none_correct.mean(),p.p_targeted_correct.mean(),p.p_generic_correct.mean()))
