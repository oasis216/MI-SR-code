import os, json
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
A=os.environ.get("MI_SR_WORK_ROOT", "work").rstrip("/") + "/"; run=A+"u1_clean/qwen3_14b_reference/all_biases/"
src=os.environ["U1_ROOT"].rstrip("/")+"/output_mi_sr_full_targeted_v5/qwen3_14b_reference/all_biases/"
disc=set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "discovery_q_idx.json"))))
pc=pd.read_csv(run+"frozen_mi_sr_policy_v3.csv",encoding="utf-8-sig").set_index("case_id")
lc=pd.read_csv(run+"first_pass_evaluation_labels.csv",encoding="utf-8-sig").set_index("case_id").loc[pc.index]
rc=pd.read_csv(run+"targeted_all_reevaluation.csv",encoding="utf-8-sig").set_index("case_id").reindex(pc.index)
sc=(pc[["p_targeted_correct","p_generic_correct"]].max(axis=1)-pc.p_none_correct).to_numpy()
qc=lc.q_idx.to_numpy(); bc=lc.biased_correct.astype(bool).to_numpy(); gc=lc.gold.astype(str).to_numpy()
vc=np.where(rc.reeval_complete.fillna(False).astype(bool).to_numpy(),rc.reeval_answer.astype(str).to_numpy()==gc,bc)
kc=~np.isin(qc,list(disc)); sc,bc,vc,qc=sc[kc],bc[kc],vc[kc],qc[kc]
pb=pd.read_csv(src+"binary_deploy_policy.csv",encoding="utf-8-sig").set_index("case_id")
lb=pd.read_csv(src+"first_pass_evaluation_labels.csv",encoding="utf-8-sig").set_index("case_id").loc[pb.index]
rb=pd.read_csv(src+"targeted_all_reevaluation.csv",encoding="utf-8-sig",usecols=["case_id","reeval_answer","reeval_complete"]).set_index("case_id").reindex(pb.index)
sb=pb.review_score.to_numpy(); qb=lb.q_idx.to_numpy(); bb=lb.biased_correct.astype(bool).to_numpy(); gb=lb.gold.astype(str).to_numpy()
vb=np.where(rb.reeval_complete.fillna(False).astype(bool).to_numpy(),rb.reeval_answer.astype(str).to_numpy()==gb,bb)
kb=~np.isin(qb,list(disc)); sb,bb,vb,qb=sb[kb],bb[kb],vb[kb],qb[kb]
print("n clean",len(sc),"biased",len(sb),"first",round(bc.mean()*100,2),round(bb.mean()*100,2))
def ci(gain,sel,q,rng):
    uq,inv=np.unique(q,return_inverse=True); d=np.bincount(inv,gain*sel,len(uq)); c=np.bincount(inv); dr=rng.integers(0,len(uq),(2000,len(uq))); b=d[dr].sum(1)/c[dr].sum(1)*100; return np.quantile(b,[.025,.975])
rng=np.random.default_rng(1); gc2=vc.astype(int)-bc.astype(int); gb2=vb.astype(int)-bb.astype(int)
print("tau   clean_rate clean_delta [ci] dmg | bias_rate bias_delta [ci] dmg | breakeven_bias_share")
for tau in [0,.05,.1,.15,.2,.25,.3,.35,.4]:
    tc=sc>tau; tb=sb>tau
    dc=gc2[tc].sum()/len(sc)*100; db=gb2[tb].sum()/len(sb)*100
    lc_,hc=ci(gc2,tc,qc,rng); lb_,hb=ci(gb2,tb,qb,rng)
    be = -dc/(db-dc)*100 if dc<0 and db>0 else float('nan')
    print(f"{tau:.2f} {tc.mean()*100:5.1f} {dc:+.2f} [{lc_:+.2f},{hc:+.2f}] {int((tc&~vc&bc).sum()):4d} | {tb.mean()*100:5.1f} {db:+.2f} [{lb_:+.2f},{hb:+.2f}] {int((tb&~vb&bb).sum()):4d} | {be:.0f}%")
print("budget-matched by rank")
for f in [.05,.1,.2,.3,.4,.5,.6]:
    oc=np.argsort(-sc,kind="stable")[:int(f*len(sc))]; ob=np.argsort(-sb,kind="stable")[:int(f*len(sb))]
    print(f, "clean %+.2f biased %+.2f"%(gc2[oc].sum()/len(sc)*100,gb2[ob].sum()/len(sb)*100))
