"""Exploratory: does the gate separate clean questions once clean variants enter its training set? Same recipe and the same
95 development questions; generic head stays frozen (no generic reviews on clean). Evaluated on the other questions, Qwen.
Post hoc, not a replacement for the frozen result."""
import os, json, pickle
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
A=os.environ.get("MI_SR_WORK_ROOT", "work").rstrip("/") + "/"; R=os.environ["U1_ROOT"].rstrip("/")+"/"; art=R+"frozen_artifacts/"
disc=set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "discovery_q_idx.json"))))
cols=json.load(open(art+"frozen_gate_v002.json"))["feature_columns"]; heads=joblib.load(art+"frozen_gate_v002_models.joblib")
def load(d):
    x=pickle.load(open(d+"deploy_gate_features_before_unblinding.pkl","rb")).set_index("case_id")
    l=pd.read_csv(d+"first_pass_evaluation_labels.csv",encoding="utf-8-sig").set_index("case_id").reindex(x.index)
    r=pd.read_csv(d+"targeted_all_reevaluation.csv",encoding="utf-8-sig",usecols=["case_id","reeval_answer","reeval_complete"]).set_index("case_id").reindex(x.index)
    b=l.biased_correct.astype(bool).to_numpy(); g=l.gold.astype(str).to_numpy()
    v=np.where(r.reeval_complete.fillna(False).astype(bool).to_numpy(),r.reeval_answer.astype(str).to_numpy()==g,b)
    return x,x.q_idx.isin(disc).to_numpy(),b,v,x.q_idx.to_numpy()
xc,dc,bc,vc,qc=load(A+"u1_clean/qwen3_14b_reference/all_biases/"); xb,db,bb,vb,qb=load(R+"output_mi_sr_full_targeted_v5/qwen3_14b_reference/all_biases/")
print("dev rows biased",db.sum(),"clean",dc.sum(),"eval biased",(~db).sum(),"clean",(~dc).sum())
def auc(y,s):
    r=pd.Series(s).rank().to_numpy(); n1=y.sum(); n0=(~y).sum(); return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
def ci(g,sel,q):
    uq,inv=np.unique(q,return_inverse=True); d=np.bincount(inv,g*sel,len(uq)); c=np.bincount(inv); dr=np.random.default_rng(1).integers(0,len(uq),(3000,len(uq))); b=d[dr].sum(1)/c[dr].sum(1)*100; return np.quantile(b,[.025,.975])
def report(name,pN,pT,pG):
    print("##",name)
    for lab,x,dv,b,v,q in (("clean",xc,dc,bc,vc,qc),("biased",xb,db,bb,vb,qb)):
        k=~dv; X=x.loc[k,cols]; n=pN.predict_proba(X)[:,1]; t=pT.predict_proba(X)[:,1]; g_=pG.predict_proba(X)[:,1]
        s=np.maximum(t,g_)-n; sel=s>0; B,V,Q=b[k],v[k],q[k]; G=V.astype(int)-B.astype(int); lo,hi=ci(G,sel,Q)
        print(f"{lab:6s} first {B.mean()*100:.2f} review {sel.mean()*100:.1f} (correct {sel[B].mean()*100:.1f} / wrong {sel[~B].mean()*100:.1f}) delta {G[sel].sum()/len(G)*100:+.2f} [{lo:+.2f},{hi:+.2f}] fix {(sel&~B&V).sum()} dmg {(sel&B&~V).sum()} AUC_pN {auc(B,n):.3f}")
report("frozen",heads["baseline_correct"],heads["targeted_correct"],heads["generic_correct"])
def fit(target,w):
    X=pd.concat([xb.loc[db,cols],xc.loc[dc,cols]]); yb=(bb if target=="b" else vb)[db]; yc=(bc if target=="b" else vc)[dc]
    y=np.r_[yb,yc].astype(int); sw=np.r_[np.ones(db.sum()),np.full(dc.sum(),w)]
    m=clone(heads["baseline_correct"]); m.fit(X,y,logisticregression__sample_weight=sw); return m
for w in (0.0001,1,7):
    report(f"retrain clean_weight={w}",fit("b",w),fit("t",w),heads["generic_correct"])
