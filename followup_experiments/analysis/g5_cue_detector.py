"""Exploratory: can the first-pass features tell whether a cue is present, and does reviewing only when a cue is predicted cut the clean loss?
Detector: same recipe, fit on the 95 development questions (biased vs clean variants), evaluated on the other questions. Qwen. Post hoc."""
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
    return x,x.q_idx.isin(disc).to_numpy(),b,v,x.q_idx.to_numpy(),l.bias_type.astype(str).to_numpy()
xc,dc,bc,vc,qc,_=load(A+"u1_clean/qwen3_14b_reference/all_biases/"); xb,db,bb,vb,qb,tb=load(R+"output_mi_sr_full_targeted_v5/qwen3_14b_reference/all_biases/")
X=pd.concat([xb.loc[db,cols],xc.loc[dc,cols]]); y=np.r_[np.ones(db.sum()),np.zeros(dc.sum())].astype(int)
det=clone(heads["baseline_correct"]).fit(X,y)
pc=det.predict_proba(xc.loc[~dc,cols])[:,1]; pb=det.predict_proba(xb.loc[~db,cols])[:,1]
s=np.r_[pb,pc]; lab=np.r_[np.ones(len(pb)),np.zeros(len(pc))].astype(bool); r=pd.Series(s).rank().to_numpy(); n1=lab.sum(); n0=(~lab).sum()
print("detector AUC %.3f; predicted cue at 0.5: biased %.1f%% clean %.1f%%"%((r[lab].sum()-n1*(n1+1)/2)/(n1*n0),(pb>.5).mean()*100,(pc>.5).mean()*100))
def score(x): return np.maximum(heads["targeted_correct"].predict_proba(x)[:,1],heads["generic_correct"].predict_proba(x)[:,1])-heads["baseline_correct"].predict_proba(x)[:,1]
rc=score(xc.loc[~dc,cols]); rb=score(xb.loc[~db,cols]); Gc=vc[~dc].astype(int)-bc[~dc].astype(int); Gb=vb[~db].astype(int)-bb[~db].astype(int)
def ci(g,sel,q):
    uq,inv=np.unique(q,return_inverse=True); d=np.bincount(inv,g*sel,len(uq)); c=np.bincount(inv); dr=np.random.default_rng(1).integers(0,len(uq),(3000,len(uq))); b=d[dr].sum(1)/c[dr].sum(1)*100; return np.quantile(b,[.025,.975])
for t in (0,.3,.5,.7,.8,.9):
    sc=(rc>0)&(pc>t); sb=(rb>0)&(pb>t); lc=ci(Gc,sc,qc[~dc]); lb=ci(Gb,sb,qb[~db])
    print(f"cue>{t}: clean review {sc.mean()*100:.1f} delta {Gc[sc].sum()/len(Gc)*100:+.2f} [{lc[0]:+.2f},{lc[1]:+.2f}] | biased review {sb.mean()*100:.1f} delta {Gb[sb].sum()/len(Gb)*100:+.2f} [{lb[0]:+.2f},{lb[1]:+.2f}]")
print(pd.Series(pb>.5).groupby(tb[~db]).mean().round(3).to_dict())
