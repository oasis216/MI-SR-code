import os
import pandas as pd, numpy as np
d=pd.read_csv(os.environ.get("MI_SR_WORK_ROOT", "work") + "/u1_extra/r8_rows_qwen3_14b_reference.csv"); print(d.columns.tolist()); print(d.head(2).T.to_string()[:600])
rng=np.random.default_rng(1)
for c in d.columns:
    if c.endswith("_hit") or c.endswith("hit"):
        x=d[c].dropna().astype(float).to_numpy(); b=rng.choice(x,(4000,len(x))).mean(1)*100
        print(c,len(x),"%.1f [%.1f,%.1f]"%(x.mean()*100,*np.quantile(b,[.025,.975])))
