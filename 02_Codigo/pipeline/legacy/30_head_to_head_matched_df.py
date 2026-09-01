# 30_head_to_head_matched_df.py
# Produces: Supplementary Table S7 (head-to-head with matched degrees of freedom)
#
# Concern addressed: the six-variable model has more parameters than the published
# predictor sets (two or three), so part of its higher C-index could reflect model size
# rather than the choice of variables. To separate the two effects we compute the
# out-of-fold C-index of subsets of size k drawn from a common pool of 20 baseline
# clinical candidates (exhaustive for k = 2 and 3; 1000 random subsets for k = 6), which
# gives the discrimination attainable with k parameters in these data. Each predictor set
# is then placed as a percentile within the distribution for its own k.
#
# Usage (the full enumeration takes a few minutes; it can be run in chunks):
#   python 30_head_to_head_matched_df.py meta
#   python 30_head_to_head_matched_df.py sets
#   python 30_head_to_head_matched_df.py chunk 2 0 190
#   python 30_head_to_head_matched_df.py chunk 3 0 1140
#   python 30_head_to_head_matched_df.py chunk 6 0 1000
#   python 30_head_to_head_matched_df.py summarise
#
# Note: Total-6 was selected in these same data, so its rank within the size-6
# distribution is optimistic. The interpretable comparisons are (a) the median gain from
# k=2 to k=6, which quantifies the pure effect of model size, and (b) whether any two- or
# three-variable subset reaches the discrimination of Total-6.

import sys,os,pickle,warnings,itertools,json
import numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from config import PROJECT_ROOT
ROOT=PROJECT_ROOT; PR=ROOT+'/02_Dados_Processados'
OUT=os.path.join(ROOT,'04_Resultados','Tabelas'); os.makedirs(OUT,exist_ok=True)
POOL=pickle.load(open(PR+'/sel_data.pkl','rb'))['POOL']
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
d=M[M['exit']>0]
EXTRA=['duration_yrs','stai','updrs3_score','updrs2_score','pigd']
CAND=[c for c in sorted(set(POOL)|set(EXTRA)) if c in d.columns]
dd=d.dropna(subset=CAND).reset_index(drop=True)
E=dd['event'].astype(bool).values; T=dd['exit'].values; X=dd[CAND].values
folds=list(KFold(5,shuffle=True,random_state=7).split(np.arange(len(dd))))
def oof(idx):
    pr=np.zeros(len(dd))
    for tr,te in folds:
        sc=StandardScaler().fit(X[np.ix_(tr,idx)])
        m=CoxPHSurvivalAnalysis(alpha=1.0).fit(sc.transform(X[np.ix_(tr,idx)]),Surv.from_arrays(E[tr],T[tr]))
        pr[te]=m.predict(sc.transform(X[np.ix_(te,idx)]))
    return float(concordance_index_censored(E,T,pr)[0])
def combos(k):
    if k<=3: return [list(c) for c in itertools.combinations(range(len(CAND)),k)]
    rng=np.random.default_rng(11)
    return [sorted(rng.choice(len(CAND),k,replace=False).tolist()) for _ in range(1000)]
mode=sys.argv[1]
if mode=='meta':
    print(json.dumps(dict(cand=CAND,n=len(dd),ev=int(E.sum()),
        n2=len(combos(2)),n3=len(combos(3)),n6=len(combos(6)))))
elif mode=='chunk':
    k=int(sys.argv[2]); a=int(sys.argv[3]); b=int(sys.argv[4])
    cs=combos(k)[a:b]
    out=[oof(c) for c in cs]
    f=os.path.join(OUT,'tabS7_dist_k%d.csv'%k)
    with open(f,'a') as fh:
        for c,v in zip(cs,out): fh.write("%s,%.6f\n"%('|'.join(map(str,c)),v))
    print("k=%d chunk %d:%d -> %d valores"%(k,a,b,len(out)))
elif mode=='sets':
    PUB={"Total-6 (this study)":["updrs_totscore","ageonset","SEX","BMI","td_pigd_ratio","NP2FREZ"],
         "Zhao 2025":["ageonset","td_pigd_ratio","updrs3_score"],
         "Olanow / STRIDE-PD":["ageonset","SEX","BMI"],
         "Santos-Lobato 2020":["ageonset","duration_yrs","updrs2_score"],
         "Chen 2021":["ageonset","duration_yrs"],
         "Eusebi 2018":["SEX","pigd","stai"]}
    pos={v:i for i,v in enumerate(CAND)}
    r=[dict(predictor_set=n,k=len(v),C=round(oof([pos[x] for x in v]),4)) for n,v in PUB.items()]
    pd.DataFrame(r).to_csv(os.path.join(OUT,'tabS7_sets.csv'),index=False); print(pd.DataFrame(r).to_string(index=False))

elif mode=='summarise':
    D={k:pd.read_csv(os.path.join(OUT,'tabS7_dist_k%d.csv'%k),header=None,names=['idx','C'])['C'].values for k in (2,3,6)}
    S=pd.read_csv(os.path.join(OUT,'tabS7_sets.csv'))
    rows=[]
    for _,r in S.sort_values('C',ascending=False).iterrows():
        k=int(r['k']); v=D[k]
        rows.append(dict(predictor_set=r['predictor_set'],k=k,C_out_of_fold=r['C'],
            median_C_for_k=round(float(np.median(v)),4),p90_C_for_k=round(float(np.percentile(v,90)),4),
            max_C_for_k=round(float(v.max()),4),percentile_within_k=round(float((v<r['C']).mean()*100),1)))
    out=pd.DataFrame(rows); out.to_csv(os.path.join(OUT,'tabS7_matched_df.csv'),index=False)
    print(out.to_string(index=False))
    t6=float(S.loc[S.predictor_set.str.contains('Total-6'),'C'].iloc[0])
    print("\nmedian C by model size: k=2 %.3f | k=3 %.3f | k=6 %.3f  (gain from 2 to 6: %+.3f)"
          %(np.median(D[2]),np.median(D[3]),np.median(D[6]),np.median(D[6])-np.median(D[2])))
    print("subsets reaching Total-6 (C %.3f): %d of %d at k=3, %d of %d at k=2"
          %(t6,(D[3]>=t6).sum(),len(D[3]),(D[2]>=t6).sum(),len(D[2])))
