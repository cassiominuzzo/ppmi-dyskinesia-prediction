# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""13b — benchmark JUSTO Cox vs RSF por validação cruzada 5-fold (out-of-fold C-index).
Corrige o C-index aparente do RSF (0,844), que era otimista por superajuste in-sample."""
from config import PROJECT_ROOT
import glob, warnings, pickle; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold
BASE=PROJECT_ROOT; PROC=BASE+'/Projeto_LID_v2/02_Dados_Processados'
M=pd.read_parquet(PROC+'/master_baseline.parquet')
for c in M.columns:
    if c!='PATNO': M[c]=pd.to_numeric(M[c],errors='coerce')
L1=['ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai']
D=M[['exit','event']+L1].dropna(); D=D[D['exit']>0].reset_index(drop=True)
def C(df,risk): return concordance_index_censored(df.event.astype(bool).values,df['exit'].values,risk)[0]
kf=KFold(5,shuffle=True,random_state=42); cox_oof=np.full(len(D),np.nan); rsf_oof=np.full(len(D),np.nan)
for tr,te in kf.split(D):
    A,B=D.iloc[tr],D.iloc[te]
    cph=CoxPHFitter(penalizer=0.01).fit(A[['exit','event']+L1],'exit','event'); cox_oof[te]=cph.predict_partial_hazard(B).values.ravel()
    y=Surv.from_arrays(event=A.event.astype(bool),time=A['exit'].values)
    rsf=RandomSurvivalForest(n_estimators=100,min_samples_leaf=20,max_features='sqrt',random_state=42,n_jobs=2).fit(A[L1],y)
    rsf_oof[te]=rsf.predict(B[L1])
print('=== Benchmark JUSTO (5-fold out-of-fold C-index) ===')
print(f'  Cox C(OOF) = {C(D,cox_oof):.3f}')
print(f'  RSF C(OOF) = {C(D,rsf_oof):.3f}')
print('  (apparent era Cox 0.680 / RSF 0.844 — RSF inflado por superajuste)')
print('Conclusão: ML não-linear', 'NÃO supera' if C(D,rsf_oof)<=C(D,cox_oof)+0.01 else 'supera', 'o Cox parcimonioso na validação.')
pickle.dump({'cox_oof_C':round(C(D,cox_oof),3),'rsf_oof_C':round(C(D,rsf_oof),3)},open(PROC+'/results_step13b.pkl','wb'))
