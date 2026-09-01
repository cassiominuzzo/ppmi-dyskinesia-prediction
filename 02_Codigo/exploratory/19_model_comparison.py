# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""19 — head-to-head JUSTO (mesma coorte, mesma CV 5-fold, mesma imputação):
L1 hand-picked (9) vs data-driven selecionadas (16) vs elastic-net (pool completo)."""
from config import PROJECT_ROOT
import glob, warnings, pickle; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
B=PROJECT_ROOT
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; SRC=B+'/02_Dados_Processados'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
M=ip(pd.read_parquet(PROC+'/master_baseline.parquet')); 
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
cur=ip(pd.read_parquet(SRC+'/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz,on='PATNO',how='inner'); cur['dsd']=(cur['visit_date']-cur['tzero_levodopa']).dt.days
base=ip(cur[cur['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index())
DD=['EDUCYRS','NP1FATG','MSEADLG','CSFSAA','NP1HALL','NP1DDS','orthostasis','NP1APAT','handed','DOMSIDE']
keepC=['PATNO']+[c for c in DD if c in base.columns]
X=M[['PATNO','exit','event','ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai','td_pigd_ratio']].merge(base[keepC],on='PATNO',how='left')
X=X[X.exit>0].reset_index(drop=True)
L1=['ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai']
DDv=['updrs2_score','ageonset','SEX','EDUCYRS','NP1FATG','MSEADLG','CSFSAA','comp_bradicinesia','BMI','handed','NP1HALL','NP1DDS','orthostasis','td_pigd_ratio','DOMSIDE','NP1APAT']
DDv=[c for c in DDv if c in X.columns]
for c in set(L1+DDv): X[c]=pd.to_numeric(X[c],errors='coerce')
y=Surv.from_arrays(X.event.astype(bool).values,X['exit'].values)
def C(risk): return concordance_index_censored(X.event.astype(bool).values,X['exit'].values,risk)[0]
kf=KFold(5,shuffle=True,random_state=42)
def cox_oof(cols):
    oof=np.full(len(X),np.nan)
    for tr,te in kf.split(X):
        im=SimpleImputer(strategy='median').fit(X.iloc[tr][cols])
        A=pd.DataFrame(im.transform(X.iloc[tr][cols]),columns=cols); A['exit']=X.iloc[tr]['exit'].values; A['event']=X.iloc[tr]['event'].values
        Te=pd.DataFrame(im.transform(X.iloc[te][cols]),columns=cols)
        cph=CoxPHFitter(penalizer=0.05).fit(A,'exit','event'); oof[te]=cph.predict_partial_hazard(Te).values.ravel()
    return C(oof)
print(f'n={len(X)}, eventos={int(X.event.sum())} (mesma coorte para os 3)')
print(f'  L1 hand-picked (9 var)   C(OOF 5-fold) = {cox_oof(L1):.3f}')
print(f'  Data-driven Cox (16 var) C(OOF 5-fold) = {cox_oof(DDv):.3f}')
# elastic-net no pool L1+DD
EN=list(dict.fromkeys(L1+DDv)); oof=np.full(len(X),np.nan)
for tr,te in kf.split(X):
    im=SimpleImputer(strategy='median').fit(X.iloc[tr][EN]); ss=StandardScaler().fit(im.transform(X.iloc[tr][EN]))
    Xtr=ss.transform(im.transform(X.iloc[tr][EN])); Xte=ss.transform(im.transform(X.iloc[te][EN]))
    m=CoxnetSurvivalAnalysis(l1_ratio=0.5,alpha_min_ratio=0.01,max_iter=100000).fit(Xtr,Surv.from_arrays(X.iloc[tr].event.astype(bool),X.iloc[tr]['exit'].values))
    a=m.alphas_[len(m.alphas_)//2]
    oof[te]=m.predict(Xte,alpha=a)
print(f'  Elastic-net (pool {len(EN)} var) C(OOF) = {C(oof):.3f}')
print('\nConclusão: se ficarem ~iguais, o modelo PARCIMONIOSO (L1) é preferível p/ a calculadora (simples, interpretável).')
