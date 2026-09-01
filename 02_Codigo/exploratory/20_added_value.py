# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""20 — GANHO INCREMENTAL (added value): quanto cada candidata sobe o C-index quando
adicionada ao modelo-base. Mede relevância PARA PREDIÇÃO (não só significância). OOF 5-fold."""
from config import PROJECT_ROOT
import glob, warnings, numpy as np, pandas as pd; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored
from sklearn.impute import SimpleImputer; from sklearn.model_selection import KFold
B=PROJECT_ROOT
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; SRC=B+'/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
M=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
cur=ip(pd.read_parquet(SRC+'/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz,on='PATNO',how='inner'); cur['dsd']=(cur['visit_date']-cur['tzero_levodopa']).dt.days
base=ip(cur[cur['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index())
CURv=['MSEADLG','NP1FATG','NP1HALL','NP1APAT','NP1DDS','orthostasis','CSFSAA','scopa_gi','updrs1_score',
      'bjlot','SDMTOTAL','hvlt_immediaterecall','VLTANIM','gds','urate','NHY','EDUCYRS','stai']
CURv=[c for c in CURv if c in base.columns]
X=M[['PATNO','exit','event','ageonset','SEX','BMI','updrs2_score','comp_bradicinesia','comp_axial','comp_marcha_pi','comp_tremor','td_pigd_ratio','pigd','upsit','moca']].merge(base[['PATNO']+CURv],on='PATNO',how='left')
X=X[X.exit>0].reset_index(drop=True)
for c in X.columns:
    if c!='PATNO': X[c]=pd.to_numeric(X[c],errors='coerce')
BASEcols=['updrs2_score','ageonset','SEX','MSEADLG','NP1FATG','comp_bradicinesia','BMI','td_pigd_ratio']
y_ev=X.event.astype(bool).values; y_t=X['exit'].values
kf=KFold(5,shuffle=True,random_state=42)
def oofC(cols):
    oof=np.full(len(X),np.nan)
    for tr,te in kf.split(X):
        im=SimpleImputer(strategy='median').fit(X.iloc[tr][cols])
        A=pd.DataFrame(im.transform(X.iloc[tr][cols]),columns=cols); A['exit']=X.iloc[tr]['exit'].values; A['event']=X.iloc[tr]['event'].values
        Te=pd.DataFrame(im.transform(X.iloc[te][cols]),columns=cols)
        cph=CoxPHFitter(penalizer=0.1).fit(A,'exit','event'); oof[te]=cph.predict_partial_hazard(Te).values.ravel()
    return concordance_index_censored(y_ev,y_t,oof)[0]
Cbase=oofC(BASEcols)
print(f'n={len(X)}, eventos={int(X.event.sum())}')
print(f'\nMODELO-BASE (8 var clínicas plausíveis): C-index = {Cbase:.3f}\n')
print('GANHO ao ADICIONAR cada candidata (ΔC; |ΔC|<0,005 ≈ não agrega):')
rows=[]
for v in ['pigd','scopa_gi','updrs1_score','comp_axial','comp_marcha_pi','comp_tremor','NP1HALL','NP1APAT','NP1DDS','orthostasis','CSFSAA','moca','bjlot','SDMTOTAL','hvlt_immediaterecall','VLTANIM','gds','urate','NHY','upsit','stai','EDUCYRS']:
    if v not in X.columns: continue
    c=oofC(BASEcols+[v]); rows.append({'candidata':v,'C_base+var':round(c,3),'ganho_dC':round(c-Cbase,4)})
r=pd.DataFrame(rows).sort_values('ganho_dC',ascending=False)
print(r.to_string(index=False))
# trocar td_pigd_ratio por pigd no base
alt=[c for c in BASEcols if c!='td_pigd_ratio']+['pigd']
print(f'\nTroca: base com PIGD (em vez da razão TD/PIGD): C = {oofC(alt):.3f}  (base atual {Cbase:.3f})')
r.to_csv(TAB+'/tab20_ganho_incremental.csv',index=False)
print('\nSalvo: tab20_ganho_incremental.csv')
