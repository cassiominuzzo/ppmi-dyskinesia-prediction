# 10_supplementary_table_S1.py
# Produces: Supplementary Table S1
# Original file in the project archive: 52_multimodal_supp_table.py

"""52 — Tabela Suplementar: valor incremental multimodal por variavel (sobre o Total-6).
Para cada variavel: n, eventos, HR(IC95) por DP, p, LRT p (vs Total-6), dC out-of-fold. Auditavel."""
from config import PROJECT_ROOT
import os
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project layout,
# inside the raw-data folder, so this table was written where nothing else could find it.
# They now come from cohort.py.
import pandas as pd, numpy as np, glob, warnings
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from scipy.stats import chi2
PR=C.PROC; TAB=C.ensure_dir(os.path.join(C.RESULTS,'Tabelas'))
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
cm=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in cm.columns and 'ageonset_x' in cm.columns: cm=cm.rename(columns={'ageonset_x':'ageonset'})
cm=ip(cm)
mb=ip(pd.read_parquet(PR+'/master_baseline.parquet'))[['PATNO','datscan_putamen_min','datscan_putamen_asym','abeta','tau','ptau']]
base=cm.merge(mb,on='PATNO',how='left')
prs=pd.read_parquet(PR+'/prs_analytic.parquet')
prs=ip(prs); prscols=[c for c in prs.columns if 'prs' in c.lower() or 'nalls' in c.lower() or 'progress' in c.lower()]
print("colunas PRS:",prscols)
base=base.merge(prs[['PATNO']+prscols],on='PATNO',how='left')
# nomes das variaveis
VARS=[('PRS_Nalls','PD-susceptibility PRS (Nalls)'),('PRS_Progression','Progression PRS'),
      ('datscan_putamen_min','DaTSCAN putaminal binding (worse side)'),('datscan_putamen_asym','DaTSCAN striatal asymmetry (laterality)'),
      ('abeta','CSF Abeta42'),('tau','CSF total tau'),('ptau','CSF phosphorylated tau')]
# ajustar nomes PRS reais
ren={}
for c in prscols:
    if 'nall' in c.lower(): ren[c]='PRS_Nalls'
    elif 'progress' in c.lower(): ren[c]='PRS_Progression'
base=base.rename(columns=ren)
d0=base[base['exit']>0].copy()
def oofC(df,cols,seed):
    kf=KFold(5,shuffle=True,random_state=seed); pr=np.zeros(len(df))
    for tr,te in kf.split(df):
        m=CoxPHFitter(penalizer=0.1).fit(df.iloc[tr][['exit','event']+cols],'exit','event')
        pr[te]=-m.predict_partial_hazard(df.iloc[te][cols]).values
    return concordance_index(df['exit'],pr,df['event'])
rows=[]
for v,lab in VARS:
    if v not in d0.columns: print("AUSENTE:",v); continue
    d=d0.dropna(subset=T6+[v]).copy().reset_index(drop=True)
    d[v+'_z']=(d[v]-d[v].mean())/d[v].std()
    n=len(d); ev=int(d.event.sum())
    m0=CoxPHFitter().fit(d[['exit','event']+T6],'exit','event')
    m1=CoxPHFitter().fit(d[['exit','event']+T6+[v+'_z']],'exit','event')
    lrt=2*(m1.log_likelihood_-m0.log_likelihood_); p_lrt=chi2.sf(lrt,1)
    h=m1.summary.loc[v+'_z']
    dc=np.mean([oofC(d,T6+[v+'_z'],s)-oofC(d,T6,s) for s in range(2)])
    rows.append({'Variable':lab,'n':n,'Events':ev,
                 'HR per SD (95% CI)':'%.2f (%.2f-%.2f)'%(h['exp(coef)'],h['exp(coef) lower 95%'],h['exp(coef) upper 95%']),
                 'p':'%.3f'%h['p'],'LRT p':'%.3f'%p_lrt,'delta C (out-of-fold)':'%+.4f'%dc})
    print("%-42s n=%d ev=%d HR=%.2f p=%.3f LRTp=%.3f dC=%+.4f"%(lab,n,ev,h['exp(coef)'],h['p'],p_lrt,dc))
tab=pd.DataFrame(rows)
tab.to_csv(TAB+'/tabS_multimodal_added_value.csv',index=False)
print("\nsalvo tabS_multimodal_added_value.csv")
