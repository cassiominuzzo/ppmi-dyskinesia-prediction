# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""22 — completude item-level: TODOS os itens MDS-UPDRS Partes I/II/III no baseline pré-levodopa
(antes diluídos em compósitos) + orthostasis. Univariado + FDR vs desfecho primário."""
from config import PROJECT_ROOT
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
B=PROJECT_ROOT
RAW=B+'/01_Dados_Brutos/MDS_UPDRS_e_Motor'; PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
out=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))[['PATNO','exit','event']]
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
def baseline_items(fn, prefix):
    d=ip(pd.read_csv(RAW+'/'+fn,low_memory=False)); d['INFODT']=pd.to_datetime(d['INFODT'],errors='coerce',format='mixed')
    items=[c for c in d.columns if c.startswith(prefix) and c not in (prefix+'RTOT',prefix+'PTOT')]
    d=d.merge(tz,on='PATNO',how='inner'); d['dsd']=(d['INFODT']-d['tzero_levodopa']).dt.days
    d=d[d['dsd']<=0].sort_values('dsd')  # pré-levodopa
    for c in items: d[c]=pd.to_numeric(d[c],errors='coerce').replace(101,np.nan)
    return d.groupby('PATNO')[items].last().reset_index(), items
parts=[('MDS-UPDRS_Part_I_29Apr2026.csv','NP1'),
       ('MDS-UPDRS_Part_I_Patient_Questionnaire_29Apr2026.csv','NP1'),
       ('MDS_UPDRS_Part_II__Patient_Questionnaire_29Apr2026.csv','NP2'),
       ('MDS-UPDRS_Part_III_29Apr2026.csv','NP3')]
M=out.copy(); allitems=[]
for fn,pf in parts:
    bi,items=baseline_items(fn,pf); M=M.merge(bi,on='PATNO',how='left'); allitems+=items
# orthostasis (curated)
cur=ip(pd.read_parquet(B+'/02_Dados_Processados/curated_cache.parquet'))
if 'orthostasis' in cur.columns:
    o=cur.groupby('PATNO')['orthostasis'].max().reset_index(); M=M.merge(o,on='PATNO',how='left'); allitems.append('orthostasis')
allitems=[c for c in dict.fromkeys(allitems) if c in M.columns]
print(f'{len(allitems)} itens individuais no baseline pré-levodopa')
d=M[M['exit']>0]; rows=[]
for c in allitems:
    s=d[['exit','event',c]].dropna()
    if len(s)<60 or s[c].nunique()<3 or int(s.event.sum())<10: continue
    sd=s[c].std()
    if sd==0 or np.isnan(sd): continue
    s=s.copy(); s[c]=(s[c]-s[c].mean())/sd
    try:
        cph=CoxPHFitter(penalizer=0.05).fit(s,'exit','event'); hr=cph.summary.loc[c]
        rows.append({'item':c,'n':len(s),'n_ev':int(s.event.sum()),'HR_DP':round(hr['exp(coef)'],3),
                     'p':float(hr['p']),'C':round(cph.concordance_index_,3)})
    except Exception: pass
R=pd.DataFrame(rows); R['p_FDR']=multipletests(R['p'],method='fdr_bh')[1]; R=R.sort_values('p').reset_index(drop=True)
R.to_csv(TAB+'/tab22_itens_individuais.csv',index=False)
print(f'\n=== ITENS INDIVIDUAIS: {len(R)} testados | {int((R.p_FDR<0.05).sum())} sig após FDR ===')
print('TOP 20 por p bruto:'); print(R.head(20).to_string(index=False))
sig=R[R.p_FDR<0.05]; print('\nSIGNIFICATIVOS após FDR<0.05:'); print(sig.to_string(index=False) if len(sig) else '  NENHUM.')
# comparar melhor item vs compósitos já no modelo (C de referência ~0.635)
print(f'\nMelhor item isolado: C={R.C.max():.3f} (vs modelo 8-var C≈0.635)')
