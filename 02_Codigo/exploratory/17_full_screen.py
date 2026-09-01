# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""
17_full_screen.py — TRIAGEM TOTAL: univariado em TODAS as candidatas baseline (pré-levodopa).
Extrai o valor pré-levodopa de cada coluna numérica do curated, exclui vazamento (NP4*, LEDD,
estado-ON, óbito, admin), roda Cox univariado (origem t=0) + FDR, e ranqueia. Marca os preditores
significativos que NÃO estão no L1 atual.
"""
from config import PROJECT_ROOT
import os, glob, re, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
B=PROJECT_ROOT
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; SRC=B+'/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
# desfecho (origem t=0) do master
M=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))
out=M[['PATNO','exit','event']].copy()
L1=['ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai']
# compósitos + responsividade já no master
extra=M[['PATNO','comp_fala','comp_tremor','comp_marcha_pi','comp_bradicinesia','comp_axial','comp_assimetria_lr','td_pigd_ratio','is_pigd','datscan_putamen_min','datscan_putamen_asym','resp']].copy()

# baseline pré-levodopa de TODAS as colunas numéricas do curated
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
cur=ip(pd.read_parquet(SRC+'/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz,on='PATNO',how='inner'); cur['dsd']=(cur['visit_date']-cur['tz' 'ero_levodopa']).dt.days
base=cur[cur['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index()
# EXCLUSÕES (vazamento/desfecho/exposição/estado-ON/admin)
EXC=re.compile(r'(NP4|updrs4|_on$|_ON$|LEDD|PDTRTMNT|Death|study_status|COHORT|enroll|analytic|EVENT_ID|SITE|YEAR|dsd|PATNO|visit|therapy|subgroup|NSD|Stage_)',re.I)
numcols=[c for c in base.select_dtypes(include=[np.number]).columns if not EXC.search(c) and base[c].notna().mean()>0.3]
print(f'candidatas curated (pós-exclusão de vazamento): {len(numcols)}')
cand=ip(base[['PATNO']+numcols]).merge(extra,on='PATNO',how='outer')
allcand=[c for c in cand.columns if c!='PATNO']
D=out.merge(cand,on='PATNO',how='left'); D=D[D['exit']>0]

rows=[]
for c in allcand:
    d=D[['exit','event',c]].copy(); d[c]=pd.to_numeric(d[c],errors='coerce'); d=d.dropna()
    if len(d)<50 or d[c].nunique()<3 or int(d.event.sum())<10: continue
    d[c]=(d[c]-d[c].mean())/d[c].std()
    try:
        cph=CoxPHFitter(penalizer=0.01).fit(d,'exit','event'); hr=cph.summary.loc[c]
        rows.append({'variavel':c,'n':len(d),'eventos':int(d.event.sum()),'HR_por_DP':round(hr['exp(coef)'],3),
                     'p':float(hr['p']),'C_index':round(cph.concordance_index_,3),'no_L1':'sim' if c in L1 else 'NÃO'})
    except Exception: pass
res=pd.DataFrame(rows)
res['p_FDR']=multipletests(res['p'],method='fdr_bh')[1]
res=res.sort_values('p').reset_index(drop=True)
res.to_csv(TAB+'/tab17_triagem_total.csv',index=False)
print(f'\n=== TRIAGEM TOTAL: {len(res)} variáveis testadas, {int((res.p_FDR<0.05).sum())} significativas (FDR<0.05) ===')
print('\nTOP 25 (por p):')
print(res.head(25)[['variavel','HR_por_DP','p','p_FDR','C_index','no_L1']].to_string(index=False))
print('\n>>> Preditores SIGNIFICATIVOS que NÃO estão no L1 (candidatos novos):')
novos=res[(res.p_FDR<0.05)&(res.no_L1=='NÃO')].head(20)
print(novos[['variavel','HR_por_DP','p_FDR','C_index']].to_string(index=False))
pickle.dump({'res':res},open(PROC+'/results_step17.pkl','wb'))
print('\nETAPA — triagem total COMPLETA (tab17_triagem_total.csv)')
