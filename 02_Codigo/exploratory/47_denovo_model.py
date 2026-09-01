# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""47 — MODELO NO DIAGNÓSTICO (de novo/BL), auditável. Testa se o modelo pode ser usado JÁ NO
DIAGNÓSTICO: compara o C usando a gravidade da visita mais PRECOCE (de novo) vs a de logo antes da
levodopa. Se similar, o Total-6 é deployável no diagnóstico (aconselhamento precoce)."""
from config import PROJECT_ROOT
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
B=PROJECT_ROOT; PR=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']; OTHER=['ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
M=ip(M); base=M[['PATNO','exit','event']+OTHER].dropna(subset=OTHER)
tz=ip(pd.read_parquet(PR+'/tzero.parquet')); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'],errors='coerce')
cur=ip(pd.read_parquet(B+'/02_Dados_Processados/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz[['PATNO','tzero_levodopa']],on='PATNO',how='inner'); cur['dsd']=(cur['visit_date']-cur['tzero_levodopa']).dt.days/365.25
cur['ut']=pd.to_numeric(cur['updrs1_score'],errors='coerce')+pd.to_numeric(cur['updrs2_score'],errors='coerce')+pd.to_numeric(cur['updrs3_score'],errors='coerce')
pre=cur[(cur['dsd']<=0)&(cur['ut'].notna())].copy()
# mais precoce (de novo) e último pré-levodopa
early=pre.sort_values('dsd').groupby('PATNO').first().reset_index()[['PATNO','ut','dsd']].rename(columns={'ut':'ut_denovo','dsd':'lead'})
last=pre.sort_values('dsd').groupby('PATNO').last().reset_index()[['PATNO','ut']].rename(columns={'ut':'ut_last'})
d=base.merge(early,on='PATNO').merge(last,on='PATNO')
d['lead_anos']=-d['lead']
print(f'n={len(d)}, eventos={int(d.event.sum())}')
print(f'lead time (anos entre a visita de novo e o início da levodopa): mediana={d.lead_anos.median():.2f}, p90={d.lead_anos.quantile(.9):.2f}')
def oof(col,seeds=range(8)):
    cols=OTHER+[col]; cs=[]
    for s in seeds:
        kf=KFold(5,shuffle=True,random_state=s); pr=np.zeros(len(d))
        for tr,te in kf.split(d):
            m=CoxPHFitter(penalizer=0.1).fit(d.iloc[tr][['exit','event']+cols],'exit','event'); pr[te]=-m.predict_partial_hazard(d.iloc[te][cols]).values
        cs.append(concordance_index(d['exit'],pr,d['event']))
    return np.mean(cs)
c_last=oof('ut_last'); c_denovo=oof('ut_denovo')
print(f'\nC (Total-6 com UPDRS de LOGO ANTES da levodopa) = {c_last:.4f}')
print(f'C (Total-6 com UPDRS DE NOVO / mais precoce)      = {c_denovo:.4f}')
print(f'ΔC (de novo - antes-levodopa) = {c_denovo-c_last:+.4f}')
print('\nLeitura: se ΔC≈0, o modelo prediz igual usando a gravidade do DIAGNÓSTICO => deployável no diagnóstico (aconselhamento precoce).')
pd.DataFrame([{'modelo':'UPDRS antes-levodopa','C':round(c_last,4)},{'modelo':'UPDRS de novo (diagnóstico)','C':round(c_denovo,4)},{'lead_mediana_anos':round(d.lead_anos.median(),2)}]).to_csv(TAB+'/tab47_denovo.csv',index=False)
