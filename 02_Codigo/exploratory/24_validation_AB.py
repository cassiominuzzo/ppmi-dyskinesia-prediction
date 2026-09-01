# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""24 — Validação dos modelos finais. Constrói dataset analítico, reporta missingness,
HRs+IC95% e C-index corrigido por otimismo (bootstrap .632-style optimism, Harrell/Steyerberg).
Modelo A = 8 var; Modelo B = 8 + NP2FREZ."""
from config import PROJECT_ROOT
import glob, warnings, pickle, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
np.random.seed(42)
B=PROJECT_ROOT
UPD=B+'/01_Dados_Brutos/MDS_UPDRS_e_Motor'; PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
mb=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
cur=ip(pd.read_parquet(B+'/02_Dados_Processados/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz,on='PATNO',how='inner'); cur['dsd']=(cur['visit_date']-cur['tzero_levodopa']).dt.days
preb=cur[cur['dsd']<=0].sort_values('dsd').groupby('PATNO').agg({'MSEADLG':'last','NP1FATG':'last','SITE':'first'}).reset_index()
p2=ip(pd.read_csv(UPD+'/MDS_UPDRS_Part_II__Patient_Questionnaire_29Apr2026.csv',low_memory=False))
p2['INFODT']=pd.to_datetime(p2['INFODT'],errors='coerce',format='mixed'); p2=p2.merge(tz,on='PATNO',how='inner')
p2['dsd']=(p2['INFODT']-p2['tzero_levodopa']).dt.days; p2=p2[p2['dsd']<=0].sort_values('dsd')
p2['NP2FREZ']=pd.to_numeric(p2['NP2FREZ'],errors='coerce').replace(101,np.nan)
fr=p2.groupby('PATNO')['NP2FREZ'].last().reset_index()
df=mb.merge(preb,on='PATNO',how='left').merge(fr,on='PATNO',how='left')
df=df[df['exit']>0].copy()
A=['updrs2_score','ageonset','SEX','MSEADLG','NP1FATG','comp_bradicinesia','BMI','td_pigd_ratio']
Bv=A+['NP2FREZ']
print(f'N={len(df)} | eventos={int(df.event.sum())}')
print('\n=== MISSINGNESS por variável ===')
for v in Bv: print(f'  {v:18s} {100*df[v].isna().mean():4.1f}% faltante')
# imputação mediana (report) — para reter N
imp=df.copy()
for v in Bv: imp[v]=pd.to_numeric(imp[v],errors='coerce').fillna(df[v].median())
def cidx(model,data,cols): return concordance_index(data['exit'],-model.predict_partial_hazard(data[cols]).values,data['event'])
def optimism_C(cols,Bboot=200):
    full=imp[['exit','event']+cols].dropna()
    m0=CoxPHFitter(penalizer=0.1).fit(full,'exit','event'); c_app=cidx(m0,full,cols)
    opt=[]
    for b in range(Bboot):
        bs=full.sample(len(full),replace=True)
        try:
            mb_=CoxPHFitter(penalizer=0.1).fit(bs,'exit','event')
            opt.append(cidx(mb_,bs,cols)-cidx(mb_,full,cols))
        except Exception: pass
    o=np.mean(opt); return c_app,c_app-o,o,m0,full
res={}
for name,cols in [('A_8var',A),('B_9var',Bv)]:
    c_app,c_corr,o,m,full=optimism_C(cols)
    res[name]={'C_app':c_app,'C_corr':c_corr,'optimism':o,'model':m,'n':len(full)}
    print(f'\n=== MODELO {name} (n={len(full)}) ===')
    print(f'  C aparente={c_app:.3f} | otimismo={o:.3f} | C CORRIGIDO={c_corr:.3f}')
    s=m.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3)
    s.columns=['HR','IC95_inf','IC95_sup','p']; print(s.to_string())
print(f"\nΔC corrigido (B−A) = {res['B_9var']['C_corr']-res['A_8var']['C_corr']:+.3f}")
pickle.dump({k:{kk:vv for kk,vv in v.items() if kk!='model'} for k,v in res.items()},open(PROC+'/results_val_AB.pkl','wb'))
# salvar HRs
for name,cols in [('A_8var',A),('B_9var',Bv)]:
    s=res[name]['model'].summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(4)
    s.columns=['HR','IC95_inf','IC95_sup','p']; s.to_csv(f'{TAB}/tab24_HR_{name}.csv')
df[['PATNO','exit','event']+Bv+['SITE']].to_parquet(PROC+'/val_dataset.parquet')
print('\nartefatos salvos (val_dataset, HRs, optimism).')
