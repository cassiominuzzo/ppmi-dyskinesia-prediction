# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

from config import PROJECT_ROOT
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
B=PROJECT_ROOT
HM=B+'/01_Dados_Brutos/Historia_Medica_e_Medicacao'; UPD=B+'/01_Dados_Brutos/MDS_UPDRS_e_Motor'
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
mb=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
out=mb[['PATNO','exit','event']].copy()
# (B1) COMORBIDADES — usar MHTERM, onset <= tzero
mc=ip(pd.read_csv(glob.glob(HM+'/Medical_Conditions*')[0],low_memory=False))
mc['t']=mc['MHTERM'].astype(str).str.lower()
defs={'diabetes':'diabet','hipertensao':'hypertens|high blood pressure','depressao':'depress',
      'ansiedade':r'anxiet|anxious','dislipidemia':'choleste|lipid|dyslipid|statin','hipotireoid':'hypothyroid'}
rows=[]
for k,pat in defs.items():
    pats=set(mc.loc[mc['t'].str.contains(pat,regex=True,na=False),'PATNO'].dropna().astype(int))
    out[k]=out['PATNO'].astype(int).isin(pats).astype(int)
    s=out[out['exit']>0][['exit','event',k]].dropna()
    if s[k].sum()<10: rows.append({'comorbid':k,'n_pos':int(s[k].sum()),'HR':np.nan,'p':np.nan}); continue
    m=CoxPHFitter(penalizer=0.05).fit(s,'exit','event'); h=m.summary.loc[k]
    rows.append({'comorbid':k,'n_pos':int(s[k].sum()),'HR':round(h['exp(coef)'],3),'p':round(float(h['p']),4),'C':round(m.concordance_index_,3)})
cdf=pd.DataFrame(rows); cdf['p_FDR']=multipletests(cdf['p'].fillna(1),method='fdr_bh')[1].round(3)
print('=== COMORBIDADES (MHTERM, baseline) ==='); print(cdf.to_string(index=False))
# (A bis) NP2FREZ — teste repetido (10 sementes) p/ ver estabilidade do ΔC
cur=ip(pd.read_parquet(B+'/02_Dados_Processados/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz,on='PATNO',how='inner'); cur['dsd']=(cur['visit_date']-cur['tzero_levodopa']).dt.days
preb=cur[cur['dsd']<=0].sort_values('dsd').groupby('PATNO')[['MSEADLG','NP1FATG']].last().reset_index()
p2=ip(pd.read_csv(UPD+'/MDS_UPDRS_Part_II__Patient_Questionnaire_29Apr2026.csv',low_memory=False))
p2['INFODT']=pd.to_datetime(p2['INFODT'],errors='coerce',format='mixed'); p2=p2.merge(tz,on='PATNO',how='inner')
p2['dsd']=(p2['INFODT']-p2['tzero_levodopa']).dt.days; p2=p2[p2['dsd']<=0].sort_values('dsd')
for c in ['NP2FREZ','NP2WALK','NP2TURN']: p2[c]=pd.to_numeric(p2[c],errors='coerce').replace(101,np.nan)
fr=p2.groupby('PATNO')[['NP2FREZ','NP2WALK','NP2TURN']].last().reset_index()
M=mb.merge(preb,on='PATNO',how='left').merge(fr,on='PATNO',how='left')
BASE=['updrs2_score','ageonset','SEX','MSEADLG','NP1FATG','comp_bradicinesia','BMI','td_pigd_ratio']
def oof(df,cols,seed):
    df=df[['exit','event']+cols].dropna(); kf=KFold(5,shuffle=True,random_state=seed); pr=np.zeros(len(df)); idx=df.reset_index(drop=True)
    for tr,te in kf.split(idx):
        try:
            m=CoxPHFitter(penalizer=0.1).fit(idx.iloc[tr][['exit','event']+cols],'exit','event'); pr[te]=-m.predict_partial_hazard(idx.iloc[te][cols]).values
        except Exception: return np.nan
    return concordance_index(idx['exit'],pr,idx['event'])
import numpy as np
for v in ['NP2FREZ','NP2WALK','NP2TURN']:
    db=[oof(M,BASE,s) for s in range(10)]; dx=[oof(M,BASE+[v],s) for s in range(10)]
    d=np.array(dx)-np.array(db)
    print(f'{v}: ΔC médio={d.mean():+.3f} (DP {d.std():.3f}, min {d.min():+.3f} max {d.max():+.3f}) sobre 10 sementes')
