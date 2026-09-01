# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""23 — fecho de completude: (A) ganho incremental OOF dos melhores extras sobre o modelo de 8 var;
(B) screen de comorbidades (Medical Conditions) e medicações basais não-dopa. FDR."""
from config import PROJECT_ROOT
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
B=PROJECT_ROOT
RAW=B+'/01_Dados_Brutos'; UPD=RAW+'/MDS_UPDRS_e_Motor'; HM=RAW+'/Historia_Medica_e_Medicacao'
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
mb=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
cur=ip(pd.read_parquet(B+'/02_Dados_Processados/curated_cache.parquet'))
cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce'); cur=cur.merge(tz,on='PATNO',how='inner')
cur['dsd']=(cur['visit_date']-cur['tzero_levodopa']).dt.days
preb=cur[cur['dsd']<=0].sort_values('dsd').groupby('PATNO')[['MSEADLG','NP1FATG']].last().reset_index()
mb=mb.merge(preb,on='PATNO',how='left')
BASE=['updrs2_score','ageonset','SEX','MSEADLG','NP1FATG','comp_bradicinesia','BMI','td_pigd_ratio']
def oof_C(df,cols):
    df=df[['exit','event']+cols].dropna(); 
    if len(df)<100: return np.nan,len(df)
    kf=KFold(5,shuffle=True,random_state=42); pr=np.zeros(len(df)); idx=df.reset_index(drop=True)
    for tr,te in kf.split(idx):
        a,b=idx.iloc[tr],idx.iloc[te]
        try:
            m=CoxPHFitter(penalizer=0.1).fit(a[['exit','event']+cols],'exit','event')
            pr[te]=-m.predict_partial_hazard(b[cols]).values
        except Exception: return np.nan,len(df)
    return concordance_index(idx['exit'],pr,idx['event']),len(df)
# itens baseline (do screen 22)
def bl_items(fn,pf,keep):
    d=ip(pd.read_csv(UPD+'/'+fn,low_memory=False)); d['INFODT']=pd.to_datetime(d['INFODT'],errors='coerce',format='mixed')
    d=d.merge(tz,on='PATNO',how='inner'); d['dsd']=(d['INFODT']-d['tzero_levodopa']).dt.days; d=d[d['dsd']<=0].sort_values('dsd')
    for c in keep: d[c]=pd.to_numeric(d.get(c),errors='coerce').replace(101,np.nan)
    return d.groupby('PATNO')[keep].last().reset_index()
it=bl_items('MDS_UPDRS_Part_II__Patient_Questionnaire_29Apr2026.csv','NP2',['NP2HOBB','NP2FREZ','NP2DRES'])
it3=bl_items('MDS-UPDRS_Part_III_29Apr2026.csv','NP3',['NP3BRADY'])
M=mb.merge(it,on='PATNO',how='left').merge(it3,on='PATNO',how='left')
if 'orthostasis' in cur.columns: M=M.merge(cur.groupby('PATNO')['orthostasis'].max().reset_index(),on='PATNO',how='left')
cb,_=oof_C(M,BASE)
print(f'=== (A) GANHO INCREMENTAL sobre modelo de 8 var (OOF C base={cb:.3f}) ===')
rows=[]
for v in ['NP2HOBB','NP2FREZ','NP2DRES','NP3BRADY','orthostasis']:
    if v in M.columns:
        c,n=oof_C(M,BASE+[v]); rows.append({'extra':v,'C_base+extra':round(c,3),'deltaC':round(c-cb,3),'n':n})
print(pd.DataFrame(rows).to_string(index=False))
# (B) COMORBIDADES
print('\n=== (B1) COMORBIDADES (Medical Conditions, baseline) ===')
mc=ip(pd.read_csv(glob.glob(HM+'/Medical_Conditions*')[0],low_memory=False))
tcol=[c for c in mc.columns if 'TERM' in c.upper() or 'COND' in c.upper() or 'DIAG' in c.upper()]
print('cols termo:',tcol[:4],'| total linhas:',len(mc))
out=mb[['PATNO','exit','event']].copy(); flags={}
if tcol:
    mc['t']=mc[tcol[0]].astype(str).str.lower()
    defs={'diabetes':'diabet','hipertensao':'hypertens|high blood','depressao':'depress','ansiedade':'anxiety|anxious',
          'dislipidemia':'choleste|lipid|statin','hipotireoid':'hypothyroid|thyroid'}
    for k,pat in defs.items():
        pats=set(mc.loc[mc['t'].str.contains(pat,regex=True,na=False),'PATNO'].dropna().astype(int))
        out[k]=out['PATNO'].astype(int).isin(pats).astype(int); flags[k]=len(pats)
    rows=[]
    for k in defs:
        s=out[out['exit']>0][['exit','event',k]].dropna()
        if s[k].sum()<10: rows.append({'comorbid':k,'n_pos':int(s[k].sum()),'p':np.nan,'HR':np.nan}); continue
        m=CoxPHFitter(penalizer=0.05).fit(s,'exit','event'); h=m.summary.loc[k]
        rows.append({'comorbid':k,'n_pos':int(s[k].sum()),'HR':round(h['exp(coef)'],3),'p':round(float(h['p']),4),'C':round(m.concordance_index_,3)})
    print(pd.DataFrame(rows).to_string(index=False))
# (B2) MEDS basais não-dopa
print('\n=== (B2) MEDICAÇÕES basais (Concomitant Med Log, started<=tzero) ===')
cm=ip(pd.read_csv(glob.glob(HM+'/Concomitant_Medication_Log*')[0],low_memory=False))
mcol=[c for c in cm.columns if 'CMTRT' in c.upper() or 'MEDNAME' in c.upper() or ('TRT' in c.upper())]
scol=[c for c in cm.columns if c.upper() in ('STARTDT','CMSTDT','STRTDT')]
print('cols med/data:',mcol[:3],scol[:2])
if mcol and scol:
    cm['m']=cm[mcol[0]].astype(str).str.lower(); cm[scol[0]]=pd.to_datetime(cm[scol[0]],errors='coerce',format='mixed')
    cm=cm.merge(tz,on='PATNO',how='left'); cm=cm[(cm[scol[0]]<=cm['tzero_levodopa'])|(cm['tzero_levodopa'].isna())]
    classes={'antidepressivo':'sertralin|fluoxet|escitalop|citalopram|paroxet|venlafax|duloxet|bupropion|mirtazap|amitript',
             'agonista_dopa':'pramipex|ropinirol|rotigotin|apomorph','benzodiazepina':'clonazep|diazep|lorazep|alprazol',
             'mao_b':'rasagilin|selegilin|safinamid','amantadina':'amantad','betabloq':'metoprol|atenolol|propranol|bisoprol',
             'antipsicotico':'quetiapin|clozapin|risperid|olanzap'}
    o2=mb[['PATNO','exit','event']].copy(); rows=[]
    for k,pat in classes.items():
        pats=set(cm.loc[cm['m'].str.contains(pat,regex=True,na=False),'PATNO'].dropna().astype(int))
        o2[k]=o2['PATNO'].astype(int).isin(pats).astype(int)
        s=o2[o2['exit']>0][['exit','event',k]].dropna()
        if s[k].sum()<10: rows.append({'classe':k,'n_pos':int(s[k].sum()),'HR':np.nan,'p':np.nan}); continue
        m=CoxPHFitter(penalizer=0.05).fit(s,'exit','event'); h=m.summary.loc[k]
        rows.append({'classe':k,'n_pos':int(s[k].sum()),'HR':round(h['exp(coef)'],3),'p':round(float(h['p']),4),'C':round(m.concordance_index_,3)})
    dfm=pd.DataFrame(rows); print(dfm.to_string(index=False))
    dfm.to_csv(TAB+'/tab23_meds_comorbid.csv',index=False)
print('\nFIM completude.')
