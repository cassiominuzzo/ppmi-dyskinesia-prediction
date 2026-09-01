# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""28 — completude: screen das tabelas clínicas NOVAS achadas na varredura 06_DADOS.
Baseline pré-levodopa, derivando escores/flags. Exclui features de vazamento. FDR global."""
from config import PROJECT_ROOT
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
B=PROJECT_ROOT; D=B+'/06_DADOS'
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
out=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))[['PATNO','exit','event']]
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
def baseline(fn, cols):
    d=ip(pd.read_csv(D+'/'+fn,low_memory=False)); d['INFODT']=pd.to_datetime(d['INFODT'],errors='coerce',format='mixed')
    d=d.merge(tz,on='PATNO',how='inner'); d['dsd']=(d['INFODT']-d['tzero_levodopa']).dt.days; d=d[d['dsd']<=0].sort_values('dsd')
    for c in cols: d[c]=pd.to_numeric(d[c],errors='coerce').replace({101:np.nan,-4:np.nan})
    return d.groupby('PATNO')[cols].last()
feats={}
# Neuro-QoL (4 formas) — escore = soma dos itens
for fn,key,pref in [('Motor MDS-UPDRS/Neuro_QoL__Upper_Extremity_Function_-_Short_Form_29Apr2026.csv','NQoL_upperUE','NQUEX'),
                    ('Motor MDS-UPDRS/Neuro_QoL__Lower_Extremity_Function__Mobility__-_Short_Form_29Apr2026.csv','NQoL_lowerLE','NQMOB'),
                    ('Non-motor Assessments/Neuro_QoL__Communication_-_Short_Form_29Apr2026.csv','NQoL_comm','NQCOG'),
                    ('Non-motor Assessments/Neuro_QoL__Cognition_Function_-_Short_Form_29Apr2026.csv','NQoL_cog','NQCOG')]:
    try:
        cols=[c for c in pd.read_csv(D+'/'+fn,nrows=1).columns if c.startswith(pref)]
        if cols: feats[key]=baseline(fn,cols).sum(axis=1,min_count=len(cols)//2)
    except Exception as e: print('skip',key,str(e)[:40])
# PDAQ-27
try:
    cols=[c for c in pd.read_csv(D+'/Non-motor Assessments/PDAQ-27_29Apr2026.csv',nrows=1).columns if c.startswith('DIFF')]
    feats['PDAQ27_total']=baseline('Non-motor Assessments/PDAQ-27_29Apr2026.csv',cols).sum(axis=1,min_count=15)
except Exception as e: print('skip pdaq',str(e)[:40])
# Freezing/Falls
ff=baseline('Medical History/Determination_of_Freezing_and_Falls_29Apr2026.csv',['FRZGT1W','FRZGT12M','FLNFR1W','FLNFR12M'])
for c in ff.columns: feats['FF_'+c]=ff[c]
# Features of Parkinsonism
fp=baseline('Medical History/Features_of_Parkinsonism_29Apr2026.csv',['FEATBRADY','FEATPOSINS','FEATRIGID','FEATTREMOR','PSGLVL'])
for c in fp.columns: feats['Park_'+c]=fp[c]
# Other Clinical Features — EXCLUI vazamento
ocf_all=[c for c in pd.read_csv(D+'/Medical History/Other_Clinical_Features_29Apr2026.csv',nrows=1).columns if c.startswith('FEAT')]
LEAK={'FEATDYSKIN','FEATMTRFLC','FEATCLRLEV','FEATNOLEVO','FEATCLRLEV'}
ocf=[c for c in ocf_all if c not in LEAK]
oc=baseline('Medical History/Other_Clinical_Features_29Apr2026.csv',ocf)
for c in oc.columns:
    if oc[c].notna().sum()>=150 and oc[c].nunique()>=2: feats['OCF_'+c]=oc[c]
# Neurological Exam (abnormality)
ne=baseline('Medical History/Neurological_Exam_29Apr2026.csv',['MTRRSP','CORDRSP','SENRSP','RFLXRSP','GAITRSP','MNTLRSP','CNRSP'])
for c in ne.columns: feats['NE_'+c]=ne[c]
# CGI
cgi=baseline('Medical History/Clinical_Global_Impression__CGI__-_Investigator_29Apr2026.csv',['INVESTAST'])
feats['CGI_invest']=cgi['INVESTAST']
# Vital signs — ortostase
vs=baseline('Medical History/Vital_Signs_29Apr2026.csv',['SYSSUP','SYSSTND','DIASUP','DIASTND','HRSUP','HRSTND'])
feats['orthostatic_drop_sys']=vs['SYSSUP']-vs['SYSSTND']; feats['orthostatic_drop_dia']=vs['DIASUP']-vs['DIASTND']; feats['HR_rise_stand']=vs['HRSTND']-vs['HRSUP']
# montar matriz e screenar
M=out.copy()
for k,s in feats.items(): M=M.merge(s.rename(k).reset_index(),on='PATNO',how='left')
M=M[M['exit']>0]; rows=[]
for k in feats:
    s=M[['exit','event',k]].dropna()
    if len(s)<80 or int(s.event.sum())<10 or s[k].nunique()<3: continue
    sd=s[k].std()
    if sd==0 or np.isnan(sd): continue
    s=s.copy(); s[k]=(s[k]-s[k].mean())/sd
    try:
        cph=CoxPHFitter(penalizer=0.05).fit(s,'exit','event'); h=cph.summary.loc[k]
        rows.append({'variavel':k[:40],'n':len(s),'n_ev':int(s.event.sum()),'HR_DP':round(h['exp(coef)'],3),'p':float(h['p']),'C':round(cph.concordance_index_,3)})
    except Exception: pass
R=pd.DataFrame(rows); R['p_FDR']=multipletests(R['p'],method='fdr_bh')[1]; R=R.sort_values('p').reset_index(drop=True)
R.to_csv(TAB+'/tab28_novas_clinicas.csv',index=False)
print(f'=== TABELAS CLÍNICAS NOVAS: {len(R)} variáveis | {int((R.p_FDR<0.05).sum())} sig após FDR ===')
print(R.head(25).to_string(index=False))
sig=R[R.p_FDR<0.05]; print('\nSIG após FDR:'); print(sig.to_string(index=False) if len(sig) else '  NENHUMA.')
