# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""
09_fix_and_rescreen.py — Correções pré-modelagem + Cox univariado (origem t=0).
(1) limpa BMI(>60)/duration(<0) -> NaN; (2) recomputa UPDRS-II/III do MESMO exame bruto pré-levodopa
dos compósitos (consistência interna); (3) origem do tempo = início da levodopa (t=0), censura à
direita (inclui todos; left-trunc/intervalar = sensibilidade); (4) incidência KM t=0; (5) Cox univariado.
Sobrescreve master_baseline.parquet (corrigido).
"""
from config import PROJECT_ROOT
import os, glob, warnings, pickle
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests
_c=[PROJECT_ROOT]; BASE=_c[0]
RAW=BASE+'/01_Dados_Brutos'; PROC=BASE+'/Projeto_LID_v2/02_Dados_Processados'
FIG=BASE+'/Projeto_LID_v2/04_Resultados/Figuras'; TAB=BASE+'/Projeto_LID_v2/04_Resultados/Tabelas'
def best(*c):
    v=[p for p in c if p and os.path.exists(p)]; return max(v,key=os.path.getsize) if v else None
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
def item04(s): s=pd.to_numeric(s,errors='coerce'); return s.where((s>=0)&(s<=4))
M=pd.read_parquet(PROC+'/master_baseline.parquet')

# (1) limpeza de outliers implausíveis
M['BMI']=pd.to_numeric(M['BMI'],errors='coerce').where(lambda s:(s>=12)&(s<=60))
M['duration_yrs']=pd.to_numeric(M['duration_yrs'],errors='coerce').where(lambda s:(s>=0)&(s<=30))
print('limpeza: BMI>60 e duration<0 -> NaN')

# (2) recomputar UPDRS-II e III do exame bruto pré-levodopa (mesma fonte dos compósitos)
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
def prelv(fn):
    d=ip(pd.read_csv(best(RAW+'/MDS_UPDRS_e_Motor/'+fn),low_memory=False)); d['INFODT']=pd.to_datetime(d['INFODT'],errors='coerce',format='mixed')
    d=d.merge(tz,on='PATNO',how='inner'); d['dsd']=(d['INFODT']-d['tzero_levodopa']).dt.days
    return d[d['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index()
I3=['NP3SPCH','NP3FACXP','NP3RIGN','NP3RIGRU','NP3RIGLU','NP3RIGRL','NP3RIGLL','NP3FTAPR','NP3FTAPL','NP3HMOVR','NP3HMOVL','NP3PRSPR','NP3PRSPL','NP3TTAPR','NP3TTAPL','NP3LGAGR','NP3LGAGL','NP3RISNG','NP3GAIT','NP3FRZGT','NP3PSTBL','NP3POSTR','NP3BRADY','NP3PTRMR','NP3PTRML','NP3KTRMR','NP3KTRML','NP3RTARU','NP3RTALU','NP3RTARL','NP3RTALL','NP3RTALJ','NP3RTCON']
I2=['NP2SPCH','NP2SALV','NP2SWAL','NP2EAT','NP2DRES','NP2HYGN','NP2HWRT','NP2HOBB','NP2TURN','NP2TRMR','NP2RISE','NP2WALK','NP2FREZ']
p3=prelv('MDS-UPDRS_Part_III_29Apr2026.csv'); p2=prelv('MDS_UPDRS_Part_II__Patient_Questionnaire_29Apr2026.csv')
for c in I3:
    if c in p3.columns: p3[c]=item04(p3[c])
for c in I2:
    if c in p2.columns: p2[c]=item04(p2[c])
u3=ip(pd.DataFrame({'PATNO':p3['PATNO'],'updrs3_raw':p3[[c for c in I3 if c in p3.columns]].sum(axis=1,min_count=30)}))
u2=ip(pd.DataFrame({'PATNO':p2['PATNO'],'updrs2_raw':p2[[c for c in I2 if c in p2.columns]].sum(axis=1,min_count=11)}))
M=M.merge(u3,on='PATNO',how='left').merge(u2,on='PATNO',how='left')
# consistência: correlação master(antigo curated) vs raw
old=pd.to_numeric(M['updrs3_score'],errors='coerce'); chk=M.dropna(subset=['updrs3_raw']); 
r=np.corrcoef(chk['updrs3_raw'],pd.to_numeric(chk['updrs3_score'],errors='coerce').fillna(chk['updrs3_raw']))[0,1]
M['updrs3_score']=M['updrs3_raw']; M['updrs2_score']=M['updrs2_raw']   # passa a usar o raw consistente
M=M.drop(columns=['updrs3_raw','updrs2_raw'])
print(f'UPDRS-III recomputado do bruto (consistente com compósitos); corr c/ curated anterior r={r:.3f}')
M.to_parquet(PROC+'/master_baseline.parquet',index=False)

# (3-4) INCIDÊNCIA KM origem t=0 (sem truncamento) — primária
M['exit']=pd.to_numeric(M['exit'],errors='coerce'); M['event']=pd.to_numeric(M['event'],errors='coerce')
D=M[M['exit']>0].copy()
km=KaplanMeierFitter().fit(D['exit'],D['event']); ci=km.confidence_interval_
print(f'\n=== INCIDÊNCIA (origem t=0=levodopa; N={len(D)}, eventos={int(D.event.sum())}) ===')
for h in [3,5,7,10]:
    s=float(km.predict(h)); idx=ci.index[ci.index<=h]
    lo,hi=(np.nan,np.nan) if len(idx)==0 else (1-ci.loc[idx[-1]].iloc[1],1-ci.loc[idx[-1]].iloc[0])
    print(f'  {h}a: {100*(1-s):.1f}%  IC95% {100*lo:.1f}-{100*hi:.1f}')

# (5) COX UNIVARIADO origem t=0
CONT=['ageonset','duration_yrs','BMI','updrs1_score','updrs2_score','updrs3_score','comp_fala','comp_tremor',
      'comp_marcha_pi','comp_bradicinesia','comp_axial','comp_assimetria_lr','td_pigd_ratio','moca','stai','upsit',
      'datscan_putamen_min','datscan_putamen_asym','MIA_CAUDATE_L','MIA_CAUDATE_R','abeta','tau','ptau','NFL_CSF']
BIN=['SEX','is_pigd','pigd','td_pigd']
rows=[]
for c in [x for x in CONT+BIN if x in M.columns]:
    d=M[['exit','event',c]].copy(); d[c]=pd.to_numeric(d[c],errors='coerce'); d=d.dropna(); d=d[d['exit']>0]
    if len(d)<30 or d[c].nunique()<2 or int(d.event.sum())<5:
        rows.append({'Preditor':c,'n':len(d),'eventos':int(d.event.sum()),'HR':np.nan}); continue
    perSD=c in CONT
    if perSD: d[c]=(d[c]-d[c].mean())/d[c].std()
    try:
        cph=CoxPHFitter().fit(d,'exit','event'); hr=cph.summary.loc[c]
        try: ph=float(proportional_hazard_test(cph,d,time_transform='rank').summary['p'].iloc[0])
        except: ph=np.nan
        rows.append({'Preditor':c+(' (/DP)' if perSD else ''),'n':len(d),'eventos':int(d.event.sum()),
                     'HR':round(hr['exp(coef)'],3),'IC inf':round(hr['exp(coef) lower 95%'],3),'IC sup':round(hr['exp(coef) upper 95%'],3),
                     'p':float(hr['p']),'C-index':round(cph.concordance_index_,3),'p_PH':round(ph,3) if ph==ph else None})
    except Exception as e:
        rows.append({'Preditor':c,'n':len(d),'eventos':int(d.event.sum()),'HR':np.nan})
res=pd.DataFrame(rows); ok=res['p'].notna(); res.loc[ok,'p_FDR']=multipletests(res.loc[ok,'p'],method='fdr_bh')[1]
res=res.sort_values('p'); res.to_csv(TAB+'/tab09_univariado_t0.csv',index=False)
print('\n=== COX UNIVARIADO (origem t=0, truncamento à esquerda = sensibilidade) ===')
print(res[['Preditor','n','eventos','HR','IC inf','IC sup','p','C-index','p_PH','p_FDR']].to_string(index=False))
sig=res[(res['p_FDR']<0.05)&res['HR'].notna()]
print(f'\nSignificativos FDR<0.05: {len(sig)} | PH violado (p<0.05): {int((res["p_PH"]<0.05).sum())}')
with open(PROC+'/results_step09.pkl','wb') as f: pickle.dump({'univariate':res},f)
print('master_baseline.parquet corrigido e re-salvo. ETAPA 4 (re-rodada t=0) COMPLETA')
