# 03_baseline_predictor_matrix.py
# Produces: Baseline predictor matrix, all candidates measured pre-levodopa
# Original file in the project archive: 06_master_baseline.py

"""
06_master_baseline.py — ETAPA 2: matriz de preditores BASAIS (mestre) + QC.
Tudo medido PRÉ-levodopa (anti-vazamento). Compósitos do exame não-tratado; DaTSCAN/LCR no
baseline; responsividade à levodopa como preditor LANDMARK (estado tratado, fora do basal puro).
CORREÇÃO: itens MDS-UPDRS válidos = 0..4; qualquer outro código (101 'não-avaliável', etc.) -> NaN.
Saídas: master_baseline.parquet, tab06_dicionario.csv, fig06_missing.png, qc impresso.
"""
from config import PROJECT_ROOT
import os, glob, re, warnings, pickle
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import cohort as C
# Realigned in September 2026: it used to write master_baseline.parquet into the previous
# project layout, inside the raw-data folder. The paths now come from cohort.py.
BASE=PROJECT_ROOT
RAW=C.RAW; PROC=C.PROC
FIG=C.ensure_dir(os.path.join(C.RESULTS,'Figuras')); SRC=BASE+'/02_Dados_Processados'
def best(*c):
    v=[p for p in c if p and os.path.exists(p)]; return max(v,key=os.path.getsize) if v else None
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
def item04(s):  # item MDS-UPDRS válido = 0..4; resto -> NaN (recodifica 101 etc.)
    s=pd.to_numeric(s,errors='coerce'); return s.where((s>=0)&(s<=4))

ab=pd.read_parquet(PROC+'/analytic_base.parquet')
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
print(f'coorte base: {len(ab)} | eventos: {int(ab.event.sum())}')

TREMOR=['NP3KTRML','NP3KTRMR','NP3PTRML','NP3PTRMR','NP3RTALJ','NP3RTALL','NP3RTALU','NP3RTARL','NP3RTARU','NP3RTCON']
BRADY=['NP3FTAPR','NP3FTAPL','NP3HMOVR','NP3HMOVL','NP3PRSPR','NP3PRSPL','NP3TTAPR','NP3TTAPL','NP3LGAGR','NP3LGAGL','NP3BRADY']
GAITP3=['NP3RISNG','NP3GAIT','NP3FRZGT','NP3PSTBL','NP3POSTR']
AXIAL=['NP3SPCH','NP3FACXP','NP3RISNG','NP3GAIT','NP3FRZGT','NP3PSTBL','NP3POSTR']
LEFT=['NP3FTAPL','NP3HMOVL','NP3KTRML','NP3LGAGL','NP3PRSPL','NP3PTRML','NP3RIGLL','NP3RIGLU','NP3RTALL','NP3RTALU','NP3TTAPL']
RIGHT=['NP3FTAPR','NP3HMOVR','NP3KTRMR','NP3LGAGR','NP3PRSPR','NP3PTRMR','NP3RIGRL','NP3RIGRU','NP3RTARL','NP3RTARU','NP3TTAPR']
TD=['NP2TRMR','NP3RTARU','NP3RTALU','NP3RTARL','NP3RTALL','NP3RTALJ','NP3PTRMR','NP3PTRML','NP3KTRMR','NP3KTRML','NP3RTCON']
def csum(df,cols,frac=0.7):
    cols=[c for c in cols if c in df.columns]; pres=df[cols].notna().sum(axis=1)
    return df[cols].sum(axis=1,min_count=1).where(pres>=int(np.ceil(frac*len(cols))))
def prelv(fname):
    d=ip(pd.read_csv(best(RAW+'/MDS_UPDRS_e_Motor/'+fname),low_memory=False))
    d['INFODT']=pd.to_datetime(d['INFODT'],errors='coerce',format='mixed')
    d=d.merge(tz,on='PATNO',how='inner'); d['dsd']=(d['INFODT']-d['tzero_levodopa']).dt.days
    return d[d['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index()
p3=prelv('MDS-UPDRS_Part_III_29Apr2026.csv'); p2=prelv('MDS_UPDRS_Part_II__Patient_Questionnaire_29Apr2026.csv')
for col in set(TREMOR+BRADY+GAITP3+AXIAL+LEFT+RIGHT+['NP3RIGN']):
    if col in p3.columns: p3[col]=item04(p3[col])          # <-- recodifica itens inválidos
for col in ['NP2SPCH','NP2WALK','NP2FREZ','NP2TRMR']:
    if col in p2.columns: p2[col]=item04(p2[col])
m=p3.merge(p2[['PATNO','NP2SPCH','NP2WALK','NP2FREZ','NP2TRMR']],on='PATNO',how='left')
C=pd.DataFrame({'PATNO':m['PATNO']})
C['comp_fala']=csum(m,['NP3SPCH'])+m.get('NP2SPCH',0).fillna(0)
C['comp_tremor']=csum(m,TREMOR); C['comp_marcha_pi']=csum(m,GAITP3)+m[['NP2WALK','NP2FREZ']].sum(axis=1,min_count=1)
C['comp_bradicinesia']=csum(m,BRADY); C['comp_axial']=csum(m,AXIAL)
C['comp_assimetria_lr']=(csum(m,LEFT)-csum(m,RIGHT)).abs()
tdv=m[[c for c in TD if c in m.columns]].mean(axis=1)
pgv=pd.concat([m[[c for c in ['NP3GAIT','NP3FRZGT','NP3PSTBL'] if c in m.columns]],m[[c for c in ['NP2WALK','NP2FREZ'] if c in m.columns]]],axis=1).mean(axis=1)
ratio=tdv/pgv.replace(0,np.nan); C['td_pigd_ratio']=ratio
C['is_pigd']=np.where(pgv.eq(0)&tdv.gt(0),0,np.where(ratio>=1.15,0,np.where(ratio<=0.90,1,np.nan))); C=ip(C)
print(f'compósitos: {len(C)} pac; is_pigd não-nulo={int(C.is_pigd.notna().sum())}')

cur=ip(pd.read_parquet(SRC+'/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz,on='PATNO',how='inner'); cur['dsd']=(cur['visit_date']-cur['tzero_levodopa']).dt.days
DAT=['MIA_PUTAMEN_L','MIA_PUTAMEN_R','MIA_CAUDATE_L','MIA_CAUDATE_R']; CSF=['CSFSAA','abeta','tau','ptau','NFL_CSF']
def base_modal(cols):
    cc=[c for c in cols if c in cur.columns]; sub=cur.dropna(subset=cc,how='all'); sub=sub[sub['dsd']<=30]
    return ip(sub.sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index())[['PATNO']+cc]
dat=base_modal(DAT); csf=base_modal(CSF)
for c in [x for x in DAT if x in dat.columns]: dat[c]=pd.to_numeric(dat[c],errors='coerce')
if {'MIA_PUTAMEN_L','MIA_PUTAMEN_R'}.issubset(dat.columns):
    dat['datscan_putamen_min']=dat[['MIA_PUTAMEN_L','MIA_PUTAMEN_R']].min(axis=1)
    dat['datscan_putamen_asym']=(dat['MIA_PUTAMEN_L']-dat['MIA_PUTAMEN_R']).abs()
for c in [x for x in CSF if x in csf.columns and x!='CSFSAA']: csf[c]=pd.to_numeric(csf[c],errors='coerce')

ITEMS=['NP3SPCH','NP3FACXP','NP3RIGN','NP3RIGRU','NP3RIGLU','NP3RIGRL','NP3RIGLL','NP3FTAPR','NP3FTAPL','NP3HMOVR','NP3HMOVL','NP3PRSPR','NP3PRSPL','NP3TTAPR','NP3TTAPL','NP3LGAGR','NP3LGAGL','NP3RISNG','NP3GAIT','NP3FRZGT','NP3PSTBL','NP3POSTR','NP3BRADY','NP3PTRMR','NP3PTRML','NP3KTRMR','NP3KTRML','NP3RTARU','NP3RTALU','NP3RTARL','NP3RTALL','NP3RTALJ','NP3RTCON']
p3r=ip(pd.read_csv(best(RAW+'/MDS_UPDRS_e_Motor/MDS-UPDRS_Part_III_29Apr2026.csv'),low_memory=False))
p3r['INFODT']=pd.to_datetime(p3r['INFODT'],errors='coerce',format='mixed')
for c in ITEMS: p3r[c]=item04(p3r[c])                      # <-- recodifica itens inválidos
p3r['U3']=p3r[ITEMS].sum(axis=1,min_count=30)
tot=p3r[p3r.PDSTATE.isin(['ON','OFF'])].groupby(['PATNO','EVENT_ID','PDSTATE'])['U3'].mean().unstack('PDSTATE')
dt=p3r[p3r.PDSTATE.isin(['ON','OFF'])].groupby(['PATNO','EVENT_ID'])['INFODT'].min()
ch=tot.join(dt).reset_index().dropna(subset=['ON','OFF']); ch=ch[ch['OFF']>0]; ch=ip(ch)
ch['resp']=((ch['OFF']-ch['ON'])/ch['OFF']*100).clip(-100,100)   # winsoriza extremos clínicos
ch=ch.merge(tz,on='PATNO',how='inner'); ch['cday']=(ch['INFODT']-ch['tzero_levodopa']).dt.days
ch=ch[ch['cday']>=0].sort_values(['PATNO','cday']).groupby('PATNO').first().reset_index()
resp=ip(ch)[['PATNO','resp','cday']].rename(columns={'cday':'resp_challenge_day'})

keep_dat=[c for c in ['datscan_putamen_min','datscan_putamen_asym','MIA_CAUDATE_L','MIA_CAUDATE_R'] if c in dat.columns]
keep_csf=[c for c in CSF if c in csf.columns]
master=(ab.merge(C[['PATNO','comp_fala','comp_tremor','comp_marcha_pi','comp_bradicinesia','comp_axial','comp_assimetria_lr','td_pigd_ratio','is_pigd']],on='PATNO',how='left')
          .merge(dat[['PATNO']+keep_dat],on='PATNO',how='left').merge(csf[['PATNO']+keep_csf],on='PATNO',how='left').merge(resp,on='PATNO',how='left'))
master.to_parquet(PROC+'/master_baseline.parquet',index=False)

PRED=[c for c in master.columns if c not in ['PATNO','entry','exit','event','prevalent_left','tzero_levodopa']]
miss=master[PRED].isna().mean()*100
print('\n=== RANGES (sanidade pós-correção) ===')
for c in ['comp_axial','comp_tremor','comp_marcha_pi','comp_bradicinesia','comp_fala','comp_assimetria_lr','datscan_putamen_min','resp','abeta','ptau']:
    if c in master.columns:
        s=pd.to_numeric(master[c],errors='coerce'); print(f'  {c:20s} min={s.min():.2f} max={s.max():.2f} mediana={s.median():.2f}')
exp={'comp_axial':28,'comp_tremor':40,'comp_marcha_pi':28,'comp_bradicinesia':44,'comp_fala':8}
viol={c:float(pd.to_numeric(master[c],errors='coerce').max()) for c in exp if c in master.columns and pd.to_numeric(master[c],errors='coerce').max()>exp[c]}
print('\nCompósitos acima do máximo teórico:', viol if viol else 'NENHUM (OK)')
pd.DataFrame({'variavel':PRED,'%_presente':(100-miss[PRED]).round(1).values}).to_csv(PROC+'/tab06_dicionario.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5)); (100-miss).sort_values().plot.barh(ax=ax,color='#4C72B0')
ax.set_xlabel('% presente'); ax.set_title('Cobertura dos preditores basais (master)'); plt.tight_layout(); plt.savefig(FIG+'/fig06_missing.png',dpi=130); plt.close()
with open(PROC+'/results_step06.pkl','wb') as f: pickle.dump({'n':len(master),'preds':PRED,'missing_pct':miss.to_dict(),'range_viol':viol},f)
print('\nmaster_baseline.parquet salvo (%d pac, %d preditores). ETAPA 2 COMPLETA'%(len(master),len(PRED)))
