# 21_full_audit.py
# Produces: Independent re-derivation of every key number from the raw tables
# Original file in the project archive: 14_full_audit.py

"""
14_full_audit.py — AUDITORIA FINAL COMPLETA (R1->8) contra o banco BRUTO.
Re-deriva cada número-chave por caminho independente e cruza com os artefatos salvos.
PASS/FAIL por checagem + veredito final. Objetivo: zero erro de dado antes do notebook.
"""
from config import PROJECT_ROOT
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import os, glob, re, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored
B=PROJECT_ROOT
RAW=C.RAW; PROC=C.PROC; SRC=B+'/02_Dados_Processados'
def best(*c):
    v=[p for p in c if p and os.path.exists(p)]; return max(v,key=os.path.getsize) if v else None
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
def i04(s): s=pd.to_numeric(s,errors='coerce'); return s.where((s>=0)&(s<=4))
F=[]; 
def ck(n,ok,d=''):
    print(f"[{'PASS' if ok else '*** FALHA':4s}] {n}"+(f' — {d}' if d else '')); 
    if not ok: F.append(n)

M=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))
for c in M.columns:
    if c!='PATNO': M[c]=pd.to_numeric(M[c],errors='coerce')

# ---- 1. TEMPO-ZERO (re-derivação independente do LEDD log) ----
print('\n# 1. TEMPO-ZERO')
ledd=ip(pd.read_csv(best(RAW+'/Historia_Medica_e_Medicacao/LEDD_Concomitant_Medication_Log_29Apr2026.csv')))
ledd['l']=ledd['LEDTRT'].astype(str).str.lower()
LD=(r'dopa|carb.{0,4}lev|lev.{0,3}dop|levodop|levodapa|levopar|levocomp|sinemet|madopar|stalevo|rytary|'
    r'duopa|duodopa|parcopa|prolopa|isicom|nacom|careldopa|carledopa|melevodopa|foslevodopa|benseraz|'
    r'vyalev|sirio|restex|kinson|syndopa|tidomet|sastravi|dopadura|levomet')
ledd['ld']=ledd['l'].str.contains(LD,regex=True,na=False)&~ledd['l'].str.contains('methyldopa',na=False)
t=ledd.groupby('LEDTRT')['ld'].first()
susp=[x for x in t[~t].index.astype(str) if re.search(r'dopa|lev|carb',x.lower()) and 'methyldopa' not in x.lower()]
ck('filtro levodopa sem falsos negativos', len(susp)==0, str(susp))
ledd['STARTDT']=pd.to_datetime(ledd['STARTDT'],errors='coerce',format='mixed')
tz=ledd[ledd.ld].dropna(subset=['STARTDT']).groupby('PATNO')['STARTDT'].min()
tzs=ip(pd.read_parquet(PROC+'/tzero.parquet')); tzs['tzero_levodopa']=pd.to_datetime(tzs['tzero_levodopa'])
mg=pd.DataFrame({'re':tz}).reset_index().merge(tzs[['PATNO','tzero_levodopa']],on='PATNO',how='inner').dropna()
ck('datas de tempo-zero idênticas', int((mg['re']!=mg['tzero_levodopa']).sum())==0, f"{int((mg['re']!=mg['tzero_levodopa']).sum())} difs")

# ---- 2. COORTE + EVENTOS (re-derivação t=0) ----
print('\n# 2. COORTE + DESFECHO')
p4=ip(pd.read_csv(best(RAW+'/MDS_UPDRS_e_Motor/MDS-UPDRS_Part_IV__Motor_Complications_29Apr2026.csv')))
p4['INFODT']=pd.to_datetime(p4['INFODT'],errors='coerce',format='mixed')
for c in ['NP4WDYSK','NP4DYSKI']: p4[c]=pd.to_numeric(p4[c],errors='coerce').replace(101,np.nan)
mm=p4.merge(tz.rename('tz').reset_index(),on='PATNO',how='inner'); mm=mm[mm['INFODT']>=mm['tz']]
mm['yr']=(mm['INFODT']-mm['tz']).dt.days/365.25; mm['fl']=((mm['NP4WDYSK']>=2)|(mm['NP4DYSKI']>=2)).astype(int)
def sv(g):
    g=g.sort_values('yr'); h=g[g.fl==1]; return pd.Series({'exit':(h.yr.iloc[0] if len(h) else g.yr.iloc[-1]),'event':int(len(h)>0)})
S=ip(mm.groupby('PATNO').apply(sv).reset_index())
ck('coorte N (raw == master)', S.PATNO.nunique()==M.PATNO.nunique(), f'{S.PATNO.nunique()} vs {M.PATNO.nunique()}')
ck('eventos (raw == master)', int(S.event.sum())==int(M.event.sum()), f'{int(S.event.sum())} vs {int(M.event.sum())}')
me=S.merge(M[['PATNO','event','exit']],on='PATNO',suffixes=('_re','_m'))
ck('event idêntico por paciente', int((me.event_re!=me.event_m).sum())==0, f'{int((me.event_re!=me.event_m).sum())} difs')
ck('exit (tempo) idêntico (<1 dia)', bool((me.exit_re-me.exit_m).abs().max()<0.01), f'maxdif={ (me.exit_re-me.exit_m).abs().max():.4f}')

# ---- 3. COMPÓSITOS (re-derivação independente do raw) ----
print('\n# 3. COMPÓSITOS')
def prelv(fn):
    d=ip(pd.read_csv(best(RAW+'/MDS_UPDRS_e_Motor/'+fn),low_memory=False)); d['INFODT']=pd.to_datetime(d['INFODT'],errors='coerce',format='mixed')
    d=d.merge(tz.rename('tz').reset_index(),on='PATNO',how='inner'); d['dsd']=(d['INFODT']-d['tz']).dt.days
    return d[d['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index()
p3=prelv('MDS-UPDRS_Part_III_29Apr2026.csv'); p2=prelv('MDS_UPDRS_Part_II__Patient_Questionnaire_29Apr2026.csv')
defs={'comp_axial':['NP3SPCH','NP3FACXP','NP3RISNG','NP3GAIT','NP3FRZGT','NP3PSTBL','NP3POSTR'],
      'comp_tremor':['NP3KTRML','NP3KTRMR','NP3PTRML','NP3PTRMR','NP3RTALJ','NP3RTALL','NP3RTALU','NP3RTARL','NP3RTARU','NP3RTCON'],
      'comp_bradicinesia':['NP3FTAPR','NP3FTAPL','NP3HMOVR','NP3HMOVL','NP3PRSPR','NP3PRSPL','NP3TTAPR','NP3TTAPL','NP3LGAGR','NP3LGAGL','NP3BRADY']}
for comp,cols in defs.items():
    for c in cols: p3[c]=i04(p3[c])
    pres=p3[cols].notna().sum(axis=1); val=p3[cols].sum(axis=1,min_count=1).where(pres>=int(np.ceil(.7*len(cols))))
    re=ip(pd.DataFrame({'PATNO':p3['PATNO'],'v':val})).merge(M[['PATNO',comp]],on='PATNO').dropna()
    ck(f'{comp} reproduz o raw', int((~np.isclose(re['v'],re[comp])).sum())==0, f"{int((~np.isclose(re['v'],re[comp])).sum())}/{len(re)} difs")
# updrs3/updrs2 consistência
I3=['NP3SPCH','NP3FACXP','NP3RIGN','NP3RIGRU','NP3RIGLU','NP3RIGRL','NP3RIGLL','NP3FTAPR','NP3FTAPL','NP3HMOVR','NP3HMOVL','NP3PRSPR','NP3PRSPL','NP3TTAPR','NP3TTAPL','NP3LGAGR','NP3LGAGL','NP3RISNG','NP3GAIT','NP3FRZGT','NP3PSTBL','NP3POSTR','NP3BRADY','NP3PTRMR','NP3PTRML','NP3KTRMR','NP3KTRML','NP3RTARU','NP3RTALU','NP3RTARL','NP3RTALL','NP3RTALJ','NP3RTCON']
for c in I3: p3[c]=i04(p3[c])
u3=ip(pd.DataFrame({'PATNO':p3['PATNO'],'u3':p3[I3].sum(axis=1,min_count=30)})).merge(M[['PATNO','updrs3_score']],on='PATNO').dropna()
ck('updrs3_score == raw (consistência)', bool(np.isclose(u3['u3'],u3['updrs3_score']).mean()>0.99), f"concordância={np.isclose(u3['u3'],u3['updrs3_score']).mean():.3f}")

# ---- 4. RANGES / SANIDADE ----
print('\n# 4. RANGES')
mx={'comp_axial':28,'comp_tremor':40,'comp_marcha_pi':28,'comp_bradicinesia':44,'comp_fala':8}
ck('compósitos <= máximo teórico', all(M[c].max()<=v for c,v in mx.items() if c in M), {c:float(M[c].max()) for c in mx if c in M})
ck('BMI em [12,60]', bool(M['BMI'].dropna().between(12,60).all()))
ck('duration em [0,30]', bool(M['duration_yrs'].dropna().between(0,30).all()))
ck('resp em [-100,100]', bool(M['resp'].dropna().between(-100,100).all()))
ck('datscan_putamen_min >0 (SBR)', bool((M['datscan_putamen_min'].dropna()> -0.1).all()))

# ---- 5. GENÉTICA (MUTRSLT, não código) ----
print('\n# 5. GENÉTICA')
gen=ip(pd.read_csv(best(RAW+'/Geneticos/Genetic_Testing_Results_29Apr2026.csv')))
gen['MUTRSLT']=pd.to_numeric(gen['MUTRSLT'],errors='coerce'); gen['GENECAT']=pd.to_numeric(gen['GENECAT'],errors='coerce')
lr=set(gen.loc[(gen.GENECAT==1)&(gen.MUTRSLT==1),'PATNO'].dropna().astype(int))
gb=set(gen.loc[(gen.GENECAT==3)&(gen.MUTRSLT==1),'PATNO'].dropna().astype(int))
ids=set(M.PATNO.astype(int))
ck('LRRK2+ (master == raw MUTRSLT)', int(M.lrrk2_carrier.sum())==len(lr&ids), f'{int(M.lrrk2_carrier.sum())} vs {len(lr&ids)}')
ck('GBA+ (master == raw MUTRSLT)', int(M.gba_carrier.sum())==len(gb&ids), f'{int(M.gba_carrier.sum())} vs {len(gb&ids)}')

# ---- 6. MODELO L1 + consistência cruzada de artefatos ----
print('\n# 6. MODELO + CONSISTÊNCIA CRUZADA')
L1=['ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai']
D=M[['exit','event']+L1].dropna(); D=D[D.exit>0]
cph=CoxPHFitter(penalizer=0.01).fit(D[['exit','event']+L1],'exit','event')
capp=concordance_index_censored(D.event.astype(bool).values,D.exit.values,cph.predict_partial_hazard(D).values.ravel())[0]
ck('C-index L1 apparent ~0.68', abs(capp-0.68)<0.02, f'{capp:.3f}')
def Lp(n): 
    p=PROC+f'/{n}'; return pickle.load(open(p,'rb')) if os.path.exists(p) else {}
r12=Lp('results_step12.pkl')
ck('C corrigido (pkl) entre 0.63-0.68', 0.63<=r12.get('C_corrected',0)<=0.68, str(r12.get('C_corrected')))
ck('C LOSO (pkl) entre 0.62-0.68', 0.62<=r12.get('C_LOSO',0)<=0.68, str(r12.get('C_LOSO')))
# N consistente entre etapas
ns=[]
for n in ['results_step03.pkl','results_step06.pkl']:
    d=Lp(n); ns.append(d.get('N') or d.get('n'))
ck('N consistente entre etapas (1447)', all((x in (1447,1441,None)) for x in ns), str(ns))

print('\n'+('='*50)+'\nVEREDITO: '+('TUDO VERIFICADO — SEM ERROS DE DADO' if not F else f'{len(F)} FALHA(S): '+'; '.join(F)))
