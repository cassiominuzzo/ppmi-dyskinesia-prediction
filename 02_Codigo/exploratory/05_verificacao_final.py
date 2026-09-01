# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

from config import PROJECT_ROOT
import os,glob,re,warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
_c=[PROJECT_ROOT]; BASE=_c[0]
RAW=BASE+'/01_Dados_Brutos'; PROC=BASE+'/Projeto_LID_v2/02_Dados_Processados'; SRC=BASE+'/02_Dados_Processados'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
F=[]; 
def chk(n,ok,d=''): print(f"[{'PASS' if ok else '*FALHA':5s}] {n}"+(f' — {d}' if d else '')); F.append(n) if not ok else None
LD=(r'dopa|carb.{0,4}lev|lev.{0,3}dop|levodop|levodapa|levopar|levocomp|sinemet|madopar|stalevo|rytary|'
    r'duopa|duodopa|parcopa|prolopa|isicom|nacom|careldopa|carledopa|melevodopa|foslevodopa|benseraz|'
    r'vyalev|sirio|restex|kinson|syndopa|tidomet|sastravi|dopadura|levomet')
ledd=ip(pd.read_csv(RAW+'/Historia_Medica_e_Medicacao/LEDD_Concomitant_Medication_Log_29Apr2026.csv'))
ledd['l']=ledd['LEDTRT'].astype(str).str.lower()
ledd['is_ld']=ledd['l'].str.contains(LD,regex=True,na=False) & ~ledd['l'].str.contains('methyldopa',na=False)
t=ledd.groupby('LEDTRT')['is_ld'].first()
susp=[x for x in t[~t].index.astype(str) if re.search(r'dopa|lev|carb',x.lower()) and 'methyldopa' not in x.lower()]
chk('Filtro levodopa SEM falsos negativos (nenhum termo dopa/lev/carb fora)', len(susp)==0, f'restam={susp}')
ledd['STARTDT']=pd.to_datetime(ledd['STARTDT'],errors='coerce',format='mixed')
tz=ledd[ledd.is_ld].dropna(subset=['STARTDT']).groupby('PATNO')['STARTDT'].min()
p4=ip(pd.read_csv(RAW+'/MDS_UPDRS_e_Motor/MDS-UPDRS_Part_IV__Motor_Complications_29Apr2026.csv'))
p4['INFODT']=pd.to_datetime(p4['INFODT'],errors='coerce',format='mixed')
for c in ['NP4WDYSK','NP4DYSKI']: p4[c]=pd.to_numeric(p4[c],errors='coerce').replace(101,np.nan)
m=p4.merge(tz.rename('tz').reset_index(),on='PATNO',how='inner'); m=m[m['INFODT']>=m['tz']]
m['yr']=(m['INFODT']-m['tz']).dt.days/365.25; m['fl']=((m['NP4WDYSK']>=2)|(m['NP4DYSKI']>=2)).astype(int)
N=m['PATNO'].nunique(); ev=int(m.groupby('PATNO')['fl'].max().sum())
ab=pd.read_parquet(PROC+'/analytic_base.parquet')
chk('Coorte N (raw vs salvo)', N==len(ab), f'{N} vs {len(ab)}')
chk('Eventos (raw vs salvo)', ev==int(ab.event.sum()), f'{ev} vs {int(ab.event.sum())}')
chk('entry<=exit e tempos>=0', bool((ab.exit>=ab.entry).all() and (ab.entry>=0).all()))
def sv(g):
    g=g.sort_values('yr'); e=g.yr.iloc[0]; h=g[g.fl==1]
    return pd.Series({'entry':e,'exit':(h.yr.iloc[0] if len(h) else g.yr.iloc[-1]),'event':int(len(h)>0)})
S=m.groupby('PATNO').apply(sv); S['lc']=((S.event==1)&(np.isclose(S.exit,S.entry))).astype(int)
ar=[int(((S.entry<h)&(S.exit>=h)).sum()) for h in [3,5,7,10]]
chk('Em-risco monotônico decrescente (3>5>7>10)', ar[0]>ar[1]>ar[2]>ar[3], str(ar))
gen=ip(pd.read_csv(RAW+'/Geneticos/Genetic_Testing_Results_29Apr2026.csv'))
gen['MUTRSLT']=pd.to_numeric(gen['MUTRSLT'],errors='coerce'); gen['GENECAT']=pd.to_numeric(gen['GENECAT'],errors='coerce')
ids=set(ab.PATNO.dropna().astype(int))
lrrk2=len(set(gen.loc[(gen.GENECAT==1)&(gen.MUTRSLT==1),'PATNO'].dropna().astype(int))&ids)
gba=len(set(gen.loc[(gen.GENECAT==3)&(gen.MUTRSLT==1),'PATNO'].dropna().astype(int))&ids)
print(f'  [info] N={N}, eventos={ev}, em-risco 3/5/7/10a={ar}, LRRK2+={lrrk2}, GBA+={gba}')
print('\n===> '+('TUDO VERIFICADO — SEM ERROS' if not F else f'{len(F)} FALHA(S): '+';'.join(F)))
