# 02_horizon_support.py
# Produces: Follow-up supporting each horizon. Table 2, at-risk row
# Original file in the project archive: 02_horizon_scoping.py

"""
02_horizon_scoping.py — Quais horizontes (anos) são suportados pelos dados?
Constrói o objeto de sobrevida no TEMPO-ZERO CORRIGIDO (levodopa) com truncamento à esquerda
e mede, para cada horizonte candidato: nº em risco, eventos acumulados, incidência cumulativa (KM).
Informa a escolha do desenho. (Também adianta a R5.)
"""
from config import PROJECT_ROOT
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import os, glob, warnings, pickle
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

_c=[PROJECT_ROOT]; BASE=_c[0] if _c else '.'
RAW=C.RAW; PROJ=C.RESULTS
PROC=C.PROC; FIG=C.ensure_dir(os.path.join(C.RESULTS,'Figuras'))
def best(*c):
    v=[p for p in c if p and os.path.exists(p)]; return max(v,key=os.path.getsize) if v else None
def ip(df): df=df.copy(); df['PATNO']=pd.to_numeric(df['PATNO'],errors='coerce').astype('Int64'); return df.dropna(subset=['PATNO'])

tz=pd.read_parquet(os.path.join(PROC,'tzero.parquet'))[['PATNO','tzero_levodopa']].dropna()
tz=ip(tz)
p4=ip(pd.read_csv(best(os.path.join(RAW,'MDS_UPDRS_e_Motor','MDS-UPDRS_Part_IV__Motor_Complications_29Apr2026.csv'))))
p4['INFODT']=pd.to_datetime(p4['INFODT'],errors='coerce',format='mixed')
for cc in ['NP4WDYSK','NP4DYSKI']: p4[cc]=p4[cc].replace(101,np.nan)
m=p4.merge(tz,on='PATNO',how='inner'); m=m[m['INFODT']>=m['tzero_levodopa']].copy()
m['yrs']=(m['INFODT']-m['tzero_levodopa']).dt.days/365.25
m['flag']=((m['NP4WDYSK']>=2)|(m['NP4DYSKI']>=2)).astype(int)
m=m.sort_values(['PATNO','yrs'])

# survival com truncamento à esquerda: entry=1a Part IV pós-levodopa; exit=evento ou última visita
def surv(g):
    g=g.sort_values('yrs'); entry=g['yrs'].iloc[0]; hit=g[g['flag']==1]
    if len(hit): ev=hit['yrs'].iloc[0]; return pd.Series({'entry':entry,'exit':ev,'event':1})
    return pd.Series({'entry':entry,'exit':g['yrs'].iloc[-1],'event':0})
S=m.groupby('PATNO').apply(surv).reset_index()
S['left_censored']=((S['event']==1)&(np.isclose(S['exit'],S['entry']))).astype(int)
print(f'Pacientes com tempo-zero levodopa + >=1 Part IV pós: {len(S)}')
print(f'Eventos primários (4.1>=2 OU 4.2>=2): {int(S.event.sum())}')
print(f'  dos quais "left-censored" (já com discinesia na 1a Part IV pós-levodopa): {int(S.left_censored.sum())}')
print(f'  => eventos INCIDENTES observáveis: {int(S.event.sum()-S.left_censored.sum())}')

# para KM/at-risk: usar entrada (truncamento) e exit; exigir exit>entry
Sv=S[S['exit']>S['entry']].copy()
kmf=KaplanMeierFitter(); kmf.fit(Sv['exit'],Sv['event'],entry=Sv['entry'])

HZ=[1,2,3,4,5,7,10,12,15]
rows=[]
for h in HZ:
    at_risk=int(((S['entry']<h)&(S['exit']>=h)).sum())              # em risco no instante h
    cum_ev=int(((S['event']==1)&(S['exit']<=h)&(S['left_censored']==0)).sum())  # eventos incidentes até h
    try:
        ci=float(1-kmf.predict(h))
    except: ci=np.nan
    rows.append({'Horizonte (anos)':h,'Em risco em t':at_risk,'Eventos incid. ate t':cum_ev,
                 'Incid. cumul. KM (%)':round(100*ci,1) if not np.isnan(ci) else None})
T=pd.DataFrame(rows)
print('\n=== SUPORTE DOS DADOS POR HORIZONTE (tempo-zero corrigido, levodopa) ===')
print(T.to_string(index=False))

# figura: curva de incidência cumulativa + barras de "em risco"
fig,ax=plt.subplots(1,2,figsize=(12,4.2))
t=np.linspace(0,16,200); ax[0].plot(t,[1-kmf.predict(x) for x in t],color='#4C72B0')
ax[0].set_title('Incidência cumulativa de discinesia problemática\n(tempo-zero=levodopa, truncamento à esquerda)')
ax[0].set_xlabel('anos desde a levodopa'); ax[0].set_ylabel('incidência cumulativa'); ax[0].set_xlim(0,15)
for h in [3,5,10]: ax[0].axvline(h,ls=':',color='gray')
ax[1].bar([str(h) for h in HZ],[r['Em risco em t'] for r in rows],color='#55A868')
for i,r in enumerate(rows): ax[1].text(i,r['Em risco em t']+8,str(r['Em risco em t']),ha='center',fontsize=8)
ax[1].set_title('Pacientes em risco por horizonte'); ax[1].set_xlabel('horizonte (anos)'); ax[1].set_ylabel('N em risco')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig02_horizon_support.png'),dpi=130); plt.close()
T.to_csv(os.path.join(PROC,'tab02_horizon_support.csv'),index=False)
print('\nfigura: fig02_horizon_support.png | tabela: tab02_horizon_support.csv')
