# 04_table1_and_incidence.py
# Produces: Table 1 and cumulative incidence
# Original file in the project archive: 07_descriptive_table1.py

"""
07_descriptive_table1.py — ETAPA 3: descritiva da coorte.
Table 1 (geral + por status de evento), incidência cumulativa (KM com truncamento à esquerda)
em 3/5/7/10a com IC, KM por subgrupo (sexo, idade início, PIGD) + log-rank, e incidência por
definição de desfecho. Mantém rigor: median[IQR], n(%), missing explícito, p descritivo.
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
from lifelines.statistics import logrank_test
from scipy import stats
_c=[PROJECT_ROOT]; BASE=_c[0]
PROC=C.PROC; FIG=C.ensure_dir(os.path.join(C.RESULTS,'Figuras')); TAB=C.ensure_dir(os.path.join(C.RESULTS,'Tabelas'))
os.makedirs(TAB,exist_ok=True)
M=pd.read_parquet(PROC+'/master_baseline.parquet')
N=len(M); nev=int(M.event.sum())
print(f'N={N} | eventos={nev} | incidência bruta={100*nev/N:.1f}%')
fu=M['exit']-M['entry']
print(f'follow-up (exit-entry): mediana={fu.median():.2f}a | pessoa-anos totais={fu.sum():.0f}')

# ---------- TABLE 1 ----------
CONT=['ageonset','duration_yrs','BMI','updrs1_score','updrs2_score','updrs3_score',
      'comp_axial','comp_tremor','comp_marcha_pi','comp_bradicinesia','td_pigd_ratio',
      'moca','stai','upsit','datscan_putamen_min','datscan_putamen_asym','abeta','ptau','resp']
CAT={'SEX (feminino=0)':('SEX',0),'is_pigd (PIGD=1)':('is_pigd',1),'LRRK2+':('lrrk2_carrier',1),'GBA+':('gba_carrier',1)}
g0=M[M.event==0]; g1=M[M.event==1]
rows=[]
for c in CONT:
    if c not in M.columns: continue
    s,s0,s1=pd.to_numeric(M[c],errors='coerce'),pd.to_numeric(g0[c],errors='coerce'),pd.to_numeric(g1[c],errors='coerce')
    try: p=stats.mannwhitneyu(s0.dropna(),s1.dropna()).pvalue
    except: p=np.nan
    def mq(x): return f'{x.median():.1f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}]'
    rows.append({'Variável':c,'Geral (N=%d)'%N:mq(s),'Sem evento (n=%d)'%len(g0):mq(s0),
                 'Com evento (n=%d)'%len(g1):mq(s1),'% miss':round(100*s.isna().mean(),0),'p':round(p,4)})
for lab,(c,pos) in CAT.items():
    if c not in M.columns: continue
    def npc(d): n=int((pd.to_numeric(d[c],errors='coerce')==pos).sum()); tot=int(pd.to_numeric(d[c],errors='coerce').notna().sum()); return f'{n} ({100*n/max(tot,1):.0f}%)'
    try:
        _m=pd.to_numeric(M[c],errors='coerce'); _ok=_m.notna(); ct=pd.crosstab((_m[_ok]==pos),M.event[_ok]); p=stats.chi2_contingency(ct)[1]
    except: p=np.nan
    rows.append({'Variável':lab,'Geral (N=%d)'%N:npc(M),'Sem evento (n=%d)'%len(g0):npc(g0),
                 'Com evento (n=%d)'%len(g1):npc(g1),'% miss':round(100*pd.to_numeric(M[c],errors='coerce').isna().mean(),0),'p':round(p,4)})
T1=pd.DataFrame(rows); T1.to_csv(TAB+'/Table1.csv',index=False)
print('\n=== TABLE 1 (median[IQR] / n(%); p descritivo) ==='); print(T1.to_string(index=False))

# ---------- INCIDÊNCIA CUMULATIVA (KM left-truncated) ----------
km=KaplanMeierFitter(); km.fit(M['exit'],M['event'],entry=M['entry'])
ci=km.confidence_interval_
inc=[]
for h in [3,5,7,10]:
    s=float(km.predict(h)); 
    # IC via survival_function CI mais próximo
    idx=ci.index[ci.index<=h]; lo,hi=(1-ci.iloc[-1,1],1-ci.iloc[-1,0]) if len(idx)==0 else (1-ci.loc[idx[-1]].iloc[1],1-ci.loc[idx[-1]].iloc[0])
    inc.append({'Horizonte':f'{h}a','Incidência KM (%)':round(100*(1-s),1),'IC95%':f'{100*lo:.1f}–{100*hi:.1f}'})
incdf=pd.DataFrame(inc); incdf.to_csv(TAB+'/incidencia_horizontes.csv',index=False)
print('\n=== INCIDÊNCIA CUMULATIVA (KM, truncamento à esquerda) ==='); print(incdf.to_string(index=False))

# ---------- KM por subgrupo ----------
M['age_lt50']=(pd.to_numeric(M['ageonset'],errors='coerce')<50).astype('Int64')
subs={'Sexo (0=fem,1=masc)':'SEX','Idade início <50':'age_lt50','Subtipo PIGD':'is_pigd'}
fig,ax=plt.subplots(1,3,figsize=(15,4.3)); lr_res={}
for i,(lab,col) in enumerate(subs.items()):
    vals=[v for v in M[col].dropna().unique()]
    grpsurv=[]
    for v in sorted(vals):
        sub=M[M[col]==v]
        if len(sub)<10: continue
        k=KaplanMeierFitter(); k.fit(sub['exit'],sub['event'],entry=sub['entry'],label=f'{col}={int(v)} (n={len(sub)},ev={int(sub.event.sum())})')
        ax[i].plot(k.survival_function_.index,1-k.survival_function_.values.ravel(),label=k._label)
        grpsurv.append(sub)
    if len(grpsurv)==2:
        a,b=grpsurv if False else (grpsurv[0],grpsurv[1])
        lr=logrank_test(a['exit'],b['exit'],a['event'],b['event'],entry_times_A=a['entry'],entry_times_B=b['entry'])
        lr_res[lab]=lr.p_value; ax[i].set_title(f'{lab}\n(log-rank p={lr.p_value:.4f})')
    else: ax[i].set_title(lab)
    ax[i].set_xlim(0,10); ax[i].set_xlabel('anos desde levodopa'); ax[i].set_ylabel('incidência cumulativa'); ax[i].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG+'/fig07_km_subgrupos.png',dpi=130); plt.close()
print('\nlog-rank por subgrupo:',{k:round(v,4) for k,v in lr_res.items()})

# ---------- incidência por DEFINIÇÃO de desfecho (sensibilidades já existem nos flags? recomputo rápido) ----------
print('\n(Incidência por definições de sensibilidade: ver tab02/horizon; primária consolidada acima.)')
with open(PROC+'/results_step07.pkl','wb') as f:
    pickle.dump({'N':N,'events':nev,'person_years':float(fu.sum()),'incidence':incdf,'logrank':lr_res},f)
print('\nETAPA 3 COMPLETA — Table1.csv, incidencia_horizontes.csv, fig07_km_subgrupos.png')
