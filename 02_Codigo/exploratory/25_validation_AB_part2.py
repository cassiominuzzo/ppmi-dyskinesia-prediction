# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""25 — Validação parte 2: C caso-completo, LOSO (leave-one-site-out), calibração (slope +
observado vs predito em 3a/5a), e decision curve analysis (DCA) em 3a. Modelos A(8) e B(9)."""
from config import PROJECT_ROOT
import glob, warnings, pickle, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
np.random.seed(42)
B=PROJECT_ROOT
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
df=pd.read_parquet(PROC+'/val_dataset.parquet')
A=['updrs2_score','ageonset','SEX','MSEADLG','NP1FATG','comp_bradicinesia','BMI','td_pigd_ratio']; Bv=A+['NP2FREZ']
def cidx(m,d,c): return concordance_index(d['exit'],-m.predict_partial_hazard(d[c]).values,d['event'])
def imp(d,cols):
    d=d.copy()
    for v in cols: d[v]=pd.to_numeric(d[v],errors='coerce').fillna(df[v].median())
    return d
print('=== (1) C CASO-COMPLETO (sem imputação) ===')
for name,cols in [('A_8var',A),('B_9var',Bv)]:
    cc=df[['exit','event']+cols].dropna()
    m=CoxPHFitter(penalizer=0.1).fit(cc,'exit','event')
    print(f'  {name}: n_completo={len(cc)} eventos={int(cc.event.sum())} | C={cidx(m,cc,cols):.3f}')
print('\n=== (2) LEAVE-ONE-SITE-OUT (pooled C) ===')
for name,cols in [('A_8var',A),('B_9var',Bv)]:
    d=imp(df,cols).dropna(subset=cols+['SITE','exit','event']); pred=np.full(len(d),np.nan); d=d.reset_index(drop=True)
    persite=[]
    for s in d['SITE'].unique():
        tr=d[d.SITE!=s]; te=d[d.SITE==s]
        if len(te)<5 or tr.event.sum()<10: continue
        try:
            m=CoxPHFitter(penalizer=0.1).fit(tr[['exit','event']+cols],'exit','event')
            r=-m.predict_partial_hazard(te[cols]).values; pred[te.index]=r
            if te.event.sum()>=8 and len(te)>=20: persite.append(concordance_index(te['exit'],r,te['event']))
        except Exception: pass
    ok=~np.isnan(pred); pooled=concordance_index(d['exit'][ok],pred[ok],d['event'][ok])
    print(f'  {name}: C pooled LOSO={pooled:.3f} | C mediano por-site={np.median(persite):.3f} (n_sites={len(persite)})')
print('\n=== (3) CALIBRAÇÃO (slope + obs vs pred) ===')
for name,cols in [('A_8var',A),('B_9var',Bv)]:
    d=imp(df,cols).dropna(subset=cols+['exit','event'])
    m=CoxPHFitter(penalizer=0.1).fit(d[['exit','event']+cols],'exit','event')
    lp=m.predict_log_partial_hazard(d[cols])
    cal=CoxPHFitter().fit(pd.DataFrame({'exit':d.exit.values,'event':d.event.values,'lp':lp.values}),'exit','event')
    slope=cal.summary.loc['lp','coef']
    print(f'  {name}: calibration slope={slope:.3f} (ideal=1.0)')
    sf=m.predict_survival_function(d[cols])
    for h in [3,5]:
        if (d.exit>=h).sum()<30: continue
        t=sf.index[np.argmin(np.abs(sf.index-h))]; risk=1-sf.loc[t].values
        d2=d.copy(); d2['risk']=risk; d2['q']=pd.qcut(d2['risk'],4,labels=False,duplicates='drop')
        obs,pre=[],[]
        for q in sorted(d2['q'].dropna().unique()):
            g=d2[d2.q==q]; km=KaplanMeierFitter().fit(g.exit,g.event)
            obs.append(1-km.predict(h)); pre.append(g['risk'].mean())
        print(f'    {h}a: pred quartis={[round(x,2) for x in pre]} | obs(KM)={[round(x,2) for x in obs]}')
print('\n=== (4) DECISION CURVE (net benefit) em 3a ===')
H=3; out={}
for name,cols in [('A_8var',A),('B_9var',Bv)]:
    d=imp(df,cols).dropna(subset=cols+['exit','event'])
    m=CoxPHFitter(penalizer=0.1).fit(d[['exit','event']+cols],'exit','event')
    sf=m.predict_survival_function(d[cols]); t=sf.index[np.argmin(np.abs(sf.index-H))]
    d=d.copy(); d['risk']=1-sf.loc[t].values; out[name]=d
N=len(out['A_8var']); kmall=KaplanMeierFitter().fit(out['A_8var'].exit,out['A_8var'].event); ev_all=1-kmall.predict(H)
print(f'  (prevalência de evento em {H}a ≈ {ev_all:.2f}; N={N})')
print(f'  {"pt":>5} {"NB_trata_todos":>14} {"NB_A":>8} {"NB_B":>8}')
for pt in [0.05,0.10,0.15,0.20,0.30]:
    nb_all=ev_all-(1-ev_all)*(pt/(1-pt))
    nbs={}
    for name in ['A_8var','B_9var']:
        d=out[name]; hi=d[d.risk>=pt]
        if len(hi)<5: nbs[name]=np.nan; continue
        km=KaplanMeierFitter().fit(hi.exit,hi.event); er=1-km.predict(H)
        nbs[name]=er*len(hi)/N-(1-er)*len(hi)/N*(pt/(1-pt))
    print(f'  {pt:>5.2f} {nb_all:>14.3f} {nbs["A_8var"]:>8.3f} {nbs["B_9var"]:>8.3f}')
print('\nFIM validação parte 2.')
