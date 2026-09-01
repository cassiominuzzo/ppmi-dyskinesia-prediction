# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""
12_validation.py — ETAPA 7: validação do modelo clínico L1.
(A) C-index apparent + correção de otimismo por bootstrap (Harrell/Steyerberg, B=200);
(B) validação interna-externa leave-one-site-out (C-index pooled out-of-site);
(C) calibração em 5a (quintis: observado-KM vs predito out-of-site) + slope;
(D) decision curve analysis em 5a (net benefit). Origem t=0.
"""
from config import PROJECT_ROOT
import os, glob, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.metrics import concordance_index_censored
_c=[PROJECT_ROOT]; BASE=_c[0]
PROC=BASE+'/Projeto_LID_v2/02_Dados_Processados'; SRC=BASE+'/02_Dados_Processados'
FIG=BASE+'/Projeto_LID_v2/04_Resultados/Figuras'; TAB=BASE+'/Projeto_LID_v2/04_Resultados/Tabelas'
SEED=42; rng=np.random.default_rng(SEED)
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
M=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))
for c in M.columns:
    if c!='PATNO': M[c]=pd.to_numeric(M[c],errors='coerce')
site=ip(pd.read_parquet(SRC+'/curated_cache.parquet'))[['PATNO','SITE']].groupby('PATNO').first().reset_index()
M=M.merge(site,on='PATNO',how='left')
L1=['ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai']
D=M[['exit','event','SITE']+L1].dropna(); D=D[D['exit']>0].reset_index(drop=True)
print(f'coorte validação: n={len(D)}, eventos={int(D.event.sum())}, sites={D.SITE.nunique()}')
def cidx(df,risk): return concordance_index_censored(df.event.astype(bool).values, df['exit'].values, risk)[0]

# ---- A. apparent + otimismo bootstrap ----
full=CoxPHFitter(penalizer=0.01).fit(D[['exit','event']+L1],'exit','event')
app=cidx(D, full.predict_partial_hazard(D).values.ravel())
B=100; opt=[]; slopes=[]
for b in range(B):
    bs=D.sample(len(D),replace=True,random_state=b)
    try:
        mb=CoxPHFitter(penalizer=0.01).fit(bs[['exit','event']+L1],'exit','event')
        c_boot=cidx(bs, mb.predict_partial_hazard(bs).values.ravel())
        c_orig=cidx(D, mb.predict_partial_hazard(D).values.ravel())
        opt.append(c_boot-c_orig)
        lp=mb.predict_log_partial_hazard(D); sl=CoxPHFitter().fit(pd.DataFrame({'exit':D.exit.values,'event':D.event.values,'lp':lp.values}),'exit','event').params_['lp']
        slopes.append(sl)
    except Exception: pass
optimism=np.mean(opt); cc=app-optimism
print(f'\n=== A. C-index ===\n  apparent={app:.3f} | otimismo={optimism:.3f} | CORRIGIDO={cc:.3f} | calib-slope(boot)={np.mean(slopes):.3f}')

# ---- B/C/D. leave-one-site-out: predições out-of-site ----
D['risk_oos']=np.nan; D['r5_oos']=np.nan
for s in D.SITE.dropna().unique():
    tr=D[D.SITE!=s]; te=D[D.SITE==s]
    if te.empty or tr.event.sum()<5: continue
    m=CoxPHFitter(penalizer=0.01).fit(tr[['exit','event']+L1],'exit','event')
    D.loc[te.index,'risk_oos']=m.predict_partial_hazard(te[L1]).values.ravel()
    sf=m.predict_survival_function(te[L1],times=[5.0]); D.loc[te.index,'r5_oos']=1-sf.iloc[0].values
oos=D.dropna(subset=['risk_oos'])
c_loso=cidx(oos, oos['risk_oos'].values)
print(f'\n=== B. Leave-one-site-out (interna-externa) ===\n  C-index pooled out-of-site = {c_loso:.3f} (n={len(oos)})')

# ---- C. calibração 5a (quintis observado-KM vs predito out-of-site) ----
cc5=oos.dropna(subset=['r5_oos']).copy(); cc5['q']=pd.qcut(cc5['r5_oos'],5,labels=False,duplicates='drop')
cal=[]
for q,g in cc5.groupby('q'):
    k=KaplanMeierFitter().fit(g['exit'],g['event']); obs=1-float(k.predict(5)); cal.append({'quintil':int(q)+1,'predito':round(g['r5_oos'].mean(),3),'observado_KM':round(obs,3),'n':len(g)})
caldf=pd.DataFrame(cal); print('\n=== C. Calibração 5a (out-of-site) ===\n'+caldf.to_string(index=False))
fig,ax=plt.subplots(figsize=(5.2,5.2)); ax.plot([0,.6],[0,.6],'--',color='gray')
ax.plot(caldf['predito'],caldf['observado_KM'],'o-',color='#4C72B0'); ax.set_xlabel('predito 5a (out-of-site)'); ax.set_ylabel('observado (KM) 5a'); ax.set_title('Calibração em 5 anos'); plt.tight_layout(); plt.savefig(FIG+'/fig12_calibracao.png',dpi=130); plt.close()

# ---- D. DCA 5a (net benefit, KM-based, out-of-site) ----
ov=1-float(KaplanMeierFitter().fit(oos['exit'],oos['event']).predict(5))
ths=np.arange(0.05,0.45,0.025); nb_m=[]; nb_all=[]
for p in ths:
    hi=oos[oos['r5_oos']>=p]
    if len(hi)>10:
        er=1-float(KaplanMeierFitter().fit(hi['exit'],hi['event']).predict(5)); fr=len(hi)/len(oos)
        nb_m.append(er*fr-(1-er)*fr*(p/(1-p)))
    else: nb_m.append(np.nan)
    nb_all.append(ov-(1-ov)*(p/(1-p)))
fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.plot(ths,nb_m,'-o',label='Modelo L1',color='#4C72B0'); ax.plot(ths,nb_all,'--',label='Tratar todos',color='#C44E52'); ax.axhline(0,color='gray',label='Tratar nenhum')
ax.set_xlabel('limiar de probabilidade'); ax.set_ylabel('net benefit'); ax.set_title('Decision Curve Analysis (5 anos)'); ax.legend(fontsize=8); ax.set_ylim(-0.02,max(0.05,np.nanmax(nb_m)*1.2)); plt.tight_layout(); plt.savefig(FIG+'/fig12_dca.png',dpi=130); plt.close()

res={'C_apparent':round(app,3),'C_optimism':round(optimism,3),'C_corrected':round(cc,3),
     'calib_slope_boot':round(np.mean(slopes),3),'C_LOSO':round(c_loso,3),'calibration':caldf,'overall_5y':round(ov,3)}
pd.DataFrame([{'metrica':'C apparent','valor':res['C_apparent']},{'metrica':'C otimismo','valor':res['C_optimism']},
  {'metrica':'C CORRIGIDO (bootstrap)','valor':res['C_corrected']},{'metrica':'C leave-one-site-out','valor':res['C_LOSO']},
  {'metrica':'calib slope (bootstrap)','valor':res['calib_slope_boot']}]).to_csv(TAB+'/tab12_validacao.csv',index=False)
caldf.to_csv(TAB+'/tab12_calibracao.csv',index=False)
with open(PROC+'/results_step12.pkl','wb') as f: pickle.dump(res,f)
print('\nETAPA 7 COMPLETA — tab12_validacao.csv, fig12_calibracao.png, fig12_dca.png')
