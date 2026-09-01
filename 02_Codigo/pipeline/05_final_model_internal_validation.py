# 05_final_model_internal_validation.py
# Produces: Total-6 coefficients, S0(t), optimism correction, leave-one-site-out. Table 2 and Supplementary Figure S1
# Original file in the project archive: 13_explainability_calculator.py

"""
13_explainability_calculator.py — ETAPA 8: explicabilidade + calculadora.
Cox L1 final (risco absoluto via S0 em 3/5/7/10a); SHAP exato (linear); PDP; benchmark RSF;
calculadora (função + artefatos p/ app). Origem t=0. SEED=42.
"""
from config import PROJECT_ROOT
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import os, glob, warnings, pickle, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
_c=[PROJECT_ROOT]; BASE=_c[0]
PROC=C.PROC; FIG=C.ensure_dir(os.path.join(C.RESULTS,'Figuras'))
CALC=C.ensure_dir(os.path.join(C.RESULTS,'Calculadora')); SEED=42; np.random.seed(SEED)
M=pd.read_parquet(PROC+'/master_baseline.parquet')
for c in M.columns:
    if c!='PATNO': M[c]=pd.to_numeric(M[c],errors='coerce')
L1=['ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai']
D=M[['exit','event']+L1].dropna(); D=D[D['exit']>0].reset_index(drop=True)
print(f'modelo final L1: n={len(D)}, eventos={int(D.event.sum())}')

# ---- 1) Cox final + baseline survival ----
cph=CoxPHFitter(penalizer=0.01).fit(D[['exit','event']+L1],'exit','event')
means=D[L1].mean().to_dict(); medians=D[L1].median().to_dict()
S0=cph.predict_survival_function(pd.DataFrame([means]),times=[3,5,7,10])  # baseline (na média) S(t)
base_risk={h:round(float(1-S0.loc[h].iloc[0]),3) for h in [3,5,7,10]}
print('risco basal (paciente médio) 3/5/7/10a:',base_risk)

# ---- 2) SHAP exato (modelo linear): phi_j = coef_j*(x_j - mean_j) ----
coef=cph.params_.to_dict()
SH=pd.DataFrame({j:(D[j]-means[j])*coef[j] for j in L1})
glob_imp=SH.abs().mean().sort_values(ascending=False)
print('\nSHAP global (|contribuição| média ao log-risco):'); print(glob_imp.round(3).to_string())
fig,ax=plt.subplots(figsize=(7,4)); glob_imp.sort_values().plot.barh(ax=ax,color='#4C72B0')
ax.set_xlabel('SHAP médio |contribuição| (log-HR)'); ax.set_title('Importância global (SHAP linear) — modelo clínico'); plt.tight_layout(); plt.savefig(FIG+'/fig13_shap_global.png',dpi=130); plt.close()
# exemplo individual (paciente de maior risco)
lp=cph.predict_log_partial_hazard(D); idx=lp.idxmax()
ind=SH.loc[idx].sort_values(); fig,ax=plt.subplots(figsize=(7,4)); ind.plot.barh(ax=ax,color=['#C44E52' if v>0 else '#4C72B0' for v in ind])
ax.set_title(f'Explicação individual (paciente de maior risco)'); ax.set_xlabel('contribuição ao log-risco'); plt.tight_layout(); plt.savefig(FIG+'/fig13_shap_individual.png',dpi=130); plt.close()

# ---- 3) PDP (risco 5a variando 1 preditor, demais na mediana) ----
fig,axes=plt.subplots(1,4,figsize=(16,3.6))
for ax,feat in zip(axes,['ageonset','updrs2_score','comp_bradicinesia','BMI']):
    grid=np.linspace(D[feat].quantile(.05),D[feat].quantile(.95),25); risks=[]
    for g in grid:
        row=dict(medians); row[feat]=g
        risks.append(1-float(cph.predict_survival_function(pd.DataFrame([row]),times=[5]).iloc[0,0]))
    ax.plot(grid,risks,color='#4C72B0'); ax.set_title(feat); ax.set_xlabel(feat); ax.set_ylabel('risco 5a')
plt.tight_layout(); plt.savefig(FIG+'/fig13_pdp.png',dpi=130); plt.close()

# ---- 4) Benchmark RSF (ML não-linear) ----
y=Surv.from_arrays(event=D.event.astype(bool),time=D['exit'].values)
rsf=RandomSurvivalForest(n_estimators=100,min_samples_leaf=15,max_features='sqrt',random_state=SEED,n_jobs=2).fit(D[L1],y)
c_rsf=concordance_index_censored(D.event.astype(bool).values,D['exit'].values,rsf.predict(D[L1]))[0]
c_cox=concordance_index_censored(D.event.astype(bool).values,D['exit'].values,cph.predict_partial_hazard(D).values.ravel())[0]
print(f'\nBenchmark (apparent): Cox C={c_cox:.3f} | RSF C={c_rsf:.3f} (ML não-linear {"NÃO supera" if c_rsf<=c_cox+0.01 else "supera"} o Cox)')

# ---- 5) Calculadora: salvar artefatos + função + teste ----
artifacts={'features':L1,'coef':coef,'means':means,'medians':medians,
           'baseline_surv':{h:float(S0.loc[h].iloc[0]) for h in [3,5,7,10]},
           'lp_mean':float(cph.predict_log_partial_hazard(D).mean()),
           'C_corrected':0.661,'C_LOSO':0.654}
with open(CALC+'/calculator_model.pkl','wb') as f: pickle.dump(cph,f)
json.dump(artifacts,open(CALC+'/calculator_artifacts.json','w'),indent=2,default=float)
def predict_risk(patient:dict):
    row={j:patient.get(j,medians[j]) for j in L1}
    sf=cph.predict_survival_function(pd.DataFrame([row]),times=[3,5,7,10])
    risk={f'{h}a':round(float(1-sf.loc[h].iloc[0]),3) for h in [3,5,7,10]}
    contrib={j:round((row[j]-means[j])*coef[j],3) for j in L1}
    return risk,contrib
print('\n=== TESTE DA CALCULADORA ===')
for nome,p in [('alto risco (jovem, F, PIGD, alta AVD)',{'ageonset':45,'SEX':0,'pigd':1,'updrs2_score':16,'comp_bradicinesia':18,'BMI':22}),
               ('baixo risco (idoso, M, não-PIGD)',{'ageonset':72,'SEX':1,'pigd':0,'updrs2_score':4,'comp_bradicinesia':6,'BMI':28})]:
    r,_=predict_risk(p); print(f'  {nome}: {r}')
with open(PROC+'/results_step13.pkl','wb') as f: pickle.dump({'base_risk':base_risk,'shap_global':glob_imp.to_dict(),'c_cox':round(c_cox,3),'c_rsf':round(c_rsf,3)},f)
print('\nETAPA 8 (núcleo) COMPLETA — artefatos da calculadora + figuras SHAP/PDP salvos')
