# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""
10_hierarchical.py — ETAPA 5: modelos hierárquicos L1->L5 + discriminação + NRI/IDI.
L1 clínico pré-especificado (não-colinear; UPDRS-III total trocado por compósitos/subtipo).
Camadas: L1 -> L1+DaTSCAN(L3) -> L1+LCR(L4) -> L5(tudo). Origem t=0. Genética = análise dedicada.
Por camada: HRs, C-index, AUC(t) 3/5/7/10 (sksurv), EPV, PH. NRI/IDI entre camadas (status 5a determinado).
"""
from config import PROJECT_ROOT
import os, glob, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
_c=[PROJECT_ROOT]; BASE=_c[0]
PROC=BASE+'/Projeto_LID_v2/02_Dados_Processados'; TAB=BASE+'/Projeto_LID_v2/04_Resultados/Tabelas'
SEED=42; np.random.seed(SEED)
M=pd.read_parquet(PROC+'/master_baseline.parquet')
for c in M.columns:
    if c!='PATNO': M[c]=pd.to_numeric(M[c],errors='coerce')
M=M[M['exit']>0].copy()

L1=['ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai']
L3add=['datscan_putamen_min','datscan_putamen_asym']; L4add=['tau','ptau']
# --- revisão: VIF do L1 ---
Xl1=M[L1].dropna(); Xs=(Xl1-Xl1.mean())/Xl1.std()
print('=== VIF do L1 (deve ser <5) ===')
vif=max(variance_inflation_factor(Xs.values,i) for i in range(len(L1)))
for i,c in enumerate(L1): print(f'  {c:18s} VIF={variance_inflation_factor(Xs.values,i):.2f}')
print(f'VIF máx L1 = {vif:.2f}', 'OK' if vif<5 else 'ATENÇÃO')

def fit_layer(cols,name):
    d=M[['exit','event']+cols].dropna(); 
    cph=CoxPHFitter(penalizer=0.01).fit(d,'exit','event')  # ridge leve p/ estabilidade
    ev=int(d.event.sum()); epv=ev/len(cols); cidx=cph.concordance_index_
    # AUC(t) sksurv (apparent)
    y=Surv.from_arrays(event=d.event.astype(bool),time=d['exit'].values)
    risk=cph.predict_partial_hazard(d).values.ravel()
    aucs={}
    tmax=d.loc[d.event==1,'exit'].max()
    for h in [3,5,7,10]:
        if h<tmax:
            try: aucs[h]=round(float(cumulative_dynamic_auc(y,y,risk,[h])[0][0]),3)
            except Exception: aucs[h]=None
        else: aucs[h]=None
    return cph,d,{'modelo':name,'n':len(d),'eventos':ev,'k':len(cols),'EPV':round(epv,1),
                  'C-index':round(cidx,3),'AUC_3a':aucs[3],'AUC_5a':aucs[5],'AUC_7a':aucs[7],'AUC_10a':aucs[10]}

print('\n=== AJUSTE DAS CAMADAS ===')
layers={'L1_clinico':L1,'L1+L3_DaTSCAN':L1+L3add,'L1+L4_LCR':L1+L4add,'L5_combinado':L1+L3add+L4add}
fits={}; rows=[]
for name,cols in layers.items():
    cph,d,info=fit_layer(cols,name); fits[name]=(cph,cols); rows.append(info)
    print(f"  {name:16s} n={info['n']:4d} ev={info['eventos']:3d} EPV={info['EPV']:4.1f} C={info['C-index']} AUC5={info['AUC_5a']}")
disc=pd.DataFrame(rows); disc.to_csv(TAB+'/tab10_discriminacao.csv',index=False)

# --- NRI/IDI entre camadas (status 5a determinado: evento<=5 -> 1; seguido>=5 -> 0) ---
def risk5(cph,cols,df):
    sf=cph.predict_survival_function(df[cols],times=[5.0]); return 1-sf.iloc[0].values
def nri_idi(old,new,oldcols,newcols):
    cf=M[['exit','event']+list(set(oldcols)|set(newcols))].dropna()   # mesmo subconjunto p/ C-index justo
    co=CoxPHFitter(penalizer=0.01).fit(cf[['exit','event']+oldcols],'exit','event')
    cn=CoxPHFitter(penalizer=0.01).fit(cf[['exit','event']+newcols],'exit','event')
    c_old,c_new=co.concordance_index_,cn.concordance_index_
    y5=np.where((cf.event==1)&(cf.exit<=5),1,np.where(cf.exit>=5,0,-1))
    ok=y5>=0; common=cf[ok]; y=y5[ok]
    ro=risk5(co,oldcols,common); rn=risk5(cn,newcols,common)
    ev=y==1; ne=y==0
    idi=(rn[ev].mean()-ro[ev].mean())-(rn[ne].mean()-ro[ne].mean())
    up=rn>ro; dn=rn<ro
    nri=((up[ev].mean()-dn[ev].mean())+(dn[ne].mean()-up[ne].mean()))
    return {'n_comp':len(cf),'eventos_5a':int(ev.sum()),'C_L1':round(c_old,3),'C_modelo':round(c_new,3),'dC':round(c_new-c_old,3),'IDI':round(idi,4),'NRI_continuo':round(nri,3)}
print('\n=== GANHO INCREMENTAL (NRI/IDI a 5 anos) ===')
nrows=[]
for name in ['L1+L3_DaTSCAN','L1+L4_LCR','L5_combinado']:
    r=nri_idi(fits['L1_clinico'][0],fits[name][0],fits['L1_clinico'][1],fits[name][1])
    r['transicao']=f'L1 -> {name}'; nrows.append(r); print(f"  L1->{name:16s} n={r['n_comp']:4d} C: {r['C_L1']}->{r['C_modelo']} (dC={r['dC']:+.3f}) IDI={r['IDI']:+.4f} NRI={r['NRI_continuo']:+.3f}")
nri=pd.DataFrame(nrows)[['transicao','n_comp','eventos_5a','C_L1','C_modelo','dC','IDI','NRI_continuo']]; nri.to_csv(TAB+'/tab10_nri_idi.csv',index=False)

# --- HRs do L5 (para inspeção) ---
print('\n=== HRs L5 (combinado) ===')
print(fits['L5_combinado'][0].summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())
with open(PROC+'/results_step10.pkl','wb') as f: pickle.dump({'disc':disc,'nri':nri,'L1':L1,'vif_max':vif},f)
print('\nETAPA 5 COMPLETA — tab10_discriminacao.csv, tab10_nri_idi.csv')
