# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""
11_competitiveness.py — ETAPA 6 (A-D). Benchmark vs publicados; responsividade (landmark);
genética dedicada (coorte enriquecida); estratos de risco. Origem t=0; C-index apparent (otimismo
corrigido na Etapa 7). Saídas: tabelas + figuras + QC.
"""
from config import PROJECT_ROOT
import os, glob, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
_c=[PROJECT_ROOT]; BASE=_c[0]
PROC=BASE+'/Projeto_LID_v2/02_Dados_Processados'; FIG=BASE+'/Projeto_LID_v2/04_Resultados/Figuras'; TAB=BASE+'/Projeto_LID_v2/04_Resultados/Tabelas'
SEED=42; np.random.seed(SEED)
M=pd.read_parquet(PROC+'/master_baseline.parquet')
for c in M.columns:
    if c!='PATNO': M[c]=pd.to_numeric(M[c],errors='coerce')
M=M[M['exit']>0].copy()
L1=['ageonset','SEX','BMI','duration_yrs','updrs2_score','pigd','comp_bradicinesia','upsit','stai']

# ============ A. BENCHMARK vs PUBLICADOS (mesma coorte) ============
print('=== A. BENCHMARK vs MODELOS PUBLICADOS ===')
M['aao_le50']=(M['ageonset']<=50).astype(float)
SETS={
 'Martinez-C (AAO<=50 + ansiedade)':['aao_le50','stai'],
 'Loo (axial/freezing/rigidez/tremor/peso/AAO)':['comp_marcha_pi','comp_bradicinesia','comp_tremor','BMI','ageonset'],
 'Nosso L1 (clinico)':L1,
}
allv=sorted(set(sum(SETS.values(),[])))
common=M[['exit','event']+allv].dropna()
print(f'coorte comum p/ benchmark: n={len(common)}, eventos={int(common.event.sum())}')
brows=[]
for name,cols in SETS.items():
    cph=CoxPHFitter(penalizer=0.02).fit(common[['exit','event']+cols],'exit','event')
    brows.append({'Modelo':name,'k':len(cols),'C-index':round(cph.concordance_index_,3)})
    print(f"  {name:46s} k={len(cols)} C={cph.concordance_index_:.3f}")
bench=pd.DataFrame(brows); bench.to_csv(TAB+'/tab11_benchmark.csv',index=False)

# ============ B. RESPONSIVIDADE À LEVODOPA (landmark) ============
print('\n=== B. RESPONSIVIDADE À LEVODOPA (desenho landmark) ===')
R=M.dropna(subset=['resp','resp_challenge_day']).copy()
R['cday_y']=R['resp_challenge_day']/365.25
R=R[R['exit']>R['cday_y']].copy()                  # livre de evento até o desafio
R['t_lm']=R['exit']-R['cday_y']
R['resp_gt50']=(R['resp']>50).astype(int)
print(f'coorte landmark: n={len(R)}, eventos pós-desafio={int(R.event.sum())}, mediana desafio={R.cday_y.median():.2f}a')
brows2=[]
for v,desc in [('resp','responsividade (% por DP)'),('resp_gt50','altamente responsivo >50%')]:
    d=R[['t_lm','event',v]].dropna(); d=d[d['t_lm']>0]
    if v=='resp': d[v]=(d[v]-d[v].mean())/d[v].std()
    cph=CoxPHFitter().fit(d,'t_lm','event'); hr=cph.summary.loc[v]
    brows2.append({'preditor':desc,'n':len(d),'eventos':int(d.event.sum()),'HR':round(hr['exp(coef)'],3),
                   'IC inf':round(hr['exp(coef) lower 95%'],3),'IC sup':round(hr['exp(coef) upper 95%'],3),'p':float(hr['p'])})
    print(f"  {desc:30s} HR={hr['exp(coef)']:.3f} ({hr['exp(coef) lower 95%']:.2f}-{hr['exp(coef) upper 95%']:.2f}) p={hr['p']:.4g}")
resp_tab=pd.DataFrame(brows2); resp_tab.to_csv(TAB+'/tab11_responsividade.csv',index=False)

# ============ C. GENÉTICA DEDICADA (coorte testada/enriquecida) ============
print('\n=== C. GENÉTICA (coorte testada) ===')
G=M[M['has_genetic']==1].copy()
print(f'coorte genotipada: n={len(G)}, eventos={int(G.event.sum())}, LRRK2+={int(G.lrrk2_carrier.sum())}, GBA+={int(G.gba_carrier.sum())}')
grows=[]
for v in ['lrrk2_carrier','gba_carrier']:
    # bruto
    d=G[['exit','event',v]].dropna(); cph=CoxPHFitter().fit(d,'exit','event'); hr=cph.summary.loc[v]
    # ajustado por idade de início e sexo
    da=G[['exit','event',v,'ageonset','SEX']].dropna(); cpha=CoxPHFitter().fit(da,'exit','event'); hra=cpha.summary.loc[v]
    grows.append({'variante':v,'n':len(d),'HR_bruto':round(hr['exp(coef)'],3),'p_bruto':float(hr['p']),
                  'HR_ajust(idade,sexo)':round(hra['exp(coef)'],3),'p_ajust':float(hra['p'])})
    print(f"  {v:14s} HR_bruto={hr['exp(coef)']:.2f} (p={hr['p']:.3g}) | ajustado={hra['exp(coef)']:.2f} (p={hra['p']:.3g})")
gen_tab=pd.DataFrame(grows); gen_tab.to_csv(TAB+'/tab11_genetica.csv',index=False)
# KM por status de portador
fig,ax=plt.subplots(figsize=(7,5))
for v,lab,col in [(G.lrrk2_carrier==1,'LRRK2+','#C44E52'),(G.gba_carrier==1,'GBA+','#DD8452'),((G.lrrk2_carrier==0)&(G.gba_carrier==0),'Não-portador','#4C72B0')]:
    s=G[v]
    if len(s)>10:
        k=KaplanMeierFitter().fit(s['exit'],s['event'],label=f'{lab} (n={len(s)},ev={int(s.event.sum())})')
        ax.plot(k.survival_function_.index,1-k.survival_function_.values.ravel(),label=k._label,color=col)
ax.set_xlim(0,10); ax.set_xlabel('anos desde levodopa'); ax.set_ylabel('incidência cumulativa'); ax.set_title('Discinesia por status genético (coorte testada)'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG+'/fig11_genetica_km.png',dpi=130); plt.close()

# ============ D. ESTRATOS DE RISCO (L1) ============
print('\n=== D. ESTRATOS DE RISCO (tercis do escore L1) ===')
d=M[['exit','event']+L1].dropna(); cph=CoxPHFitter(penalizer=0.01).fit(d,'exit','event')
d['lp']=cph.predict_log_partial_hazard(d); d['grupo']=pd.qcut(d['lp'],3,labels=['Baixo','Intermediário','Alto'])
fig,ax=plt.subplots(figsize=(7,5)); inc5={}
for g,col in [('Baixo','#4C72B0'),('Intermediário','#DD8452'),('Alto','#C44E52')]:
    s=d[d.grupo==g]; k=KaplanMeierFitter().fit(s['exit'],s['event'],label=g)
    ax.plot(k.survival_function_.index,1-k.survival_function_.values.ravel(),label=f'{g} (n={len(s)},ev={int(s.event.sum())})',color=col)
    inc5[g]=round(100*(1-float(k.predict(5))),1)
lr=logrank_test(d[d.grupo=='Baixo']['exit'],d[d.grupo=='Alto']['exit'],d[d.grupo=='Baixo']['event'],d[d.grupo=='Alto']['event'])
ax.set_xlim(0,10); ax.set_xlabel('anos desde levodopa'); ax.set_ylabel('incidência cumulativa'); ax.set_title(f'Estratos de risco L1 (log-rank baixo vs alto p={lr.p_value:.2g})'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG+'/fig11_estratos.png',dpi=130); plt.close()
print(f'  incidência 5a por grupo: {inc5} | log-rank baixo-vs-alto p={lr.p_value:.3g}')
with open(PROC+'/results_step11.pkl','wb') as f: pickle.dump({'bench':bench,'resp':resp_tab,'gen':gen_tab,'inc5_strata':inc5,'logrank_strata':lr.p_value},f)
print('\nETAPA 6 (A-D) COMPLETA — tab11_benchmark/responsividade/genetica.csv, fig11_*')
