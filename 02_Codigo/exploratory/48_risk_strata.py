# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""48 — ESTRATOS DE RISCO + enriquecimento de ensaio (auditável). Divide o risco predito (5a) do
Total-6 em baixo/intermediário/alto; reporta incidência observada e HR por estrato; calcula o
enriquecimento de um ensaio clínico ao recrutar só o estrato alto (redução do tamanho amostral)."""
from config import PROJECT_ROOT
import glob, warnings, pickle, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import norm
B=PROJECT_ROOT; PR=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
S=pickle.load(open(PR+'/sel_data.pkl','rb')); d=S['cc'].reset_index(drop=True)
T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
d=d[['exit','event']+T6].dropna(); d=d[d['exit']>0].reset_index(drop=True)
cph=CoxPHFitter(penalizer=0.05).fit(d,'exit','event')
sf=cph.predict_survival_function(d,times=[5]); d['risco5']=1-sf.iloc[0].values
# estratos por tercis do risco predito
d['estrato']=pd.qcut(d['risco5'],3,labels=['Baixo','Intermediário','Alto'])
print('=== ESTRATOS DE RISCO (tercis do risco predito em 5 anos) ===')
rows=[]
for e in ['Baixo','Intermediário','Alto']:
    g=d[d.estrato==e]; km=KaplanMeierFitter().fit(g.exit,g.event); inc5=1-float(km.predict(5))
    rows.append({'estrato':e,'n':len(g),'risco_predito_medio':round(g.risco5.mean(),3),'incidencia_obs_5a':round(inc5,3),'eventos':int(g.event.sum())})
    print(f'  {e:14s} n={len(g)} | risco predito médio={g.risco5.mean():.1%} | incidência observada 5a={inc5:.1%}')
# HR alto vs baixo
d['alto']=(d.estrato=='Alto').astype(int); m=CoxPHFitter().fit(d[['exit','event','alto']],'exit','event'); h=m.summary.loc['alto']
print(f'\n  HR (Alto vs resto)={h["exp(coef)"]:.2f} (IC {h["exp(coef) lower 95%"]:.2f}-{h["exp(coef) upper 95%"]:.2f}) p={h["p"]:.4g}')
# ENRIQUECIMENTO DE ENSAIO: detectar RRR de 30% na incidência de 5a
def nperarm(p1,rrr=0.30,alpha=0.05,power=0.8):
    p2=p1*(1-rrr); za=norm.ppf(1-alpha/2); zb=norm.ppf(power)
    return (za+zb)**2*(p1*(1-p1)+p2*(1-p2))/(p1-p2)**2
p_all=1-float(KaplanMeierFitter().fit(d.exit,d.event).predict(5)); p_hi=rows[2]['incidencia_obs_5a']
n_all=nperarm(p_all); n_hi=nperarm(p_hi)
print(f'\n=== ENRIQUECIMENTO DE ENSAIO (detectar 30% de redução relativa em 5a, poder 80%) ===')
print(f'  recrutando TODOS (incid. {p_all:.1%}): {int(np.ceil(n_all))} por braço')
print(f'  recrutando só ALTO risco (incid. {p_hi:.1%}): {int(np.ceil(n_hi))} por braço')
print(f'  => redução de ~{100*(1-n_hi/n_all):.0f}% no tamanho amostral (fator de enriquecimento {n_all/n_hi:.2f}x)')
pd.DataFrame(rows).to_csv(TAB+'/tab48_estratos.csv',index=False)
