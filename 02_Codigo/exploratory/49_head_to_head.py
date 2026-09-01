# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""49 — HEAD-TO-HEAD vs modelos publicados (auditável). Re-ajusta os CONJUNTOS DE PREDITORES de cada
modelo publicado nos NOSSOS dados (desfecho estrito 4.1>=2 OU 4.2>=2, relógio da levodopa, MESMA
subamostra), e compara o C corrigido por otimismo com o Total-6. Testa a ESCOLHA DE VARIÁVEIS deles.
Preditores de tratamento (LEDD) são exposição pós-basal -> comparação é dos preditores BASAIS."""
from config import PROJECT_ROOT
import glob, warnings, numpy as np, pandas as pd; warnings.filterwarnings('ignore')
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
B=PROJECT_ROOT; PR=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
M=ip(M)
d=M[M['exit']>0].copy()
# conjuntos de preditores (parte BASAL de cada modelo publicado)
MODELS={
 'Total-6 (nosso)':['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ'],
 'Olanow/STRIDE-PD':['ageonset','SEX','BMI'],                          # jovem, feminino, baixo peso
 'Eusebi 2018':['SEX','pigd','stai'],                                  # feminino, PIGD, ansiedade
 'Zhao 2025':['ageonset','td_pigd_ratio','updrs3_score'],             # jovem, acinético-rígido
 'Chen 2021 (clínico)':['ageonset','duration_yrs'],                    # jovem, duração
 'Santos-Lobato 2020':['ageonset','duration_yrs','updrs2_score'],     # onset, duração, sintoma inicial
}
allv=sorted(set(v for s in MODELS.values() for v in s))
dd=d.dropna(subset=allv).reset_index(drop=True)             # MESMA subamostra p/ todos
print(f'Subamostra idêntica (todos os preditores presentes): n={len(dd)}, eventos={int(dd.event.sum())}')
y=Surv.from_arrays(dd['event'].astype(bool),dd['exit'].values); idx=np.arange(len(dd))
def optC(cols,Bb=60):
    def fp(tr,pr):
        sc=StandardScaler().fit(dd.iloc[tr][cols]); m=CoxPHSurvivalAnalysis(alpha=1.0).fit(sc.transform(dd.iloc[tr][cols]),Surv.from_arrays(dd['event'].astype(bool).values[tr],dd['exit'].values[tr])); return m.predict(sc.transform(dd.iloc[pr][cols]))
    ca=concordance_index_censored(dd['event'].astype(bool),dd['exit'].values,fp(idx,idx))[0]; opt=[]
    for b in range(Bb):
        bs=np.random.choice(idx,len(dd),replace=True)
        cb=concordance_index_censored(dd['event'].astype(bool).values[bs],dd['exit'].values[bs],fp(bs,bs))[0]
        co=concordance_index_censored(dd['event'].astype(bool),dd['exit'].values,fp(bs,idx))[0]; opt.append(cb-co)
    return ca,ca-np.mean(opt)
print('\n=== HEAD-TO-HEAD (C corrigido por otimismo, mesma subamostra, desfecho estrito) ===')
res={}
for n,cols in MODELS.items():
    ca,cc=optC(cols); res[n]=(len(cols),ca,cc); 
base=res['Total-6 (nosso)'][2]
rows=[]
for n,(k,ca,cc) in res.items():
    print(f'  {n:22s} ({k} var) C_apar={ca:.3f} C_corr={cc:.3f} ΔC_vs_Total6={cc-base:+.3f}')
    rows.append({'modelo':n,'n_var':k,'C_aparente':round(ca,3),'C_corrigido':round(cc,3),'dC_vs_Total6':round(cc-base,3)})
pd.DataFrame(rows).to_csv(TAB+'/tab49_head_to_head.csv',index=False)
print(f'\nTotal-6 C corrigido = {base:.3f}. Ranking:')
for n,(k,ca,cc) in sorted(res.items(),key=lambda x:-x[1][2]): print(f'   {cc:.3f}  {n}')
