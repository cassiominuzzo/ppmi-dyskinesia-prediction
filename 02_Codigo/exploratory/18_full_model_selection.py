# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""
18_full_model_selection.py — Modelo DATA-DRIVEN com todas as variáveis (elastic-net Cox + CV).
Pool curado (leakage-free, 1 variável por construto): demografia, motor/compósitos, não-motor
(Part I, ansiedade, apatia, fadiga, DDS), humor, autonômico, sono, olfato, cognição (memória,
fluência, visuoespacial, velocidade, working memory), impulsividade, imagem, LCR.
EXCLUI vazamento: NP4*, LEDD, estado-ON, pm_* (marcos de progressão = complicações motoras), óbito.
Seleção por elastic-net com alpha escolhido por CV (cv.glmnet style) -> C-index honesto out-of-fold.
"""
from config import PROJECT_ROOT
import glob, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
B=PROJECT_ROOT
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; SRC=B+'/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
SEED=42
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
# desfecho + compósitos (master)
M=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))
COMP=['comp_fala','comp_tremor','comp_marcha_pi','comp_bradicinesia','comp_axial','comp_assimetria_lr','td_pigd_ratio','datscan_putamen_min','datscan_putamen_asym']
mst=M[['PATNO','exit','event']+COMP].copy()
# baseline curated pré-levodopa
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
cur=ip(pd.read_parquet(SRC+'/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz,on='PATNO',how='inner'); cur['dsd']=(cur['visit_date']-cur['tzero_levodopa']).dt.days
base=ip(cur[cur['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index())
# pool CURADO (1 por construto, leakage-free)
CUR=['ageonset','SEX','EDUCYRS','BMI','duration_yrs','handed','fampd','APOE_e4','DOMSIDE',  # demografia/genética
     'updrs2_score','updrs1_score','pigd','NHY','MSEADLG',                                  # motor/função
     'NP1COG','NP1HALL','NP1DPRS','NP1ANXS','NP1APAT','NP1FATG','NP1DDS',                    # não-motor Part I
     'gds','stai','scopa','orthostasis','ess','rem','upsit',                                # humor/autonômico/sono/olfato
     'moca','hvlt_immediaterecall','VLTANIM','bjlot','SDMTOTAL','lns',                       # cognição (1/domínio)
     'quip','urate','CSFSAA','MIA_PUTAMEN_L','MIA_PUTAMEN_R','MIA_CAUDATE_L']                # impulsividade/LCR/imagem
CUR=[c for c in CUR if c in base.columns]
X=base[['PATNO']+CUR].merge(mst,on='PATNO',how='inner')
# imagem: derivar mínimo/assimetria do putâmen e descartar L/R brutos
X['putamen_min']=X[['MIA_PUTAMEN_L','MIA_PUTAMEN_R']].min(axis=1)
X=X.drop(columns=['MIA_PUTAMEN_L','MIA_PUTAMEN_R','datscan_putamen_min'])  # usa o putamen_min recomputado + asym
feats=[c for c in X.columns if c not in ('PATNO','exit','event')]
for c in feats: X[c]=pd.to_numeric(X[c],errors='coerce')
X=X[X['exit']>0].reset_index(drop=True)
# excluir variáveis quase-constantes
feats=[c for c in feats if X[c].nunique()>=3 or set(X[c].dropna().unique())<= {0,1}]
print(f'pool final de candidatas: {len(feats)} | n={len(X)}, eventos={int(X.event.sum())}')
y=Surv.from_arrays(event=X.event.astype(bool).values, time=X['exit'].values)

# grade de alphas a partir de um ajuste completo
imp=SimpleImputer(strategy='median'); sc=StandardScaler()
Xall=sc.fit_transform(imp.fit_transform(X[feats]))
net=CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=100000, fit_baseline_model=False).fit(Xall,y)
alphas=net.alphas_[::max(1,len(net.alphas_)//25)]   # ~25 alphas

# CV 5-fold: risco out-of-fold para cada alpha (cv.glmnet style); imput/scale DENTRO do fold
kf=KFold(5,shuffle=True,random_state=SEED); oof=np.full((len(X),len(alphas)),np.nan)
for tr,te in kf.split(X):
    im=SimpleImputer(strategy='median').fit(X.iloc[tr][feats]); ss=StandardScaler().fit(im.transform(X.iloc[tr][feats]))
    Xtr=ss.transform(im.transform(X.iloc[tr][feats])); Xte=ss.transform(im.transform(X.iloc[te][feats]))
    m=CoxnetSurvivalAnalysis(l1_ratio=0.5,alphas=alphas,alpha_min_ratio=0.01,max_iter=100000).fit(Xtr,Surv.from_arrays(X.iloc[tr].event.astype(bool),X.iloc[tr]['exit'].values))
    for j,a in enumerate(alphas):
        try: oof[te,j]=m.predict(Xte,alpha=a)
        except Exception: pass
cidx=[concordance_index_censored(X.event.astype(bool).values,X['exit'].values,oof[:,j])[0] if not np.isnan(oof[:,j]).all() else np.nan for j in range(len(alphas))]
jbest=int(np.nanargmax(cidx)); abest=alphas[jbest]; Cbest=cidx[jbest]
print(f'\n=== ELASTIC-NET (seleção por CV) ===')
print(f'C-index honesto (out-of-fold 5-fold) = {Cbest:.3f}  (vs L1 hand-picked LOSO 0,654)')

# modelo final no alpha selecionado -> variáveis selecionadas
coef=pd.Series(net.predict  if False else 0, dtype=float)
cf=pd.Series(net.coef_[:, np.argmin(np.abs(net.alphas_-abest))], index=feats)
sel=cf[cf.abs()>1e-6].sort_values(key=lambda s:-s.abs())
print(f'\nvariáveis SELECIONADAS ({len(sel)} de {len(feats)}), por |coef| (HR=exp(coef por DP)):')
out=pd.DataFrame({'variavel':sel.index,'coef':sel.values,'HR_por_DP':np.exp(sel.values).round(3)})
print(out.to_string(index=False))
out.to_csv(TAB+'/tab18_modelo_datadriven.csv',index=False)
pickle.dump({'C_oof':round(Cbest,3),'selected':out,'n_cand':len(feats)},open(PROC+'/results_step18.pkl','wb'))
print('\nETAPA — modelo data-driven COMPLETO (tab18_modelo_datadriven.csv)')
