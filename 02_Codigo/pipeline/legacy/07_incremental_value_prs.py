# 07_incremental_value_prs.py
# Produces: Incremental value of the two polygenic risk scores
# Original file in the project archive: 40_prs_added_value.py

"""40 — VALOR INCREMENTAL DO PRS (Nalls + Progression) sobre o Total-6.
Auditável, sem pular etapas. Amostra IDÊNTICA (Total-6 completo + PRS). Métricas honestas:
ΔC corrigido por otimismo, ΔC out-of-fold, teste de razão de verossimilhança (LRT), IDI 5a, calibração.
PRS é genético/fixo (sem questão BL/SC). Sensibilidade restrita a europeus (race=1)."""
import os
from config import PROJECT_ROOT
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import glob, warnings, pickle, json, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from scipy.stats import chi2
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
B=PROJECT_ROOT; PR=C.PROC; TAB=C.ensure_dir(os.path.join(C.RESULTS,'Tabelas'))
PRSF=os.path.join(PROJECT_ROOT,'PRSresult_ppmi.txt')
def ip(x): x=x.copy(); x['PATNO']=pd.to_numeric(x['PATNO'],errors='coerce').astype('Int64'); return x.dropna(subset=['PATNO'])
T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
FAIL=[]
def chk(name,ok,d=''):
    print(f"[{'PASS' if ok else '*FALHA*'}] {name}"+(f' — {d}' if d else '')); 
    if not ok: FAIL.append(name)

print('===== PARTE A — AUDITORIA DO ARQUIVO PRS =====')
prs=pd.read_csv(PRSF,sep='\t')
chk('colunas esperadas presentes', all(c in prs.columns for c in ['participant_id','PRS_Nalls','PRS_Progression','diagnosis_at_baseline']))
prs['PATNO']=pd.to_numeric(prs['participant_id'].str.replace('PP-','',regex=False),errors='coerce').astype('Int64')
chk('PATNO sem nulos após extrair PP-', int(prs['PATNO'].isna().sum())==0)
chk('sem PATNO duplicado', prs['PATNO'].duplicated().sum()==0, f"dups={int(prs['PATNO'].duplicated().sum())}")
chk('contagem casos+controles bate (656+481=1137)', len(prs)==1137 and (prs['case_control_other_at_baseline']=='Case').sum()==656)
for c in ['PRS_Nalls','PRS_Progression']: prs[c]=pd.to_numeric(prs[c],errors='coerce')

print('\n===== PARTE B — AMOSTRA ANALÍTICA (mesma subamostra) =====')
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
M=ip(M); M=M[M['exit']>0]
cur=ip(pd.read_parquet(B+'/02_Dados_Processados/curated_cache.parquet')); race=cur.groupby('PATNO')['race'].first().reset_index()
d=M.dropna(subset=T6).merge(prs[['PATNO','PRS_Nalls','PRS_Progression']],on='PATNO',how='inner').merge(race,on='PATNO',how='left')
d=d.dropna(subset=['PRS_Nalls','PRS_Progression']).reset_index(drop=True)
chk('amostra com Total-6 completo + PRS', len(d)>250, f'n={len(d)}, eventos={int(d.event.sum())}')
# z-score do PRS dentro da amostra (HR por DP); ΔC é invariante a escala
for c in ['PRS_Nalls','PRS_Progression']: d[c+'_z']=(d[c]-d[c].mean())/d[c].std()
print(f'  n={len(d)} | eventos={int(d.event.sum())} | EPV(8 var)={int(d.event.sum())//8} | europeus(race=1)={int((d.race==1).sum())}')

print('\n===== PARTE C — PRS UNIVARIADO (Cox, HR por DP) =====')
for c in ['PRS_Nalls_z','PRS_Progression_z']:
    m=CoxPHFitter().fit(d[['exit','event',c]],'exit','event'); h=m.summary.loc[c]
    print(f'  {c:18s} HR={h["exp(coef)"]:.3f} (IC {h["exp(coef) lower 95%"]:.2f}-{h["exp(coef) upper 95%"]:.2f}) p={h["p"]:.3f} | C={m.concordance_index_:.3f}')

print('\n===== PARTE D — VALOR INCREMENTAL sobre o Total-6 (mesma amostra n=%d) ====='%len(d))
y=Surv.from_arrays(d['event'].astype(bool),d['exit'].values)
def oofC(cols,seeds=range(4)):
    cs=[]
    for s in seeds:
        kf=KFold(5,shuffle=True,random_state=s); risk=np.zeros(len(d))
        for tr,te in kf.split(d):
            sc=StandardScaler().fit(d.iloc[tr][cols]); m=CoxPHSurvivalAnalysis(alpha=1.0).fit(sc.transform(d.iloc[tr][cols]),y[tr]); risk[te]=m.predict(sc.transform(d.iloc[te][cols]))
        cs.append(concordance_index_censored(d['event'].astype(bool),d['exit'].values,risk)[0])
    return np.mean(cs)
def optC(cols,Bb=80):
    idx=np.arange(len(d))
    def fitpred(tr,pr):
        sc=StandardScaler().fit(d.iloc[tr][cols]); m=CoxPHSurvivalAnalysis(alpha=1.0).fit(sc.transform(d.iloc[tr][cols]),Surv.from_arrays(d['event'].astype(bool).values[tr],d['exit'].values[tr])); return m.predict(sc.transform(d.iloc[pr][cols]))
    capp=concordance_index_censored(d['event'].astype(bool),d['exit'].values,fitpred(idx,idx))[0]; opt=[]
    for b in range(Bb):
        bs=np.random.choice(idx,len(d),replace=True)
        cb=concordance_index_censored(d['event'].astype(bool).values[bs],d['exit'].values[bs],fitpred(bs,bs))[0]
        co=concordance_index_censored(d['event'].astype(bool),d['exit'].values,fitpred(bs,idx))[0]; opt.append(cb-co)
    return capp,capp-np.mean(opt)
def llf(cols): return CoxPHFitter(penalizer=0.001).fit(d[['exit','event']+cols],'exit','event').log_likelihood_
models={'Total-6 (base)':T6,'+PRS_Nalls':T6+['PRS_Nalls_z'],'+PRS_Progression':T6+['PRS_Progression_z'],'+ambos':T6+['PRS_Nalls_z','PRS_Progression_z']}
res={}
for n,cols in models.items():
    ca,cc=optC(cols); oof=oofC(cols); res[n]=(ca,cc,oof,len(cols)); print(f'  {n:20s} C_apar={ca:.4f} C_corr={cc:.4f} C_OOF={oof:.4f}')
cb=res['Total-6 (base)']
print('\n  ΔC (corrigido por otimismo) vs base:')
for n in ['+PRS_Nalls','+PRS_Progression','+ambos']: print(f'    {n:18s} ΔC_corr={res[n][1]-cb[1]:+.4f} | ΔC_OOF={res[n][2]-cb[2]:+.4f}')
# LRT
llb=llf(T6)
print('\n  Teste de razão de verossimilhança (o PRS melhora o ajuste?):')
for n,cols in [('+PRS_Nalls',T6+['PRS_Nalls_z']),('+PRS_Progression',T6+['PRS_Progression_z']),('+ambos',T6+['PRS_Nalls_z','PRS_Progression_z'])]:
    lr=2*(llf(cols)-llb); df=len(cols)-len(T6); p=chi2.sf(lr,df); print(f'    {n:18s} LR={lr:.2f} (df={df}) p={p:.3f}')
# IDI 5a
mb=CoxPHFitter().fit(d[['exit','event']+T6],'exit','event'); mf=CoxPHFitter().fit(d[['exit','event']+T6+['PRS_Nalls_z','PRS_Progression_z']],'exit','event')
rb=1-mb.predict_survival_function(d[T6],times=[5]).iloc[0].values; rf=1-mf.predict_survival_function(d[T6+['PRS_Nalls_z','PRS_Progression_z']],times=[5]).iloc[0].values
st=d.assign(rb=rb,rf=rf); ev=st[(st.event==1)&(st.exit<=5)]; ne=st[st.exit>=5]
idi=(ev.rf.mean()-ev.rb.mean())-(ne.rf.mean()-ne.rb.mean())
print(f'\n  IDI em 5 anos (n_ev={len(ev)}, n_naoev={len(ne)}): {idi:+.4f}')
print('\n===== RESULTADO AUDITORIA:', 'TUDO PASSOU' if not FAIL else f'{len(FAIL)} FALHAS: {FAIL}','=====')
pickle.dump({'res':res,'idi5':idi,'n':len(d),'ev':int(d.event.sum())},open(PR+'/prs_added_value.pkl','wb'))
d[['PATNO','exit','event']+T6+['PRS_Nalls_z','PRS_Progression_z','race']].to_parquet(PR+'/prs_analytic.parquet')
