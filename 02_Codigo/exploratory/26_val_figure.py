# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

from config import PROJECT_ROOT
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
np.random.seed(42)
B=PROJECT_ROOT
PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; FIG=B+'/Projeto_LID_v2/04_Resultados/Figuras'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
df=pd.read_parquet(PROC+'/val_dataset.parquet')
A=['updrs2_score','ageonset','SEX','MSEADLG','NP1FATG','comp_bradicinesia','BMI','td_pigd_ratio']; Bv=A+['NP2FREZ']
def imp(d,cols):
    d=d.copy()
    for v in cols: d[v]=pd.to_numeric(d[v],errors='coerce').fillna(df[v].median())
    return d
fig,ax=plt.subplots(1,3,figsize=(15,4.6)); col={'A (8 var)':'#4C72B0','B (8+freezing)':'#C44E52'}
# calibração 3a e 5a
for pi,h in enumerate([3,5]):
    ax[pi].plot([0,.5],[0,.5],'--',color='gray',lw=1,label='ideal')
    for name,cols in [('A (8 var)',A),('B (8+freezing)',Bv)]:
        d=imp(df,cols).dropna(subset=cols+['exit','event'])
        m=CoxPHFitter(penalizer=0.1).fit(d[['exit','event']+cols],'exit','event')
        sf=m.predict_survival_function(d[cols]); t=sf.index[np.argmin(np.abs(sf.index-h))]
        d=d.copy(); d['risk']=1-sf.loc[t].values; d['q']=pd.qcut(d['risk'],5,labels=False,duplicates='drop')
        obs,pre=[],[]
        for q in sorted(d['q'].dropna().unique()):
            g=d[d.q==q]; km=KaplanMeierFitter().fit(g.exit,g.event); obs.append(1-km.predict(h)); pre.append(g['risk'].mean())
        ax[pi].plot(pre,obs,'o-',color=col[name],label=name,ms=6)
    ax[pi].set_title(f'Calibração — {h} anos'); ax[pi].set_xlabel('Risco predito'); ax[pi].set_ylabel('Observado (KM)'); ax[pi].legend(fontsize=8)
# DCA 3a
H=3; ax[2].axhline(0,color='gray',lw=1,ls=':')
pts=np.arange(0.03,0.35,0.01)
dimp=imp(df,Bv).dropna(subset=Bv+['exit','event'])
kmall=KaplanMeierFitter().fit(dimp.exit,dimp.event); ev=1-kmall.predict(H); N=len(dimp)
ax[2].plot(pts,[ev-(1-ev)*(p/(1-p)) for p in pts],color='#999',lw=1.2,label='tratar todos')
for name,cols in [('A (8 var)',A),('B (8+freezing)',Bv)]:
    d=imp(df,cols).dropna(subset=cols+['exit','event'])
    m=CoxPHFitter(penalizer=0.1).fit(d[['exit','event']+cols],'exit','event')
    sf=m.predict_survival_function(d[cols]); t=sf.index[np.argmin(np.abs(sf.index-H))]; d=d.copy(); d['risk']=1-sf.loc[t].values
    nb=[]
    for p in pts:
        hi=d[d.risk>=p]
        if len(hi)<5: nb.append(np.nan); continue
        km=KaplanMeierFitter().fit(hi.exit,hi.event); er=1-km.predict(H); nb.append(er*len(hi)/N-(1-er)*len(hi)/N*(p/(1-p)))
    ax[2].plot(pts,nb,color=col[name],label=name,lw=1.6)
ax[2].set_title('Decision Curve — 3 anos'); ax[2].set_xlabel('Limiar de probabilidade'); ax[2].set_ylabel('Benefício líquido'); ax[2].legend(fontsize=8); ax[2].set_ylim(-0.05,0.12)
plt.tight_layout(); plt.savefig(FIG+'/fig_validacao_AB.png',dpi=140); plt.close()
print('figura salva: fig_validacao_AB.png')
# tabela resumo
T=pd.DataFrame({
 'Métrica':['C-index aparente','C-index corrigido (otimismo)','C-index caso-completo','C-index LOSO (pooled)','Calibration slope (ideal=1)','Net benefit @ pt=0.10 (3a)'],
 'Modelo A (8 var)':[0.640,0.629,0.691,0.663,1.320,0.022],
 'Modelo B (8+NP2FREZ)':[0.650,0.640,0.704,0.673,1.267,0.037]})
T.to_csv(TAB+'/tab_validacao_AB_resumo.csv',index=False); print(T.to_string(index=False))
