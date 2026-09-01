# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""
08_univariate_screen.py — ETAPA 4. PARTE A: pré-auditoria do dataset de modelagem
(extração + metodologia). PARTE B: Cox univariado (truncamento à esquerda) + PH + FDR.
Só preditores BASAIS pré-levodopa entram na triagem; responsividade (landmark) e portadores
LRRK2/GBA (coorte enriquecida) ficam para análises dedicadas.
"""
from config import PROJECT_ROOT
import os, glob, warnings, pickle
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests
_c=[PROJECT_ROOT]; BASE=_c[0]
RAW=BASE+'/01_Dados_Brutos'; PROC=BASE+'/Projeto_LID_v2/02_Dados_Processados'
FIG=BASE+'/Projeto_LID_v2/04_Resultados/Figuras'; TAB=BASE+'/Projeto_LID_v2/04_Resultados/Tabelas'
def best(*c):
    v=[p for p in c if p and os.path.exists(p)]; return max(v,key=os.path.getsize) if v else None
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
def item04(s): s=pd.to_numeric(s,errors='coerce'); return s.where((s>=0)&(s<=4))
M=pd.read_parquet(PROC+'/master_baseline.parquet'); FLAG=[]
def chk(n,ok,d=''): print(f"  [{'PASS' if ok else '*FALHA':5s}] {n}"+(f' — {d}' if d else '')); FLAG.append(n) if not ok else None

print('===== PARTE A — PRÉ-AUDITORIA =====')
# A1 ranges plausíveis
print('\nA1. Plausibilidade de ranges:')
RANGES={'ageonset':(20,95),'BMI':(12,60),'duration_yrs':(0,30),'updrs3_score':(0,132),
        'comp_axial':(0,28),'comp_tremor':(0,40),'comp_marcha_pi':(0,28),'comp_bradicinesia':(0,44),
        'datscan_putamen_min':(-0.1,5),'resp':(-100,100),'moca':(0,30),'upsit':(0,40),'abeta':(50,2000),'ptau':(0,80)}
for c,(lo,hi) in RANGES.items():
    if c in M.columns:
        s=pd.to_numeric(M[c],errors='coerce'); bad=int(((s<lo)|(s>hi)).sum())
        chk(f'{c} dentro de [{lo},{hi}]', bad==0, f'{bad} fora' if bad else '')
# A2 re-extração independente: comp_axial vs bruto
print('\nA2. Re-extração independente (comp_axial vs Part III bruto):')
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
p3=ip(pd.read_csv(best(RAW+'/MDS_UPDRS_e_Motor/MDS-UPDRS_Part_III_29Apr2026.csv'),low_memory=False))
p3['INFODT']=pd.to_datetime(p3['INFODT'],errors='coerce',format='mixed'); p3=p3.merge(tz,on='PATNO',how='inner')
p3['dsd']=(p3['INFODT']-p3['tzero_levodopa']).dt.days; p3=p3[p3['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index()
AX=['NP3SPCH','NP3FACXP','NP3RISNG','NP3GAIT','NP3FRZGT','NP3PSTBL','NP3POSTR']
for c in AX: p3[c]=item04(p3[c])
pres=p3[AX].notna().sum(axis=1); ax_re=p3[AX].sum(axis=1,min_count=1).where(pres>=int(np.ceil(.7*7)))
re=ip(pd.DataFrame({'PATNO':p3['PATNO'],'ax_re':ax_re})).merge(M[['PATNO','comp_axial']],on='PATNO',how='inner')
both=re.dropna(subset=['ax_re','comp_axial']); mism=int((~np.isclose(both['ax_re'],both['comp_axial'])).sum())
chk('comp_axial reproduz o bruto (0 divergências)', mism==0, f'{mism}/{len(both)} divergem')
# A3 re-extração UPDRS-III baseline vs curated
ITEMS=['NP3SPCH','NP3FACXP','NP3RIGN','NP3RIGRU','NP3RIGLU','NP3RIGRL','NP3RIGLL','NP3FTAPR','NP3FTAPL','NP3HMOVR','NP3HMOVL','NP3PRSPR','NP3PRSPL','NP3TTAPR','NP3TTAPL','NP3LGAGR','NP3LGAGL','NP3RISNG','NP3GAIT','NP3FRZGT','NP3PSTBL','NP3POSTR','NP3BRADY','NP3PTRMR','NP3PTRML','NP3KTRMR','NP3KTRML','NP3RTARU','NP3RTALU','NP3RTARL','NP3RTALL','NP3RTALJ','NP3RTCON']
for c in ITEMS: p3[c]=item04(p3[c])
p3['u3_re']=p3[ITEMS].sum(axis=1,min_count=30)
u=ip(p3[['PATNO','u3_re']]).merge(M[['PATNO','updrs3_score']],on='PATNO',how='inner').dropna()
r=np.corrcoef(u['u3_re'],u['updrs3_score'])[0,1] if len(u)>10 else np.nan
chk('UPDRS-III baseline correlaciona com curated (r>0.9)', r>0.9, f'r={r:.3f}, MAD={np.abs(u.u3_re-u.updrs3_score).mean():.1f}')
# A4 estrutura de sobrevida
chk('entry<exit (todos em risco)', bool((M['exit']>M['entry']).mean()>0.98), f'{100*(M.exit>M.entry).mean():.0f}% ok')
chk('event binário 0/1', set(M.event.dropna().unique())<= {0,1,0.0,1.0})
# A5 janela de baseline imagem/LCR
print('\nA5. (nota) clínico/compósitos usam visita dsd<=0; DaTSCAN/LCR usam janela dsd<=30 (baseline). resp e LRRK2/GBA EXCLUÍDOS da triagem basal.')

print('\n===== PARTE B — COX UNIVARIADO (truncamento à esquerda) =====')
CONT=['ageonset','duration_yrs','BMI','updrs1_score','updrs2_score','updrs3_score','comp_fala','comp_tremor',
      'comp_marcha_pi','comp_bradicinesia','comp_axial','comp_assimetria_lr','td_pigd_ratio','moca','stai','upsit',
      'datscan_putamen_min','datscan_putamen_asym','MIA_CAUDATE_L','MIA_CAUDATE_R','abeta','tau','ptau','NFL_CSF']
BIN=['SEX','is_pigd','pigd','td_pigd']
rows=[]
for c in [x for x in CONT+BIN if x in M.columns]:
    d=M[['entry','exit','event',c]].copy(); d[c]=pd.to_numeric(d[c],errors='coerce'); d=d.dropna(); d=d[d['exit']>d['entry']]
    if len(d)<30 or d[c].nunique()<2 or int(d.event.sum())<5: 
        rows.append({'Preditor':c,'n':len(d),'eventos':int(d.event.sum()),'HR':np.nan,'nota':'poucos dados'}); continue
    perSD = c in CONT
    if perSD: d[c]=(d[c]-d[c].mean())/d[c].std()
    try:
        cph=CoxPHFitter().fit(d,duration_col='exit',event_col='event',entry_col='entry')
        hr=cph.summary.loc[c]
        try: ph=float(proportional_hazard_test(cph,d,time_transform='rank').summary['p'].iloc[0])
        except: ph=np.nan
        rows.append({'Preditor':c+(' (/DP)' if perSD else ''),'n':len(d),'eventos':int(d.event.sum()),
                     'HR':round(hr['exp(coef)'],3),'IC inf':round(hr['exp(coef) lower 95%'],3),'IC sup':round(hr['exp(coef) upper 95%'],3),
                     'p':float(hr['p']),'C-index':round(cph.concordance_index_,3),'p_PH':round(ph,3) if ph==ph else None})
    except Exception as e:
        rows.append({'Preditor':c,'n':len(d),'eventos':int(d.event.sum()),'HR':np.nan,'nota':str(e)[:30]})
res=pd.DataFrame(rows)
ok=res['p'].notna()
res.loc[ok,'p_FDR']=multipletests(res.loc[ok,'p'],method='fdr_bh')[1]
res=res.sort_values('p')
print(res.to_string(index=False))
res.to_csv(TAB+'/tab08_univariado.csv',index=False)

# forest dos significativos (FDR<0.05)
sig=res[(res['p_FDR']<0.05)&res['HR'].notna()].copy()
if len(sig):
    sig=sig.sort_values('HR'); fig,ax=plt.subplots(figsize=(8,max(3,0.4*len(sig))))
    y=range(len(sig)); ax.errorbar(sig['HR'],y,xerr=[sig['HR']-sig['IC inf'],sig['IC sup']-sig['HR']],fmt='o',color='#4C72B0',capsize=3)
    ax.axvline(1,color='gray',ls='--'); ax.set_yticks(list(y)); ax.set_yticklabels(sig['Preditor']); ax.set_xlabel('HR (contínuos por DP) — desfecho primário')
    ax.set_title('Preditores univariados significativos (FDR<0.05)'); plt.tight_layout(); plt.savefig(FIG+'/fig08_forest_univariado.png',dpi=130); plt.close()
print(f'\nSignificativos (FDR<0.05): {len(sig)} | violam PH (p_PH<0.05): {int((res["p_PH"]<0.05).sum())}')
with open(PROC+'/results_step08.pkl','wb') as f: pickle.dump({'univariate':res,'audit_fail':FLAG},f)
print('\n===== AUDITORIA: '+('TUDO OK' if not FLAG else f'{len(FLAG)} FALHA(S): '+';'.join(FLAG))+' =====')
print('ETAPA 4 COMPLETA — tab08_univariado.csv, fig08_forest_univariado.png')
