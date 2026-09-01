# 13_supplementary_table_S4.py
# Produces: Supplementary Table S4, levodopa responsiveness across thresholds
# Original file in the project archive: 50_responsividade_robustez.py

"""50 — ROBUSTEZ da responsividade (auditável): >40/50/60% + contínua; HR, p, ΔC OOF, PH."""
import os, glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
import cohort as C

# Rewritten in September 2026. It pointed at the previous project layout, so it wrote into
# the raw-data folder, and it printed the confidence interval of the responsiveness hazard
# ratio without saving it: the interval quoted in the Results and in Supplementary Table
# S16 therefore came from a file that no script produced. Both are fixed here.
PR = C.PROC; TAB = C.ensure_dir(os.path.join(C.RESULTS, 'Tabelas'))
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
mb=ip(pd.read_parquet(PR+'/master_baseline.parquet'))[['PATNO','resp']].dropna()
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
M=ip(M); d=M[M['exit']>0].dropna(subset=T6).merge(mb,on='PATNO',how='inner').reset_index(drop=True)
print('n=%d, eventos=%d'%(len(d),int(d.event.sum())))
def oof(cols,seed):
    kf=KFold(5,shuffle=True,random_state=seed); pr=np.zeros(len(d))
    for tr,te in kf.split(d):
        m=CoxPHFitter(penalizer=0.1).fit(d.iloc[tr][['exit','event']+cols],'exit','event'); pr[te]=-m.predict_partial_hazard(d.iloc[te][cols]).values
    return concordance_index(d['exit'],pr,d['event'])
cb=np.mean([oof(T6,s) for s in range(6)])
print('Total-6 base C_OOF=%.4f\n'%cb)
rows=[]; print('variante            | n_pos | HR (IC95)          | p        | dC_OOF   | pos/12')
d['respz']=(d.resp-d.resp.mean())/d.resp.std()
m=CoxPHFitter(penalizer=0.02).fit(d[['exit','event']+T6+['respz']],'exit','event'); h=m.summary.loc['respz']
dc=np.array([oof(T6+['respz'],s)-oof(T6,s) for s in range(6)])
print('continua (por DP)   |   -   | %.2f (%.2f-%.2f) | %.4f  | %+.4f | %d'%(h['exp(coef)'],h['exp(coef) lower 95%'],h['exp(coef) upper 95%'],h['p'],dc.mean(),int((dc>0).sum())))
rows.append(['continua',round(h['exp(coef)'],3),round(h['exp(coef) lower 95%'],3),round(h['exp(coef) upper 95%'],3),round(h['p'],4),round(dc.mean(),4),round(dc.std(ddof=1),4),int((dc>0).sum())])
for thr in [40,50,60]:
    col='r%d'%thr; d[col]=(d.resp>thr).astype(int); npos=int(d[col].sum())
    m=CoxPHFitter(penalizer=0.02).fit(d[['exit','event']+T6+[col]],'exit','event'); h=m.summary.loc[col]
    dc=np.array([oof(T6+[col],s)-oof(T6,s) for s in range(6)])
    print('>%d%%                | %d(%.0f%%) | %.2f (%.2f-%.2f) | %.5f | %+.4f | %d'%(thr,npos,100*npos/len(d),h['exp(coef)'],h['exp(coef) lower 95%'],h['exp(coef) upper 95%'],h['p'],dc.mean(),int((dc>0).sum())))
    rows.append(['>%d%%'%thr,round(h['exp(coef)'],3),round(h['exp(coef) lower 95%'],3),round(h['exp(coef) upper 95%'],3),round(h['p'],5),round(dc.mean(),4),round(dc.std(ddof=1),4),int((dc>0).sum())])
dd=d[['exit','event']+T6+['r50']]; ph=proportional_hazard_test(CoxPHFitter(penalizer=0.02).fit(dd,'exit','event'),dd)
print('\nPH (resp>50%%): p=%.3f (%s)'%(ph.summary.loc['r50','p'],'OK' if ph.summary.loc['r50','p']>0.05 else 'viola'))
R=pd.DataFrame(rows,columns=['variante','HR','IC_inf','IC_sup','p','dC_OOF','dC_dp','pos_de_6'])
R.to_csv(TAB+'/tab50_responsividade_robustez.csv',index=False)
# the >50% row is the one the Results quote, saved on its own so the interval has a source
r50=R[R.variante=='>50%'].iloc[0]
pd.DataFrame({'metric':['HR_resp50','IC_inf','IC_sup','p','dC_medio','dC_dp'],
              'valor':[r50.HR,r50.IC_inf,r50.IC_sup,r50.p,r50.dC_OOF,r50.dC_dp]}).to_csv(
    TAB+'/tab46b_responsividade.csv',index=False)
print('\nLeitura: HR>1 e dC>0 em TODOS os cortes => achado robusto ao corte.')
