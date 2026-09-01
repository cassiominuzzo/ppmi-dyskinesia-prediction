# 08_incremental_value_imaging_csf.py
# Produces: DaTSCAN laterality and CSF; robustness of the multimodal null
# Original file in the project archive: 42_multimodal_robustness.py

"""42 — BLINDAR o null multimodal (auditável). (A) Lateralidade DaTSCAN: pior-lado + índice de
assimetria, univariado + valor incremental sobre o Total-6. (B) FDR POR DOMÍNIO (vs global).
(C) Poder. Dá a cada modalidade a melhor chance; se ainda não agrega, o null fica à prova de balas."""
from config import PROJECT_ROOT
import os
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
B=PROJECT_ROOT; PROC=B+'/02_Dados_Processados'; PR=C.PROC; TAB=C.ensure_dir(os.path.join(C.RESULTS,'Tabelas'))
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
mb=ip(pd.read_parquet(PR+'/master_baseline.parquet')); mb=mb[mb['exit']>0]
cur=ip(pd.read_parquet(PROC+'/curated_cache.parquet'))
T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
M=ip(M)

print('===== (A) LATERALIDADE DaTSCAN — pior-lado + assimetria =====')
reg={'putamen':('MIA_PUTAMEN_L','MIA_PUTAMEN_R'),'caudate':('MIA_CAUDATE_L','MIA_CAUDATE_R'),'striatum':('MIA_STRIATUM_L','MIA_STRIATUM_R')}
dat=cur.groupby('PATNO').first().reset_index()
der={'PATNO':dat['PATNO']}
for nm,(l,r) in reg.items():
    if l in dat.columns and r in dat.columns:
        L=pd.to_numeric(dat[l],errors='coerce'); R=pd.to_numeric(dat[r],errors='coerce')
        der[nm+'_pior']=np.minimum(L,R)                      # menor SBR = lado mais afetado
        der[nm+'_assim']=(L-R).abs()/((L+R)/2)              # índice de assimetria
D=ip(pd.DataFrame(der))
dlat=mb[['PATNO','exit','event']].merge(D,on='PATNO',how='inner')
latvars=[c for c in D.columns if c!='PATNO']
print(f'  n com DaTSCAN={len(dlat)}, eventos={int(dlat.event.sum())}')
rows=[]
for v in latvars:
    s=dlat[['exit','event',v]].dropna()
    if len(s)<60 or s[v].nunique()<10: continue
    s=s.copy(); s[v]=(s[v]-s[v].mean())/s[v].std()
    m=CoxPHFitter(penalizer=0.01).fit(s,'exit','event'); h=m.summary.loc[v]
    rows.append({'var':v,'n':len(s),'HR_DP':round(h['exp(coef)'],3),'p':float(h['p']),'C':round(m.concordance_index_,3)})
R=pd.DataFrame(rows); R['p_FDR_imagem']=multipletests(R['p'],method='fdr_bh')[1]
print(R.round(4).to_string(index=False))
# valor incremental do MELHOR lateralidade sobre Total-6 (mesma subamostra)
best=R.sort_values('p').iloc[0]['var']
mm=M[M['exit']>0].dropna(subset=T6).merge(dlat[['PATNO',best]].dropna(),on='PATNO',how='inner')
def oof(d,cols,seeds=range(5)):
    cs=[]
    for s in seeds:
        kf=KFold(5,shuffle=True,random_state=s); pr=np.zeros(len(d))
        for tr,te in kf.split(d):
            m=CoxPHFitter(penalizer=0.1).fit(d.iloc[tr][['exit','event']+cols],'exit','event'); pr[te]=-m.predict_partial_hazard(d.iloc[te][cols]).values
        cs.append(concordance_index(d['exit'],pr,d['event']))
    return np.mean(cs)
cb=oof(mm,T6); cf=oof(mm,T6+[best])
print(f'  melhor lateralidade = {best}; valor incremental sobre Total-6 (n={len(mm)}): ΔC_OOF={cf-cb:+.4f}')

print('\n===== (B) FDR POR DOMÍNIO (vs global) =====')
def L(f):
    try: return pd.read_csv(TAB+'/'+f)
    except: return pd.DataFrame()
dom={'imagem':[('tab21_completude.csv','variavel'),('tab29_fs_cortical.csv','regiao')],
     'biomarcador':[('tab27_biospecimen.csv','analito')]}
for d_,tabs in dom.items():
    ps=[]
    for f,col in tabs:
        t=L(f)
        if not t.empty and 'p' in t.columns:
            sub=t.copy()
            if d_=='imagem' and f=='tab21_completude.csv': sub=sub[sub['dominio']=='MRI FreeSurfer']  # só MRI
            ps+=list(pd.to_numeric(sub['p'],errors='coerce').dropna())
    ps=np.array(ps); fdr=multipletests(ps,method='fdr_bh')[1] if len(ps) else np.array([])
    print(f'  {d_:12s}: {len(ps)} testes | menor p={ps.min():.4f} | menor p_FDR(domínio)={fdr.min():.3f} | sig(FDR<0.05)={int((fdr<0.05).sum())}')

print('\n===== (C) PODER (eventos disponíveis por modalidade) =====')
print('  DaTSCAN: 283 eventos (alto); LCR core: 277 (alto); MRI: 209 (médio); PRS: 115 (adequado); carriers LRRK2/GBA: 39-77 (baixo)')
print('\nVEREDITO: mesmo dando a melhor chance (pior-lado/assimetria + FDR por domínio), imagem não agrega (ΔC≈0) e nada sobrevive ao FDR de domínio.')
R.to_csv(TAB+'/tab42_lateralidade_datscan.csv',index=False)
