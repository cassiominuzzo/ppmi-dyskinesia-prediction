# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""21 — completude: screen univariado (FDR) dos domínios FORA do curated com boa cobertura."""
from config import PROJECT_ROOT
import glob, os, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
B=PROJECT_ROOT
RAW=B+'/01_Dados_Brutos'; PROC=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
out=ip(pd.read_parquet(PROC+'/master_baseline.parquet'))[['PATNO','exit','event']]
tz=ip(pd.read_parquet(PROC+'/tzero.parquet')[['PATNO','tzero_levodopa']].dropna()); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'])
def screen(df, cols, label):
    d=out.merge(df,on='PATNO',how='left'); d=d[d['exit']>0]; rows=[]
    for c in cols:
        s=d[['exit','event',c]].copy(); s[c]=pd.to_numeric(s[c],errors='coerce'); s=s.dropna()
        if len(s)<60 or s[c].nunique()<5 or int(s.event.sum())<10: continue
        sd=s[c].std();
        if sd==0 or np.isnan(sd): continue
        s[c]=(s[c]-s[c].mean())/sd
        try:
            cph=CoxPHFitter(penalizer=0.05).fit(s,'exit','event'); hr=cph.summary.loc[c]
            rows.append({'dominio':label,'variavel':str(c)[:38],'n':len(s),'n_ev':int(s.event.sum()),
                         'HR_DP':round(hr['exp(coef)'],3),'p':float(hr['p']),'C':round(cph.concordance_index_,3)})
        except Exception: pass
    return pd.DataFrame(rows)
res=[]
# 1) BIOQUÍMICA — pivot longo->largo, baseline pré-levodopa (LCOLLDT <= tzero)
bc=ip(pd.read_csv(RAW+'/Biomarcadores_LCR_Sangue/Blood_Chemistry___Hematology_29Apr2026.csv',low_memory=False))
bc['LSIRES']=pd.to_numeric(bc['LSIRES'],errors='coerce'); bc['LCOLLDT']=pd.to_datetime(bc['LCOLLDT'],errors='coerce',format='mixed')
bc=bc.merge(tz,on='PATNO',how='left'); bc=bc[(bc['LCOLLDT']<=bc['tzero_levodopa'])|(bc['tzero_levodopa'].isna())]  # pré-levodopa
bc=bc.dropna(subset=['LSIRES']).sort_values('LCOLLDT')
bcw=bc.groupby(['PATNO','LTSTNAME'])['LSIRES'].last().unstack().reset_index(); bcw=ip(bcw)
bcols=[c for c in bcw.columns if c!='PATNO' and bcw[c].notna().mean()>0.5 and pd.to_numeric(bcw[c],errors='coerce').notna().sum()>200]
res.append(screen(bcw,bcols,'Bioquímica/sangue'))
# 2) MRI FreeSurfer — BL, normalizar por volume intracraniano se houver
fs=ip(pd.read_csv(RAW+'/Imagem_DaTSCAN_MRI/FS7_ASEG_VOL_29Apr2026.csv',low_memory=False))
fs=fs[fs.get('EVENT_ID','BL')=='BL'] if 'EVENT_ID' in fs.columns else fs
fs=fs.groupby('PATNO').first().reset_index(); fs=ip(fs)
etiv=[c for c in fs.columns if 'eTIV' in c or 'IntraCranial' in c or 'ICV' in c]
fcols=[c for c in fs.select_dtypes(include=[np.number]).columns if c not in ('PATNO','REC_ID') and 'eTIV' not in c and fs[c].notna().mean()>0.4]
if etiv:
    icv=pd.to_numeric(fs[etiv[0]],errors='coerce')
    for c in fcols: fs[c]=pd.to_numeric(fs[c],errors='coerce')/icv  # fração do ICV
    print(f'FreeSurfer normalizado por {etiv[0]}')
res.append(screen(fs,fcols,'MRI FreeSurfer'))
R=pd.concat([r for r in res if len(r)],ignore_index=True)
R['p_FDR']=multipletests(R['p'],method='fdr_bh')[1]
R=R.sort_values('p').reset_index(drop=True)
R.to_csv(TAB+'/tab21_completude.csv',index=False)
print(f'\n=== COMPLETUDE: {len(R)} variáveis extras testadas | {int((R.p_FDR<0.05).sum())} sig após FDR ===')
print('TOP 18 por p bruto:'); print(R.head(18).to_string(index=False))
sig=R[R.p_FDR<0.05]; print('\nSIGNIFICATIVAS após FDR<0.05:'); print(sig.to_string(index=False) if len(sig) else '  NENHUMA.')
