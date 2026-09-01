# SUPERSEDED. Exploratory analysis from the model-selection phase.
# Retained to document how the six predictors were chosen. The numbers it
# produces are NOT those reported in the manuscript; see pipeline/ for those.

"""46 — PREDIÇÃO DINÂMICA (landmark), auditável. Em marcos de 1/2/3 anos de levodopa, entre pacientes
ainda SEM discinesia, compara o modelo ESTÁTICO (só baseline Total-6) vs DINÂMICO (Total-6 + UPDRS e
LEDD ATUAIS no marco). Se atualizar a informação melhora o C, a predição dinâmica agrega — algo que
NENHUM modelo publicado de LID faz (todos são estáticos)."""
from config import PROJECT_ROOT
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
B=PROJECT_ROOT; PR=B+'/Projeto_LID_v2/02_Dados_Processados'; TAB=B+'/Projeto_LID_v2/04_Resultados/Tabelas'
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
# baseline Total-6 + desfecho (relógio da levodopa)
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
M=ip(M); base=M[['PATNO','exit','event']+T6].dropna(subset=T6)
# longitudinal: UPDRS-total e LEDD por visita, no relógio da levodopa
tz=ip(pd.read_parquet(PR+'/tzero.parquet')); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'],errors='coerce')
cur=ip(pd.read_parquet(B+'/02_Dados_Processados/curated_cache.parquet')); cur['visit_date']=pd.to_datetime(cur['visit_date'],errors='coerce')
cur=cur.merge(tz[['PATNO','tzero_levodopa']],on='PATNO',how='inner'); cur['yr']=(cur['visit_date']-cur['tzero_levodopa']).dt.days/365.25
cur['updrs_total_v']=pd.to_numeric(cur['updrs1_score'],errors='coerce')+pd.to_numeric(cur['updrs2_score'],errors='coerce')+pd.to_numeric(cur['updrs3_score'],errors='coerce')
lg=cur[['PATNO','yr','updrs_total_v','LEDD']].dropna()
def oofC(d,cols,seeds=range(6)):
    d=d[['t','e']+cols].dropna();
    if len(d)<60 or d['e'].sum()<15: return np.nan,len(d),int(d['e'].sum())
    cs=[]
    for s in seeds:
        kf=KFold(5,shuffle=True,random_state=s); pr=np.zeros(len(d)); idx=d.reset_index(drop=True)
        ok=True
        for tr,te in kf.split(idx):
            try: m=CoxPHFitter(penalizer=0.2).fit(idx.iloc[tr][['t','e']+cols],'t','e'); pr[te]=-m.predict_partial_hazard(idx.iloc[te][cols]).values
            except: ok=False
        if ok: cs.append(concordance_index(idx['t'],pr,idx['e']))
    return (np.mean(cs) if cs else np.nan),len(d),int(d['e'].sum())
print('=== PREDIÇÃO DINÂMICA — landmark (marco) ===')
print(f'{"marco":>6} {"em risco":>9} {"eventos":>8} {"C estático":>11} {"C dinâmico":>11} {"ΔC":>8}')
rows=[]
for L in [1,2,3]:
    ar=base[base['exit']>L].copy()                     # sem discinesia no marco
    ar['t']=ar['exit']-L; ar['e']=ar['event']          # novo relógio a partir do marco
    # estado ATUAL no marco: visita mais próxima em [L-1, L+1]
    w=lg[(lg.yr>=L-1)&(lg.yr<=L+1)].copy(); w['dist']=(w.yr-L).abs()
    cw=w.sort_values('dist').groupby('PATNO').first().reset_index()[['PATNO','updrs_total_v','LEDD']]
    d=ar.merge(cw,on='PATNO',how='inner')
    cs,n,ev=oofC(d,T6); cd,_,_=oofC(d,T6+['updrs_total_v','LEDD'])
    if not np.isnan(cs):
        print(f'{L:>5}a {n:>9} {ev:>8} {cs:>11.4f} {cd:>11.4f} {cd-cs:>+8.4f}')
        rows.append({'marco_anos':L,'em_risco':n,'eventos':ev,'C_estatico':round(cs,4),'C_dinamico':round(cd,4),'dC':round(cd-cs,4)})
pd.DataFrame(rows).to_csv(TAB+'/tab46_dynamic_landmark.csv',index=False)
print('\nLeitura: ΔC>0 => atualizar UPDRS/LEDD no marco melhora a predição (valor da dinâmica).')
