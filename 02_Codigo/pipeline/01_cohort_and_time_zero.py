# 01_cohort_and_time_zero.py
# Produces: Cohort definition and time-zero (levodopa initiation). Figure 1 numbers
# Original file in the project archive: 01b_cohort_tzero_coverage_CANONICO.py

"""
04_levodopa_filter_fix.py — CORREÇÃO do filtro de levodopa (falsos negativos) + genética.
1) Filtro robusto de levodopa (marcas, abreviações, erros de grafia); audita que nenhum termo
   remanescente contém dopa/lev/carb (exceto methyldopa).
2) Re-deriva tempo-zero, coorte, eventos, horizontes; mede o DELTA vs o salvo.
3) Carriers genéticos corrigidos via MUTRSLT (positivo), não pelo código da variante.
4) Sobrescreve tzero.parquet, analytic_base.parquet, tab02/tab03 corrigidos.
"""
from config import PROJECT_ROOT
import os, glob, re, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from lifelines import KaplanMeierFitter
import cohort as C
# Realigned in September 2026: it used to write into the previous project layout, inside
# the raw-data folder, so tzero.parquet and analytic_base.parquet never reached
# 04_Dados_processados. The paths now come from cohort.py, the single definition of where
# things live. Verified against the stored files: both reproduce exactly.
BASE=PROJECT_ROOT
RAW=C.RAW; PROC=C.PROC; SRC=os.path.join(BASE,'02_Dados_Processados')
def best(*c):
    v=[p for p in c if p and os.path.exists(p)]; return max(v,key=os.path.getsize) if v else None
def ip(df): df=df.copy(); df['PATNO']=pd.to_numeric(df['PATNO'],errors='coerce').astype('Int64'); return df.dropna(subset=['PATNO'])

# ---- 1) FILTRO ROBUSTO DE LEVODOPA ----
LD=(r'dopa|carb.{0,4}lev|lev.{0,3}dop|levodop|levodapa|levopar|levocomp|sinemet|madopar|stalevo|rytary|'
    r'duopa|duodopa|parcopa|prolopa|isicom|nacom|careldopa|carledopa|melevodopa|foslevodopa|'
    r'benseraz|vyalev|sirio|restex|kinson|syndopa|tidomet|sastravi|dopadura|levomet')
def is_levodopa(s):
    s=str(s).lower()
    if 'methyldopa' in s: return False
    return bool(re.search(LD,s))
ledd=ip(pd.read_csv(best(os.path.join(RAW,'Historia_Medica_e_Medicacao','LEDD_Concomitant_Medication_Log_29Apr2026.csv'))))
ledd['STARTDT']=pd.to_datetime(ledd['STARTDT'],errors='coerce',format='mixed')
ledd['is_ld']=ledd['LEDTRT'].apply(is_levodopa)
terms=ledd.groupby('LEDTRT')['is_ld'].first()
unmatched_susp=[t for t in terms[~terms].index.astype(str) if re.search(r'dopa|lev|carb',t.lower()) and 'methyldopa' not in t.lower()]
print('AUDITORIA do filtro: termos LEVODOPA =',int(terms.sum()),'| não-LD =',int((~terms).sum()))
print('Termos remanescentes suspeitos (contêm dopa/lev/carb, fora methyldopa):',unmatched_susp)
print('Exemplos NÃO-LD (conferir agonistas/IMAO/COMT):',sorted(terms[~terms].index.astype(str))[:12])

# ---- 2) tempo-zero NOVO vs ANTIGO ----
tz_new=ledd[ledd.is_ld].dropna(subset=['STARTDT']).groupby('PATNO')['STARTDT'].min()
old=ip(pd.read_parquet(os.path.join(PROC,'tzero.parquet')))
old['tzero_levodopa']=pd.to_datetime(old['tzero_levodopa'])
cmp=pd.DataFrame({'new':tz_new}).reset_index().merge(old[['PATNO','tzero_levodopa']].rename(columns={'tzero_levodopa':'old'}),on='PATNO',how='outer')
n_new_only=int(cmp['old'].isna().__and__(cmp['new'].notna()).sum())
both=cmp.dropna(subset=['new','old']); earlier=int((both['new']<both['old']).sum())
print(f'\nDELTA tempo-zero: cobertura {old.tzero_levodopa.notna().sum()} -> {tz_new.notna().sum()} (+{n_new_only} novos pacientes)')
print(f'  pacientes cujo tempo-zero ficou MAIS CEDO com o filtro corrigido: {earlier}')
if earlier: print('  mediana do adianto (anos):',round(((both.old-both.new).dt.days/365.25)[both.new<both.old].median(),2))

# ---- 3) coorte + eventos NOVOS ----
p4=ip(pd.read_csv(best(os.path.join(RAW,'MDS_UPDRS_e_Motor','MDS-UPDRS_Part_IV__Motor_Complications_29Apr2026.csv'))))
p4['INFODT']=pd.to_datetime(p4['INFODT'],errors='coerce',format='mixed')
for cc in ['NP4WDYSK','NP4DYSKI']: p4[cc]=pd.to_numeric(p4[cc],errors='coerce').replace(101,np.nan)
m=p4.merge(tz_new.rename('tz').reset_index(),on='PATNO',how='inner'); m=m[m['INFODT']>=m['tz']]
m['yr']=(m['INFODT']-m['tz']).dt.days/365.25
m['fl']=((m['NP4WDYSK']>=2)|(m['NP4DYSKI']>=2)).astype(int)
def surv(g):
    g=g.sort_values('yr'); e=g['yr'].iloc[0]; hit=g[g.fl==1]
    if len(hit): return pd.Series({'entry':e,'exit':hit['yr'].iloc[0],'event':1})
    return pd.Series({'entry':e,'exit':g['yr'].iloc[-1],'event':0})
S=ip(m.groupby('PATNO').apply(surv).reset_index())
S['prevalent_left']=((S.event==1)&(np.isclose(S.exit,S.entry))).astype(int)
ab_old=pd.read_parquet(os.path.join(PROC,'analytic_base.parquet'))
print(f'\nDELTA coorte: N {len(ab_old)} -> {len(S)} | eventos {int(ab_old.event.sum())} -> {int(S.event.sum())}')

# ---- 4) GENÉTICA corrigida (carrier = MUTRSLT positivo) ----
gen=ip(pd.read_csv(best(os.path.join(RAW,'Geneticos','Genetic_Testing_Results_29Apr2026.csv'))))
gen['MUTRSLT']=pd.to_numeric(gen['MUTRSLT'],errors='coerce'); gen['GENECAT']=pd.to_numeric(gen['GENECAT'],errors='coerce')
print('\nGENECAT x MUTRSLT (linhas):'); print(pd.crosstab(gen['GENECAT'],gen['MUTRSLT']).to_string())
ids=set(S['PATNO'].dropna().astype(int))
has_gen=set(gen.loc[gen['GENECAT'].notna(),'PATNO'].dropna().astype(int))
# GENECAT: 1=LRRK2, 2=SNCA, 3=GBA (confirmar pela crosstab acima)
lrrk2=set(gen.loc[(gen.GENECAT==1)&(gen.MUTRSLT==1),'PATNO'].dropna().astype(int))
gba=set(gen.loc[(gen.GENECAT==3)&(gen.MUTRSLT==1),'PATNO'].dropna().astype(int))
anycarrier=set(gen.loc[gen.MUTRSLT==1,'PATNO'].dropna().astype(int))
print(f'\nGenética na coorte (N={len(ids)}): testados(GENECAT)={len(has_gen&ids)} | '
      f'carrier QUALQUER(MUTRSLT=1)={len(anycarrier&ids)} | LRRK2+={len(lrrk2&ids)} | GBA+={len(gba&ids)}')

# DaTSCAN/LCR (curated)
cur=ip(pd.read_parquet(os.path.join(SRC,'curated_cache.parquet')))
has_dat=set(cur.groupby('PATNO')[['MIA_PUTAMEN_L','MIA_PUTAMEN_R']].apply(lambda g:g.notna().any().any()).pipe(lambda s:s[s].index).astype(int))
csfcols=[x for x in ['CSFSAA','abeta','tau','ptau','NFL_CSF'] if x in cur.columns]
has_csf=set(cur.groupby('PATNO')[csfcols].apply(lambda g:g.notna().any().any()).pipe(lambda s:s[s].index).astype(int))
def pc(s): n=len(s&ids); return f'{n} ({100*n/len(ids):.0f}%)'
cov=pd.DataFrame([
 {'Modalidade':'Genético TESTADO (GENECAT)','Cobertura':pc(has_gen)},
 {'Modalidade':'  carrier QUALQUER (MUTRSLT+)','Cobertura':pc(anycarrier)},
 {'Modalidade':'  LRRK2+ (GENECAT=1 & MUTRSLT+)','Cobertura':pc(lrrk2)},
 {'Modalidade':'  GBA+ (GENECAT=3 & MUTRSLT+)','Cobertura':pc(gba)},
 {'Modalidade':'DaTSCAN','Cobertura':pc(has_dat)},
 {'Modalidade':'LCR','Cobertura':pc(has_csf)},
 {'Modalidade':'INTERSEÇÃO genét+DaTSCAN+LCR','Cobertura':pc(has_gen&has_dat&has_csf)},
])
print('\n=== COBERTURA CORRIGIDA ==='); print(cov.to_string(index=False))

# ---- 5) horizontes novos ----
print('\n=== HORIZONTES (filtro corrigido) ===')
for h in [3,5,7,10]:
    ar=int(((S.entry<h)&(S.exit>=h)).sum()); ev=int(((S.event==1)&(S.exit<=h)&(S.prevalent_left==0)).sum())
    print(f'  {h}a: em risco={ar}, eventos incid.={ev}')

# ---- 6) SOBRESCREVER artefatos corrigidos ----
old2=old.copy(); old2=old2.drop(columns=['tzero_levodopa']).merge(tz_new.rename('tzero_levodopa').reset_index(),on='PATNO',how='outer')
old2.to_parquet(os.path.join(PROC,'tzero.parquet'),index=False)
# base analítica: re-anexar clínico pré-levodopa
cur2=cur.merge(tz_new.rename('tz').reset_index(),on='PATNO',how='inner'); cur2['dsd']=(pd.to_datetime(cur2['visit_date'],errors='coerce')-cur2['tz']).dt.days
L1=[x for x in ['ageonset','SEX','BMI','duration_yrs','updrs1_score','updrs2_score','updrs3_score','pigd','td_pigd','moca','stai','upsit'] if x in cur2.columns]
preb=ip(cur2[cur2['dsd']<=0].sort_values(['PATNO','dsd']).groupby('PATNO').last().reset_index())[['PATNO']+L1]
for x in L1: preb[x]=pd.to_numeric(preb[x],errors='coerce')
S['has_genetic']=S.PATNO.astype(int).isin(has_gen).astype(int)
S['has_datscan']=S.PATNO.astype(int).isin(has_dat).astype(int)
S['has_csf']=S.PATNO.astype(int).isin(has_csf).astype(int)
S['lrrk2_carrier']=S.PATNO.astype(int).isin(lrrk2).astype(int)
S['gba_carrier']=S.PATNO.astype(int).isin(gba).astype(int)
S.merge(preb,on='PATNO',how='left').to_parquet(os.path.join(PROC,'analytic_base.parquet'),index=False)
cov.to_csv(os.path.join(PROC,'tab03_cobertura.csv'),index=False)
print('\nArtefatos corrigidos sobrescritos (tzero, analytic_base, tab03). FIX COMPLETO')
