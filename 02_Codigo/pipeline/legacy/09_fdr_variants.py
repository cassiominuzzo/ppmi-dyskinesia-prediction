# 09_fdr_variants.py
# Produces: False-discovery-rate variants for the multimodal screen
# Original file in the project archive: 43_fdr_variants.py

"""43 — Variantes de FDR (auditável): (1) global, (2) por domínio, (3) SÓ nos p<0,05 (global),
(4) SÓ nos p<0,05 por domínio. A (3)/(4) são ANTI-CONSERVADORAS (correção frouxa) — mostradas
para completude: se nem assim imagem/biomarcador/genética sobrevivem, o null está blindado."""
from config import PROJECT_ROOT
import os
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from statsmodels.stats.multitest import multipletests
B=PROJECT_ROOT; TAB=C.ensure_dir(os.path.join(C.RESULTS,'Tabelas'))
def L(f):
    try: return pd.read_csv(TAB+'/'+f)
    except: return pd.DataFrame()
BIO={'abeta','tau','ptau','asyn','CSFSAA','CSFSAA_assay','nfl_serum','NFL_CSF','urate','total_di_18_1_BMP','total_di_22_6_BMP','_2_2__di_22_6_BMP'}
GEN={'APOE_e4','fampd','fampd_bin','lrrk2_carrier','gba_carrier','GRS__DNA','PRS_Nalls','PRS_Progression'}
def dom_row(v,src,dom21=None):
    v=str(v)
    if src=='cortical' or src=='lat' or (src=='sangueMRI' and dom21=='MRI FreeSurfer') or 'MIA' in v or 'datscan' in v or v.startswith(('lh_','rh_')): return 'Neuroimagem'
    if src=='biosp' or (src=='sangueMRI' and dom21!='MRI FreeSurfer') or v in BIO: return 'Biomarcador'
    if v in GEN: return 'Genética'
    return 'Clínica'
rows=[]
for f,col,src in [('tab17_triagem_total.csv','variavel','curated'),('tab22_itens_individuais.csv','item','itens'),
                  ('tab28_novas_clinicas.csv','variavel','novas'),('tab27_biospecimen.csv','analito','biosp'),
                  ('tab29_fs_cortical.csv','regiao','cortical'),('tab42_lateralidade_datscan.csv','var','lat')]:
    t=L(f)
    if t.empty or 'p' not in t.columns: continue
    for _,r in t.iterrows():
        rows.append({'var':str(r[col]),'p':float(r['p']),'dom':dom_row(r[col],src)})
t21=L('tab21_completude.csv')
for _,r in t21.iterrows(): rows.append({'var':str(r['variavel']),'p':float(r['p']),'dom':dom_row(r['variavel'],'sangueMRI',r.get('dominio'))})
# PRS
rows+=[{'var':'PRS_Nalls','p':0.477,'dom':'Genética'},{'var':'PRS_Progression','p':0.146,'dom':'Genética'}]
allv=pd.DataFrame(rows).dropna(subset=['p']).drop_duplicates('var')
print(f'Total de testes consolidados: {len(allv)} | por domínio:', allv['dom'].value_counts().to_dict())
def summ(df,label):
    if len(df)==0: print(f'  {label}: (vazio)'); return
    fdr=multipletests(df['p'],method='fdr_bh')[1]; sig=df[fdr<0.05]
    print(f'  {label}: n={len(df)} | sobrevivem FDR<0.05: {len(sig)}'+(' -> '+', '.join(sig["var"].head(8)) if len(sig) else ''))
print('\n=== (1) FDR GLOBAL (todos os testes) ===')
summ(allv,'global')
print('\n=== (2) FDR POR DOMÍNIO ===')
for d in ['Clínica','Neuroimagem','Biomarcador','Genética']: summ(allv[allv.dom==d],d)
sigonly=allv[allv['p']<0.05]
print(f'\n=== (3) FDR SÓ nos p<0,05 (GLOBAL) — ANTI-CONSERVADOR — n={len(sigonly)} ===')
summ(sigonly,'só significativos (global)')
print('\n=== (4) FDR SÓ nos p<0,05 POR DOMÍNIO — ANTI-CONSERVADOR ===')
for d in ['Clínica','Neuroimagem','Biomarcador','Genética']:
    summ(sigonly[sigonly.dom==d],d)
# quais imagem/biomarcador/genética passam na versão frouxa?
print('\n=== FOCO: imagem/biomarcador/genética que sobrevivem na versão FROUXA (3) ===')
so=sigonly.copy(); so['fdr']=multipletests(so['p'],method='fdr_bh')[1]
nonclin=so[(so.dom!='Clínica')&(so.fdr<0.05)]
print(nonclin[['var','dom','p','fdr']].round(4).to_string(index=False) if len(nonclin) else '  NENHUMA variável não-clínica sobrevive nem na correção frouxa.')
allv.to_csv(TAB+'/tab43_fdr_variantes.csv',index=False)
