# 00_data_inventory.py
# Produces: Inventory of all raw PPMI tables (data audit)
# Original file in the project archive: 00_data_inventory.py

"""
00_data_inventory.py — INVENTÁRIO COMPLETO de todas as tabelas brutas (data audit).
Para CADA CSV em 01_Dados_Brutos: domínio, linhas, nº pacientes, EVENT_ID, longitudinal?,
colunas de data, colunas-chave. Gera INVENTARIO_DADOS.md + catalogo.csv.
Objetivo: "olhar tudo perfeitamente" antes de qualquer modelagem (CLAUDE.md §4).
"""
from config import PROJECT_ROOT
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import os, glob, subprocess, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

_c=[PROJECT_ROOT]; BASE=_c[0] if _c else '.'
RAW=C.RAW; PROJ=C.RESULTS
DOC=C.ensure_dir(os.path.join(C.RESULTS,'Documentacao')); PROC=C.PROC
os.makedirs(DOC,exist_ok=True); os.makedirs(PROC,exist_ok=True)

def nrows(path):
    try: return int(subprocess.run(['wc','-l',path],capture_output=True,text=True).stdout.split()[0])-1
    except: return -1

DATE_HINT=('DT','DATE','INFODT','ORIG_ENTRY','LAST_UPDATE','STARTDT','STOPDT','RUNDATE','visit_date')
rows=[]
files=sorted(glob.glob(os.path.join(RAW,'*','*.csv')))
print(f'inventariando {len(files)} CSVs...')
for f in files:
    domain=os.path.basename(os.path.dirname(f)); name=os.path.basename(f)
    try:
        cols=list(pd.read_csv(f,nrows=0).columns)
    except Exception as e:
        rows.append({'Dominio':domain,'Arquivo':name,'Linhas':-1,'erro':str(e)[:40]}); continue
    n=nrows(f)
    has_p='PATNO' in cols; has_e='EVENT_ID' in cols
    npat=nev=np.nan
    if has_p:
        use=['PATNO']+(['EVENT_ID'] if has_e else [])
        try:
            s=pd.read_csv(f,usecols=use,low_memory=False)
            npat=pd.to_numeric(s['PATNO'],errors='coerce').nunique()
            if has_e: nev=s['EVENT_ID'].nunique()
        except: pass
    datecols=[c for c in cols if any(h.lower() in c.lower() for h in DATE_HINT)]
    longit='Sim' if (has_p and not np.isnan(npat) and npat>0 and n>npat*1.2) else ('Não' if has_p else '?')
    rows.append({'Dominio':domain,'Arquivo':name.replace('_29Apr2026','').replace('.csv',''),
                 'Linhas':n,'Pacientes':int(npat) if not np.isnan(npat) else 0,
                 'EVENT_ID':'Sim' if has_e else 'Não','Longitudinal':longit,
                 'N_colunas':len(cols),'Datas':', '.join(datecols[:4]),
                 'Colunas_exemplo':', '.join([c for c in cols if c not in ('PATNO','EVENT_ID','REC_ID','PAG_NAME')][:6])})
cat=pd.DataFrame(rows)
cat.to_csv(os.path.join(PROC,'catalogo_tabelas.csv'),index=False)

# markdown agrupado por domínio
lines=['# Inventário de Dados Brutos — PPMI (cut 29-Abr-2026)','',
       f'_Gerado automaticamente por `03_Scripts/00_data_inventory.py`. {len(files)} CSVs em `01_Dados_Brutos`._','',
       f'**Total:** {len(cat)} tabelas | {int(cat.Linhas[cat.Linhas>0].sum()):,} linhas | domínios: {cat.Dominio.nunique()}','']
for dom in sorted(cat.Dominio.unique()):
    sub=cat[cat.Dominio==dom].sort_values('Linhas',ascending=False)
    lines+=[f'## {dom} ({len(sub)} tabelas)','','| Tabela | Linhas | Pac. | EVENT_ID | Long. | Datas | Colunas (exemplo) |',
            '|---|--:|--:|:--:|:--:|---|---|']
    for _,r in sub.iterrows():
        lines.append(f"| {r['Arquivo']} | {r['Linhas']:,} | {r.get('Pacientes',0)} | {r.get('EVENT_ID','?')} | {r.get('Longitudinal','?')} | {r.get('Datas','')} | {r.get('Colunas_exemplo','')} |")
    lines.append('')
open(os.path.join(DOC,'INVENTARIO_DADOS.md'),'w').write('\n'.join(lines))
print('OK ->', os.path.join(DOC,'INVENTARIO_DADOS.md'))
print(f'tabelas={len(cat)} | longitudinais={int((cat.Longitudinal=="Sim").sum())} | com EVENT_ID={int((cat.EVENT_ID=="Sim").sum())}')
print('\nTop domínios por nº de tabelas:')
print(cat.Dominio.value_counts().to_string())
