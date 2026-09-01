# 12_competing_risk.py
# Produces: Aalen-Johansen competing-risk adjustment, factor c(t) in Table 2
# Original file in the project archive: 45_competing_risk.py

"""45 — RISCO COMPETITIVO com morte (auditável). Compara a incidência cumulativa de discinesia
problemática pelo Kaplan-Meier ingênuo (morte = censura) vs Aalen-Johansen (morte = risco
competitivo). O ingênuo superestima; o AJ dá a probabilidade absoluta CORRETA (relevante em 10a)."""
from config import PROJECT_ROOT
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import KaplanMeierFitter, AalenJohansenFitter
import cohort as C, os
# Realigned in September 2026: it used to write into the previous project layout, inside
# the raw-data folder, so its output never reached 03_Resultados. PROC and TAB now come
# from cohort.py, which is the single definition of where things live.
B=PROJECT_ROOT; PR=C.PROC; TAB=C.ensure_dir(os.path.join(C.RESULTS,'Tabelas'))
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
mb=ip(pd.read_parquet(PR+'/master_baseline.parquet')); mb=mb[mb['exit']>0]
cur=ip(pd.read_parquet(B+'/02_Dados_Processados/curated_cache.parquet'))
dth=cur.groupby('PATNO').agg({'Death_Status':'max','Death_Date':'first'}).reset_index()
tz=ip(pd.read_parquet(PR+'/tzero.parquet')); tz['tzero_levodopa']=pd.to_datetime(tz['tzero_levodopa'],errors='coerce')
d=mb[['PATNO','exit','event']].merge(dth,on='PATNO',how='left').merge(tz[['PATNO','tzero_levodopa']],on='PATNO',how='left')
d['Death_Date']=pd.to_datetime(d['Death_Date'],errors='coerce'); d['death_yr']=(d['Death_Date']-d['tzero_levodopa']).dt.days/365.25
# tipo competitivo: 1=discinesia, 2=morte sem discinesia (após exit), 0=censura
d['type']=0; d['time']=d['exit']
d.loc[d.event==1,'type']=1
comp=(d.event==0)&(d.Death_Status==1)&(d.death_yr.notna())&(d.death_yr>d.exit)
d.loc[comp,'type']=2; d.loc[comp,'time']=d.loc[comp,'death_yr']
d=d[d.time>0]
print(f'coorte N={len(d)} | discinesia(1)={int((d.type==1).sum())} | morte-competitiva(2)={int((d.type==2).sum())} | censura(0)={int((d.type==0).sum())}')
HOR=[3,5,7,10]
# KM ingênuo (morte = censura)
km=KaplanMeierFitter().fit(d.time, (d.type==1).astype(int))

# Aalen-Johansen (morte = competitivo).
#
# Corrigido em setembro de 2026. lifelines não aceita tempos empatados neste estimador e os
# desempata somando um ruído de 1e-4 a cada tempo; o argumento `seed` é None por omissão, de
# modo que duas execuções sobre os mesmos dados devolvem incidências diferentes. Não era um
# detalhe: c(3) oscilava entre 0.9962 e 0.9972 e c(10) entre 0.9191 e 0.9195, ou seja a
# terceira casa decimal das quatro constantes que a calculadora usa e que o rodapé da
# Tabela 2 publica. O valor publicado, 0.920 em 10 anos, era uma realização do sorteio.
#
# A correção não é fixar uma semente e torcer para que seja a mesma de antes, o que apenas
# congelaria um sorteio. O desempate é ruído em torno de uma quantidade bem definida, então
# a estimativa é a média sobre muitos desempates, e a dispersão entre eles é reportada para
# que se veja o tamanho do ruído. Com REPLICATES sorteios a média é estável na quarta casa.
REPLICATES=400
SEED=20260901
def cif_at(t,df,col):
    idx=df.index[df.index<=t]; return float(df.loc[idx[-1],col]) if len(idx) else 0.0
draws={t:[] for t in HOR}
for r in range(REPLICATES):
    aj=AalenJohansenFitter(calculate_variance=False,seed=SEED+r).fit(
        d.time, d.type, event_of_interest=1)
    ajc=aj.cumulative_density_
    for t in HOR: draws[t].append(cif_at(t,ajc,ajc.columns[0]))
print(f'\n=== INCIDÊNCIA CUMULATIVA de discinesia problemática ===')
print(f'(Aalen-Johansen: média de {REPLICATES} desempates, sementes {SEED}..{SEED+REPLICATES-1})')
print(f'{"horizonte":>10} {"KM ingênuo":>12} {"Aalen-Johansen":>16} {"superestimação":>16} {"ruído do desempate":>20}')
rows=[]
for t in HOR:
    naive=1-float(km.predict(t)); v=np.array(draws[t]); cif=float(v.mean())
    print(f'{t:>9}a {naive*100:>11.1f}% {cif*100:>15.1f}% {(naive-cif)*100:>15.1f} p.p. '
          f'{(v.max()-v.min())*100:>17.3f} p.p.')
    rows.append({'horizonte_anos':t,'KM_ingenuo':round(naive,4),'Aalen_Johansen':round(cif,4),
                 'superestimacao_pp':round((naive-cif)*100,2),
                 'c_t':round(cif/naive,4),
                 'AJ_min':round(float(v.min()),5),'AJ_max':round(float(v.max()),5),
                 'desempates':REPLICATES})
pd.DataFrame(rows).to_csv(TAB+'/tab45_risco_competitivo.csv',index=False)
print('\nfator c(t) = Aalen-Johansen / Kaplan-Meier ingênuo, o que a calculadora aplica:')
print('   '+', '.join(f'c({r["horizonte_anos"]})={r["c_t"]:.4f}' for r in rows))
print('\nLeitura: a diferença cresce com o tempo (a morte compete mais tarde). A calculadora deve')
print('reportar a incidência de Aalen-Johansen como probabilidade absoluta correta, sobretudo em 10a.')
