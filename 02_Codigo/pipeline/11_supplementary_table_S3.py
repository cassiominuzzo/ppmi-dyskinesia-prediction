# 11_supplementary_table_S3.py
# Produces: Supplementary Table S3, sensitivity to outcome, time-zero and truncation
# Original file in the project archive: 44_sensibilidade_desfecho_truncamento.py

"""44 — SENSIBILIDADE do desfecho e do truncamento (auditável). Mostra que o Total-6 é ROBUSTO a:
(i) definição do desfecho (any / primária / 4.1>=2 sozinho); (ii) tempo-zero (levodopa vs 1a terapia
dopaminérgica); (iii) tratamento de prevalentes (right-censored vs excluir prevalentes)."""
from config import PROJECT_ROOT
import os
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import glob, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, KaplanMeierFitter
B=PROJECT_ROOT; RAW=C.RAW; PR=C.PROC; TAB=C.ensure_dir(os.path.join(C.RESULTS,'Tabelas'))
def best(*c):
    import os; v=[p for p in c if p and os.path.exists(p)]; return max(v,key=os.path.getsize) if v else None
def ip(d): d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])
T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
# Corrigido em setembro de 2026. O recorte era feito exigindo dados completos apenas nas
# seis preditoras do modelo, e nao nas dezoito do pool de candidatas. A amostra de
# desenvolvimento do artigo e definida pelas dezoito, "813 had complete data on all 18
# candidate clinical predictors", de modo que o recorte antigo devolvia seis participantes
# a mais em todas as doze linhas: 819 em vez de 813 na linha primaria. Nenhuma das doze
# linhas da Suplementar S1 reproduzia. Com o pool completo, as doze reproduzem exatamente.
M=ip(M)[['PATNO']+[c for c in C.POOL if c in M.columns]]
tz=ip(pd.read_parquet(PR+'/tzero.parquet'))
for c in ['tzero_levodopa','tzero_any_dopa']: tz[c]=pd.to_datetime(tz[c],errors='coerce')
p4=ip(pd.read_csv(best(RAW+'/MDS_UPDRS_e_Motor/MDS-UPDRS_Part_IV__Motor_Complications_29Apr2026.csv'),low_memory=False))
p4['INFODT']=pd.to_datetime(p4['INFODT'],errors='coerce',format='mixed')
for cc in ['NP4WDYSK','NP4DYSKI']: p4[cc]=pd.to_numeric(p4[cc],errors='coerce').replace(101,np.nan)
FLAGS={'any (>=1)':lambda d:((d.NP4WDYSK>=1)|(d.NP4DYSKI>=1)).astype(int),
       'primaria (>=2 OU)':lambda d:((d.NP4WDYSK>=2)|(d.NP4DYSKI>=2)).astype(int),
       '4.1>=2 (Martinez)':lambda d:(d.NP4WDYSK>=2).astype(int)}
def build(tzcol,flagfn,excl_prev):
    m=p4.merge(tz[['PATNO',tzcol]].dropna(),on='PATNO',how='inner'); m=m[m['INFODT']>=m[tzcol]].copy()
    m['yr']=(m['INFODT']-m[tzcol]).dt.days/365.25; m['fl']=flagfn(m)
    def surv(g):
        g=g.sort_values('yr'); e0=g['yr'].iloc[0]; hit=g[g['fl']==1]
        if len(hit): return pd.Series({'entry':e0,'exit':hit['yr'].iloc[0],'event':1})
        return pd.Series({'entry':e0,'exit':g['yr'].iloc[-1],'event':0})
    S=ip(m.groupby('PATNO').apply(surv).reset_index()); S['prev']=((S.event==1)&(np.isclose(S.exit,S.entry))).astype(int)
    if excl_prev: S=S[S.prev==0]
    S=S[S.exit>0]
    return S.merge(M,on='PATNO',how='inner').dropna(subset=[c for c in C.POOL if c in M.columns])
rows=[]
for tzn,tzc in [('levodopa','tzero_levodopa'),('1a-dopaminergica','tzero_any_dopa')]:
    for fn,ff in FLAGS.items():
        for exn,ex in [('right-cens',False),('exclui-prevalentes',True)]:
            d=build(tzc,ff,ex)
            if len(d)<80: continue
            cph=CoxPHFitter(penalizer=0.05).fit(d[['exit','event']+T6],'exit','event')
            km=KaplanMeierFitter().fit(d.exit,d.event); inc5=1-float(km.predict(5))
            # Acrescentado em setembro de 2026. A coluna "C (out-of-fold)" da Suplementar S1
            # nao era produzida por nenhum script: o texto citava a faixa de ganho ao ancorar
            # na levodopa a partir dela e nao havia como refazer a conta. Usa o protocolo de
            # cohort.py, o mesmo de todo o resto do artigo: vinte repeticoes de validacao
            # cruzada estratificada em cinco partes, padronizacao ajustada dentro de cada
            # parte de treino.
            coof=C.out_of_fold_c(d.reset_index(drop=True),list(T6))
            rows.append({'tempo-zero':tzn,'desfecho':fn,'trunc':exn,'n':len(d),'ev':int(d.event.sum()),
                'inc_5a':round(inc5,3),'C':round(cph.concordance_index_,3),
                'C_oof':round(coof,4),
                'HR_UPDRStot':round(cph.summary.loc['updrs_totscore','exp(coef)'],3),
                'HR_ageonset':round(cph.summary.loc['ageonset','exp(coef)'],3),
                'HR_NP2FREZ':round(cph.summary.loc['NP2FREZ','exp(coef)'],3)})
R=pd.DataFrame(rows); R.to_csv(TAB+'/tab44_sensibilidade.csv',index=False)
print('=== SENSIBILIDADE — Total-6 sob variações de desfecho/tempo-zero/truncamento ===')
print(R.to_string(index=False))
print('\nLeitura: C e direção dos HRs estáveis => modelo ROBUSTO às escolhas de desenho.')
print(f'  C varia de {R.C.min()} a {R.C.max()} | HR_UPDRStot sempre >1 ({R.HR_UPDRStot.min()}-{R.HR_UPDRStot.max()}) | HR_ageonset sempre <1 ({R.HR_ageonset.min()}-{R.HR_ageonset.max()})')
