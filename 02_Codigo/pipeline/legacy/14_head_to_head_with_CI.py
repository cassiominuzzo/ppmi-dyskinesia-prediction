# 14_head_to_head_with_CI.py
# Produces: Head-to-head vs published predictor sets with paired bootstrap. Supplementary Figure S2
# Original file in the project archive: 67_head_to_head_CI.py

"""67 - Head-to-head com incerteza.

Calcula, para cada conjunto de preditores publicado e para o Total-6:
  - escore de risco out-of-fold (5-fold repetido 5x, para nao usar dado de treino)
  - C-index com IC 95% por bootstrap
  - delta-C vs Total-6 com IC 95% por bootstrap PAREADO (mesma replica para os dois
    modelos, que e a comparacao correta quando avaliados nos mesmos pacientes)

Todos os conjuntos sao reajustados na MESMA subamostra completa, de modo que a
comparacao e da ESCOLHA DE VARIAVEIS, nao dos coeficientes publicados originais.
"""
from config import PROJECT_ROOT
import os
import cohort as C
# Realigned in September 2026: the paths used to point at the previous project
# layout, inside the raw-data folder. They now come from cohort.py, the single
# definition of where things live.
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

np.random.seed(42)
B = PROJECT_ROOT
PR = C.PROC
TAB = C.ensure_dir(os.path.join(C.RESULTS,'Tabelas'))


def ip(d):
    d = d.copy()
    d['PATNO'] = pd.to_numeric(d['PATNO'], errors='coerce').astype('Int64')
    return d.dropna(subset=['PATNO'])


M = pd.read_parquet(PR + '/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns:
    M = M.rename(columns={'ageonset_x': 'ageonset'})
M = ip(M)
d = M[M['exit'] > 0].copy()

MODELS = {
    'Total-6 (this study)': ['updrs_totscore', 'ageonset', 'SEX', 'BMI', 'td_pigd_ratio', 'NP2FREZ'],
    'Zhao 2025':            ['ageonset', 'td_pigd_ratio', 'updrs3_score'],
    'Olanow / STRIDE-PD':   ['ageonset', 'SEX', 'BMI'],
    'Santos-Lobato 2020':   ['ageonset', 'duration_yrs', 'updrs2_score'],
    'Chen 2021':            ['ageonset', 'duration_yrs'],
    'Eusebi 2018':          ['SEX', 'pigd', 'stai'],
}

allv = sorted(set(v for s in MODELS.values() for v in s))
dd = d.dropna(subset=allv).reset_index(drop=True)
n = len(dd)
E = dd['event'].astype(bool).values
T = dd['exit'].values
print('subamostra identica para todos: n=%d, eventos=%d' % (n, int(E.sum())), flush=True)


def cidx(E, T, S):
    """Harrell C vetorizado (pares comparaveis; empates de escore valem 0.5)."""
    Ti = T[:, None]; Tj = T[None, :]
    a = (Ti < Tj) & E[:, None]
    b = (Tj < Ti) & E[None, :]
    Si = S[:, None]; Sj = S[None, :]
    eq = 0.5 * (Si == Sj)
    den = a.sum() + b.sum()
    if den == 0:
        return np.nan
    num = (a * ((Si > Sj) + eq)).sum() + (b * ((Sj > Si) + eq)).sum()
    return num / den


def oof(cols, n_rep=5):
    """escore de risco out-of-fold, media de n_rep particoes 5-fold"""
    acc = np.zeros(n)
    for r in range(n_rep):
        kf = KFold(5, shuffle=True, random_state=100 + r)
        pr = np.zeros(n)
        for tr, te in kf.split(np.arange(n)):
            sc = StandardScaler().fit(dd.iloc[tr][cols])
            m = CoxPHSurvivalAnalysis(alpha=1.0).fit(
                sc.transform(dd.iloc[tr][cols]), Surv.from_arrays(E[tr], T[tr]))
            pr[te] = m.predict(sc.transform(dd.iloc[te][cols]))
        acc = acc + (pr - pr.mean()) / pr.std()
    return acc / n_rep


S = {k: oof(v) for k, v in MODELS.items()}
names = list(MODELS)
ref = names[0]

print('\nvalidacao da implementacao vetorizada vs scikit-survival:', flush=True)
for k in names[:3]:
    print('  %-22s  vetorizada=%.6f  sksurv=%.6f'
          % (k, cidx(E, T, S[k]), concordance_index_censored(E, T, S[k])[0]), flush=True)

Cpt = {k: cidx(E, T, S[k]) for k in names}

NB = 500
idx = np.arange(n)
bootC = {k: [] for k in names}
bootD = {k: [] for k in names if k != ref}
ok = 0
for b in range(NB):
    bs = np.random.choice(idx, n, replace=True)
    Eb, Tb = E[bs], T[bs]
    if Eb.sum() < 10:
        continue
    cs = {k: cidx(Eb, Tb, S[k][bs]) for k in names}
    if any(np.isnan(v) for v in cs.values()):
        continue
    ok += 1
    for k in names:
        bootC[k].append(cs[k])
        if k != ref:
            bootD[k].append(cs[ref] - cs[k])

print('\nreplicas bootstrap validas: %d' % ok, flush=True)
rows = []
print('\n%-22s %2s  %-22s  %-30s %8s'
      % ('Predictor set', 'k', 'C (95% CI)', 'dC vs Total-6 (95% CI)', 'p'), flush=True)
for k in names:
    lo, hi = np.percentile(bootC[k], [2.5, 97.5])
    cstr = '%.3f (%.3f-%.3f)' % (Cpt[k], lo, hi)
    if k == ref:
        print('%-22s %2d  %-22s  %-30s %8s' % (k, len(MODELS[k]), cstr, 'reference', ''), flush=True)
        rows.append(dict(model=k, k=len(MODELS[k]), C=Cpt[k], C_lo=lo, C_hi=hi,
                         dC=0.0, d_lo=np.nan, d_hi=np.nan, p=np.nan))
    else:
        dv = np.array(bootD[k])
        dlo, dhi = np.percentile(dv, [2.5, 97.5])
        dpt = Cpt[ref] - Cpt[k]
        p = max(2 * min((dv <= 0).mean(), (dv >= 0).mean()), 1.0 / ok)
        print('%-22s %2d  %-22s  %-30s %8.4f'
              % (k, len(MODELS[k]), cstr, '%+.3f (%+.3f to %+.3f)' % (dpt, dlo, dhi), p), flush=True)
        rows.append(dict(model=k, k=len(MODELS[k]), C=Cpt[k], C_lo=lo, C_hi=hi,
                         dC=dpt, d_lo=dlo, d_hi=dhi, p=p))

pd.DataFrame(rows).to_csv(TAB + '/tab67_head_to_head_CI.csv', index=False)
print('\nsalvo tab67_head_to_head_CI.csv', flush=True)
