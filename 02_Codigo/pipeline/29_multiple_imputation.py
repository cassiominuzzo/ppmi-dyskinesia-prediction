# 29_multiple_imputation.py
# Produces: Supplementary Table S6 (multiple imputation as a sensitivity analysis)
#
# Missingness in the eligible PPMI cohort has two distinct structures:
#   (a) participants with a pre-levodopa assessment but one or more variables missing
#       -> imputable;
#   (b) participants with no pre-levodopa assessment at all, for whom every candidate
#       predictor is missing -> not imputable, because there is no observed baseline
#       information to impute from. Imputing 18 variables from nothing would fabricate
#       the baseline rather than recover it.
# The imputation is therefore restricted to (a) plus the complete cases.
#
# Following White and Royston (Stat Med 2009), the imputation model includes the event
# indicator and the Nelson-Aalen estimate of the cumulative hazard, so that the imputed
# values are compatible with the survival model that will be fitted.

import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from lifelines import CoxPHFitter, NelsonAalenFitter
from lifelines.utils import concordance_index
from scipy import stats
from config import PROJECT_ROOT

import cohort as C

# Rewritten in September 2026, for the same reason as 28_model_assumptions.py: the
# candidate pool came from sel_data.pkl, a pickle that survives only inside the previous
# project layout, so Supplementary Table S3 had no producer. The pool now comes from
# cohort.py and the output is written beside the other results.
PR = C.PROC
TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))

M_IMPUTATIONS = 20
PENALIZER = 0.05
TOTAL6 = ["updrs_totscore", "ageonset", "SEX", "BMI", "td_pigd_ratio", "NP2FREZ"]
DISCRETE = {"SEX": (0, 1), "NP2FREZ": (0, 4)}
np.random.seed(42)

POOL = list(C.POOL)
M = pd.read_parquet(os.path.join(PR, "candidate_matrix.parquet"))
if "ageonset" not in M.columns and "ageonset_x" in M.columns:
    M = M.rename(columns={"ageonset_x": "ageonset"})
e = M[M["exit"] > 0].copy().reset_index(drop=True)
P = [c for c in POOL if c in e.columns]
n_missing = e[P].isna().sum(axis=1)

no_baseline = e[n_missing == len(P)]
imputable = e[n_missing < len(P)].reset_index(drop=True)
complete = e[n_missing == 0]

print("eligible                         : %4d (%d events, %.1f%%)"
      % (len(e), int(e.event.sum()), 100 * e.event.mean()))
print("  complete cases                 : %4d (%d events, %.1f%%)"
      % (len(complete), int(complete.event.sum()), 100 * complete.event.mean()))
print("  partially missing (imputable)  : %4d (%d events, %.1f%%)"
      % (len(imputable) - len(complete), int(imputable.event.sum() - complete.event.sum()),
         100 * (imputable.event.sum() - complete.event.sum()) / (len(imputable) - len(complete))))
print("  no pre-levodopa assessment     : %4d (%d events, %.1f%%)  NOT imputed"
      % (len(no_baseline), int(no_baseline.event.sum()), 100 * no_baseline.event.mean()))

# auxiliary variables for a survival-compatible imputation model
na = NelsonAalenFitter().fit(imputable["exit"], imputable["event"])
imputable["_H"] = na.cumulative_hazard_at_times(imputable["exit"]).values
imputable["_D"] = imputable["event"].astype(float)
IMPCOLS = P + ["_H", "_D"]

# ---- reference: complete-case model ----
ccfit = CoxPHFitter(penalizer=PENALIZER).fit(complete[TOTAL6 + ["exit", "event"]], "exit", "event")
cc_beta = ccfit.params_[TOTAL6].values
cc_se = ccfit.standard_errors_[TOTAL6].values
cc_c = concordance_index(complete["exit"], -ccfit.predict_partial_hazard(complete[TOTAL6]), complete["event"])

# ---- multiple imputation ----
betas, ses, cidx = [], [], []
for m in range(M_IMPUTATIONS):
    imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                           max_iter=10, random_state=1000 + m)
    X = pd.DataFrame(imp.fit_transform(imputable[IMPCOLS]), columns=IMPCOLS)
    for v, (lo, hi) in DISCRETE.items():
        X[v] = X[v].round().clip(lo, hi)
    X["exit"] = imputable["exit"].values
    X["event"] = imputable["event"].values
    f = CoxPHFitter(penalizer=PENALIZER).fit(X[TOTAL6 + ["exit", "event"]], "exit", "event")
    betas.append(f.params_[TOTAL6].values)
    ses.append(f.standard_errors_[TOTAL6].values)
    cidx.append(concordance_index(X["exit"], -f.predict_partial_hazard(X[TOTAL6]), X["event"]))

betas = np.array(betas); ses = np.array(ses)

# ---- Rubin's rules ----
qbar = betas.mean(axis=0)
ubar = (ses ** 2).mean(axis=0)
B = betas.var(axis=0, ddof=1)
Ttot = ubar + (1 + 1 / M_IMPUTATIONS) * B
se_pool = np.sqrt(Ttot)
lam = ((1 + 1 / M_IMPUTATIONS) * B) / Ttot
df = (M_IMPUTATIONS - 1) / np.maximum(lam ** 2, 1e-12)
tcrit = stats.t.ppf(0.975, df)

rows = []
for i, v in enumerate(TOTAL6):
    rows.append(dict(predictor=v,
                     cc_beta=round(cc_beta[i], 4), cc_se=round(cc_se[i], 4),
                     mi_beta=round(qbar[i], 4), mi_se=round(se_pool[i], 4),
                     mi_lo=round(qbar[i] - tcrit[i] * se_pool[i], 4),
                     mi_hi=round(qbar[i] + tcrit[i] * se_pool[i], 4),
                     fmi=round(float(lam[i]), 3),
                     diff_in_cc_se=round((qbar[i] - cc_beta[i]) / cc_se[i], 2)))
out = pd.DataFrame(rows)
print("\nCoefficients, complete case (n=%d) versus multiple imputation (n=%d, m=%d):"
      % (len(complete), len(imputable), M_IMPUTATIONS))
print(out.to_string(index=False))
print("\napparent C-index  complete case %.3f | multiple imputation %.3f (range %.3f to %.3f)"
      % (cc_c, np.mean(cidx), min(cidx), max(cidx)))
print("largest shift in any coefficient: %.2f complete-case standard errors"
      % np.abs(out["diff_in_cc_se"]).max())

out.to_csv(os.path.join(TAB, "tabS6_multiple_imputation.csv"), index=False)
pd.DataFrame(dict(imputation=range(1, M_IMPUTATIONS + 1), c_index=cidx)).to_csv(
    os.path.join(TAB, "tabS6_mi_cindex.csv"), index=False)
print("\nsaved tabS6_multiple_imputation.csv")
