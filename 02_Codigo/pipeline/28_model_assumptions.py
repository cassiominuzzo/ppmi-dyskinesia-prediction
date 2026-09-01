# 28_model_assumptions.py
# Produces: Supplementary Table S5 (proportional hazards and linearity of the Total-6 model)
# Proportional hazards: Schoenfeld residual test, under two time transforms.
# Linearity: likelihood-ratio test of restricted cubic splines (4 knots at the 5th, 35th,
# 65th and 95th percentiles) against the linear term, for each continuous predictor.
# Assumption checks are run on the unpenalized fit, because a ridge-penalized partial
# likelihood is not valid for a likelihood-ratio test; the published model uses a ridge
# penalty of 0.05 for coefficient stability.

import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats
from config import PROJECT_ROOT

import cohort as C

# Rewritten in September 2026. It used to load the development sample from sel_data.pkl,
# a pickle that survives only inside the previous project layout, so the script could not
# run as shipped and Supplementary Table S10 had no producer. The cohort now comes from
# cohort.py, the single definition every other analysis uses, and the output is written
# beside the other results rather than into the raw-data folder.
TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))

TOTAL6 = ["updrs_totscore", "ageonset", "SEX", "BMI", "td_pigd_ratio", "NP2FREZ"]
CONTINUOUS = ["updrs_totscore", "ageonset", "BMI", "td_pigd_ratio"]

d = C.build_cohort(with_metadata=False).reset_index(drop=True)
X = d[TOTAL6 + ["exit", "event"]]
print("development sample: n = %d, events = %d" % (len(d), int(d.event.sum())))

# ---- proportional hazards ----
fit = CoxPHFitter().fit(X, "exit", "event")
ph = []
for tt in ("rank", "km"):
    s = proportional_hazard_test(fit, X, time_transform=tt).summary
    for v in TOTAL6:
        ph.append(dict(predictor=v, time_transform=tt,
                       chi2=round(float(s.loc[v, "test_statistic"]), 3),
                       p=round(float(s.loc[v, "p"]), 3)))
ph = pd.DataFrame(ph)
print("\nProportional hazards (Schoenfeld):")
print(ph.pivot(index="predictor", columns="time_transform", values="p").to_string())
print("  predictors with p < 0.05:", int((ph["p"] < 0.05).sum()))


def rcs(x, knots):
    """Restricted cubic spline basis (Harrell); k knots give k-2 non-linear terms."""
    k = np.asarray(knots, float); K = len(k)
    cols = []
    for j in range(K - 2):
        num = (np.maximum(x - k[j], 0) ** 3
               - np.maximum(x - k[K - 2], 0) ** 3 * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
               + np.maximum(x - k[K - 1], 0) ** 3 * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2]))
        cols.append(num / (k[K - 1] - k[0]) ** 2)
    return np.column_stack(cols)


# ---- linearity ----
ll0 = fit.log_likelihood_
lin = []
for v in CONTINUOUS:
    kn = np.unique(np.percentile(d[v].values, [5, 35, 65, 95]))
    if len(kn) < 4:
        lin.append(dict(predictor=v, df=np.nan, LR_chi2=np.nan, p=np.nan,
                        note="too few distinct knots; variable is near-discrete"))
        continue
    B = rcs(d[v].values, kn)
    dd = X.copy()
    for i in range(B.shape[1]):
        dd["%s_spline%d" % (v, i + 1)] = B[:, i]
    m = CoxPHFitter().fit(dd, "exit", "event")
    lr = 2 * (m.log_likelihood_ - ll0)
    p = stats.chi2.sf(lr, B.shape[1])
    lin.append(dict(predictor=v, df=B.shape[1], LR_chi2=round(lr, 3), p=round(p, 3), note=""))
lin = pd.DataFrame(lin)
print("\nLinearity (restricted cubic splines vs linear):")
print(lin.to_string(index=False))
print("  predictors with evidence of non-linearity:", int((lin["p"] < 0.05).sum()))

ph.to_csv(os.path.join(TAB, "tabS5_proportional_hazards.csv"), index=False)
lin.to_csv(os.path.join(TAB, "tabS5_linearity.csv"), index=False)
print("\nsaved tabS5_proportional_hazards.csv and tabS5_linearity.csv")
