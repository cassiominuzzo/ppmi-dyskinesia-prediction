# 36_sample_size_riley.py
# Produces: the sample-size justification required by TRIPOD+AI.
#
# The three criteria of Riley and colleagues for the minimum sample size of a
# prognostic model with a time-to-event outcome:
#   1. expected shrinkage of the coefficients no greater than 10%
#   2. difference between apparent and adjusted Nagelkerke R2 no greater than 0.05
#   3. the overall risk estimated with a margin of error no greater than 5 points
#
# The first two use the apparent Cox-Snell R2, obtained from the likelihood-ratio
# statistic of the unpenalised model, since the criteria are defined for maximum
# likelihood estimates.

import json
import os

import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
SHRINKAGE_TARGET = 0.90
R2_DIFFERENCE_TARGET = 0.05
RISK_MARGIN_TARGET = 0.05

df = C.build_cohort()
n = len(df)
events = int(df.event.sum())
parameters = len(C.TOTAL6)

model = CoxPHFitter(penalizer=0.0).fit(df[C.TOTAL6 + ["exit", "event"]], "exit", "event")
lr = float(model.log_likelihood_ratio_test().test_statistic)
loglik_null = float(model.log_likelihood_) - lr / 2

r2_cox_snell = 1 - np.exp(-lr / n)
r2_max = 1 - np.exp(2 * loglik_null / n)
r2_nagelkerke = r2_cox_snell / r2_max

print(f"n = {n}, events = {events}, parameters = {parameters}, "
      f"events per parameter = {events / parameters:.1f}")
print(f"likelihood-ratio statistic = {lr:.2f}")
print(f"apparent Cox-Snell R2 = {r2_cox_snell:.4f}, maximum possible = {r2_max:.4f}, "
      f"Nagelkerke R2 = {r2_nagelkerke:.4f}")


def required_n(shrinkage, r2):
    return parameters / ((shrinkage - 1) * np.log(1 - r2 / shrinkage))


n_criterion_1 = required_n(SHRINKAGE_TARGET, r2_cox_snell)
shrinkage_2 = r2_cox_snell / (r2_cox_snell + R2_DIFFERENCE_TARGET * r2_max)
n_criterion_2 = required_n(shrinkage_2, r2_cox_snell)

print(f"\ncriterion 1, expected shrinkage no greater than "
      f"{100 * (1 - SHRINKAGE_TARGET):.0f}%: minimum n = {n_criterion_1:.0f} "
      f"-> {'met' if n >= n_criterion_1 else 'NOT met'}")
print(f"criterion 2, difference in Nagelkerke R2 no greater than {R2_DIFFERENCE_TARGET}: "
      f"implied shrinkage {shrinkage_2:.3f}, minimum n = {n_criterion_2:.0f} "
      f"-> {'met' if n >= n_criterion_2 else 'NOT met'}")

km = KaplanMeierFitter().fit(df.exit, df.event)
interval = km.confidence_interval_survival_function_
criterion_3 = {}
for horizon in (3, 5):
    idx = max(np.searchsorted(interval.index.values, horizon, side="right") - 1, 0)
    lo = 1 - float(interval.iloc[idx, 1])
    hi = 1 - float(interval.iloc[idx, 0])
    incidence = C.km_incidence(df, horizon)
    margin = (hi - lo) / 2
    criterion_3[horizon] = {"incidence": incidence, "lo": lo, "hi": hi, "margin": margin,
                            "met": bool(margin <= RISK_MARGIN_TARGET)}
    print(f"criterion 3 at {horizon} years: incidence {100 * incidence:.1f}% "
          f"(95% CI {100 * lo:.1f} to {100 * hi:.1f}), margin {100 * margin:.1f} points "
          f"-> {'met' if margin <= RISK_MARGIN_TARGET else 'NOT met'}")

van_houwelingen = 1 - parameters / lr
print(f"\nheuristic shrinkage factor of van Houwelingen = {van_houwelingen:.3f}")

summary = {
    "n": n, "events": events, "parameters": parameters,
    "events_per_parameter": events / parameters,
    "likelihood_ratio": lr,
    "r2_cox_snell": float(r2_cox_snell), "r2_max": float(r2_max),
    "r2_nagelkerke": float(r2_nagelkerke),
    "criterion_1_n": float(n_criterion_1), "criterion_1_met": bool(n >= n_criterion_1),
    "criterion_2_shrinkage": float(shrinkage_2), "criterion_2_n": float(n_criterion_2),
    "criterion_2_met": bool(n >= n_criterion_2),
    "criterion_3": {str(k): v for k, v in criterion_3.items()},
    "van_houwelingen_shrinkage": float(van_houwelingen),
}
with open(os.path.join(OUT, "tab36_sample_size_riley.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"\nwritten to {OUT}")
