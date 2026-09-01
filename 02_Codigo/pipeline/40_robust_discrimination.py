# 40_robust_discrimination.py
# Produces: censoring-robust measures of discrimination and overall accuracy.
#
# Harrell's concordance depends on the censoring distribution of the sample in
# which it is computed, and this cohort is heavily censored: 165 events in 813
# with a median follow-up of 2.4 years. Uno's concordance weights pairs by the
# inverse probability of remaining uncensored and is consistent under that
# censoring, so it is the value that should be quoted alongside Harrell's.
#
# The cumulative-dynamic AUC gives discrimination at the horizons the calculator
# actually reports, rather than a single figure summed over all follow-up times.
#
# The Brier score is the mean squared error of the predicted survival probability,
# so it captures discrimination and calibration together. The index of prediction
# accuracy rescales it against a Kaplan-Meier model that uses no covariates:
#
#     IPA = 1 - Brier(model) / Brier(Kaplan-Meier)
#
# All three are computed out of fold: within each training fold the model is
# refitted and the survival function is predicted for the held-out participants,
# so nothing is evaluated in the data that fitted it.

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (brier_score, concordance_index_censored,
                            concordance_index_ipcw, cumulative_dynamic_auc)
from sksurv.util import Surv

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
HORIZONS = np.array([3.0, 5.0, 7.0])
REPEATS = 10

df = C.build_cohort()
y = Surv.from_arrays(df.event.astype(bool).values, df.exit.values)
x = df[C.TOTAL6].values
print(f"cohort: n={len(df)}, events={int(df.event.sum())}, "
      f"censored {100 * (1 - df.event.mean()):.1f}%")
print(f"median follow-up {df.exit.median():.2f} years, "
      f"at risk at 3/5/7 years: {[int((df.exit >= h).sum()) for h in HORIZONS]}")

harrell, uno, auc_by_horizon, brier_model, brier_null = [], [], [], [], []

for r in range(REPEATS):
    risk = np.zeros(len(df))
    survival = np.zeros((len(df), len(HORIZONS)))
    for train, test in StratifiedKFold(C.CV_FOLDS, shuffle=True,
                                       random_state=C.CV_SEED + r).split(x, df.event):
        scaler = StandardScaler().fit(x[train])
        model = CoxPHSurvivalAnalysis(alpha=1.0).fit(scaler.transform(x[train]), y[train])
        risk[test] = model.predict(scaler.transform(x[test]))
        functions = model.predict_survival_function(scaler.transform(x[test]))
        for i, fn in enumerate(functions):
            survival[test[i], :] = fn(HORIZONS)

    harrell.append(concordance_index_censored(df.event.astype(bool).values,
                                              df.exit.values, risk)[0])
    uno.append(concordance_index_ipcw(y, y, risk, tau=float(HORIZONS[-1]))[0])
    auc, _ = cumulative_dynamic_auc(y, y, risk, HORIZONS)
    auc_by_horizon.append(auc)
    times, scores = brier_score(y, y, survival, HORIZONS)
    brier_model.append(scores)

    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter().fit(df.exit, df.event)
    reference = np.tile(
        np.array([float(km.survival_function_at_times(h).iloc[0]) for h in HORIZONS]),
        (len(df), 1))
    _, null_scores = brier_score(y, y, reference, HORIZONS)
    brier_null.append(null_scores)

harrell = np.asarray(harrell)
uno = np.asarray(uno)
auc_by_horizon = np.vstack(auc_by_horizon)
brier_model = np.vstack(brier_model)
brier_null = np.vstack(brier_null)
ipa = 1 - brier_model.mean(axis=0) / brier_null.mean(axis=0)

print(f"\nHarrell concordance, out of fold: {harrell.mean():.4f} "
      f"(sd across repetitions {harrell.std():.4f})")
print(f"Uno concordance (IPCW, tau = {HORIZONS[-1]:.0f} years): {uno.mean():.4f} "
      f"(sd {uno.std():.4f})")
print(f"difference, Uno minus Harrell: {uno.mean() - harrell.mean():+.4f}")

print(f"\n{'horizon':>9s}{'AUC':>9s}{'Brier':>9s}{'Brier null':>12s}{'IPA':>9s}")
for j, horizon in enumerate(HORIZONS):
    print(f"{horizon:8.0f}y{auc_by_horizon[:, j].mean():9.4f}"
          f"{brier_model[:, j].mean():9.4f}{brier_null[:, j].mean():12.4f}{ipa[j]:9.4f}")

integrated = np.trapz(auc_by_horizon.mean(axis=0), HORIZONS) / (HORIZONS[-1] - HORIZONS[0])
print(f"\nmean AUC across the three horizons: {integrated:.4f}")
print("IPA is the proportional reduction in squared error against a Kaplan-Meier")
print("model with no covariates; it is the overall accuracy measure TRIPOD asks for.")

summary = {
    "n": len(df), "events": int(df.event.sum()),
    "repeats": REPEATS, "horizons": HORIZONS.tolist(),
    "harrell_c": float(harrell.mean()), "harrell_sd": float(harrell.std()),
    "uno_c": float(uno.mean()), "uno_sd": float(uno.std()),
    "auc": {str(int(h)): float(auc_by_horizon[:, j].mean()) for j, h in enumerate(HORIZONS)},
    "brier": {str(int(h)): float(brier_model[:, j].mean()) for j, h in enumerate(HORIZONS)},
    "brier_null": {str(int(h)): float(brier_null[:, j].mean()) for j, h in enumerate(HORIZONS)},
    "ipa": {str(int(h)): float(ipa[j]) for j, h in enumerate(HORIZONS)},
    "mean_auc": float(integrated),
}
with open(os.path.join(OUT, "tab40_robust_discrimination.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
pd.DataFrame({"horizon": HORIZONS,
              "AUC": auc_by_horizon.mean(axis=0),
              "Brier": brier_model.mean(axis=0),
              "Brier_null": brier_null.mean(axis=0),
              "IPA": ipa}).to_csv(os.path.join(OUT, "tab40_by_horizon.csv"), index=False)
print(f"\nwritten to {OUT}")
