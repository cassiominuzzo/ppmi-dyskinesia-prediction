# 32_internal_performance.py
# Produces: the internal-performance paragraph of the Results.
#           Apparent and optimism-corrected C-index, apparent, optimism-corrected and
#           out-of-fold calibration slope, calibration at 3, 5 and 7 years with the
#           integrated calibration index, and leave-one-site-out cross-validation.
#
# Optimism is corrected by the bootstrap of Harrell: a model is refitted in each
# resample, its performance is measured in that resample and in the original data,
# and the mean difference is subtracted from the apparent value.

import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
BOOTSTRAP = 500
HORIZONS = (3, 5, 7)

df = C.build_cohort()
print(f"cohort: n={len(df)}, events={int(df.event.sum())}")

apparent_model = C.fit_total6(df)
apparent_lp = C.linear_predictor(apparent_model, df)
c_apparent = concordance_index(df.exit, -apparent_lp, df.event)


def calibration_slope(lp, data):
    frame = pd.DataFrame({"lp": lp, "exit": data.exit.values, "event": data.event.values})
    return float(CoxPHFitter().fit(frame, "exit", "event").params_["lp"])


slope_apparent = calibration_slope(apparent_lp, df)

rng = np.random.default_rng(C.BOOT_SEED)
optimism_c, optimism_slope = [], []
for _ in range(BOOTSTRAP):
    idx = rng.integers(0, len(df), len(df))
    resample = df.iloc[idx].reset_index(drop=True)
    try:
        model = C.fit_total6(resample)
    except Exception:
        continue
    lp_in = C.linear_predictor(model, resample)
    lp_out = C.linear_predictor(model, df)
    optimism_c.append(
        concordance_index(resample.exit, -lp_in, resample.event)
        - concordance_index(df.exit, -lp_out, df.event)
    )
    try:
        optimism_slope.append(calibration_slope(lp_in, resample) - calibration_slope(lp_out, df))
    except Exception:
        pass

optimism_c = np.asarray(optimism_c)
c_corrected = c_apparent - optimism_c.mean()
lo, hi = np.percentile(c_apparent - optimism_c, [2.5, 97.5])
slope_corrected = slope_apparent - float(np.mean(optimism_slope))

print(f"\napparent C            {c_apparent:.4f}")
print(f"optimism              {optimism_c.mean():.4f}  ({len(optimism_c)} resamples)")
print(f"optimism-corrected C  {c_corrected:.4f} (95% CI {lo:.3f} to {hi:.3f})")

c_out_of_fold = C.out_of_fold_c(df)
lp_oof = C.out_of_fold_linear_predictor(df)
slope_oof = calibration_slope(lp_oof, df)
print(f"out-of-fold C         {c_out_of_fold:.4f}")
print(f"\ncalibration slope: apparent {slope_apparent:.3f}, "
      f"optimism-corrected {slope_corrected:.3f}, out of fold {slope_oof:.3f}")


def restricted_cubic_spline(x, knots):
    knots = np.asarray(knots, dtype=float)
    k = len(knots)
    scale = (knots[-1] - knots[0]) ** (2 / 3)
    basis = [x]
    for j in range(k - 2):
        term = (
            np.maximum(x - knots[j], 0) ** 3
            - np.maximum(x - knots[k - 2], 0) ** 3 * (knots[-1] - knots[j]) / (knots[-1] - knots[k - 2])
            + np.maximum(x - knots[-1], 0) ** 3 * (knots[k - 2] - knots[j]) / (knots[-1] - knots[k - 2])
        )
        basis.append(term / scale)
    return np.column_stack(basis)


centre = float(apparent_lp.mean())
calibration = {}
print()
for horizon in HORIZONS:
    predicted = np.clip(C.absolute_risk(apparent_model, lp_oof, horizon, centre), 1e-6, 1 - 1e-6)
    cloglog = np.log(-np.log(1 - predicted))
    basis = restricted_cubic_spline(cloglog, np.quantile(cloglog, [0.10, 0.50, 0.90]))
    names = [f"z{i}" for i in range(basis.shape[1])]
    frame = pd.DataFrame(basis, columns=names)
    frame["exit"] = df.exit.values
    frame["event"] = df.event.values
    flexible_fit = CoxPHFitter().fit(frame, "exit", "event")
    s0 = C.baseline_survival_at(flexible_fit, horizon)
    lp_flex = basis @ flexible_fit.params_[names].values
    observed_flexible = 1 - s0 ** np.exp(lp_flex - lp_flex.mean())
    absolute = np.abs(observed_flexible - predicted)
    observed = C.km_incidence(df, horizon)
    calibration[horizon] = {
        "observed": observed,
        "predicted_mean": float(predicted.mean()),
        "OE": float(observed / predicted.mean()),
        "ICI": float(absolute.mean()),
        "E50": float(np.median(absolute)),
        "E90": float(np.quantile(absolute, 0.90)),
        "S0": C.baseline_survival_at(apparent_model, horizon),
    }
    print(f"{horizon} years: observed {100 * observed:.1f}%, predicted {100 * predicted.mean():.1f}%, "
          f"O:E {observed / predicted.mean():.2f}, ICI {100 * absolute.mean():.1f} pp, "
          f"E90 {100 * np.quantile(absolute, 0.90):.1f} pp")

# ------------------------------------------------------- leave one site out
rows = []
for site in sorted(df.SITE.dropna().unique()):
    test = df[df.SITE == site]
    train = df[df.SITE != site]
    if len(test) < 10 or test.event.sum() < 2:
        continue
    try:
        model = C.fit_total6(train)
        score = concordance_index(test.exit, -C.linear_predictor(model, test), test.event)
    except Exception:
        continue
    rows.append({"SITE": int(site), "n": len(test), "events": int(test.event.sum()), "C": float(score)})

sites = pd.DataFrame(rows)
weighted = float(np.average(sites.C, weights=sites.events))
print(f"\nleave-one-site-out: {len(sites)} of {df.SITE.nunique()} sites evaluable "
      f"({sites.n.sum()} participants, {sites.events.sum()} events)")
print(f"  median C {sites.C.median():.3f} (IQR {sites.C.quantile(.25):.3f} to {sites.C.quantile(.75):.3f}), "
      f"event-weighted mean {weighted:.3f}, range {sites.C.min():.3f} to {sites.C.max():.3f}")
print(f"  median events per evaluable site: {sites.events.median():.0f}")

summary = {
    "n": len(df), "events": int(df.event.sum()),
    "C_apparent": float(c_apparent), "optimism": float(optimism_c.mean()),
    "C_corrected": float(c_corrected), "C_corrected_lo": float(lo), "C_corrected_hi": float(hi),
    "C_out_of_fold": float(c_out_of_fold),
    "slope_apparent": slope_apparent, "slope_corrected": slope_corrected, "slope_out_of_fold": slope_oof,
    "bootstrap": int(len(optimism_c)),
    "calibration": {str(k): v for k, v in calibration.items()},
    "loso": {"sites": len(sites), "n": int(sites.n.sum()), "events": int(sites.events.sum()),
             "median": float(sites.C.median()), "q1": float(sites.C.quantile(.25)),
             "q3": float(sites.C.quantile(.75)), "weighted_mean": weighted,
             "min": float(sites.C.min()), "max": float(sites.C.max()),
             "median_events_per_site": float(sites.events.median())},
}
with open(os.path.join(OUT, "tab32_internal_performance.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
sites.to_csv(os.path.join(OUT, "tab32_leave_one_site_out.csv"), index=False)
pd.DataFrame({"PATNO": df.PATNO, "lp_out_of_fold": lp_oof}).to_csv(
    os.path.join(OUT, "tab32_out_of_fold_linear_predictor.csv"), index=False)
print(f"\nwritten to {OUT}")
