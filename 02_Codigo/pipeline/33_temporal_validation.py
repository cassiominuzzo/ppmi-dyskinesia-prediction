# 33_temporal_validation.py
# Produces: the temporal-validation subsection of the Results.
#
# PPMI recruited in two waves and no participant belongs to both. The model is
# refitted in the earlier wave alone, the coefficients and the baseline hazard are
# frozen, and the frozen model is applied to the later wave, which it has never
# seen. This is a validation in participants, in time and in part in place, since
# the later wave includes sites that contributed nobody to the earlier one.

import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
BOOTSTRAP = 2000
HORIZONS = (2, 3)

df = C.build_cohort()
df = df[df.enroll_phase.notna()].reset_index(drop=True)
early = df[df.enroll_phase == 1].reset_index(drop=True)
late = df[df.enroll_phase == 2].reset_index(drop=True)

print(f"earlier wave (development): n={len(early)}, events={int(early.event.sum())}, "
      f"sites={early.SITE.nunique()}")
print(f"later wave (validation):    n={len(late)}, events={int(late.event.sum())}, "
      f"sites={late.SITE.nunique()}")
sites_early, sites_late = set(early.SITE), set(late.SITE)
print(f"sites contributing only to the later wave: {len(sites_late - sites_early)}")
print(f"median follow-up: earlier {early.exit.median():.2f} years, later {late.exit.median():.2f} years")
print(f"maximum follow-up in the later wave: {late.exit.max():.2f} years")

model = C.fit_total6(early)
lp_late = C.linear_predictor(model, late)
c_late = concordance_index(late.exit, -lp_late, late.event)

rng = np.random.default_rng(7)
boot = []
for _ in range(BOOTSTRAP):
    idx = rng.integers(0, len(late), len(late))
    sample = late.iloc[idx]
    if sample.event.sum() < 3:
        continue
    try:
        boot.append(concordance_index(sample.exit, -C.linear_predictor(model, sample), sample.event))
    except Exception:
        pass
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nC in the later wave: {c_late:.3f} (95% CI {lo:.3f} to {hi:.3f}, {len(boot)} valid resamples)")

frame = pd.DataFrame({"lp": lp_late, "exit": late.exit.values, "event": late.event.values})
slope_fit = CoxPHFitter().fit(frame, "exit", "event")
slope_row = slope_fit.summary.loc["lp"]
print(f"calibration slope:   {slope_fit.params_['lp']:.3f} "
      f"(95% CI {slope_row['coef lower 95%']:.3f} to {slope_row['coef upper 95%']:.3f}), "
      f"p = {slope_row['p']:.3f}")

centre = float(C.linear_predictor(model, early).mean())
calibration = {}
for horizon in HORIZONS:
    predicted = C.absolute_risk(model, lp_late, horizon, centre)
    observed = C.km_incidence(late, horizon)
    at_risk = int((late.exit >= horizon).sum())
    calibration[horizon] = {"observed": observed, "predicted_mean": float(predicted.mean()),
                            "OE": float(observed / predicted.mean()), "at_risk": at_risk}
    print(f"  {horizon} years: observed {100 * observed:.1f}%, predicted {100 * predicted.mean():.1f}%, "
          f"O:E {observed / predicted.mean():.2f} ({at_risk} still at risk)")

print("\ncumulative incidence by wave, for the case-mix argument:")
for label, wave in (("earlier", early), ("later", late)):
    values = [100 * C.km_incidence(wave, h) for h in (1, 2, 3)]
    print(f"  {label:8s} 1 year {values[0]:5.1f}%  2 years {values[1]:5.1f}%  3 years {values[2]:5.1f}%")

print("\nbaseline case-mix (median unless stated):")
for variable in C.TOTAL6:
    print(f"  {variable:18s} earlier {early[variable].median():7.2f}   later {late[variable].median():7.2f}")

full = C.fit_total6(df)
print("\nhazard ratios per unit, earlier wave against the full cohort:")
for variable in C.TOTAL6:
    print(f"  {variable:18s} {np.exp(model.params_[variable]):.4f}   {np.exp(full.params_[variable]):.4f}")

summary = {
    "early": {"n": len(early), "events": int(early.event.sum()), "sites": int(early.SITE.nunique()),
              "median_followup": float(early.exit.median())},
    "late": {"n": len(late), "events": int(late.event.sum()), "sites": int(late.SITE.nunique()),
             "sites_new": len(sites_late - sites_early),
             "median_followup": float(late.exit.median()), "max_followup": float(late.exit.max())},
    "C": float(c_late), "C_lo": float(lo), "C_hi": float(hi),
    "slope": float(slope_fit.params_["lp"]),
    "slope_lo": float(slope_row["coef lower 95%"]), "slope_hi": float(slope_row["coef upper 95%"]),
    "slope_p": float(slope_row["p"]),
    "calibration": {str(k): v for k, v in calibration.items()},
    "coefficients_early": {v: float(np.exp(model.params_[v])) for v in C.TOTAL6},
    "coefficients_full": {v: float(np.exp(full.params_[v])) for v in C.TOTAL6},
}
with open(os.path.join(OUT, "tab33_temporal_validation.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"\nwritten to {OUT}")
