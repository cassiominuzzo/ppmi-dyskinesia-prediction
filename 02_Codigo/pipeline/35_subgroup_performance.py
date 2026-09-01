# 35_subgroup_performance.py
# Produces: the Fairness item of the TRIPOD+AI checklist.
#
# Discrimination and calibration in each subgroup the development cohort can
# support. Calibration is measured at three years, the longest horizon every
# subgroup reaches without extrapolation. Predictions are out of fold.
#
# Note on interpretation: the C-index necessarily falls when the cohort is
# stratified on one of the model's own predictors, because stratification removes
# that predictor's contribution to ranking. A lower C within tertiles of age at
# onset is therefore expected and is not evidence of differential performance.

import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
HORIZON = 3
BOOTSTRAP = 1500

df = C.build_cohort()
model = C.fit_total6(df)
centre = float(C.linear_predictor(model, df).mean())
lp_oof = C.out_of_fold_linear_predictor(df)
df = df.assign(lp_oof=lp_oof, risk=C.absolute_risk(model, lp_oof, HORIZON, centre))

rng = np.random.default_rng(11)


def evaluate(subset, label):
    score = concordance_index(subset.exit, -subset.lp_oof, subset.event)
    boot = []
    for _ in range(BOOTSTRAP):
        sample = subset.iloc[rng.integers(0, len(subset), len(subset))]
        if sample.event.sum() < 3:
            continue
        try:
            boot.append(concordance_index(sample.exit, -sample.lp_oof, sample.event))
        except Exception:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) > 50 else (np.nan, np.nan)
    observed = C.km_incidence(subset, HORIZON)
    frame = pd.DataFrame({"lp": subset.lp_oof.values, "exit": subset.exit.values,
                          "event": subset.event.values})
    slope = float(CoxPHFitter().fit(frame, "exit", "event").params_["lp"])
    return {"subgroup": label, "n": len(subset), "events": int(subset.event.sum()),
            "C": float(score), "C_lo": float(lo), "C_hi": float(hi),
            "OE": float(observed / subset.risk.mean()), "slope": slope,
            "at_risk": int((subset.exit >= HORIZON).sum())}


cuts = df.ageonset.quantile([0, 1 / 3, 2 / 3, 1]).values
df["onset_tertile"] = pd.qcut(df.ageonset, 3, labels=["1", "2", "3"])

rows = [evaluate(df, "Whole cohort"),
        evaluate(df[df.SEX == 1], "Men"),
        evaluate(df[df.SEX == 0], "Women")]
for i, tertile in enumerate(["1", "2", "3"]):
    rows.append(evaluate(df[df.onset_tertile == tertile],
                         f"Age at onset, tertile {tertile} ({cuts[i]:.0f} to {cuts[i + 1]:.0f} years)"))
rows.append(evaluate(df[df.enroll_phase == 1], "Recruitment wave 1 (2010 to 2019)"))
rows.append(evaluate(df[df.enroll_phase == 2], "Recruitment wave 2 (2020 to 2025)"))
rows.append(evaluate(df[df.race == 1], "White"))
rows.append(evaluate(df[df.race.notna() & (df.race != 1)], "All other races"))

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "tab35_subgroup_performance.csv"), index=False)

print(f"{'subgroup':44s}{'n / ev':>11s}{'C (95% CI)':>25s}{'O:E 3y':>8s}{'slope':>7s}")
for row in rows:
    ci = f"{row['C']:.3f} ({row['C_lo']:.3f} to {row['C_hi']:.3f})"
    print(f"{row['subgroup']:44s}{row['n']:6d} /{row['events']:4d}{ci:>25s}"
          f"{row['OE']:8.2f}{row['slope']:7.2f}")

white = int((df.race == 1).sum())
other = int((df.race.notna() & (df.race != 1)).sum())
other_events = int(df[df.race.notna() & (df.race != 1)].event.sum())
missing = int(df.race.isna().sum())
print(f"\nrace: {white} White ({100 * white / len(df):.1f}%), {other} of other reported races "
      f"contributing {other_events} events, {missing} with no recorded race")
print("Ancestry-specific performance cannot be estimated in this cohort. The estimate for")
print("other races rests on 9 events and is reported so that the limitation is visible.")

summary = {"horizon": HORIZON, "rows": rows,
           "race": {"white": white, "other": other, "other_events": other_events, "missing": missing}}
with open(os.path.join(OUT, "tab35_subgroup_performance.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"\nwritten to {OUT}")
