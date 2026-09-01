# 34_decision_curve_internal.py
# Produces: the internal decision-curve analysis.
#
# The decision modelled is whether, at the moment levodopa is started, to counsel
# the patient about the risk of dyskinesia and consider a levodopa-sparing strategy.
# Net benefit is computed from out-of-fold predicted risk at five years, so the
# curve is not the in-sample optimism of the fitted model. Within the group the
# model flags, the event rate is estimated by Kaplan-Meier, which keeps censored
# participants in the calculation.
#
#     net benefit = P(flagged) * [ observed(flagged) - (1 - observed(flagged)) * pt/(1-pt) ]

import json
import os

import numpy as np
import pandas as pd

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
HORIZON = 5

df = C.build_cohort()
model = C.fit_total6(df)
centre = float(C.linear_predictor(model, df).mean())
lp_oof = C.out_of_fold_linear_predictor(df)
risk = C.absolute_risk(model, lp_oof, HORIZON, centre)
df = df.assign(risk=risk)

incidence_all = C.km_incidence(df, HORIZON)
print(f"cohort: n={len(df)}, events={int(df.event.sum())}")
print(f"observed cumulative incidence at {HORIZON} years: {100 * incidence_all:.1f}%")

rows = []
for threshold in np.arange(0.05, 0.505, 0.01):
    flagged = df[df.risk >= threshold]
    proportion = len(flagged) / len(df)
    if len(flagged) < 5 or flagged.event.sum() < 1:
        net_benefit = 0.0
    else:
        observed = C.km_incidence(flagged, HORIZON)
        odds = threshold / (1 - threshold)
        net_benefit = observed * proportion - (1 - observed) * proportion * odds
    treat_all = incidence_all - (1 - incidence_all) * (threshold / (1 - threshold))
    rows.append({"threshold": round(float(threshold), 3),
                 "net_benefit_model": float(net_benefit),
                 "net_benefit_treat_all": float(treat_all),
                 "net_benefit_treat_none": 0.0,
                 "proportion_flagged": float(proportion)})

curve = pd.DataFrame(rows)
curve.to_csv(os.path.join(OUT, "tab34_decision_curve.csv"), index=False)

print(f"\n{'threshold':>10s}{'model':>10s}{'treat all':>11s}{'treat none':>12s}{'flagged':>10s}")
for _, row in curve[curve.threshold.round(2).isin([0.05, 0.10, 0.15, 0.20, 0.24, 0.30, 0.35, 0.40, 0.50])].iterrows():
    print(f"{100 * row.threshold:9.0f}%{row.net_benefit_model:10.4f}"
          f"{row.net_benefit_treat_all:11.4f}{0.0:12.4f}{100 * row.proportion_flagged:9.0f}%")

superior = curve[(curve.net_benefit_model > curve.net_benefit_treat_all) & (curve.net_benefit_model > 0)]

# The first version of this reported the lowest and the highest threshold at which the
# model wins, which is only the same thing as a range if the winning thresholds are
# contiguous. Here they are not: the model falls below treating everyone between 11% and
# 14%, and its net benefit dips just below zero at 47% and 48%. Reporting 6% to 50% as a
# single band was therefore wrong, and it reached the manuscript. The bands are now
# computed by walking the sorted thresholds and breaking wherever one is skipped.
step = float(np.round(np.diff(np.sort(curve.threshold.unique())).min(), 4))
bands, start, prev = [], None, None
for t in sorted(superior.threshold):
    if start is None:
        start = t
    elif t - prev > step * 1.5:
        bands.append((start, prev))
        start = t
    prev = t
if start is not None:
    bands.append((start, prev))
print("\nthe model gives higher net benefit than both default strategies at " +
      ", ".join(f"{100 * a:.0f}% to {100 * b:.0f}%" for a, b in bands))
missed = curve[~curve.threshold.isin(superior.threshold)]
for _, r in missed.iterrows():
    why = ("treating everyone is better" if r.net_benefit_model <= r.net_benefit_treat_all
           else "net benefit is not above zero")
    print(f"   not at {100 * r.threshold:.0f}%: {why}")

at20 = curve[curve.threshold.round(2) == 0.20].iloc[0]
gain = at20.net_benefit_model - at20.net_benefit_treat_all
avoided = 1000 * gain / (0.20 / 0.80)
print(f"at a threshold of 20%: net benefit {at20.net_benefit_model:+.4f} against "
      f"{at20.net_benefit_treat_all:+.4f} for treating all, equivalent to avoiding "
      f"{avoided:.0f} false positives per 1,000 patients without missing any true positive")

summary = {
    "horizon": HORIZON,
    "observed_incidence": float(incidence_all),
    "superior_bands": [[float(a), float(b)] for a, b in bands],
    "superior_from": float(superior.threshold.min()),
    "superior_to": float(superior.threshold.max()),
    "at_20pct": {"model": float(at20.net_benefit_model),
                 "treat_all": float(at20.net_benefit_treat_all),
                 "flagged": float(at20.proportion_flagged),
                 "false_positives_avoided_per_1000": float(avoided)},
}
with open(os.path.join(OUT, "tab34_decision_curve.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"\nwritten to {OUT}")
