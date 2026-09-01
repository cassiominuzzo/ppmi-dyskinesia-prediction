# 44_classification_metrics.py
# Produces: sensitivity, specificity and predictive values at clinically usable
#           thresholds of five-year predicted risk.
#
# Decision-curve analysis is the better tool for judging whether a model is worth
# acting on, because it weighs a false positive against a false negative at each
# threshold rather than treating them as equivalent. But clinicians and reviewers
# ask for the classification table, and it answers a different and legitimate
# question: if I flag everyone above this risk, how many of the people who go on
# to develop dyskinesia will I have caught, and how many of those I flag will
# actually develop it?
#
# Censoring is handled by estimating the incidence at five years separately inside
# the flagged and unflagged groups by Kaplan-Meier, rather than by counting events,
# which would treat a participant censored at two years as a non-event:
#
#     sensitivity = P(flagged) * incidence(flagged) / incidence(overall)
#     specificity = P(not flagged) * (1 - incidence(not flagged)) / (1 - incidence(overall))
#     positive predictive value = incidence(flagged)
#     negative predictive value = 1 - incidence(not flagged)
#
# Predicted risk is out of fold, so no participant is classified by a model that
# saw them.

import json
import os

import numpy as np
import pandas as pd

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
HORIZON = 5
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

df = C.build_cohort()
model = C.fit_total6(df)
centre = float(C.linear_predictor(model, df).mean())
lp_oof = C.out_of_fold_linear_predictor(df)
df = df.assign(risk=C.absolute_risk(model, lp_oof, HORIZON, centre))

overall = C.km_incidence(df, HORIZON)
print(f"cohort: n={len(df)}, events={int(df.event.sum())}")
print(f"observed cumulative incidence at {HORIZON} years: {100 * overall:.1f}%")
print(f"predicted risk, median {100 * df.risk.median():.1f}%, "
      f"range {100 * df.risk.min():.1f}% to {100 * df.risk.max():.1f}%")

rows = []
for threshold in THRESHOLDS:
    flagged = df[df.risk >= threshold]
    unflagged = df[df.risk < threshold]
    if len(flagged) < 20 or len(unflagged) < 20:
        continue
    proportion = len(flagged) / len(df)
    incidence_flagged = C.km_incidence(flagged, HORIZON)
    incidence_unflagged = C.km_incidence(unflagged, HORIZON)
    sensitivity = proportion * incidence_flagged / overall
    specificity = (1 - proportion) * (1 - incidence_unflagged) / (1 - overall)
    rows.append({
        "threshold": threshold,
        "flagged_n": len(flagged),
        "flagged_pct": 100 * proportion,
        "sensitivity": 100 * sensitivity,
        "specificity": 100 * specificity,
        "ppv": 100 * incidence_flagged,
        "npv": 100 * (1 - incidence_unflagged),
        "number_needed_to_flag": 1 / incidence_flagged if incidence_flagged > 0 else np.nan,
    })

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "tab44_classification_metrics.csv"), index=False)

print(f"\n{'threshold':>10s}{'flagged':>10s}{'sens':>8s}{'spec':>8s}{'PPV':>8s}{'NPV':>8s}{'NNF':>7s}")
for row in rows:
    print(f"{100 * row['threshold']:9.0f}%{row['flagged_n']:6d} "
          f"({row['flagged_pct']:2.0f}%){row['sensitivity']:8.1f}{row['specificity']:8.1f}"
          f"{row['ppv']:8.1f}{row['npv']:8.1f}{row['number_needed_to_flag']:7.1f}")

print("\nNNF is the number needed to flag: how many patients must be flagged for one")
print("of them to develop problematic dyskinesia within five years.")
print(f"With no model, flagging everyone gives a positive predictive value equal to the")
print(f"overall incidence, {100 * overall:.1f}%, and a number needed to flag of {1 / overall:.1f}.")

best = max(rows, key=lambda r: r["sensitivity"] + r["specificity"])
print(f"\nthe threshold maximising sensitivity plus specificity is "
      f"{100 * best['threshold']:.0f}%, giving {best['sensitivity']:.0f}% sensitivity and "
      f"{best['specificity']:.0f}% specificity")
print("No single threshold is recommended: the right one depends on what follows a")
print("positive result, which is why the decision curve in script 34 is the primary")
print("statement of clinical usefulness.")

summary = {"horizon": HORIZON, "overall_incidence": float(overall),
           "rows": rows, "best_youden_threshold": float(best["threshold"])}
with open(os.path.join(OUT, "tab44_classification_metrics.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"\nwritten to {OUT}")
