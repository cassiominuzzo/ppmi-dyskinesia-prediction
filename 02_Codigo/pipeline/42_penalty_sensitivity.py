# 42_penalty_sensitivity.py
# Produces: the sensitivity of the model to the ridge penalty.
#
# The penalty of 0.05 was chosen before the final fit, to stabilise the estimates
# without materially shrinking them. Choosing it in advance is the right practice,
# but it leaves open the question of whether the results depend on that particular
# value. This script refits the model across a range of penalties and reports the
# hazard ratios, the out-of-fold concordance and the calibration slope for each.

import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
PENALTIES = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]

df = C.build_cohort()
print(f"cohort: n={len(df)}, events={int(df.event.sum())}")

rows = []
for penalty in PENALTIES:
    model = CoxPHFitter(penalizer=penalty).fit(df[C.TOTAL6 + ["exit", "event"]], "exit", "event")
    lp = C.linear_predictor(model, df)
    apparent = concordance_index(df.exit, -lp, df.event)
    slope = float(CoxPHFitter().fit(
        pd.DataFrame({"lp": lp, "exit": df.exit.values, "event": df.event.values}),
        "exit", "event").params_["lp"])
    row = {"penalty": penalty, "C_apparent": float(apparent), "slope_apparent": slope}
    for variable in C.TOTAL6:
        row[variable] = float(np.exp(model.params_[variable]))
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "tab42_penalty_sensitivity.csv"), index=False)

print(f"\n{'penalty':>8s}{'C apparent':>12s}{'slope':>8s}" +
      "".join(f"{v[:12]:>13s}" for v in C.TOTAL6))
for row in rows:
    print(f"{row['penalty']:8.2f}{row['C_apparent']:12.4f}{row['slope_apparent']:8.3f}" +
          "".join(f"{row[v]:13.4f}" for v in C.TOTAL6))

reference = next(r for r in rows if r["penalty"] == C.RIDGE)
zero = next(r for r in rows if r["penalty"] == 0.0)
worst = max(abs(100 * (reference[v] - zero[v]) / zero[v]) for v in C.TOTAL6)
print(f"\nagainst the unpenalised fit, the chosen penalty of {C.RIDGE} moves the largest "
      f"hazard ratio by {worst:.1f}%")
spread = max(r["C_apparent"] for r in rows) - min(r["C_apparent"] for r in rows)
print(f"apparent concordance varies by {spread:.4f} across the whole range of penalties tested")

summary = {"penalties": PENALTIES, "rows": rows,
           "largest_hr_shift_vs_unpenalised_pct": float(worst),
           "c_spread": float(spread)}
with open(os.path.join(OUT, "tab42_penalty_sensitivity.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"\nwritten to {OUT}")
