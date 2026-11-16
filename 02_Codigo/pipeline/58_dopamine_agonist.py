# 58_dopamine_agonist.py
# Produces: tab58_dopamine_agonist.json, the dopamine-agonist rows of Supplementary Table
#           S11: agonist use at time-zero, unadjusted and adjusted for the six predictors,
#           with the change in out-of-fold C-index when it is added to them.
#
# Why this script exists. The Results report that dopamine-agonist use was not associated
# with the outcome, adjusted HR 0.96 (95% CI 0.71 to 1.29) and unadjusted 1.09 (0.80 to
# 1.49). Those values came from the August 2026 audit, which corrected the agonist search
# (227 users, not 226 or 220), but no script in the pipeline produced them and no table
# carried them. This script recomputes both with the method that 31_amantadine_handling.py
# uses for amantadine, so the two treatments are reported the same way: an unpenalised Cox
# model with agonist use alone, and agonist use added to the six predictors, standardised,
# under the ridge penalty of the final model. It stops if the published values do not
# reproduce.
#
# Agonist use comes from tab27_treatments_per_patient.csv, written by
# 27_treatments_at_time_zero.py with the audited drug-name searches of cohort.py.
#
# Usage:
#   python 58_dopamine_agonist.py

import json
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

import cohort as C

TAB = os.path.join(C.RESULTS, "Tabelas")
TREATMENTS = os.path.join(TAB, "tab27_treatments_per_patient.csv")
OUT = os.path.join(TAB, "tab58_dopamine_agonist.json")


def adjusted_hr(data, extra):
    """Same procedure as hazard_ratio() in 31_amantadine_handling.py."""
    cols = list(C.TOTAL6)
    d = data[cols + [extra, "exit", "event"]].copy()
    for c in cols:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    fit = CoxPHFitter(penalizer=C.RIDGE).fit(d, "exit", "event")
    row = fit.summary.loc[extra]
    return (float(np.exp(fit.params_[extra])), float(row["exp(coef) lower 95%"]),
            float(row["exp(coef) upper 95%"]), float(row["p"]))


def main():
    if not os.path.exists(TREATMENTS):
        raise SystemExit("run 27_treatments_at_time_zero.py first: %s is missing" % os.path.basename(TREATMENTS))
    df = C.build_cohort(with_metadata=False)
    df["PATNO"] = pd.to_numeric(df.PATNO, errors="coerce").astype("Int64")
    t = pd.read_csv(TREATMENTS)
    t["PATNO"] = pd.to_numeric(t.PATNO, errors="coerce").astype("Int64")
    df = df.merge(t[["PATNO", "agonist"]], on="PATNO", how="left")
    df["AG"] = df.agonist.fillna(False).astype(bool).astype(int)
    assert (len(df), int(df.event.sum()), int(df.AG.sum())) == (813, 165, 227), (len(df), int(df.AG.sum()))

    crude = CoxPHFitter().fit(df[["AG", "exit", "event"]], "exit", "event")
    r = crude.summary.loc["AG"]
    unadj = (float(np.exp(crude.params_["AG"])), float(r["exp(coef) lower 95%"]),
             float(r["exp(coef) upper 95%"]), float(r["p"]))
    adj = adjusted_hr(df, "AG")
    c_ref = C.out_of_fold_c(df)
    c_with = C.out_of_fold_c(df, C.TOTAL6 + ["AG"])
    print("agonist users: %d of %d (%.1f%%)" % (df.AG.sum(), len(df), 100 * df.AG.mean()))
    print("unadjusted HR %.2f (%.2f to %.2f), p = %.3f" % unadj)
    print("adjusted HR   %.2f (%.2f to %.2f), p = %.3f, change in out-of-fold C %+.4f"
          % (adj + (c_with - c_ref,)))
    published = {"unadjusted": "1.09 (0.80 to 1.49)", "adjusted": "0.96 (0.71 to 1.29)"}
    got = {"unadjusted": "%.2f (%.2f to %.2f)" % unadj[:3], "adjusted": "%.2f (%.2f to %.2f)" % adj[:3]}
    assert got == published, ("the published agonist hazard ratios do not reproduce", got, published)
    print("both published hazard ratios reproduce")
    json.dump({"n": len(df), "events": int(df.event.sum()), "users": int(df.AG.sum()),
               "unadjusted_HR": unadj[0], "unadjusted_lo": unadj[1], "unadjusted_hi": unadj[2], "unadjusted_p": unadj[3],
               "adjusted_HR": adj[0], "adjusted_lo": adj[1], "adjusted_hi": adj[2], "adjusted_p": adj[3],
               "C_reference": c_ref, "C_with_agonist": c_with, "delta_C": c_with - c_ref},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
