# 56_candidate_hazard_ratios.py
# Produces: tab56_candidate_hazard_ratios.csv, Supplementary Table S4 of the November 2026
#           revision: the unadjusted hazard ratio of every one of the 18 routine clinical
#           candidates, and its hazard ratio adjusted for the six predictors of the final
#           model: mutually adjusted for the six themselves, as in Table 2, and for each of
#           the other twelve, as a seventh term added to those six.
#
# Why this script exists. Until November 2026 the table carried the six selected predictors
# only, so a reader could not see what the selection chose from. Showing every candidate,
# as earlier analyses of PPMI did for their own candidates, answers the obvious question of
# why a variable strongly associated with the outcome on its own, such as Part III of the
# MDS-UPDRS, is not in the model. TRIPOD 2015 (item 14b) asks for exactly this when the
# unadjusted associations have been computed. The adjusted column answers the follow-up
# question, whether the candidate still carries information once the six are known; each
# seventh term is fitted with the same ridge penalty as the final model. Because the six were
# chosen on cross-validated discrimination and not on p values, the table also gives, for each
# of the twelve, the change in out-of-fold C-index when it is added to the six, by the
# protocol of cohort.out_of_fold_c used for every comparison of discrimination in the paper.
# The repetitions share their seeds, so the two models are compared on the same folds.
#
# The six selected rows must reproduce tab53_unadjusted_hazard_ratios.csv, the table this one
# replaces, to the last digit: same sample, same unpenalised univariable Cox model, same
# penalised six-variable model, same increments. The script stops if they do not.
#
# Usage:
#   python 56_candidate_hazard_ratios.py

import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

import cohort as C

TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
OUT = os.path.join(TAB, "tab56_candidate_hazard_ratios.csv")
PREVIOUS = os.path.join(TAB, "tab53_unadjusted_hazard_ratios.csv")

# Increments chosen so that one step is clinically readable and close to one standard
# deviation in this sample: sums of many items per 5 or 10 points, single items and stages
# per point, the PIGD subscore (a mean of five items, 0 to 4) per 0.25 points.
CANDIDATES = [
    # the six of the final model, in the order of Table 2
    ("updrs_totscore", "Total MDS-UPDRS (Parts I to III)", 10, "per 10 points"),
    ("ageonset", "Age at onset", 10, "per 10 years"),
    ("SEX", "Sex", 1, "male versus female"),
    ("BMI", "Body-mass index", 5, "per 5 kg/m2"),
    ("td_pigd_ratio", "TD/PIGD ratio", 1, "per unit"),
    ("NP2FREZ", "Freezing of gait (item 2.13)", 1, "per point"),
    # the other twelve, grouped by what they measure
    ("updrs1_score", "MDS-UPDRS Part I", 5, "per 5 points"),
    ("updrs2_score", "MDS-UPDRS Part II", 5, "per 5 points"),
    ("updrs3_score", "MDS-UPDRS Part III", 10, "per 10 points"),
    ("comp_bradicinesia", "Bradykinesia composite (11 Part III items)", 5, "per 5 points"),
    ("pigd", "PIGD subscore", 0.25, "per 0.25 points"),
    ("hy", "Hoehn and Yahr stage", 1, "per stage"),
    ("Clinical_Stage", "PPMI clinical stage", 1, "per stage"),
    ("MSEADLG", "Modified Schwab and England scale", 10, "per 10 points"),
    ("NP1FATG", "Fatigue (item 1.13)", 1, "per point"),
    ("NP1DPRS", "Depressed mood (item 1.3)", 1, "per point"),
    ("scopa_gi", "SCOPA-AUT gastrointestinal subscale", 1, "per point"),
    ("EDUCYRS", "Years of education", 5, "per 5 years"),
]


def hr(model, col, k):
    r = model.summary.loc[col]
    return (float(np.exp(k * r["coef"])), float(np.exp(k * r["coef lower 95%"])),
            float(np.exp(k * r["coef upper 95%"])), float(r["p"]))


def main():
    assert sorted(c for c, *_ in CANDIDATES) == sorted(C.POOL), "the table must cover the 18-candidate pool"
    assert [c for c, *_ in CANDIDATES[:6]] == list(C.TOTAL6)
    d = C.build_cohort(with_metadata=False)
    assert (len(d), int(d.event.sum())) == (813, 165)
    assert not d[C.POOL].isna().any().any(), "the development sample is complete on the pool by definition"

    adjusted = C.fit_total6(d)
    base_c = C.out_of_fold_c(d)
    assert round(base_c, 3) == 0.696, "the out-of-fold C of the six must be the published 0.696"
    rows = []
    for col, label, k, unit in CANDIDATES:
        uni = CoxPHFitter(penalizer=0.0).fit(d[[col, "exit", "event"]], "exit", "event")
        u = hr(uni, col, k)
        row = {"variable": col, "candidate": label, "increment": unit, "selected": col in C.TOTAL6,
               "unadjusted_HR": u[0], "unadjusted_lo": u[1], "unadjusted_hi": u[2], "unadjusted_p": u[3]}
        model = adjusted if col in C.TOTAL6 else C.fit_total6(d, predictors=list(C.TOTAL6) + [col])
        a = hr(model, col, k)
        row.update({"adjusted_HR": a[0], "adjusted_lo": a[1], "adjusted_hi": a[2], "adjusted_p": a[3]})
        row["delta_c"] = (np.nan if col in C.TOTAL6
                          else C.out_of_fold_c(d, predictors=list(C.TOTAL6) + [col]) - base_c)
        rows.append(row)
    table = pd.DataFrame(rows)

    # the six rows must equal the table they replace
    old = pd.read_csv(PREVIOUS)
    new6 = table[table.selected].reset_index(drop=True)
    for c in ["unadjusted_HR", "unadjusted_lo", "unadjusted_hi", "unadjusted_p",
              "adjusted_HR", "adjusted_lo", "adjusted_hi", "adjusted_p"]:
        assert np.allclose(new6[c].values, old[c].values, rtol=0, atol=1e-12), c
    print("the six selected rows reproduce tab53_unadjusted_hazard_ratios.csv exactly")

    for _, r in table.iterrows():
        adj = ("%s %.2f (%.2f to %.2f) p=%.4f" % ("in the model" if r.selected else "added to the six",
                                                  r.adjusted_HR, r.adjusted_lo, r.adjusted_hi, r.adjusted_p))
        print("  %-44s %-18s unadjusted %.2f (%.2f to %.2f) p=%.4f | %s%s"
              % (r.candidate, r.increment, r.unadjusted_HR, r.unadjusted_lo, r.unadjusted_hi,
                 r.unadjusted_p, adj, "" if r.selected else " | change in C %+.4f" % r.delta_c))
    print("out-of-fold C of the six: %.4f" % base_c)
    table.to_csv(OUT, index=False)
    print("\nwritten %s" % OUT)


if __name__ == "__main__":
    main()
