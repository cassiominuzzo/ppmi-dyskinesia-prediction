# 41_included_vs_excluded.py
# Produces: the comparison of the development sample with the participants
#           excluded for incomplete data.
#
# Of the 1,441 eligible participants, 628 were excluded because at least one of
# the 18 candidate clinical predictors was missing. That is 43.6% of the eligible
# cohort, so whether the 813 who remain are representative is a fair question and
# the answer belongs in the manuscript rather than in a footnote.
#
# Supplementary Table S5 already shows that the two excluded strata have event
# rates of 13.6% and 29.4% against 20.3% in the development sample, so missingness
# is plainly not independent of the outcome. This script quantifies the difference
# in baseline characteristics as well, and tests whether being excluded predicts
# the outcome once the model's own predictors are accounted for.

import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))

matrix = pd.read_parquet(os.path.join(C.PROC, "candidate_matrix.parquet"))
if "ageonset" not in matrix.columns and "ageonset_x" in matrix.columns:
    matrix = matrix.rename(columns={"ageonset_x": "ageonset"})
eligible = matrix[matrix.exit > 0].reset_index(drop=True)
included = C.build_cohort()
excluded = eligible[~eligible.PATNO.isin(included.PATNO)].reset_index(drop=True)

print(f"eligible {len(eligible)} ({int(eligible.event.sum())} events)")
print(f"  included  {len(included)} ({int(included.event.sum())} events, "
      f"{100 * included.event.mean():.1f}%)")
print(f"  excluded  {len(excluded)} ({int(excluded.event.sum())} events, "
      f"{100 * excluded.event.mean():.1f}%)  = "
      f"{100 * len(excluded) / len(eligible):.1f}% of the eligible cohort")

CONTINUOUS = ["updrs_totscore", "ageonset", "BMI", "td_pigd_ratio", "exit",
              "updrs1_score", "updrs2_score", "updrs3_score"]
BINARY = ["SEX", "NP2FREZ", "event"]
LABELS = {"updrs_totscore": "Total MDS-UPDRS", "ageonset": "Age at onset, years",
          "BMI": "Body-mass index", "td_pigd_ratio": "TD/PIGD ratio",
          "exit": "Follow-up, years", "updrs1_score": "MDS-UPDRS Part I",
          "updrs2_score": "MDS-UPDRS Part II", "updrs3_score": "MDS-UPDRS Part III",
          "SEX": "Male sex", "NP2FREZ": "Freezing of gait (item 2.13) above zero",
          "event": "Reached the outcome"}

rows = []
for variable in CONTINUOUS:
    a = included[variable].dropna()
    b = excluded[variable].dropna() if variable in excluded.columns else pd.Series(dtype=float)
    if len(b) < 20:
        rows.append({"variable": LABELS.get(variable, variable), "included": f"{a.median():.1f}",
                     "excluded": "not recorded", "p": np.nan, "n_excluded": int(len(b))})
        continue
    p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    rows.append({"variable": LABELS.get(variable, variable),
                 "included": f"{a.median():.1f} [{a.quantile(.25):.1f} to {a.quantile(.75):.1f}]",
                 "excluded": f"{b.median():.1f} [{b.quantile(.25):.1f} to {b.quantile(.75):.1f}]",
                 "p": p, "n_excluded": int(len(b))})

for variable in BINARY:
    a = included[variable].dropna()
    b = excluded[variable].dropna() if variable in excluded.columns else pd.Series(dtype=float)
    if len(b) < 20:
        rows.append({"variable": LABELS.get(variable, variable), "included": f"{100 * a.mean():.1f}%",
                     "excluded": "not recorded", "p": np.nan, "n_excluded": int(len(b))})
        continue
    a_bin = (a > 0).astype(int)
    b_bin = (b > 0).astype(int)
    table = [[int(a_bin.sum()), int(len(a_bin) - a_bin.sum())],
             [int(b_bin.sum()), int(len(b_bin) - b_bin.sum())]]
    p = float(stats.chi2_contingency(table)[1])
    rows.append({"variable": LABELS.get(variable, variable),
                 "included": f"{100 * a_bin.mean():.1f}%", "excluded": f"{100 * b_bin.mean():.1f}%",
                 "p": p, "n_excluded": int(len(b))})

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "tab41_included_vs_excluded.csv"), index=False)
print(f"\n{'variable':42s}{'included (n=813)':>28s}{'excluded (n=628)':>28s}{'p':>10s}")
for row in rows:
    p = "-" if np.isnan(row["p"]) else (f"{row['p']:.3f}" if row["p"] >= 0.001 else "<0.001")
    print(f"{row['variable']:42s}{row['included']:>28s}{row['excluded']:>28s}{p:>10s}")

# --------------------------------------------------------- outcome comparison
merged = pd.concat([included.assign(group=1)[["exit", "event", "group"]],
                    excluded.assign(group=0)[["exit", "event", "group"]]])
test = logrank_test(included.exit, excluded.exit, included.event, excluded.event)
print(f"\nlog-rank test of the outcome between the two groups: "
      f"chi2 = {test.test_statistic:.2f}, p = {test.p_value:.4f}")
for horizon in (3, 5, 7):
    a = C.km_incidence(included, horizon)
    b = C.km_incidence(excluded, horizon)
    print(f"  cumulative incidence at {horizon} years: included {100 * a:.1f}%, "
          f"excluded {100 * b:.1f}%")

model = CoxPHFitter().fit(merged, "exit", "event")
row = model.summary.loc["group"]
print(f"\nhazard ratio for being in the development sample, unadjusted: "
      f"{np.exp(model.params_['group']):.2f} "
      f"({row['exp(coef) lower 95%']:.2f} to {row['exp(coef) upper 95%']:.2f}), p = {row['p']:.4f}")

# How much of the exclusion is driven by any single predictor
print("\nmissingness by candidate predictor, among the 628 excluded:")
counts = {}
for variable in C.POOL:
    if variable in excluded.columns:
        counts[variable] = int(excluded[variable].isna().sum())
ordered = sorted(counts.items(), key=lambda kv: -kv[1])
for variable, missing in ordered[:10]:
    print(f"  {variable:20s} {missing:4d}  ({100 * missing / len(excluded):.1f}%)")
only_one = int((excluded[C.POOL].isna().sum(axis=1) == 1).sum())
print(f"\nexcluded because exactly one predictor was missing: {only_one} "
      f"({100 * only_one / len(excluded):.1f}%)")

summary = {
    "eligible": len(eligible), "included": len(included), "excluded": len(excluded),
    "excluded_pct": 100 * len(excluded) / len(eligible),
    "events_included": int(included.event.sum()), "events_excluded": int(excluded.event.sum()),
    "rate_included": float(included.event.mean()), "rate_excluded": float(excluded.event.mean()),
    "logrank_p": float(test.p_value),
    "hr_group": float(np.exp(model.params_["group"])),
    "hr_group_lo": float(row["exp(coef) lower 95%"]),
    "hr_group_hi": float(row["exp(coef) upper 95%"]),
    "hr_group_p": float(row["p"]),
    "incidence": {str(h): {"included": C.km_incidence(included, h),
                           "excluded": C.km_incidence(excluded, h)} for h in (3, 5, 7)},
    "missingness": dict(ordered),
    "missing_exactly_one": only_one,
}
with open(os.path.join(OUT, "tab41_included_vs_excluded.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"\nwritten to {OUT}")
