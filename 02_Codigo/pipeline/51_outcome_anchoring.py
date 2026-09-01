# 51_outcome_anchoring.py
# Produces: Supplementary Table S17, the functional consequence of crossing the outcome
#           threshold, and the numbers for the anchoring sentence in the Results.
#
# The question. The outcome of this study is a threshold on a scale, and the fairest
# criticism of any such outcome is that a threshold is a convention. This script asks
# whether crossing it is followed by a measurable loss of function in the same
# participants, which converts the outcome from a convention into an event with a
# demonstrated consequence.
#
# The tautology that has to be avoided. The outcome is a score of 2 or more on MDS-UPDRS
# item 4.1 or item 4.2, and item 4.2 is itself the functional impact of dyskinesia. Using
# the full outcome here would amount to showing that a functional-impact rating predicts a
# functional scale. The exposure is therefore item 4.1 alone, the time spent with
# dyskinesia, which is a duration and not an impairment rating: a score of 2 means more
# than a quarter of the waking day. That version of the outcome is already reported as a
# sensitivity analysis in Supplementary Table S1.
#
# The design. The functional measure is the Modified Schwab and England scale, recorded at
# the same visits throughout follow-up. Each participant contributes their own before and
# after, so the comparison is within a person rather than between people. A linear mixed
# model with a random intercept carries three terms of interest:
#
#   time            the ordinary decline with disease duration, which happens to everyone
#   crossed         a step at the moment of crossing, the primary estimand
#   years after     a change in the rate of decline once the threshold has been crossed
#
# A pure-progression explanation predicts a smooth trajectory: no step, no bend. A
# consequential outcome predicts at least one of the two. The step is pre-specified as
# primary because it is the quantity a clinician would recognise.
#
# The step is estimated from measurements taken at the same visits, so it is concurrent
# rather than prospective. A concurrent drop could in principle be a bad day rather than a
# lasting change, so a second, stricter analysis asks whether the decrement is still there
# one year or more after crossing.
#
# Usage:  python 51_outcome_anchoring.py

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import statsmodels.formula.api as smf

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))


def read(folder, pattern, cols):
    d = pd.read_csv(C._find(folder, pattern), low_memory=False)
    d["PATNO"] = pd.to_numeric(d.PATNO, errors="coerce")
    d["INFODT"] = pd.to_datetime(d.INFODT, format="%m/%Y", errors="coerce")
    return d.dropna(subset=["PATNO"])[["PATNO", "INFODT"] + cols]


coh = C.build_cohort(with_metadata=False)
tz = pd.read_parquet(os.path.join(C.PROC, "tzero.parquet"))[["PATNO", "tzero_levodopa"]].dropna()
tz["PATNO"] = pd.to_numeric(tz.PATNO, errors="coerce")
tz["tzero_levodopa"] = pd.to_datetime(tz.tzero_levodopa)

# ------------------------------------------------------------------ exposure -------
p4 = read("MDS_UPDRS_e_Motor", "MDS-UPDRS_Part_IV*", ["NP4WDYSK"])
p4["NP4WDYSK"] = pd.to_numeric(p4.NP4WDYSK, errors="coerce")
p4 = p4.merge(tz, on="PATNO")
p4["t"] = (p4.INFODT - p4.tzero_levodopa).dt.days / 365.25
p4 = p4[p4.PATNO.isin(coh.PATNO) & (p4.t > 0)]
cross = p4[p4.NP4WDYSK >= 2].groupby("PATNO").t.min().rename("t_cross").reset_index()
print("participants crossing item 4.1 >= 2 after time-zero: %d of %d" % (len(cross), len(coh)))

# ------------------------------------------------------------------- outcome -------
se = read("MDS_UPDRS_e_Motor", "Modified_Schwab*", ["MSEADLG"])
se["MSEADLG"] = pd.to_numeric(se.MSEADLG, errors="coerce")
se = se.dropna(subset=["MSEADLG"]).merge(tz, on="PATNO")
se["t"] = (se.INFODT - se.tzero_levodopa).dt.days / 365.25
se = se[se.PATNO.isin(coh.PATNO) & (se.t > 0)]

d = (se.merge(cross, on="PATNO", how="left")
       .merge(coh[["PATNO", "ageonset", "updrs_totscore", "SEX"]], on="PATNO"))
d["crossed"] = ((d.t_cross.notna()) & (d.t >= d.t_cross)).astype(int)
d["after"] = np.where(d.crossed == 1, d.t - d.t_cross, 0.0)
print("measurements: %d in %d participants (%d of them cross)"
      % (len(d), d.PATNO.nunique(), d[d.t_cross.notna()].PATNO.nunique()))

# ---------------------------------------------------------------- mixed model ------
d["age_c"] = d.ageonset - d.ageonset.mean()
d["updrs_c"] = d.updrs_totscore - d.updrs_totscore.mean()
m = smf.mixedlm("MSEADLG ~ t + crossed + after + age_c + updrs_c + SEX",
                d, groups=d.PATNO).fit(method="powell")
print("\n" + "=" * 70)
print("MIXED MODEL: Modified Schwab and England over follow-up")
print("=" * 70)
NAMES = {"t": "time since levodopa, per year", "crossed": "crossed item 4.1 >= 2 (step)",
         "after": "years since crossing (slope change)", "age_c": "age at onset, per year",
         "updrs_c": "baseline total MDS-UPDRS, per point", "SEX": "male sex"}
res = {}
for k, lab in NAMES.items():
    b, se_b, p = m.params[k], m.bse[k], m.pvalues[k]
    res[k] = dict(beta=float(b), lo=float(b - 1.96 * se_b), hi=float(b + 1.96 * se_b), p=float(p))
    print("   %-38s %+7.3f  (%+.3f to %+.3f)   p = %.2g" % (lab, b, b - 1.96 * se_b, b + 1.96 * se_b, p))

# ------------------------------------------------- the competing explanation ------
# The obvious alternative is that people who cross were simply declining faster all along,
# and that the step is their steeper slope showing up at the moment they happen to cross.
# Allowing each participant their own rate of decline tests that directly: if the step is
# really a faster slope in disguise, it should shrink towards zero once every participant
# has a slope of their own.
print("\n" + "=" * 70)
print("COMPETING EXPLANATION: were the crossers simply declining faster all along?")
print("=" * 70)
ms = smf.mixedlm("MSEADLG ~ t + crossed + after + age_c + updrs_c + SEX", d,
                 groups=d.PATNO, re_formula="~t").fit(method="powell")
b, sb, p = ms.params["crossed"], ms.bse["crossed"], ms.pvalues["crossed"]
print("   step with a random slope for every participant: %+.3f (%+.3f to %+.3f), p = %.2g"
      % (b, b - 1.96 * sb, b + 1.96 * sb, p))
print("   step with a random intercept only:              %+.3f" % m.params["crossed"])
res["crossed_random_slope"] = dict(beta=float(b), lo=float(b - 1.96 * sb),
                                   hi=float(b + 1.96 * sb), p=float(p))

# ------------------------------------------- is the decrement still there later? ---
print("\n" + "=" * 70)
print("IS THE DECREMENT SUSTAINED? measurements one year or more after crossing")
print("=" * 70)
d["window"] = np.select(
    [d.t_cross.isna(), d.t < d.t_cross, d.after < 1],
    ["never crossed", "before crossing", "within the first year after"],
    default="one year or more after")
for w in ("never crossed", "before crossing", "within the first year after", "one year or more after"):
    s = d[d.window == w]
    print("   %-30s %5d measurements, %3d participants, median %3.0f, mean %5.1f"
          % (w, len(s), s.PATNO.nunique(), s.MSEADLG.median(), s.MSEADLG.mean()))

late = d[d.window.isin(["before crossing", "one year or more after"])].copy()
late["late"] = (late.window == "one year or more after").astype(int)
m2 = smf.mixedlm("MSEADLG ~ t + late + age_c + updrs_c + SEX", late, groups=late.PATNO).fit(method="powell")
b, sb, p = m2.params["late"], m2.bse["late"], m2.pvalues["late"]
print("\n   within the crossers only, before against one year or more after,")
print("   adjusted for time since levodopa: %+.3f points (%+.3f to %+.3f), p = %.2g"
      % (b, b - 1.96 * sb, b + 1.96 * sb, p))
res["sustained"] = dict(beta=float(b), lo=float(b - 1.96 * sb), hi=float(b + 1.96 * sb), p=float(p))

# ------------------------------------------------ a threshold a clinician knows ----
print("\n" + "=" * 70)
print("PROPORTION BELOW 80 ON THE SCALE, WHICH IS THE USUAL MARK OF NEEDING HELP")
print("=" * 70)
for w in ("never crossed", "before crossing", "one year or more after"):
    s = d[d.window == w]
    print("   %-30s %5.1f%% of measurements  (%d of %d)"
          % (w, 100 * (s.MSEADLG < 80).mean(), (s.MSEADLG < 80).sum(), len(s)))

summary = dict(n_cross=int(len(cross)), n_measurements=int(len(d)),
               n_participants=int(d.PATNO.nunique()),
               n_crossers_with_data=int(d[d.t_cross.notna()].PATNO.nunique()),
               model=res,
               below80={w: float(100 * (d[d.window == w].MSEADLG < 80).mean())
                        for w in d.window.unique()})
json.dump(summary, open(os.path.join(OUT, "tab51_outcome_anchoring.json"), "w"), indent=1)
d.to_csv(os.path.join(OUT, "tab51_anchoring_measurements.csv"), index=False)
print("\nwritten tab51_outcome_anchoring.json and tab51_anchoring_measurements.csv")
