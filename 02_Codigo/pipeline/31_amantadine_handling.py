# 31_amantadine_handling.py
# Produces: Supplementary Table S8 (amantadine sensitivity analyses) and the
#           corrected pre-levodopa dopaminergic dose variable of Supplementary Table S1.
#
# Amantadine is the only drug in routine use with established anti-dyskinetic
# efficacy, so it occupies two distinct positions in this study. It is included in
# the levodopa-equivalent conversion alongside drugs whose effect on the outcome
# runs the other way, which is a measurement problem; and, when started during
# follow-up, it can suppress dyskinesia and prevent a participant from crossing the
# outcome threshold, which is an outcome problem. Analyses A to C address the
# second, analysis D the first.

import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
D31 = pd.Timedelta(days=C.WINDOW_DAYS)

df = C.build_cohort()
log, tzero = C.load_medication_log()
log = log.merge(tzero, on="PATNO")
log = log[log.PATNO.isin(df.PATNO)]

name = log.LEDTRT.astype(str).str.lower()
active = log[(log.STARTDT <= log.tz + D31) & ((log.STOPDT.isna()) | (log.STOPDT >= log.tz - D31))]

# ---------------------------------------------------------------- exposure groups
amantadine = log[name.str.contains(C.RE_AMANTADINE, na=False, regex=True)]
baseline_users = set(
    amantadine[
        (amantadine.STARTDT <= amantadine.tz + D31)
        & ((amantadine.STOPDT.isna()) | (amantadine.STOPDT >= amantadine.tz - D31))
    ].PATNO
)
later = (
    amantadine[amantadine.STARTDT > amantadine.tz + D31]
    .groupby("PATNO").STARTDT.min()
)
later = later[~later.index.isin(baseline_users)]

df["AM"] = df.PATNO.isin(baseline_users).astype(int)
print(f"cohort: n={len(df)}, events={int(df.event.sum())}")
print(f"amantadine active at time-zero: {int(df.AM.sum())} ({100 * df.AM.mean():.1f}%)")

start_year = {p: (s - tzero[p]).days / 365.25 for p, s in later.items() if p in set(df.PATNO)}
df["am_start"] = df.PATNO.map(start_year)
started = df[df.am_start.notna()]
before = int(((started.event == 1) & (started.exit <= started.am_start)).sum())
after = int(((started.event == 1) & (started.exit > started.am_start)).sum())
free = int((started.event == 0).sum())
print(f"started amantadine during follow-up: {len(started)} "
      f"({before} event before, {after} event after, {free} event free)")

# ------------------------------------------------- A. amantadine as a covariate
def hazard_ratio(data, extra, predictors=None):
    cols = list(predictors or C.TOTAL6)
    d = data[cols + [extra, "exit", "event"]].copy()
    for c in cols:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    fit = CoxPHFitter(penalizer=C.RIDGE).fit(d, "exit", "event")
    row = fit.summary.loc[extra]
    return (float(np.exp(fit.params_[extra])),
            float(row["exp(coef) lower 95%"]), float(row["exp(coef) upper 95%"]), float(row["p"]))

crude = CoxPHFitter().fit(df[["AM", "exit", "event"]], "exit", "event")
crude_row = crude.summary.loc["AM"]
adjusted = hazard_ratio(df, "AM")
c_reference = C.out_of_fold_c(df)
c_with_amantadine = C.out_of_fold_c(df, C.TOTAL6 + ["AM"])
print(f"\nA. crude    HR {np.exp(crude.params_['AM']):.2f} "
      f"({crude_row['exp(coef) lower 95%']:.2f} to {crude_row['exp(coef) upper 95%']:.2f}), "
      f"p = {crude_row['p']:.3f}")
print(f"A. adjusted HR {adjusted[0]:.2f} ({adjusted[1]:.2f} to {adjusted[2]:.2f}), p = {adjusted[3]:.3f}, "
      f"delta C {c_with_amantadine - c_reference:+.3f}")

# ------------------------------------------------ B. exclusion of baseline users
non_users = df[df.AM == 0].reset_index(drop=True)
c_refit = C.out_of_fold_c(non_users)
frozen = C.fit_total6(df)
lp_all = C.linear_predictor(frozen, df)
from lifelines.utils import concordance_index
c_frozen_non = concordance_index(df.exit[df.AM == 0], -lp_all[df.AM.values == 0], df.event[df.AM == 0])
c_frozen_users = concordance_index(df.exit[df.AM == 1], -lp_all[df.AM.values == 1], df.event[df.AM == 1])
print(f"\nB. refit in {len(non_users)} non-users: C {c_refit:.3f} (delta {c_refit - c_reference:+.3f})")
print(f"B. frozen model in the {len(non_users)} non-users: C {c_frozen_non:.3f}")
print(f"B. frozen model in the {int(df.AM.sum())} users:     C {c_frozen_users:.3f}")

# ------------------------------------------------ C. outcome masking, bounded above
part4 = C.load_part_iv()
candidates = []
for _, row in df[df.event == 0].dropna(subset=["am_start"]).iterrows():
    hist = part4[(part4.PATNO == row.PATNO) & (part4.t > 0) & (part4.t <= row.am_start)]
    worst = np.nanmax(pd.concat([hist.NP4DYSKI, hist.NP4WDYSK]).values) if len(hist) else np.nan
    candidates.append({"PATNO": row.PATNO, "am_start": row.am_start,
                       "assessments": len(hist), "worst_before": worst})
candidates = pd.DataFrame(candidates)
with_signal = candidates[candidates.worst_before >= 1]
print(f"\nC. of the {len(candidates)} event-free starters: "
      f"{len(with_signal)} scored 1 before starting, "
      f"{int((candidates.worst_before == 0).sum())} scored 0, "
      f"{int(candidates.worst_before.isna().sum())} had no assessment")

def reclassify(ids):
    data = df.set_index("PATNO").copy()
    for p in ids:
        data.loc[p, "exit"] = max(float(candidates.set_index("PATNO").loc[p, "am_start"]), 1 / 365.25)
        data.loc[p, "event"] = 1
    data = data.reset_index()
    return int(data.event.sum()), C.out_of_fold_c(data)

events_plausible, c_plausible = reclassify(with_signal.PATNO.tolist())
events_extreme, c_extreme = reclassify(candidates.PATNO.tolist())
print(f"C. reclassifying the {len(with_signal)} plausible: {events_plausible} events, "
      f"delta C {c_plausible - c_reference:+.3f}")
print(f"C. reclassifying all {len(candidates)} (bound):    {events_extreme} events, "
      f"delta C {c_extreme - c_reference:+.3f}")

# ------------------------------------- D. dopaminergic dose, with and without amantadine
dose = active.copy()
dose["LEDD_n"] = pd.to_numeric(dose.LEDD, errors="coerce")
dose["is_am"] = dose.LEDTRT.astype(str).str.lower().str.contains(C.RE_AMANTADINE, na=False, regex=True)
am_dose = dose[dose.is_am].groupby("PATNO").LEDD_n.sum()
total_dose = dose.groupby("PATNO").LEDD_n.sum()
am_dose = am_dose[am_dose > 0]
share = 100 * (am_dose / total_dose.reindex(am_dose.index)).median()
print(f"\nD. in the {len(am_dose)} baseline users amantadine contributed a median of "
      f"{am_dose.median():.0f} mg (IQR {am_dose.quantile(.25):.0f} to {am_dose.quantile(.75):.0f}) "
      f"of a median total of {total_dose.reindex(am_dose.index).median():.0f} mg, "
      f"that is {share:.0f}% of the converted dose")

summary = {
    "n": len(df), "events": int(df.event.sum()),
    "baseline_users": int(df.AM.sum()),
    "started_during_followup": {"total": len(started), "event_before": before,
                                "event_after": after, "event_free": free},
    "masking_candidates": {"with_prior_score": len(with_signal),
                           "score_zero": int((candidates.worst_before == 0).sum()),
                           "no_assessment": int(candidates.worst_before.isna().sum())},
    "A_crude": {"HR": float(np.exp(crude.params_["AM"])),
                "lo": float(crude_row["exp(coef) lower 95%"]),
                "hi": float(crude_row["exp(coef) upper 95%"]), "p": float(crude_row["p"])},
    "A_adjusted": {"HR": adjusted[0], "lo": adjusted[1], "hi": adjusted[2], "p": adjusted[3],
                   "delta_C": c_with_amantadine - c_reference},
    "B": {"C_refit": c_refit, "delta_C": c_refit - c_reference,
          "C_frozen_non_users": float(c_frozen_non), "C_frozen_users": float(c_frozen_users)},
    "C": {"plausible_events": events_plausible, "plausible_delta_C": c_plausible - c_reference,
          "extreme_events": events_extreme, "extreme_delta_C": c_extreme - c_reference},
    "D": {"median_amantadine_mg": float(am_dose.median()),
          "median_total_mg": float(total_dose.reindex(am_dose.index).median()),
          "median_share_pct": float(share)},
    "C_reference": c_reference,
}
with open(os.path.join(OUT, "tab31_amantadine.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
candidates.to_csv(os.path.join(OUT, "tab31_masking_candidates.csv"), index=False)
print(f"\nwritten to {OUT}")
