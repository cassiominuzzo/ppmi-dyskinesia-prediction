# 39_drug_name_quality_control.py
# Produces: the drug-name audit of the medication log and the time-zero sensitivity
#           analysis it motivated.
#
# The PPMI medication log records free text, so the drug-name search decides which
# entries count as levodopa, as a dopamine agonist or as amantadine. Getting this
# wrong matters twice over: time-zero is the first levodopa record, and the
# treatment rows of Table 1 count exposures at that moment.
#
# Two findings from this audit are carried into the analysis:
#   Clarium is piribedil and therefore a dopamine agonist, not levodopa. Entries
#   recording the brand name alone were previously uncounted.
#   Dopicar, Levocomp and Sinement are levodopa preparations that the original
#   search missed. For five participants this places time-zero later than the true
#   date, so they are excluded in a sensitivity analysis.

import json
import os

import numpy as np
import pandas as pd

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
D31 = pd.Timedelta(days=C.WINDOW_DAYS)
SHIFT_THRESHOLD_DAYS = C.WINDOW_DAYS

df = C.build_cohort()
log, tzero = C.load_medication_log()
in_cohort = log[log.PATNO.isin(df.PATNO)]

name = in_cohort.LEDTRT.astype(str).str.lower()
is_amantadine = name.str.contains(C.RE_AMANTADINE, na=False, regex=True)
is_agonist = name.str.contains(C.RE_AGONIST, na=False, regex=True)
is_levodopa = in_cohort.LEDTRT.apply(C.is_levodopa)

unclassified = in_cohort[~(is_amantadine | is_agonist | is_levodopa)]
print("drug names not matched by any search, in the development cohort:")
print(unclassified.LEDTRT.astype(str).str.lower().str.strip()
      .value_counts().head(20).to_string())

# ---------------------------------------------------- effect on the time-zero date
tzero_extended = (
    log[log.LEDTRT.apply(lambda s: C.is_levodopa(s, extended=True))]
    .groupby("PATNO").STARTDT.min().rename("tz_extended")
)
comparison = pd.concat([tzero, tzero_extended], axis=1).reindex(df.PATNO.values)
comparison.index = df.PATNO.values
shift_days = (comparison.tz - comparison.tz_extended).dt.days
affected = shift_days[shift_days > SHIFT_THRESHOLD_DAYS]

print(f"\nparticipants whose time-zero would move earlier by more than "
      f"{SHIFT_THRESHOLD_DAYS} days: {len(affected)} of {len(df)}")
if len(affected):
    detail = log[log.PATNO.isin(affected.index)
                 & log.LEDTRT.apply(lambda s: C.is_levodopa(s, extended=True))
                 & ~log.LEDTRT.apply(C.is_levodopa)]
    for patno, days in affected.items():
        drugs = sorted(set(detail[detail.PATNO == patno].LEDTRT.astype(str).str.strip()))
        print(f"  {patno}: {int(days)} days, unmatched entry {', '.join(drugs)}")

# ------------------------------------------------------------ sensitivity analysis
retained = df[~df.PATNO.isin(affected.index)].reset_index(drop=True)
c_full = C.out_of_fold_c(df)
c_retained = C.out_of_fold_c(retained)
print(f"\nout-of-fold C: full cohort {c_full:.4f} (n={len(df)}, events={int(df.event.sum())}) "
      f"against {c_retained:.4f} (n={len(retained)}, events={int(retained.event.sum())}), "
      f"difference {c_retained - c_full:+.4f}")

model_full = C.fit_total6(df)
model_retained = C.fit_total6(retained)
print("\nhazard ratios per unit:")
print(f"{'':20s}{'full':>10s}{'retained':>10s}")
for variable in C.TOTAL6:
    print(f"{variable:20s}{np.exp(model_full.params_[variable]):10.4f}"
          f"{np.exp(model_retained.params_[variable]):10.4f}")

# ------------------------------------------------ exposure counts at time-zero
merged = in_cohort.merge(tzero, on="PATNO")
active = merged[(merged.STARTDT <= merged.tz + D31)
                & ((merged.STOPDT.isna()) | (merged.STOPDT >= merged.tz - D31))]
active_name = active.LEDTRT.astype(str).str.lower()
agonist_users = active[active_name.str.contains(C.RE_AGONIST, na=False, regex=True)].PATNO.nunique()
amantadine_users = active[active_name.str.contains(C.RE_AMANTADINE, na=False, regex=True)].PATNO.nunique()
print(f"\nexposures at time-zero with the audited searches: "
      f"{agonist_users} on a dopamine agonist ({100 * agonist_users / len(df):.1f}%), "
      f"{amantadine_users} on amantadine ({100 * amantadine_users / len(df):.1f}%)")

summary = {
    "affected": {int(k): int(v) for k, v in affected.items()},
    "n_affected": len(affected),
    "C_full": float(c_full), "C_retained": float(c_retained),
    "delta_C": float(c_retained - c_full),
    "n_retained": len(retained), "events_retained": int(retained.event.sum()),
    "agonist_at_time_zero": int(agonist_users),
    "amantadine_at_time_zero": int(amantadine_users),
    "hazard_ratios_full": {v: float(np.exp(model_full.params_[v])) for v in C.TOTAL6},
    "hazard_ratios_retained": {v: float(np.exp(model_retained.params_[v])) for v in C.TOTAL6},
}
with open(os.path.join(OUT, "tab39_drug_name_quality_control.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
unclassified.LEDTRT.astype(str).str.lower().str.strip().value_counts().to_csv(
    os.path.join(OUT, "tab39_unmatched_drug_names.csv"))
print(f"\nwritten to {OUT}")
