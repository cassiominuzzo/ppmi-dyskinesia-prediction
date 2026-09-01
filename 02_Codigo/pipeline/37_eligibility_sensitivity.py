# 37_eligibility_sensitivity.py
# Produces: an eligibility row of Supplementary Table S3.
#
# A small number of participants in the development sample were enrolled in PPMI
# cohorts other than the Parkinson's disease cohort, including participants with
# scans showing no evidence of dopaminergic deficit, whose diagnosis a reader may
# reasonably question. They are excluded here as a sensitivity analysis.
#
# Methodological note. When first run with ten cross-validation repetitions this
# comparison appeared to cost 0.017 of C-index. Repeating it with twenty
# repetitions and independent seeds showed that the apparent effect was
# fold-partition noise: the true difference is 0.003. The comparison is therefore
# run here at the full twenty repetitions, and both the frozen-prediction and the
# refitted contrasts are reported, because they answer different questions.

import json
import os

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))

df = C.build_cohort()
excluded = df[(df.COHORT != 1) | (df.subgroup == "SWEDD")]
retained = df[~df.PATNO.isin(excluded.PATNO)].reset_index(drop=True)

print(f"cohort: n={len(df)}, events={int(df.event.sum())}")
print(f"outside the Parkinson's disease cohort or labelled SWEDD: "
      f"{len(excluded)} participants with {int(excluded.event.sum())} event(s)")
print(excluded.groupby(["COHORT", "subgroup"]).agg(n=("PATNO", "size"),
                                                   events=("event", "sum")).to_string())

scores_full = C.out_of_fold_c(df, return_all=True)
scores_retained = C.out_of_fold_c(retained, return_all=True)
print(f"\nrefitted, out of fold, {C.CV_REPEATS} repetitions:")
print(f"  full cohort   n={len(df):4d} events={int(df.event.sum()):4d}  "
      f"C = {scores_full.mean():.4f} (sd across repetitions {scores_full.std():.4f})")
print(f"  retained      n={len(retained):4d} events={int(retained.event.sum()):4d}  "
      f"C = {scores_retained.mean():.4f} (sd {scores_retained.std():.4f})")
print(f"  difference = {scores_retained.mean() - scores_full.mean():+.4f}")

lp_oof = C.out_of_fold_linear_predictor(df)
df = df.assign(lp_oof=lp_oof)
kept = df[~df.PATNO.isin(excluded.PATNO)]
c_frozen_full = concordance_index(df.exit, -df.lp_oof, df.event)
c_frozen_kept = concordance_index(kept.exit, -kept.lp_oof, kept.event)
print(f"\nsame predictions, evaluation restricted:")
print(f"  full cohort {c_frozen_full:.4f} against retained {c_frozen_kept:.4f} "
      f"(difference {c_frozen_kept - c_frozen_full:+.4f})")

model_full = C.fit_total6(df)
model_kept = C.fit_total6(retained)
print("\nhazard ratios per unit:")
print(f"{'':20s}{'full':>10s}{'retained':>10s}")
for variable in C.TOTAL6:
    print(f"{variable:20s}{np.exp(model_full.params_[variable]):10.4f}"
          f"{np.exp(model_kept.params_[variable]):10.4f}")

summary = {
    "excluded_n": len(excluded), "excluded_events": int(excluded.event.sum()),
    "excluded_detail": excluded.groupby(["COHORT", "subgroup"]).size().to_dict().__str__(),
    "C_full": float(scores_full.mean()), "C_retained": float(scores_retained.mean()),
    "delta_C": float(scores_retained.mean() - scores_full.mean()),
    "C_frozen_full": float(c_frozen_full), "C_frozen_retained": float(c_frozen_kept),
    "n_retained": len(retained), "events_retained": int(retained.event.sum()),
    "hazard_ratios_full": {v: float(np.exp(model_full.params_[v])) for v in C.TOTAL6},
    "hazard_ratios_retained": {v: float(np.exp(model_kept.params_[v])) for v in C.TOTAL6},
}
with open(os.path.join(OUT, "tab37_eligibility_sensitivity.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
excluded[["PATNO", "COHORT", "subgroup", "event", "exit"]].to_csv(
    os.path.join(OUT, "tab37_excluded_participants.csv"), index=False)
print(f"\nwritten to {OUT}")
