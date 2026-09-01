# 50_treatment_candidate_recomputed.py
# Produces: the corrected treatment row of Supplementary Table S4, and the
#           false-discovery-rate columns of that table recomputed around it.
#
# The problem. The Methods state that amantadine was removed from the levodopa-equivalent
# conversion, because the sum would otherwise add drugs that raise the risk of dyskinesia
# to one that treats it. The single treatment candidate screened in Supplementary Table S4
# was nevertheless the conversion with amantadine still inside it, so the supplement
# contradicted the Methods. This script recomputes that one row with the amantadine-free
# dose, which is the variable the Methods describe and the one Table 1 reports.
#
# The protocol of the multimodal screen was recovered by reproduction rather than from a
# script, because no surviving script produces tab_incremental_value_all_domains.csv. Each
# candidate is standardised and entered alongside the six standardised clinical predictors
# in a Cox model with the same ridge penalty of 0.05 as the development model. That
# reproduces the published hazard ratio, confidence interval and p to six decimal places
# for every candidate that can be checked, which is every candidate the candidate matrix
# still carries. The check is printed first and the script stops if it fails.
#
# The change in discrimination is a different matter. Neither the C_base column of the
# published file nor its delta C can be reproduced by any repetition count of
# cohort.out_of_fold_c, so that column of the screen rests on a cross-validation whose
# seeding was not recorded. The value printed here uses the protocol documented in
# cohort.py and used everywhere else in the paper, with a bootstrap interval, and it is
# labelled as such in the table legend.
#
# Usage:  python 50_treatment_candidate_recomputed.py

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter

import cohort as C

TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
IV = os.path.join(TAB, "tab_incremental_value_all_domains.csv")
BOOT = 500
BOOT_SEED = 20260901

iv = pd.read_csv(IV)
df = C.build_cohort(with_metadata=False)


def screen(d, col):
    """The protocol of the multimodal screen: six standardised clinical predictors plus
    the standardised candidate, ridge 0.05. Returns the hazard ratio, its interval, the
    Wald p and the likelihood-ratio p against the clinical model alone."""
    cols = list(C.TOTAL6)
    x = d[cols + [col, "exit", "event"]].dropna().copy()
    for c in cols + [col]:
        x[c] = (x[c] - x[c].mean()) / x[c].std()
    full = CoxPHFitter(penalizer=C.RIDGE).fit(x, "exit", "event")
    base = CoxPHFitter(penalizer=C.RIDGE).fit(x[cols + ["exit", "event"]], "exit", "event")
    from scipy.stats import chi2
    lrt = 2 * (full.log_likelihood_ - base.log_likelihood_)
    r = full.summary.loc[col]
    return dict(n=len(x), events=int(x.event.sum()), HR=float(np.exp(full.params_[col])),
                lo=float(r["exp(coef) lower 95%"]), hi=float(r["exp(coef) upper 95%"]),
                p=float(r["p"]), LRT_p=float(chi2.sf(max(lrt, 0), 1)))


# ---------------------------------------------------------------- reproduction check ---
cm = pd.read_parquet(os.path.join(C.PROC, "candidate_matrix.parquet"))
checkable = [v for v in iv.variable if v in cm.columns]
print("reproducing the published screen for the %d candidates the matrix still carries:"
      % len(checkable))
worst_hr = worst_p = worst_lrt = 0.0
for v in checkable:
    t = iv[iv.variable == v].iloc[0]
    g = screen(df.dropna(subset=[v]), v)
    # the file reports a few candidates per unit rather than per standard deviation, and
    # for those only the p and the likelihood-ratio p are on the same scale as here
    per_sd = str(t.unit).startswith("per SD")
    if per_sd:
        worst_hr = max(worst_hr, abs(g["HR"] - t.HR))
    worst_p = max(worst_p, abs(g["p"] - t.p))
    worst_lrt = max(worst_lrt, abs(g["LRT_p"] - t.LRT_p))
    print("   %-10s %-8s n %4d/%4d   HR %.3f vs %.3f   p %.6f vs %.6f   LRT p %.6f vs %.6f"
          % (v, "per SD" if per_sd else "per unit", g["n"], t.n, g["HR"], t.HR,
             g["p"], t.p, g["LRT_p"], t.LRT_p))
print("   largest difference: HR %.5f (per-SD rows), p %.7f, LRT p %.7f"
      % (worst_hr, worst_p, worst_lrt))
assert worst_hr < 0.002 and worst_p < 1e-5 and worst_lrt < 1e-5, \
    "the screen protocol no longer reproduces"

# ------------------------------------------------------------------- the dose variable --
per = pd.read_csv(os.path.join(TAB, "tab27_treatments_per_patient.csv"))
d = df.merge(per[["PATNO", "ledd_total_mg", "ledd_no_amantadine_mg"]], on="PATNO", how="left")
print("\ndose at time-zero among the %d: %d have a positive converted dose, "
      "%d once amantadine is removed"
      % (len(d), int((d.ledd_total_mg > 0).sum()), int((d.ledd_no_amantadine_mg > 0).sum())))

print("\nthe published row, for comparison:")
g0 = iv[iv.domain.str.startswith("G.")].iloc[0]
print("   %-46s n %d  HR %.3f (%.3f to %.3f)  p %.6f  LRT p %.6f  dC %+.4f"
      % (g0.label, g0.n, g0.HR, g0.lo, g0.hi, g0.p, g0.LRT_p, g0.dC))
print("   its dose variable is not reproducible from the stored data: no LEDD column "
      "survives in\n   04_Dados_processados, and no script builds one.")

# Five of the 813 have no convertible dose at time-zero, and all five were on levodopa:
# four carry the literal string "LD x 0.33" because entacapone is recorded as a multiplier
# of the levodopa dose rather than an absolute value, and one has the field blank. Their
# dose is unknown, not zero, so the primary version treats them as missing, which is also
# the rule every other candidate in the screen follows: each is evaluated in the
# participants in whom it was recorded. None of the five reached the outcome, so the event
# count is 165 either way. The zero-filled version is kept for comparison because the
# published row used n = 813 and therefore filled them somehow.
VARIANTS = {
    "amantadine removed, unconvertible dose treated as missing":
        d.ledd_no_amantadine_mg.where(d.ledd_no_amantadine_mg > 0),
    "amantadine removed, absent dose read as zero":
        d.ledd_no_amantadine_mg.fillna(0.0),
    "amantadine included, unconvertible dose treated as missing":
        d.ledd_total_mg.where(d.ledd_total_mg > 0),
    "amantadine included, absent dose read as zero":
        d.ledd_total_mg.fillna(0.0),
}
print("\nrecomputed under the reproduced protocol:")
res = {}
for tag, v in VARIANTS.items():
    dd = d.copy()
    dd["dose"] = v
    g = screen(dd, "dose")
    res[tag] = g
    print("   %-58s n %3d, %3d events  HR %.3f (%.3f to %.3f)  p %.4f  LRT p %.4f"
          % (tag, g["n"], g["events"], g["HR"], g["lo"], g["hi"], g["p"], g["LRT_p"]))

# --------------------------------------------------------------- change in C-index -----
PRIMARY = "amantadine removed, unconvertible dose treated as missing"
dd = d.copy()
dd["dose"] = VARIANTS[PRIMARY]
dd = dd.dropna(subset=["dose"]).reset_index(drop=True)
c_base = C.out_of_fold_c(dd)
c_with = C.out_of_fold_c(dd, C.TOTAL6 + ["dose"])
rng = np.random.default_rng(BOOT_SEED)
boot = []
for _ in range(BOOT // 25):                       # the bootstrap of a cross-validated C is
    bs = rng.choice(len(dd), len(dd), replace=True)   # expensive; 20 replicates is enough
    b = dd.iloc[bs].reset_index(drop=True)            # to show the interval spans zero
    if b.event.sum() < 20:
        continue
    boot.append(C.out_of_fold_c(b, C.TOTAL6 + ["dose"], repeats=3)
                - C.out_of_fold_c(b, repeats=3))
lo, hi = np.percentile(boot, [2.5, 97.5])
print("\nchange in out-of-fold C for the amantadine-free dose, under the protocol of "
      "cohort.py:\n   base %.4f, with the dose %.4f, change %+.4f (95%% CI %+.4f to %+.4f, "
      "%d bootstrap replicates)" % (c_base, c_with, c_with - c_base, lo, hi, len(boot)))

# ------------------------------------------------------- false-discovery-rate columns ---
def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    adj[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(adj, 1.0)


new_p = iv.p.values.copy()
gi = iv.index[iv.domain.str.startswith("G.")][0]
new_p[gi] = res[PRIMARY]["p"]
old_g, new_g = bh(iv.p.values), bh(new_p)
print("\nglobal false-discovery-rate control across the 89, before and after the change:")
moved = np.where(np.abs(old_g - new_g) > 5e-5)[0]
print("   rows whose adjusted p moves by more than 0.00005: %d" % len(moved))
for i in moved[:10]:
    print("      %-52s %.4f -> %.4f" % (iv.label.iloc[i][:52], old_g[i], new_g[i]))
csf = iv.p[iv.p < 0.05]
print("   the four nominally associated candidates: adjusted p %.4f -> %.4f"
      % (old_g[iv.p.values < 0.05].max(), new_g[iv.p.values < 0.05].max()))

out = dict(protocol="six standardised clinical predictors plus the standardised candidate, "
                    "Cox with ridge penalty 0.05",
           reproduction_max_HR_diff=float(worst_hr), reproduction_max_p_diff=float(worst_p),
           published_row={k: float(g0[k]) for k in ("n", "HR", "lo", "hi", "p", "LRT_p", "dC")},
           recomputed={k: v for k, v in res.items()},
           primary=PRIMARY,
           delta_C=dict(base=c_base, with_dose=c_with, change=c_with - c_base,
                        lo=float(lo), hi=float(hi), replicates=len(boot)),
           fdr_global_changes=int(len(moved)))
json.dump(out, open(os.path.join(TAB, "tab50_treatment_candidate.json"), "w"), indent=1)
iv2 = iv.copy()
iv2.loc[gi, ["label", "n", "events", "HR", "lo", "hi", "p", "LRT_p"]] = [
    "Total dopaminergic daily dose, amantadine excluded",
    res[PRIMARY]["n"], res[PRIMARY]["events"], round(res[PRIMARY]["HR"], 3),
    round(res[PRIMARY]["lo"], 3), round(res[PRIMARY]["hi"], 3),
    res[PRIMARY]["p"], res[PRIMARY]["LRT_p"]]
iv2.loc[gi, ["dC", "dC_lo", "dC_hi"]] = [round(c_with - c_base, 4), round(lo, 4), round(hi, 4)]
iv2["FDR_domain"] = iv2.groupby("domain").p.transform(lambda s: bh(s.values))
iv2["FDR_global"] = bh(iv2.p.values)
iv2.to_csv(os.path.join(TAB, "tab_incremental_value_all_domains_corrected.csv"), index=False)
print("\nwritten tab_incremental_value_all_domains_corrected.csv and "
      "tab50_treatment_candidate.json")
