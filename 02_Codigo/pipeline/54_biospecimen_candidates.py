# 54_biospecimen_candidates.py
# Produces: the corrected rows of the lysosomal assays in Supplementary Table S8, and the
#           false-discovery-rate columns of that table recomputed around them.
#
# The problem. 52_multimodal_screen.py read these assays by keeping the last row in the file
# for each participant, but the results file holds several rows per participant under one
# test name. So pS65 ubiquitin was the standard deviation of the replicates rather than the
# measurement, for everyone; glucocerebrosidase activity mixed three specimens on two units;
# and progranulin mixed two specimens. Those three rows of the table did not measure what
# their labels said. Script 52 now reads each assay from one specimen and one reported
# quantity, and this script carries the result into the table the supplement is built from.
#
# The three lipids keep their numbers, because they were already all from cerebrospinal
# fluid, and the script stops if they no longer reproduce. They gain the specimen in their
# label. The change in discrimination of the three corrected rows uses the protocol of
# cohort.py with a bootstrap interval, the same as the dose row in
# 50_treatment_candidate_recomputed.py.
#
# Which specimen to read is a choice, so every assay is also run in every specimen and
# reported quantity available, and the result is written to tab54_biospecimen_candidates.json.
#
# Run after 50_treatment_candidate_recomputed.py, which writes the table this one amends, and
# after 52_multimodal_screen.py --write.
#
# Usage:  python 54_biospecimen_candidates.py

import importlib.util
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cohort as C

TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
TABLE = os.path.join(TAB, "tab_incremental_value_all_domains_corrected.csv")
BOOT_REPLICATES = 20        # the same as the dose row in script 50
BOOT_SEED = 20261103

spec = importlib.util.spec_from_file_location(
    "screen52", os.path.join(os.path.dirname(os.path.abspath(__file__)), "52_multimodal_screen.py"))
m52 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m52)

df = C.build_cohort(with_metadata=False)
df["PATNO"] = pd.to_numeric(df.PATNO).astype("Int64")
tz = None                   # biospecimen() takes the baseline visit and does not need time-zero
merged = df.merge(m52.biospecimen(tz), on="PATNO", how="left")
iv = pd.read_csv(TABLE)
original = pd.read_csv(os.path.join(TAB, "tab_incremental_value_all_domains.csv")).set_index("variable")


def delta_c(sub, col, seed):
    """Change in out-of-fold C from adding the candidate, under the protocol of cohort.py, in
    the participants in whom it was recorded, with a bootstrap interval."""
    sub = sub.reset_index(drop=True)
    base = C.out_of_fold_c(sub)
    full = C.out_of_fold_c(sub, list(C.TOTAL6) + [col])
    rng = np.random.default_rng(seed)
    boot, tries = [], 0
    while len(boot) < BOOT_REPLICATES and tries < 10 * BOOT_REPLICATES:
        tries += 1
        b = sub.iloc[rng.choice(len(sub), len(sub), replace=True)].reset_index(drop=True)
        if b.event.sum() < 10:      # five folds need events in every fold
            continue
        boot.append(C.out_of_fold_c(b, list(C.TOTAL6) + [col], repeats=3) - C.out_of_fold_c(b, repeats=3))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return base, full, float(lo), float(hi), len(boot)


# ----------------------------------------------- the lipids must reproduce, label aside ---
print("lipids, which were already all from cerebrospinal fluid:")
for v in ("bio_gl2", "bio_glccer", "bio_sm"):
    t = iv[iv.variable == v].iloc[0]
    g = m52.screen(merged, v)
    print("   %-12s n %d/%d  HR %.3f vs %.3f  p %.5f vs %.5f" % (v, g["n"], t.n, g["HR"], t.HR, g["p"], t.p))
    assert g["n"] == t.n and abs(g["HR"] - t.HR) < 0.002 and abs(g["p"] - t.p) < 1e-4, v
    iv.loc[iv.variable == v, "label"] = m52.BIOSPECIMEN_LABEL[v]

# ------------------------------------------------------------- the three corrected rows ---
before, after = {}, {}
print("\nthe three corrected rows:")
for k, v in enumerate(("bio_progranulin", "bio_gcase", "bio_ps65ub")):
    i = iv.index[iv.variable == v][0]
    before[v] = {c: (float(original.loc[v, c]) if c != "label" else original.loc[v, c])
                 for c in ("label", "n", "events", "HR", "lo", "hi", "p", "LRT_p", "dC")}
    g = m52.screen(merged, v)
    sub = merged[list(C.TOTAL6) + [v, "exit", "event"]].dropna()
    base, full, lo, hi, nb = delta_c(sub, v, BOOT_SEED + k)
    after[v] = dict(label=m52.BIOSPECIMEN_LABEL[v], n=g["n"], events=g["events"], HR=g["HR"], lo=g["lo"],
                    hi=g["hi"], p=g["p"], LRT_p=g["LRT_p"], min_HR=g["min_HR_85pct"], C_base=base,
                    C_with=full, dC=full - base, dC_lo=lo, dC_hi=hi, replicates=nb)
    print("   %-16s n %d -> %d, events %d -> %d, HR %.3f -> %.3f (%.3f to %.3f), p %.3f -> %.3f, dC %+.4f -> %+.4f (%+.4f to %+.4f)"
          % (v, before[v]["n"], g["n"], before[v]["events"], g["events"], before[v]["HR"], g["HR"], g["lo"], g["hi"],
             before[v]["p"], g["p"], before[v]["dC"], full - base, lo, hi))
    iv.loc[i, ["label", "n", "events", "HR", "lo", "hi", "p", "LRT_p", "min_HR_80pct"]] = [
        m52.BIOSPECIMEN_LABEL[v], g["n"], g["events"], round(g["HR"], 3), round(g["lo"], 3), round(g["hi"], 3),
        g["p"], g["LRT_p"], g["min_HR_85pct"]]
    iv.loc[i, ["dC", "dC_lo", "dC_hi", "C_base", "C_with", "dC_pt"]] = [
        round(full - base, 4), round(lo, 4), round(hi, 4), round(base, 4), round(full, 4), round(full - base, 4)]
    for c in ("dC_txt", "excl_zero", "se_dC", "P_pos"):
        if c in iv.columns:
            iv[c] = iv[c].astype(object)
            iv.loc[i, c] = np.nan

# ---------------------------------------------------- false-discovery-rate columns ---
old_global = iv.FDR_global.copy()
iv["FDR_domain"] = iv.groupby("domain").p.transform(lambda s: m52.bh(s.values))
iv["FDR_global"] = m52.bh(iv.p.values)
nominal = iv[iv.p < 0.05]
print("\nnominal at p<0.05: %d (%s)" % (len(nominal), ", ".join(nominal.label.str[:24])))
print("surviving global control: %d; surviving within a domain: %d (%s)"
      % (int((iv.FDR_global < 0.05).sum()), int((iv.FDR_domain < 0.05).sum()),
         ", ".join(iv[iv.FDR_domain < 0.05].label.str[:24])))
print("global adjusted p of the nominal markers: %s (was %s)"
      % (sorted({round(float(x), 4) for x in nominal.FDR_global}),
         sorted({round(float(x), 4) for x in old_global[iv.p < 0.05]})))
print("detectable hazard ratio below 1.30: %d; below 1.35: %d"
      % (int((iv.min_HR_80pct < 1.30).sum()), int((iv.min_HR_80pct < 1.35).sum())))

# ------------------------------------------------------ every assay in every specimen ---
rows = m52.biospecimen_rows()
every = []
chosen = {(t, s, u): k for k, (t, s, u) in m52.BIOSPECIMEN.items()}
print("\nevery assay in every specimen and reported quantity, in the development sample:")
for (test, specimen, unit), one in rows.groupby(["test", "specimen", "unit"]):
    if unit == "stdev":        # the spread of the replicates, not a measurement
        continue
    x = one.groupby("PATNO", as_index=False).value.last().rename(columns={"value": "alt"})
    d = df.merge(x, on="PATNO", how="left")
    g = m52.screen(d, "alt")
    if g is None:
        continue
    used = (test, specimen, unit) in chosen
    every.append(dict(test=test, specimen=specimen, unit=unit, used=used, n=g["n"], events=g["events"],
                      HR=g["HR"], lo=g["lo"], hi=g["hi"], p=g["p"], zeros=int((d.alt == 0).sum())))
    print("   %-15s %-12s %-15s %-5s n %3d, %3d events  HR %.3f (%.3f to %.3f)  p %.3f  zeros %d"
          % (test, specimen, unit, "used" if used else "", g["n"], g["events"], g["HR"], g["lo"], g["hi"], g["p"],
             every[-1]["zeros"]))

# what the global adjustment would be if pS65 ubiquitin were read from cerebrospinal fluid
alt = [r for r in every if r["test"] == "ps65 ubiquitin" and r["specimen"] == "CSF"][0]
p_alt = iv.p.values.copy()
k = iv.index[iv.variable == "bio_ps65ub"][0]
p_alt[k] = alt["p"]
fdr_alt = float(m52.bh(p_alt)[k])
print("\npS65 ubiquitin in cerebrospinal fluid, p %.3f: global adjusted p %.3f" % (alt["p"], fdr_alt))

iv.to_csv(TABLE, index=False)
json.dump(dict(before=before, after=after, every_specimen=every, ps65_csf_fdr_global=fdr_alt,
               nominal=list(nominal.label), fdr_global_nominal=sorted({float(x) for x in nominal.FDR_global})),
          open(os.path.join(TAB, "tab54_biospecimen_candidates.json"), "w"), indent=1)
print("\nwritten tab_incremental_value_all_domains_corrected.csv and tab54_biospecimen_candidates.json")
