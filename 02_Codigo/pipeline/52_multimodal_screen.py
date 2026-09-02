# 52_multimodal_screen.py
# Produces: Supplementary Table S4, the incremental value of 89 multimodal candidate
#           predictors over the six-variable clinical model.
#
# Why this script exists. The screen was the largest single analysis in the supplement and
# it was the one piece of the paper with no producer: tab_incremental_value_all_domains.csv
# sat in 03_Resultados with no script that wrote it, so 89 hazard ratios, their p values and
# the two false-discovery-rate columns rested on a file nobody could regenerate. The
# protocol itself had already been recovered by reproduction in
# 50_treatment_candidate_recomputed.py, but only for the 45 candidates the candidate matrix
# still carried. The other 44, every genetic marker, every MRI volume, every subregional
# DaTSCAN binding and every lysosomal assay, existed only inside that orphan file.
#
# This script rebuilds all 89 from the raw downloads and reruns the screen. It is written to
# be checked rather than believed: --probe prints, candidate by candidate, the sample size
# and event count it recovers beside the published ones, and the default run prints the
# published estimate beside the recomputed one for every candidate. Where a candidate does
# not reproduce, the script says so instead of quietly overwriting the number.
#
# The protocol, as recovered: each candidate is standardised and entered alongside the six
# standardised clinical predictors in a Cox model with the same ridge penalty of 0.05 as the
# development model, in the participants in whom that candidate was recorded. The Wald p and
# the likelihood-ratio p against the clinical model alone are invariant to rescaling the
# candidate, so the eight binary or count candidates that the table reports per unit are
# additionally fitted unstandardised to recover their hazard ratio on the natural scale.
#
# The change in out-of-fold concordance is the one column that cannot be reproduced. Neither
# the C_base column of the published file nor its delta can be recovered at any repetition
# count of cohort.out_of_fold_c, so the published screen used a cross-validation whose
# seeding was never recorded. Where this script reports a change in C it uses the protocol
# documented in cohort.py and used everywhere else in the paper, and the legend of
# Supplementary Table S4 says so.
#
# Timing, recovered the same way. Everything from the curated cut, which is every clinical
# scale, every cerebrospinal fluid and blood assay and the striatal binding, is the last
# visit on or before time-zero, taken column by column rather than for the panel as a whole:
# reading a panel together silently discards a marker measured at an earlier visit than its
# neighbours, and doing so moves cerebrospinal neurofilament light off its published sample
# by one participant. Subregional binding is the last scan on or before time-zero, dated by
# the scan itself. Morphometry and the lysosomal assays are the baseline visit, which is the
# only one at which they were run. Genotype has no visit, so a carrier flag is carried for
# the whole cohort and a participant who was never sequenced is a non-carrier rather than a
# missing value, which is what makes those four candidates span all 813.
#
# Usage:
#   python 52_multimodal_screen.py --probe   # coverage only, writes nothing, fast
#   python 52_multimodal_screen.py           # full screen, compares, writes nothing
#   python 52_multimodal_screen.py --write   # full screen, rewrites the table

import argparse
import csv
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter
from scipy.stats import chi2, norm

import cohort as C
from config import PROJECT_ROOT

TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
PUBLISHED = os.path.join(TAB, "tab_incremental_value_all_domains.csv")
REBUILT = os.path.join(TAB, "tab_incremental_value_all_domains_rebuilt.csv")
PER_UNIT = {"orthostasis", "quip_any", "GBA", "LRRK2", "SNCA", "APOE_e4", "any_mut",
            "CSFSAA_pos"}
_CURATED = None

# The constant the published Supplementary Table S4 was built with. See screen() for how it
# was recovered and for why it is 85 per cent power rather than the 80 the header claims.
PUBLISHED_K = 3.0


def ip(d):
    d = d.copy()
    d["PATNO"] = pd.to_numeric(d["PATNO"], errors="coerce").astype("Int64")
    return d.dropna(subset=["PATNO"])


def raw(*parts):
    return os.path.join(C.RAW, *parts)


# ============================================================== candidate construction ===
def curated(tz):
    """The curated cut, dated in days since levodopa, read once and cached."""
    global _CURATED
    if _CURATED is None:
        cur = ip(pd.read_parquet(os.path.join(PROJECT_ROOT, "02_Dados_Processados",
                                              "curated_cache.parquet")))
        cur["visit_date"] = pd.to_datetime(cur["visit_date"], errors="coerce")
        cur = cur.merge(tz, on="PATNO", how="inner")
        cur["dsd"] = (cur["visit_date"] - cur["tzero_levodopa"]).dt.days
        _CURATED = cur
    return _CURATED


def curated_at(tz, columns):
    """The last pre-levodopa value of each column, taken one column at a time so that a
    marker measured at an earlier visit than its neighbours is not discarded with them."""
    cur = curated(tz)
    out = None
    for c in [c for c in columns if c in cur.columns]:
        sub = cur[cur["dsd"] <= 0].dropna(subset=[c])
        last = sub.sort_values(["PATNO", "dsd"]).groupby("PATNO").last().reset_index()
        piece = ip(last)[["PATNO", c]]
        out = piece if out is None else out.merge(piece, on="PATNO", how="outer")
    return out


def genetics(tz):
    """Carrier status from the genetic testing log. A carrier is a participant with a
    positive result (MUTRSLT = 1), not merely one in whom a variant code was entered:
    the code records which gene was interrogated, and reading it as a carrier flag turns
    everyone who was tested into a carrier. A gene-specific carrier additionally needs that
    gene's own variant code filled in, because a positive result is recorded on the row of
    the panel that found it and the category field alone attributes one LRRK2 positive to a
    gene whose code is blank. A participant never sequenced is recorded as a non-carrier
    rather than as missing, which is the convention the published screen used and the
    reason these four candidates cover the whole development sample."""
    g = ip(pd.read_csv(raw("Geneticos", "Genetic_Testing_Results_29Apr2026.csv")))
    for c in ["MUTRSLT", "LRRKCD", "SNCACD", "GBACD"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    positive = g[g["MUTRSLT"] == 1]
    out = pd.DataFrame({"PATNO": tz["PATNO"].drop_duplicates().sort_values().values})
    for name, code in [("LRRK2", "LRRKCD"), ("SNCA", "SNCACD"), ("GBA", "GBACD")]:
        carriers = set(positive.loc[positive[code].notna(), "PATNO"])
        out[name] = out["PATNO"].isin(carriers).astype(int)
    out["any_mut"] = out["PATNO"].isin(set(positive["PATNO"])).astype(int)
    return out


def apoe():
    """Number of e4 alleles, from the genotype string recorded in the biospecimen file."""
    rows = []
    with open(raw("Biomarcadores_LCR_Sangue",
                  "Current_Biospecimen_Analysis_Results_29Apr2026.csv"),
              newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        i_pat, i_name = header.index("PATNO"), header.index("TESTNAME")
        i_val = header.index("TESTVALUE")
        for row in reader:
            if len(row) > max(i_pat, i_name, i_val) and \
                    row[i_name].strip().lower() == "apoe genotype":
                rows.append((row[i_pat], row[i_val]))
    d = ip(pd.DataFrame(rows, columns=["PATNO", "genotype"]))
    d = d[d["genotype"].astype(str).str.contains("4|3|2", regex=True)]
    d["APOE_e4"] = d["genotype"].astype(str).str.count("4")
    return d.groupby("PATNO", as_index=False)["APOE_e4"].max()


def prs():
    """The three polygenic scores. They are standardised inside the screen like every
    other continuous candidate, so they enter here on their native scale."""
    p = pd.read_csv(os.path.join(PROJECT_ROOT, "PRSresult_ppmi.txt"), sep="\t")
    p["PATNO"] = pd.to_numeric(p["participant_id"].str.replace("PP-", "", regex=False),
                               errors="coerce").astype("Int64")
    for c in ["PRS_Nalls", "PRS_Progression", "PRS_all"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["PATNO"]).rename(columns={"PRS_Nalls": "PRS_Nalls_z",
                                                   "PRS_Progression": "PRS_Progression_z"})
    return p[["PATNO", "PRS_Nalls_z", "PRS_Progression_z", "PRS_all"]]


def saa(tz):
    """Seed amplification assay status at the last cerebrospinal fluid sample taken before
    levodopa. The curated cut codes the result 1 for positive and 0 for negative, and
    reserves 2 and 3 for the indeterminate and the not-run; those fourteen are missing,
    not negative, and calling them negative would understate the positive fraction."""
    s = curated_at(tz, ["CSFSAA"])
    s["CSFSAA_pos"] = pd.to_numeric(s["CSFSAA"], errors="coerce")
    return s.loc[s["CSFSAA_pos"].isin([0, 1]), ["PATNO", "CSFSAA_pos"]]


def datscan_striatal(tz):
    """The three striatal summaries the screen carries beside bilateral binding, built from
    the curated regional values. The worse side is the lower of the two, because lower
    binding is worse. The asymmetry index is the absolute difference divided by the mean of
    the two sides, the conventional form: a raw difference of 0.3 means something quite
    different in a striatum binding at 1.0 than in one binding at 3.0, and the undivided
    version is the one place in this reconstruction where using the obvious quantity
    instead of the conventional one changes the answer, from 1.008 to 0.963."""
    d = curated_at(tz, ["MIA_PUTAMEN_L", "MIA_PUTAMEN_R", "MIA_CAUDATE_L", "MIA_CAUDATE_R",
                        "MIA_STRIATUM_BILAT"])
    for c in d.columns.drop("PATNO"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["datscan_put_worse"] = d[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
    d["datscan_cau_worse"] = d[["MIA_CAUDATE_L", "MIA_CAUDATE_R"]].min(axis=1)
    d["datscan_put_asym"] = ((d["MIA_PUTAMEN_L"] - d["MIA_PUTAMEN_R"]).abs()
                             / ((d["MIA_PUTAMEN_L"] + d["MIA_PUTAMEN_R"]) / 2))
    return d[["PATNO", "datscan_put_worse", "datscan_cau_worse", "datscan_put_asym",
              "MIA_STRIATUM_BILAT"]]


SUBREGIONS = {
    "dat_precaudate": "PRECAUDATE", "dat_poscaudate": "POSCAUDATE",
    "dat_precomm_putamen": "PRECOMMISSURAL_PUTAMEN",
    "dat_poscomm_putamen": "POSCOMMISSURAL_PUTAMEN",
    "dat_predorsal_putamen": "PREDORSALPUTAMEN",
    "dat_preventral_putamen": "PREVENTRALPUTAMEN",
    "dat_posdorsal_putamen": "POSDORSALPUTAMEN",
    "dat_posventral_putamen": "POSVENTRALPUTAMEN",
}


def datscan_subregional(tz):
    """Subregional striatal binding from the Xing core laboratory quantification, at the
    last scan before levodopa. The bilateral column the laboratory reports is used rather
    than the mean of its own left and right columns: the two agree to three decimals but
    not exactly, because the bilateral value is computed over the combined volume of
    interest rather than averaged after the fact."""
    x = ip(pd.read_csv(raw("Imagem_DaTSCAN_MRI",
                           "Xing_Core_Lab_-_Quant_SBR_29Apr2026.csv"), low_memory=False))
    x["DATSCAN_DATE"] = pd.to_datetime(x["DATSCAN_DATE"], errors="coerce", format="mixed")
    x = x.merge(tz, on="PATNO", how="inner")
    x['dsd'] = (x['DATSCAN_DATE'] - x['tzero_levodopa']).dt.days
    x = x[x['dsd'] <= 0]
    out = pd.DataFrame({"PATNO": x["PATNO"].values, "dsd": x["dsd"].values})
    for name, stem in SUBREGIONS.items():
        bilateral = f"{stem}_REF_CWM"
        if bilateral in x.columns:
            out[name] = pd.to_numeric(x[bilateral], errors="coerce").values
    have = [c for c in SUBREGIONS if c in out.columns]
    out = out.dropna(subset=have, how="all")
    out = out.sort_values(["PATNO", "dsd"]).groupby("PATNO").last().reset_index()
    return ip(out).drop(columns=["dsd"])


LOBES = {
    "mri_cth_frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal",
                        "parsopercularis", "parsorbitalis", "parstriangularis",
                        "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                        "frontalpole"],
    "mri_cth_parietal": ["superiorparietal", "inferiorparietal", "supramarginal",
                         "postcentral", "precuneus"],
    "mri_cth_temporal": ["superiortemporal", "middletemporal", "inferiortemporal",
                         "bankssts", "fusiform", "transversetemporal", "entorhinal",
                         "temporalpole", "parahippocampal"],
    "mri_cth_occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "mri_cth_cingulate": ["rostralanteriorcingulate", "caudalanteriorcingulate",
                          "posteriorcingulate", "isthmuscingulate"],
}
ASEG_PAIRED = {
    "mri_accumbens": "Accumbens_area", "mri_amygdala": "Amygdala",
    "mri_hippocampus": "Hippocampus", "mri_thalamus": "Thalamus",
    "mri_putamen": "Putamen", "mri_pallidum": "Pallidum", "mri_caudate": "Caudate",
    "mri_ventricle": "Lateral_Ventricle",
}
ASEG_WHOLE = {
    "mri_cortex": "CortexVol", "mri_total_gray": "TotalGrayVol",
    "mri_subcort_gray": "SubCortGrayVol", "mri_white_matter": "CerebralWhiteMatterVol",
    "mri_wm_hypo": "WM_hypointensities",
}


def mri():
    """Baseline FreeSurfer 7 morphometry. Volumes are expressed per unit of estimated
    total intracranial volume, so that head size does not masquerade as atrophy; the
    intracranial volume itself is carried unscaled as its own candidate. Cortical
    thickness is already scale-free and is averaged over the two hemispheres within
    each lobe."""
    a = ip(pd.read_csv(raw("Imagem_DaTSCAN_MRI", "FS7_ASEG_VOL_29Apr2026.csv"),
                       low_memory=False))
    a = a[a["EVENT_ID"] == "BL"].groupby("PATNO", as_index=False).first()
    icv = pd.to_numeric(a["EstimatedTotalIntraCranialVol"], errors="coerce")
    out = pd.DataFrame({"PATNO": a["PATNO"].values, "mri_icv": icv.values})
    for name, stem in ASEG_PAIRED.items():
        left = pd.to_numeric(a[f"Left_{stem}"], errors="coerce")
        right = pd.to_numeric(a[f"Right_{stem}"], errors="coerce")
        out[name] = ((left + right) / icv).values
    for name, col in ASEG_WHOLE.items():
        out[name] = (pd.to_numeric(a[col], errors="coerce") / icv).values

    t = ip(pd.read_csv(raw("Imagem_DaTSCAN_MRI", "FS7_APARC_CTH_29Apr2026.csv"),
                       low_memory=False))
    t = t[t["EVENT_ID"] == "BL"].groupby("PATNO", as_index=False).first()
    cth = pd.DataFrame({"PATNO": t["PATNO"].values})
    for name, regions in LOBES.items():
        cols = [f"{h}_{r}" for h in ("lh", "rh") for r in regions if f"{h}_{r}" in t.columns]
        cth[name] = t[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).values
    # The published global thickness averages the sixty-eight parcels together with the two
    # hemispheric means FreeSurfer already reports, so the hemispheric means are counted
    # twice. That is reproduced here rather than silently corrected, because this is the
    # file the supplement was built from. It makes no difference to anything: averaging the
    # sixty-eight parcels alone gives the same hazard ratio of 0.980 with p 0.784 instead
    # of 0.783, and the candidate is null either way.
    parcels = sorted({c[3:] for c in t.columns
                      if c.startswith("lh_") and c != "lh_MeanThickness"})
    every = [f"{h}_{r}" for h in ("lh", "rh") for r in parcels if f"{h}_{r}" in t.columns]
    cth["mri_cth_global"] = t[every + ["lh_MeanThickness", "rh_MeanThickness"]].apply(
        pd.to_numeric, errors="coerce").mean(axis=1).values
    return out.merge(cth, on="PATNO", how="outer")


BIOSPECIMEN = {
    "bio_gl2": "total gl2", "bio_glccer": "total glccer", "bio_sm": "total sm",
    "bio_progranulin": "progranulin", "bio_gcase": "gcase activity",
    "bio_ps65ub": "ps65 ubiquitin",
}


def biospecimen(tz):
    """The lysosomal and mitophagy assays at the baseline visit. These were run once, on
    the baseline draw, in the sub-studies that requested them; a later sample exists for
    some participants but it is post-levodopa by construction, so restricting to baseline
    is what keeps the candidate pre-treatment. The results file is read line by line
    because one record carries an unbalanced quote that stops the C parser partway
    through."""
    wanted = {v: k for k, v in BIOSPECIMEN.items()}
    rows = []
    path = raw("Biomarcadores_LCR_Sangue",
               "Current_Biospecimen_Analysis_Results_29Apr2026.csv")
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        i_pat, i_name = header.index("PATNO"), header.index("TESTNAME")
        i_val, i_ev = header.index("TESTVALUE"), header.index("CLINICAL_EVENT")
        for row in reader:
            if len(row) <= max(i_pat, i_name, i_val, i_ev):
                continue
            key = wanted.get(row[i_name].strip().lower())
            if key and row[i_ev] == "BL":
                rows.append((row[i_pat], key, row[i_val]))
    d = ip(pd.DataFrame(rows, columns=["PATNO", "assay", "value"]))
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value"]).groupby(["PATNO", "assay"], as_index=False).last()
    return d.pivot(index="PATNO", columns="assay", values="value").reset_index()


def ledd():
    """The one treatment candidate of the screen, from the per-participant conversion that
    27_treatments_at_time_zero.py writes, reading an absent conversion as a zero dose so
    that the row spans all 813 the way the published one did.

    This is the single candidate of the eighty-nine that does not reproduce, and it is the
    one already known not to: 50_treatment_candidate_recomputed.py established that no
    levodopa-equivalent column survives in the processed data and no script builds one, so
    the published dose variable cannot be recovered, only approximated. The approximation
    here gives 1.132 with p 0.061 against the published 1.092 with p 0.153. Neither is the
    number the supplement reports. The published row used the conversion with amantadine
    still inside it, which contradicts the Methods, and script 50 replaces it with the
    amantadine-free dose: 1.10, p 0.160. That is the row Supplementary Table S4 carries,
    and this one exists so that the discrepancy is visible rather than buried."""
    path = os.path.join(TAB, "tab27_treatments_per_patient.csv")
    if not os.path.exists(path):
        return pd.DataFrame({"PATNO": [], "LEDD": []})
    per = ip(pd.read_csv(path))
    return pd.DataFrame({"PATNO": per["PATNO"],
                         "LEDD": pd.to_numeric(per["ledd_total_mg"],
                                               errors="coerce").fillna(0.0)})


def csf_composite(frame, ids):
    """The first principal component of alpha-synuclein, total tau and phosphorylated tau.
    The three measure overlapping biology and each was screened on its own; the component
    is carried so that the screen also asks whether their shared variation, which is better
    measured than any one of them, carries signal the individual markers do not. Amyloid
    beta is deliberately not in it: adding a fourth marker measured in fewer participants
    would cost a sixth of the sample to buy a construct that is no longer a summary of the
    synucleinopathy markers.

    Two details decide the number. The component is extracted inside the development
    sample, not across everyone the assays were ever run on, so that the weights describe
    the participants the model is about. And the sign of a principal component is
    arbitrary, so it is fixed: the three markers all load in the same direction, and the
    component is oriented to load positively on them, which makes a high score mean high
    levels of all three and a hazard ratio below one mean what a reader expects it to."""
    cols = ["asyn", "tau", "ptau"]
    if not all(c in frame.columns for c in cols):
        return pd.DataFrame({"PATNO": [], "csf_composite": []})
    sub = frame.loc[frame["PATNO"].isin(ids), ["PATNO"] + cols].dropna()
    if len(sub) < 20:
        return pd.DataFrame({"PATNO": [], "csf_composite": []})
    x = sub[cols].apply(lambda s: (s - s.mean()) / s.std()).values
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    loading = vt[0] * (-1 if vt[0].sum() < 0 else 1)
    return pd.DataFrame({"PATNO": sub["PATNO"].values, "csf_composite": x @ loading})


CURATED_CLINICAL = [
    "bjlot", "DVT_FAS", "DVS_JLO_MSSAE", "DVS_LNS", "TMT_A", "COG_COMPOSITE_INT",
    "DVS_BNT", "DVT_TOTAL_RECALL", "moca", "DVT_RECOG_DISC_INDEX", "TMT_B", "DVT_SDM",
    "DVT_SFTANIM", "DVT_RETENTION", "DVT_DELAYED_RECALL", "clockdraw",
    "orthostasis", "scopa_cv", "stai_state", "rem", "scopa_gi", "scopa_sex", "scopa_pm",
    "scopa_therm", "quip_any", "gds", "scopa_ur", "upsit", "ess", "scopa", "stai_trait",
]
CURATED_ASSAY = ["asyn", "tau", "ptau", "NFL_CSF", "abeta", "total_di_18_1_BMP",
                 "total_di_22_6_BMP", "nfl_serum", "urate"]


def build_candidates(tz, ids):
    """Every candidate of the screen, on one row per participant."""
    frame = curated_at(tz, CURATED_CLINICAL)
    for piece in [curated_at(tz, CURATED_ASSAY),
                  datscan_striatal(tz), datscan_subregional(tz), mri(),
                  biospecimen(tz), genetics(tz), apoe(), prs(), saa(tz), ledd()]:
        frame = frame.merge(ip(piece), on="PATNO", how="outer")
    for c in frame.columns.drop("PATNO"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.merge(csf_composite(frame, ids), on="PATNO", how="left")


# ================================================================== the screen itself ====
def screen(d, col):
    """Six standardised clinical predictors plus the candidate, ridge 0.05. The hazard
    ratio is reported on the scale the published table uses for that candidate; the p
    values do not depend on that choice."""
    base_cols = list(C.TOTAL6)
    x = d[base_cols + [col, "exit", "event"]].dropna().copy()
    if len(x) < 30 or x.event.sum() < 5 or x[col].std() == 0:
        return None
    for c in base_cols:
        x[c] = (x[c] - x[c].mean()) / x[c].std()
    scaled = x.copy()
    scaled[col] = (x[col] - x[col].mean()) / x[col].std()
    full = CoxPHFitter(penalizer=C.RIDGE).fit(scaled, "exit", "event")
    base = CoxPHFitter(penalizer=C.RIDGE).fit(x[base_cols + ["exit", "event"]],
                                              "exit", "event")
    reported = CoxPHFitter(penalizer=C.RIDGE).fit(x, "exit", "event") \
        if col in PER_UNIT else full
    r = reported.summary.loc[col]
    lrt = 2 * (full.log_likelihood_ - base.log_likelihood_)

    # The smallest hazard ratio the screen could have detected for this candidate, which is
    # what tells a reader whether a null is an absent effect or an unexamined one. Schoenfeld
    # gives it in closed form for a Cox model: with d events and a predictor of standard
    # deviation s, an effect of log hazard ratio b is detected with power 1-B at two-sided
    # level a when b x s x sqrt(d) = z_(1-a/2) + z_(1-B). So the detectable ratio is
    # exp(K / (sqrt(d) x s)), with s equal to one for the standardised continuous candidates
    # and to the sample standard deviation for the eight reported per unit.
    #
    # Two constants are returned because the published table used the second. Solving for K
    # across all 88 checkable rows of Supplementary Table S4 gives 3.00 with a spread of
    # 0.03, which is exactly the scatter that rounding the printed value to two decimals
    # produces. But K = 3.00 is not the 80% the column header claims: z_0.975 + z_0.80 is
    # 2.8016, and 3.00 solves to z of 1.04, which is 85% power. The published numbers are
    # right for 85% and about 0.02 too large for 80%.
    #
    # Both are given so the discrepancy is visible and so the table can be checked against
    # the constant it was actually built with, rather than the one it says it used.
    events = int(x.event.sum())
    spread = 1.0 if col not in PER_UNIT else float(x[col].std())
    root = np.sqrt(events) * spread
    k80 = norm.ppf(0.975) + norm.ppf(0.80)
    return dict(n=len(x), events=events, HR=float(r["exp(coef)"]),
                lo=float(r["exp(coef) lower 95%"]), hi=float(r["exp(coef) upper 95%"]),
                p=float(r["p"]), LRT_p=float(chi2.sf(max(lrt, 0), 1)),
                min_HR_80pct=round(float(np.exp(k80 / root)), 3),
                min_HR_85pct=round(float(np.exp(PUBLISHED_K / root)), 3))


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    adj[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(adj, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true",
                        help="print coverage against the published table and stop")
    parser.add_argument("--write", action="store_true", help="write the rebuilt table")
    args = parser.parse_args()

    published = pd.read_csv(PUBLISHED)
    tz = ip(pd.read_parquet(os.path.join(C.PROC, "tzero.parquet"))
            [["PATNO", "tzero_levodopa"]].dropna())
    tz["tzero_levodopa"] = pd.to_datetime(tz["tzero_levodopa"])
    df = C.build_cohort(with_metadata=False)
    print("development sample: n=%d, %d events" % (len(df), int(df.event.sum())))

    candidates = build_candidates(tz, set(df.PATNO))
    print("candidate frame: %d participants, %d candidate columns"
          % (len(candidates), len(candidates.columns) - 1))

    absent = [v for v in published.variable
              if v not in candidates.columns and v not in df.columns]
    if absent:
        print("NOT REBUILT (%d): %s" % (len(absent), ", ".join(absent)))

    merged = df.merge(candidates[["PATNO"] + [c for c in candidates.columns
                                              if c != "PATNO" and c not in df.columns]],
                      on="PATNO", how="left")

    # -------------------------------------------------------------------------- probe --
    if args.probe:
        print("\n%-24s %-12s %-12s %s" % ("candidate", "rebuilt", "published", "coverage"))
        exact = 0
        for _, row in published.iterrows():
            v = row.variable
            if v not in merged.columns:
                print("%-24s %-12s %-12s not rebuilt"
                      % (v, "-", "%d/%d" % (row.n, row.events)))
                continue
            sub = merged[list(C.TOTAL6) + [v, "exit", "event"]].dropna()
            same = len(sub) == row.n and int(sub.event.sum()) == row.events
            exact += same
            print("%-24s %-12s %-12s %s"
                  % (v, "%d/%d" % (len(sub), int(sub.event.sum())),
                     "%d/%d" % (row.n, row.events), "exact" if same else "differs"))
        print("\n%d of %d candidates recover the published sample exactly"
              % (exact, len(published)))
        return

    # ------------------------------------------------------------------- full screen ---
    print("\n%-24s %-22s %-22s %s"
          % ("candidate", "rebuilt HR (p)", "published HR (p)", "agreement"))
    rows, agree = [], 0
    for _, row in published.iterrows():
        v = row.variable
        got = screen(merged, v) if v in merged.columns else None
        if got is None:
            keep = dict(row)
            keep["source"] = "published, not rebuilt"
            rows.append(keep)
            print("%-24s %-22s %-22s not rebuilt"
                  % (v, "-", "%.3f (%.4f)" % (row.HR, row.p)))
            continue
        same = abs(got["HR"] - row.HR) < 0.002 and abs(got["p"] - row.p) < 1e-4
        agree += same
        print("%-24s %-22s %-22s %s"
              % (v, "%.3f (%.4f)" % (got["HR"], got["p"]),
                 "%.3f (%.4f)" % (row.HR, row.p), "match" if same else "DIFFERS"))
        rows.append(dict(domain=row.domain, variable=v, label=row.label, unit=row.unit,
                         n=got["n"], events=got["events"], HR=round(got["HR"], 3),
                         lo=round(got["lo"], 3), hi=round(got["hi"], 3), p=got["p"],
                         LRT_p=got["LRT_p"], min_HR_80pct=got["min_HR_80pct"], min_HR_85pct=got["min_HR_85pct"],
                         source="rebuilt"))
    print("\n%d of %d candidates reproduce the published hazard ratio and p"
          % (agree, len(published)))

    out = pd.DataFrame(rows)

    # The false-discovery adjustment is computed over the dose row the supplement actually
    # reports, which is the amantadine-free one from 50_treatment_candidate_recomputed.py,
    # not the approximation rebuilt here. The two differ, 0.160 against 0.061, and a single
    # p value shifts every adjusted p in a Benjamini-Hochberg step-up: leaving the
    # approximation in moves the global adjustment of the four cerebrospinal fluid
    # candidates from 0.774 to 0.771 and would put this table quietly out of step with the
    # supplement over a row that neither of them reports.
    corrected = os.path.join(TAB, "tab50_treatment_candidate.json")
    p_for_fdr = out.p.values.copy()
    if os.path.exists(corrected):
        with open(corrected, encoding="utf-8") as fh:
            fixed = json.load(fh)
        dose = out.index[out.domain.str.startswith("G.")]
        if len(dose):
            p_for_fdr[dose[0]] = fixed["recomputed"][fixed["primary"]]["p"]
            print("dose row: false-discovery adjustment uses the amantadine-free p of "
                  "%.4f from script 50, not the %.4f rebuilt here"
                  % (p_for_fdr[dose[0]], out.p.iloc[dose[0]]))
    else:
        print("script 50 has not been run, so the false-discovery columns use the dose "
              "row rebuilt here and will not match the supplement")
    out["FDR_domain"] = pd.Series(p_for_fdr, index=out.index).groupby(
        out.domain).transform(lambda s: bh(s.values))
    out["FDR_global"] = bh(p_for_fdr)
    print("nominally associated at p<0.05: %d; surviving global false-discovery control "
          "at 0.05: %d" % (int((out.p < 0.05).sum()), int((out.FDR_global < 0.05).sum())))
    if args.write:
        out.to_csv(REBUILT, index=False)
        print("written %s" % REBUILT)
    else:
        print("run with --write to save the rebuilt table")


if __name__ == "__main__":
    main()
