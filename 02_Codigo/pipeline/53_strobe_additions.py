# 53_strobe_additions.py
# Produces: the quantities the STROBE checklist asks for that no earlier script reported,
#           and the descriptive numbers of the Discussion that until now had no producer.
#
#   tab53_strobe_additions.json         every number below, with its definition
#   tab53_unadjusted_hazard_ratios.csv  Supplementary Table of unadjusted hazard ratios
#
# Why this script exists. Restructuring the manuscript against STROBE in October 2026
# exposed two kinds of gap. The first is items the paper never reported: total person-time
# (14c), an incidence rate (15), how often participants were assessed (6a), unadjusted
# estimates beside the adjusted ones (16a) and the boundaries of the risk tertiles in
# Figure 2 (16b). The second is numbers the paper did report that no script produced: the
# timing of the predictor visit, the participants examined in the ON state, those assessed
# in the calendar month of levodopa initiation, the eligible participants with no
# pre-levodopa assessment, those whose TD/PIGD ratio is undefined, and the two worked
# patients of the Discussion with their centiles. They came from an audit whose code was not
# kept. Every one is recomputed here from the raw tables.
#
# One finding changed the text. The six predictors do not come from one visit. Total
# MDS-UPDRS, body-mass index, age at onset and sex are taken from the last visit on or
# before time-zero in the PPMI Curated Data Cut, which carries only some scheduled visits
# and lies a median of six months before levodopa initiation. The TD/PIGD ratio and
# freezing of gait are derived from the raw MDS-UPDRS forms, which include every visit and
# the unscheduled symptomatic-therapy visit, and lie a median of a month before. The
# manuscript had described all six as taken at a single visit a median of 31 days before;
# 31 days is right for the second group only. Both timings are reported below.
#
# Usage:
#   python 53_strobe_additions.py

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter

import cohort as C

TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
OUT_JSON = os.path.join(TAB, "tab53_strobe_additions.json")
OUT_HR = os.path.join(TAB, "tab53_unadjusted_hazard_ratios.csv")
SIX_MONTHS = 180   # days; PPMI dates carry month precision, so this is six calendar months

# the increments at which Table 2 reports each hazard ratio
INCREMENT = {"updrs_totscore": (10, "per 10 points"), "ageonset": (10, "per 10 years"),
             "SEX": (1, "male versus female"), "BMI": (5, "per 5 kg/m2"),
             "td_pigd_ratio": (1, "per unit"), "NP2FREZ": (1, "per point")}
LABEL = {"updrs_totscore": "Total MDS-UPDRS (I+II+III)", "ageonset": "Age at onset",
         "SEX": "Sex", "BMI": "Body-mass index", "td_pigd_ratio": "TD/PIGD ratio",
         "NP2FREZ": "Freezing of gait (item 2.13)"}

# the two patients of the Discussion
PATIENTS = {
    "man_68": {"updrs_totscore": 32, "ageonset": 68, "SEX": 1, "BMI": 30,
               "td_pigd_ratio": 3.0, "NP2FREZ": 0},
    "woman_55": {"updrs_totscore": 53, "ageonset": 55, "SEX": 0, "BMI": 24,
                 "td_pigd_ratio": 0.8, "NP2FREZ": 1},
}


def ip(d):
    d = d.copy()
    d["PATNO"] = pd.to_numeric(d["PATNO"], errors="coerce").astype("Int64")
    return d.dropna(subset=["PATNO"])


def last_before(frame, datecol, tz):
    """The last record on or before time-zero, one per participant. Ties on the date are
    broken by file order, which is what 03_baseline_predictor_matrix.py does."""
    m = frame.merge(tz, on="PATNO")
    m["dsd"] = (m[datecol] - m["tzero_levodopa"]).dt.days
    return m[m.dsd <= 0].sort_values(["PATNO", "dsd"]).groupby("PATNO").tail(1)


def raw_pigd_mean(tz):
    """The postural-instability gait-difficulty mean on the raw-form visit, computed exactly
    as 03_baseline_predictor_matrix.py computes the denominator of the TD/PIGD ratio:
    items recoded to 0 to 4, last record on or before time-zero, Part III gait, freezing
    and postural stability with Part II walking and freezing."""
    def item04(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.where((s >= 0) & (s <= 4))

    def prelv(name):
        x = ip(pd.read_csv(os.path.join(C.RAW, "MDS_UPDRS_e_Motor", name), low_memory=False))
        x["INFODT"] = pd.to_datetime(x.INFODT, errors="coerce", format="mixed")
        x = x.merge(tz, on="PATNO", how="inner")
        x["dsd"] = (x.INFODT - x.tzero_levodopa).dt.days
        return x[x.dsd <= 0].sort_values(["PATNO", "dsd"]).groupby("PATNO").last().reset_index()

    p3 = prelv("MDS-UPDRS_Part_III_29Apr2026.csv")
    p2 = prelv("MDS_UPDRS_Part_II__Patient_Questionnaire_29Apr2026.csv")
    for c in ("NP3GAIT", "NP3FRZGT", "NP3PSTBL"):
        p3[c] = item04(p3[c])
    for c in ("NP2WALK", "NP2FREZ"):
        p2[c] = item04(p2[c])
    m = p3.merge(p2[["PATNO", "NP2WALK", "NP2FREZ"]], on="PATNO", how="left")
    m["pigd_raw"] = m[["NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP2WALK", "NP2FREZ"]].mean(axis=1)
    m["PATNO"] = m.PATNO.astype(int)
    return m[["PATNO", "pigd_raw"]]


def timing(pre, datecol):
    gap = -pre["dsd"]
    month = pre[datecol].dt.to_period("M") == pre["tzero_levodopa"].dt.to_period("M")
    return {"median_days": float(gap.median()), "q1_days": float(gap.quantile(.25)),
            "q3_days": float(gap.quantile(.75)),
            "within_six_months_pct": float(100 * (gap <= SIX_MONTHS).mean()),
            "same_calendar_month_n": int(month.sum()),
            "same_calendar_month_pct": float(100 * month.mean())}


def ci(model, col, k):
    r = model.summary.loc[col]
    return (float(np.exp(k * r["coef"])), float(np.exp(k * r["coef lower 95%"])),
            float(np.exp(k * r["coef upper 95%"])), float(r["p"]))


def main():
    out = {}
    d = C.build_cohort(with_metadata=True)
    ids = set(d.PATNO.astype(int))
    n, events = len(d), int(d.event.sum())

    # ---- STROBE 14c and 15: person-time and incidence rate
    py = float(d.exit.sum())
    out["person_years"] = py
    out["rate_per_1000_py"] = 1000 * events / py
    print("person-time %.0f years; %.1f events per 1,000 person-years" % (py, out["rate_per_1000_py"]))

    tz = ip(pd.read_parquet(os.path.join(C.PROC, "tzero.parquet")))[["PATNO", "tzero_levodopa"]]
    tz["tzero_levodopa"] = pd.to_datetime(tz.tzero_levodopa)

    # ---- STROBE 6a: how often motor complications were assessed after levodopa
    p4 = ip(pd.read_csv(os.path.join(C.RAW, "MDS_UPDRS_e_Motor",
                                     "MDS-UPDRS_Part_IV__Motor_Complications_29Apr2026.csv"),
                        low_memory=False, usecols=["PATNO", "INFODT"]))
    p4["INFODT"] = pd.to_datetime(p4.INFODT, errors="coerce", format="mixed")
    m = p4.merge(tz, on="PATNO")
    m = m[(m.INFODT >= m.tzero_levodopa) & m.PATNO.astype(int).isin(ids)]
    m = m.sort_values(["PATNO", "INFODT"]).drop_duplicates(["PATNO", "INFODT"])
    gap = m.groupby("PATNO").INFODT.diff().dt.days.dropna() / 30.44
    per = m.groupby("PATNO").size()
    out["part_iv_interval_months"] = {"median": float(gap.median()), "q1": float(gap.quantile(.25)),
                                      "q3": float(gap.quantile(.75))}
    out["part_iv_assessments_per_participant"] = {"median": float(per.median()),
                                                  "q1": float(per.quantile(.25)),
                                                  "q3": float(per.quantile(.75))}
    print("Part IV assessed every %.1f months (median); %d assessments per participant (median)"
          % (gap.median(), per.median()))

    # ---- timing of the predictor visits, by source
    cur = ip(pd.read_excel(C._find("Curated_Data_Cut", "PPMI_Curated_Data_Cut_Public_*.xlsx"),
                           sheet_name=0,
                           usecols=lambda c: c in ["PATNO", "EVENT_ID", "visit_date",
                                                   "age_at_visit", "agediag", "hy", "race"]))
    cur["visit_date"] = pd.to_datetime(cur.visit_date, errors="coerce")
    cpre = last_before(cur, "visit_date", tz)
    cpre = cpre[cpre.PATNO.astype(int).isin(ids)]
    p3 = ip(pd.read_csv(os.path.join(C.RAW, "MDS_UPDRS_e_Motor", "MDS-UPDRS_Part_III_29Apr2026.csv"),
                        low_memory=False))
    p3["INFODT"] = pd.to_datetime(p3.INFODT, errors="coerce", format="mixed")
    rpre = last_before(p3, "INFODT", tz)
    rpre = rpre[rpre.PATNO.astype(int).isin(ids)]
    out["timing_curated_visit"] = timing(cpre, "visit_date")
    out["timing_raw_mdsupdrs_visit"] = timing(rpre, "INFODT")
    for k in ("timing_curated_visit", "timing_raw_mdsupdrs_visit"):
        t = out[k]
        print("%-28s median %.0f days (IQR %.0f to %.0f), %.1f%% within six months, %d (%.1f%%) in the month of initiation"
              % (k, t["median_days"], t["q1_days"], t["q3_days"], t["within_six_months_pct"],
                 t["same_calendar_month_n"], t["same_calendar_month_pct"]))

    # ---- the motor examination used for the TD/PIGD ratio, by medication state
    on = rpre[rpre.PDSTATE == "ON"]
    tr = pd.read_csv(os.path.join(TAB, "tab27_treatments_per_patient.csv"))
    tr["PATNO"] = pd.to_numeric(tr.PATNO, errors="coerce")
    on_ag = on[["PATNO"]].astype({"PATNO": "float"}).merge(tr[["PATNO", "agonist"]], on="PATNO", how="left")
    out["exam_on_state_n"] = int(len(on))
    out["exam_on_state_pct"] = float(100 * len(on) / n)
    out["exam_on_state_on_agonist_n"] = int(on_ag.agonist.fillna(False).astype(bool).sum())
    print("motor examination in the ON state: %d (%.1f%%), %d of them on a dopamine agonist"
          % (len(on), out["exam_on_state_pct"], out["exam_on_state_on_agonist_n"]))

    # ---- eligible participants with no pre-levodopa assessment, and undefined TD/PIGD
    mat = pd.read_parquet(os.path.join(C.PROC, "candidate_matrix.parquet"))
    if "ageonset" not in mat.columns and "ageonset_x" in mat.columns:
        mat = mat.rename(columns={"ageonset_x": "ageonset"})
    elig = mat[mat.exit > 0]
    pool = [c for c in C.POOL if c in elig.columns]
    missing = elig[pool].isna().sum(axis=1)
    none = elig[missing == len(pool)]
    some = elig[missing < len(pool)]
    out["eligible_n"] = int(len(elig))
    out["no_prelevodopa_assessment_n"] = int(len(none))
    out["no_prelevodopa_event_rate_pct"] = float(100 * none.event.mean())
    out["no_prelevodopa_median_followup"] = float(none.exit.median())
    out["development_median_followup"] = float(d.exit.median())
    out["with_some_assessment_n"] = int(len(some))
    # The ratio is computed in 03_baseline_predictor_matrix.py from the raw forms, so the
    # reason it is missing has to be read on that same visit. The pigd column of the
    # candidate matrix comes from the curated cut, a different and earlier visit, and
    # comparing the two attributes 35 of the missing ratios to the wrong cause.
    raw_pigd = raw_pigd_mean(tz)
    missing_ratio = some[some.td_pigd_ratio.isna()][["PATNO"]].copy()
    missing_ratio["PATNO"] = missing_ratio.PATNO.astype(int)
    missing_ratio = missing_ratio.merge(raw_pigd, on="PATNO", how="left")
    undefined = missing_ratio[missing_ratio.pigd_raw == 0]
    out["tdpigd_missing_n"] = int(len(missing_ratio))
    out["tdpigd_missing_because_pigd_zero_n"] = int(len(undefined))
    out["tdpigd_undefined_n"] = int(len(undefined))
    out["tdpigd_undefined_pct"] = float(100 * len(undefined) / len(some))
    print("eligible %d; %d with no pre-levodopa assessment (event rate %.1f%%, median follow-up %.1f years); "
          "%d with at least one, of whom %d (%.1f%%) have an undefined TD/PIGD ratio"
          % (len(elig), len(none), out["no_prelevodopa_event_rate_pct"], out["no_prelevodopa_median_followup"],
             len(some), len(undefined), out["tdpigd_undefined_pct"]))

    # ---- STROBE 16a: unadjusted hazard ratios beside the adjusted ones
    adj = C.fit_total6(d)
    rows = []
    for col in C.TOTAL6:
        k, unit = INCREMENT[col]
        uni = CoxPHFitter(penalizer=0.0).fit(d[[col, "exit", "event"]], "exit", "event")
        u = ci(uni, col, k)
        a = ci(adj, col, k)
        rows.append({"predictor": LABEL[col], "increment": unit,
                     "unadjusted_HR": u[0], "unadjusted_lo": u[1], "unadjusted_hi": u[2], "unadjusted_p": u[3],
                     "adjusted_HR": a[0], "adjusted_lo": a[1], "adjusted_hi": a[2], "adjusted_p": a[3]})
    hr = pd.DataFrame(rows)
    hr.to_csv(OUT_HR, index=False)
    print("\nunadjusted and adjusted hazard ratios:")
    for _, r in hr.iterrows():
        print("  %-30s %-20s unadjusted %.2f (%.2f to %.2f) p=%.4f | adjusted %.2f (%.2f to %.2f)"
              % (r.predictor, r.increment, r.unadjusted_HR, r.unadjusted_lo, r.unadjusted_hi,
                 r.unadjusted_p, r.adjusted_HR, r.adjusted_lo, r.adjusted_hi))

    # ---- STROBE 16b: boundaries of the risk tertiles of Figure 2, as five-year risk
    beta = adj.params_[list(C.TOTAL6)]
    centre = float((d[list(C.TOTAL6)].mean() * beta).sum())
    lp = d[list(C.TOTAL6)].values @ beta.values - centre
    cal = json.load(open(os.path.join(TAB, "tab32_internal_performance.json"), encoding="utf-8"))
    s0_5 = cal["calibration"]["5"]["S0"]
    comp = pd.read_csv(os.path.join(TAB, "tab45_risco_competitivo.csv"))
    c5 = float(comp.loc[comp.horizonte_anos == 5, "c_t"].iloc[0])

    def risk5(x):
        return (1 - s0_5 ** np.exp(x)) * c5

    # the breaks are the 1/3 and 2/3 quantiles of the linear predictor, exactly as
    # figuras_R/figuras.R cuts the tertiles it draws in Figure 2
    tert = pd.qcut(lp, 3, labels=False)
    bounds = [float(np.quantile(lp, 1 / 3)), float(np.quantile(lp, 2 / 3))]
    out["tertile_sizes"] = [int((tert == g).sum()) for g in (0, 1, 2)]
    out["tertile_bounds_risk5_pct"] = [100 * risk5(b) for b in bounds]
    print("\nrisk tertiles of %s participants; boundaries at a five-year risk of %.1f%% and %.1f%%"
          % (out["tertile_sizes"], *out["tertile_bounds_risk5_pct"]))

    # ---- the two patients of the Discussion
    allrisk = risk5(lp)
    out["patients"] = {}
    for name, x in PATIENTS.items():
        v = sum(beta[c] * x[c] for c in C.TOTAL6) - centre
        r = float(risk5(v))
        out["patients"][name] = {"risk5_pct": 100 * r, "centile": float(100 * (allrisk < r).mean())}
        print("%-9s five-year risk %.1f%%, centile %.1f" % (name, 100 * r, out["patients"][name]["centile"]))

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwritten %s and %s" % (os.path.basename(OUT_JSON), os.path.basename(OUT_HR)))


if __name__ == "__main__":
    main()
