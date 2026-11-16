# 04b_table1.py
# Produces: Table 1 of the manuscript, the development sample described, overall and by
#           recruitment wave.
#
# Why this script exists rather than 04_table1_and_incidence.py. That script describes the
# eligible cohort, 1,447 participants with 303 events, split by whether the outcome
# occurred. Table 1 of the manuscript describes something different: the 813 participants
# with a complete set of the six predictors, the sample the model was actually fitted in,
# with 165 events. The two are both legitimate and neither substitutes for the other, but
# only the first had a script. Every number in Table 1 therefore came from a calculation
# nobody could repeat, which for the table that defines who the paper is about is the worst
# place to have that gap. This script closes it.
#
# The six clinical rows come from cohort.py, the single definition of the development
# sample. The four treatment rows come from 27_treatments_at_time_zero.py, which reads what
# each participant was actually prescribed on the day levodopa started. The dose rows are
# medians among those who were taking the drug in question, not among all 813, because a
# median dose that averages in the people not on the drug is not a dose.
#
# Continuous rows are median [interquartile range] and counts are n (%), rounded half away
# from zero rather than half to even, because Python rounds 0.5 down and the difference
# shows up in a table full of percentages.
#
# Usage:
#   python 04b_table1.py            # print the table and compare against the manuscript
#   python 04b_table1.py --write    # also write Table1_development_sample.csv

import argparse
import os
import warnings
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cohort as C

TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
OUT = os.path.join(TAB, "Table1_development_sample.csv")
TREATMENTS = os.path.join(TAB, "tab27_treatments_per_patient.csv")

# what the manuscript currently prints, so that a divergence is visible instead of silent
PUBLISHED = {
    "Age at onset, years": "62.0 [54.9-68.3]",
    "Male, n (%)": "522 (64%)",
    "Total MDS-UPDRS (I+II+III)": "42.0 [32.0-53.0]",
    "Body-mass index, kg/m2": "26.2 [23.9-29.6]",
    "TD/PIGD ratio": "1.6 [0.8-3.0]",
    "Freezing of gait present (item 2.13 above zero), n (%)": "146 (18%)",
    "Problematic dyskinesia (events), n (%)": "165 (20%)",
    "Follow-up from levodopa initiation, years": "2.4 [1.3-4.7]",
    "Levodopa daily dose at time-zero, mg": "300 [250-450]",
    "Total dopaminergic daily dose at time-zero, amantadine excluded, mg": "400 [300-600]",
    "Dopamine agonist at time-zero, n (%)": "227 (28%)",
    "Amantadine at time-zero, n (%)": "66 (8%)",
}

# rows added in October 2026 for STROBE item 14a, which the manuscript did not yet print
NEW_ROWS = {"Age at levodopa initiation, years",
            "Disease duration from diagnosis to levodopa initiation, years",
            "White race, n (%)", "Hoehn and Yahr stage 2 or higher, n (%)"}


def half_up(value, places):
    """Round half away from zero. Python rounds 0.5 to the nearest even digit, which turns
    64.5% into 64% and 8.5% into 8%, so a table of percentages needs the school rule."""
    if pd.isna(value):
        return value
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def iqr(series, places=1):
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    fmt = "%%.%df" % places
    return (fmt % half_up(med, places)) + " [" + (fmt % half_up(q1, places)) + "-" + \
        (fmt % half_up(q3, places)) + "]", len(s)


def count(mask, total):
    n = int(pd.Series(mask).fillna(False).astype(bool).sum())
    return "%d (%d%%)" % (n, int(half_up(100 * n / total, 0))), n


# Added in October 2026, for STROBE item 14a. Age at levodopa initiation and disease
# duration have to be computed here rather than read off the curated cut, because the
# cut's own age and duration_yrs refer to enrolment in PPMI, not to the start of levodopa,
# and in this cohort the two are often years apart. Age at the predictor visit
# (age_at_visit, which PPMI derives from the birth date) is carried forward to time-zero,
# and duration is that age minus age at diagnosis (agediag, also from the birth date).
# Hoehn and Yahr is the stage the candidate matrix already holds for the same visit.
def predictor_visit_ages(ids):
    def ip(x):
        x = x.copy()
        x["PATNO"] = pd.to_numeric(x["PATNO"], errors="coerce").astype("Int64")
        return x.dropna(subset=["PATNO"])

    tz = ip(pd.read_parquet(os.path.join(C.PROC, "tzero.parquet")))[["PATNO", "tzero_levodopa"]]
    tz["tzero_levodopa"] = pd.to_datetime(tz.tzero_levodopa)
    cur = ip(pd.read_excel(C._find("Curated_Data_Cut", "PPMI_Curated_Data_Cut_Public_*.xlsx"),
                           sheet_name=0,
                           usecols=lambda c: c in ["PATNO", "visit_date", "age_at_visit", "agediag"]))
    cur["visit_date"] = pd.to_datetime(cur.visit_date, errors="coerce")
    cur = cur.merge(tz, on="PATNO")
    cur["dsd"] = (cur.visit_date - cur.tzero_levodopa).dt.days
    pre = cur[cur.dsd <= 0].sort_values(["PATNO", "dsd"]).groupby("PATNO").tail(1)
    pre = pre[pre.PATNO.astype(int).isin(ids)].copy()
    pre["age_at_levodopa"] = pre.age_at_visit + (-pre.dsd) / 365.25
    pre["duration_at_levodopa"] = pre.age_at_levodopa - pre.agediag
    pre["PATNO"] = pre.PATNO.astype("Int64")
    return pre[["PATNO", "age_at_levodopa", "duration_at_levodopa"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    d = C.build_cohort(with_metadata=True)
    n = len(d)
    print("development sample: n=%d, %d events" % (n, int(d.event.sum())))
    d["PATNO"] = pd.to_numeric(d["PATNO"], errors="coerce").astype("Int64")
    d = d.merge(predictor_visit_ages(set(d.PATNO.astype(int))), on="PATNO", how="left")

    if not os.path.exists(TREATMENTS):
        raise SystemExit("run 27_treatments_at_time_zero.py first: %s is missing"
                         % os.path.basename(TREATMENTS))
    t = pd.read_csv(TREATMENTS)
    t["PATNO"] = pd.to_numeric(t["PATNO"], errors="coerce").astype("Int64")
    d = d.merge(t, on="PATNO", how="left")

    # Added in November 2026: the two recruitment waves side by side, because the temporal
    # validation fits the model in the earlier wave and applies it to the later one, and a
    # reader judging that validation needs the case-mix of each (TRIPOD+AI item 20c asks
    # for exactly this comparison). The wave is PPMI's own enroll_phase, the split used by
    # 33_temporal_validation.py. The two waves are compared by the absolute standardised
    # difference, the difference in means or proportions over the pooled standard
    # deviation, rather than by p values: the question is how far apart the samples are,
    # not whether 392 and 421 participants can detect a difference. It is not given for the
    # events and the follow-up, which describe what happened after time-zero, not case-mix.
    early = d[d.enroll_phase == 1]
    late = d[d.enroll_phase == 2]
    assert len(early) + len(late) == n, "every participant must belong to one wave"
    assert (len(early), int(early.event.sum()), len(late), int(late.event.sum())) == (392, 145, 421, 20), \
        "the waves no longer match 33_temporal_validation.py and the Results"
    enrol = "visit_date_cur" if "visit_date_cur" in d.columns else "visit_date"
    for label, wave in (("earlier", early), ("later", late)):
        print("%s wave: n=%d, %d events, first visit %s to %s"
              % (label, len(wave), int(wave.event.sum()), wave[enrol].min().date(), wave[enrol].max().date()))
    groups = [("PPMI development sample, n = %d" % n, d),
              ("Earlier wave, enrolled 2010 to 2019, n = %d" % len(early), early),
              ("Later wave, enrolled 2020 to 2025, n = %d" % len(late), late)]
    SMD = "Standardised difference, earlier against later wave"

    # Doses are described among those actually taking the drug. Averaging in the untreated
    # would report a dose nobody was prescribed.
    specs = [
        ("Age at onset, years", 1, lambda x: x.ageonset, True),
        ("Age at levodopa initiation, years", 1, lambda x: x.age_at_levodopa, True),
        ("Disease duration from diagnosis to levodopa initiation, years", 1,
         lambda x: x.duration_at_levodopa, True),
        ("Male, n (%)", None, lambda x: x.SEX == 1, True),
        ("White race, n (%)", None, lambda x: x.race == 1, True),
        ("Total MDS-UPDRS (I+II+III)", 1, lambda x: x.updrs_totscore, True),
        ("Hoehn and Yahr stage 2 or higher, n (%)", None, lambda x: x.hy >= 2, True),
        ("Body-mass index, kg/m2", 1, lambda x: x.BMI, True),
        ("TD/PIGD ratio", 1, lambda x: x.td_pigd_ratio, True),
        ("Freezing of gait present (item 2.13 above zero), n (%)", None, lambda x: x.NP2FREZ > 0, True),
        ("Problematic dyskinesia (events), n (%)", None, lambda x: x.event == 1, False),
        ("Follow-up from levodopa initiation, years", 1, lambda x: x.exit, False),
        ("Levodopa daily dose at time-zero, mg", 0, lambda x: x.levodopa_mg.where(x.levodopa_mg > 0), True),
        ("Total dopaminergic daily dose at time-zero, amantadine excluded, mg", 0,
         lambda x: x.ledd_no_amantadine_mg.where(x.ledd_no_amantadine_mg > 0), True),
        ("Dopamine agonist at time-zero, n (%)", None, lambda x: x.agonist == True, True),
        ("Amantadine at time-zero, n (%)", None, lambda x: x.amantadine == True, True),
    ]

    def standardised_difference(places, a, b):
        if places is None:
            p, q = (s.fillna(False).astype(bool).mean() for s in (a, b))
            pooled = np.sqrt((p * (1 - p) + q * (1 - q)) / 2)
        else:
            a, b = (pd.to_numeric(s, errors="coerce").dropna() for s in (a, b))
            p, q = a.mean(), b.mean()
            pooled = np.sqrt((a.var() + b.var()) / 2)
        return "%.2f" % half_up(abs(p - q) / pooled, 2)

    rows = []
    for label, places, get, baseline in specs:
        row = {"Characteristic": label}
        for name, g in groups:
            s = get(g)
            row[name] = count(s, len(g))[0] if places is None else iqr(s, places)[0]
        row[SMD] = standardised_difference(places, get(early), get(late)) if baseline else ""
        rows.append(row)
    levodopa = d.levodopa_mg.where(d.levodopa_mg > 0)
    total = d.ledd_no_amantadine_mg.where(d.ledd_no_amantadine_mg > 0)

    table = pd.DataFrame(rows)
    column = table.columns[1]
    print()
    print(table.to_string(index=False))
    print("dose rows by wave are among those taking the drug: levodopa %d and %d, any dopaminergic %d and %d"
          % (int(levodopa[d.enroll_phase == 1].notna().sum()), int(levodopa[d.enroll_phase == 2].notna().sum()),
             int(total[d.enroll_phase == 1].notna().sum()), int(total[d.enroll_phase == 2].notna().sum())))
    race = d.race.value_counts(dropna=False)
    print("race, for the legend: White %d, Black %d, Asian %d, other or multiracial %d, not recorded %d"
          % (race.get(1.0, 0), race.get(2.0, 0), race.get(3.0, 0), race.get(4.0, 0), int(d.race.isna().sum())))
    print("Hoehn and Yahr, for the legend:", d.hy.value_counts().sort_index().to_dict())
    print()
    print("%-70s %-22s %s" % ("Characteristic", "this script", "manuscript"))
    disagree = 0
    for _, r in table.iterrows():
        want = PUBLISHED.get(r.Characteristic, "")
        if r.Characteristic in NEW_ROWS:
            print("%-70s %-22s %s" % (r.Characteristic[:70], r[column], "new row"))
            continue
        same = r[column] == want
        disagree += not same
        print("%-70s %-22s %-22s %s"
              % (r.Characteristic[:70], r[column], want, "" if same else "DIFFERS"))
    published = len(table) - len(NEW_ROWS)
    print("\n%d of %d previously published cells reproduce the manuscript; %d rows are new"
          % (published - disagree, published, len(NEW_ROWS)))
    print("median follow-up is the median of exit, the time from levodopa to the event or "
          "to censoring;\nthe dose rows are medians among those taking the drug, %d on "
          "levodopa and %d on any\ndopaminergic drug once amantadine is set aside"
          % (int(levodopa.notna().sum()), int(total.notna().sum())))

    if args.write:
        table.to_csv(OUT, index=False)
        print("\nwritten %s" % OUT)
    else:
        print("\nrun with --write to save the table")


if __name__ == "__main__":
    main()
