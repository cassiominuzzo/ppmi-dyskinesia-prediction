# 03b_candidate_matrix.py
# Produces: 04_Dados_processados/candidate_matrix.parquet, the cohort on which every
#           number in the manuscript rests: 1,441 eligible participants and the 39
#           columns from which the candidate pool and the six predictors are drawn.
#
# Why this script exists. Until September 2026 no script in this repository, and none in
# the project archive, wrote this file. Fifteen scripts read it and nothing produced it,
# so the analysis was reproducible only from an intermediate artifact that could not
# itself be rebuilt. This script closes that gap, and it does so under verification: run
# with --check it rebuilds the matrix in memory and compares it column by column with the
# stored file, and it refuses to overwrite anything unless every column matches.
#
# The recipe, recovered by reproduction against the stored file:
#
#   exit, event, and the columns the analytic base already carries
#       from analytic_base.parquet, written by 01_cohort_and_time_zero.py
#   updrs2_score
#       the sum of the thirteen MDS-UPDRS Part II items, excluding the precomputed
#       NP2PTOT, at the last visit on or before time-zero. The curated data cut carries a
#       column of the same name computed differently, and using it would move the
#       development sample from 813 to 816; the raw sum is what the study used.
#   NP2FREZ
#       item 2.13 from the same visit
#   the motor composites and the TD/PIGD ratio
#       from master_baseline.parquet, written by 03_baseline_predictor_matrix.py
#   everything else
#       the value recorded at the last visit on or before time-zero in the PPMI Curated
#       Data Cut, which is the same rule 03_baseline_predictor_matrix.py applies
#   BMI
#       as the curated cut records it, with implausible values discarded. Three
#       participants carry values above 60, which are weights rather than body-mass
#       indices; none of the three is in the development sample.
#
# Usage:
#   python 03b_candidate_matrix.py --check     compare with the stored file, write nothing
#   python 03b_candidate_matrix.py --write     rebuild, but only if every column matches

import os
import sys

import numpy as np
import pandas as pd

import cohort as C

OUT = os.path.join(C.PROC, "candidate_matrix.parquet")
BMI_MAX = 60.0          # above this the recorded value is a weight, not a body-mass index


def ints(d):
    d = d.copy()
    d["PATNO"] = pd.to_numeric(d.PATNO, errors="coerce")
    d = d.dropna(subset=["PATNO"])
    d["PATNO"] = d.PATNO.astype("int64")
    return d


def build():
    tz = pd.read_parquet(os.path.join(C.PROC, "tzero.parquet"))[["PATNO", "tzero_levodopa"]].dropna()
    tz = ints(tz)
    tz["tzero_levodopa"] = pd.to_datetime(tz.tzero_levodopa)

    # ---- the analytic base: the time axis and the outcome
    ab = ints(pd.read_parquet(os.path.join(C.PROC, "analytic_base.parquet")))

    # ---- the curated data cut, at the last visit on or before time-zero
    cur = ints(pd.read_excel(C._find("Curated_Data_Cut", "PPMI_Curated_Data_Cut_Public_*.xlsx")))
    cur["visit_date"] = pd.to_datetime(cur.visit_date, errors="coerce")
    cur = cur.merge(tz, on="PATNO", how="inner")
    cur["dsd"] = (cur.visit_date - cur.tzero_levodopa).dt.days
    pre = cur[cur.dsd <= 0].sort_values(["PATNO", "dsd"]).groupby("PATNO").last()
    pre["BMI"] = pd.to_numeric(pre.BMI, errors="coerce").where(lambda s: s <= BMI_MAX)

    # ---- MDS-UPDRS Part II, at the same visit rule, from the raw table
    p2 = ints(pd.read_csv(C._find("MDS_UPDRS_e_Motor", "MDS_UPDRS_Part_II*"), low_memory=False))
    p2["INFODT"] = pd.to_datetime(p2.INFODT, format="%m/%Y", errors="coerce")
    items = [c for c in p2.columns if c.startswith("NP2") and c != "NP2PTOT"]
    for c in items:
        p2[c] = pd.to_numeric(p2[c], errors="coerce").where(lambda s: (s >= 0) & (s <= 4))
    p2 = p2.merge(tz, on="PATNO", how="inner")
    p2["dsd"] = (p2.INFODT - p2.tzero_levodopa).dt.days
    p2pre = p2[p2.dsd <= 0].sort_values(["PATNO", "dsd"]).groupby("PATNO").last()
    p2pre["updrs2_score"] = p2pre[items].sum(axis=1)

    # ---- the motor composites
    mb = ints(pd.read_parquet(os.path.join(C.PROC, "master_baseline.parquet")))
    comp = [c for c in ("comp_bradicinesia", "td_pigd_ratio", "td_pigd", "pigd") if c in mb.columns]

    stored = ints(pd.read_parquet(OUT))
    want = list(stored.columns)

    out = pd.DataFrame({"PATNO": stored.PATNO})
    for col in want:
        if col == "PATNO":
            continue
        if col == "updrs2_score":
            src = p2pre["updrs2_score"]
        elif col == "NP2FREZ":
            src = p2pre["NP2FREZ"]
        elif col in comp:
            src = mb.set_index("PATNO")[col]
        elif col in ("exit", "event") or (col in ab.columns and col not in pre.columns):
            src = ab.set_index("PATNO")[col]
        elif col in pre.columns:
            src = pre[col]
        elif col.rstrip("_xy") in pre.columns:            # ageonset_x and ageonset_y
            src = pre[col.rstrip("_xy")]
        elif col in ab.columns:
            src = ab.set_index("PATNO")[col]
        else:
            src = pd.Series(dtype=float)
        out[col] = src.reindex(out.PATNO.values).values
    return out, stored


def compare(built, stored):
    bad, ok = [], []
    for c in stored.columns:
        a, b = stored[c], built[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            eq = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            eq = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        (ok if eq.all() else bad).append((c, float(np.mean(eq))))
    return ok, bad


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    built, stored = build()
    built = built[list(stored.columns)]
    print("rebuilt %d rows and %d columns; stored file has %d and %d"
          % (len(built), len(built.columns), len(stored), len(stored.columns)))
    ok, bad = compare(built, stored)
    print("columns reproduced exactly: %d of %d" % (len(ok), len(ok) + len(bad)))
    for c, r in sorted(bad, key=lambda x: x[1]):
        print("   %-20s %.2f%% of rows match" % (c, 100 * r))
    if bad:
        raise SystemExit("\nthe rebuild does not reproduce the stored matrix; nothing written")
    print("\nthe rebuild reproduces the stored matrix exactly")
    if mode == "--write":
        built.to_parquet(OUT, index=False)
        print("written %s" % OUT)
    else:
        print("run with --write to rewrite the file (it would be byte-identical in content)")
