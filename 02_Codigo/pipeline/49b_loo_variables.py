# 49b_loo_variables.py
# Produces: 04_Dados_processados/loo2024_variables.parquet, the seven baseline factors
#           that Loo and colleagues report, operationalised in PPMI.
#
# Why this is needed. Loo 2024 (Parkinsonism and Related Disorders 126:107054) is a
# cross-cohort machine-learning study rather than a published equation: it reports which
# baseline characteristics carried predictive value, not a set of coefficients. The
# head-to-head comparison in this paper refits every published predictor set in these
# data, so what has to be reproduced is the choice of variables, and that choice is taken
# from the seven factors the paper itself names:
#
#   axial symptoms .................. comp_axial, the axial composite of MDS-UPDRS Part III
#   freezing of gait ................ NP2FREZ, item 2.13
#   rigidity in the lower limbs ..... mean of NP3RIGRL and NP3RIGLL
#   resting tremor .................. comp_tremor, the Part III tremor composite, which covers
#                                     rest, postural and kinetic tremor and is therefore broader
#                                     than the resting tremor the paper names
#   higher body weight .............. BMI, which is the weight measure this study carries
#   later age at onset .............. ageonset
#   visuospatial ability ............ Benton judgement of line orientation, raw score
#
# Two of the seven are not in the candidate matrix and are built here from the raw tables,
# at the same visit the other predictors use: the last study visit on or before time-zero,
# which is the rule of 03_baseline_predictor_matrix.py.
#
# The mapping is a judgement, and it is stated in the legend of Supplementary Table S15 so
# that a reader can disagree with it. Direction is not imposed: every set, this one
# included, is refitted in these data.
#
# Usage:  python 49b_loo_variables.py

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cohort as C

OUT = os.path.join(C.PROC, "loo2024_variables.parquet")


def ip(d):
    d = d.copy()
    d["PATNO"] = pd.to_numeric(d.PATNO, errors="coerce")
    return d.dropna(subset=["PATNO"]).astype({"PATNO": "int64"})


tz = pd.read_parquet(os.path.join(C.PROC, "tzero.parquet"))[["PATNO", "tzero_levodopa"]].dropna()
tz = ip(tz)
tz["tzero_levodopa"] = pd.to_datetime(tz.tzero_levodopa)


def at_baseline(path, cols):
    """The last visit on or before time-zero, the rule used for every other predictor."""
    d = ip(pd.read_csv(path, low_memory=False))
    d["INFODT"] = pd.to_datetime(d.INFODT, errors="coerce", format="mixed")
    d = d.merge(tz, on="PATNO", how="inner")
    d["dsd"] = (d.INFODT - d.tzero_levodopa).dt.days
    d = d[d.dsd <= 0].sort_values(["PATNO", "dsd"]).groupby("PATNO").last().reset_index()
    return d[["PATNO"] + [c for c in cols if c in d.columns]]


p3 = at_baseline(C._find("MDS_UPDRS_e_Motor", "MDS-UPDRS_Part_III_*.csv"), ["NP3RIGRL", "NP3RIGLL"])
for c in ("NP3RIGRL", "NP3RIGLL"):
    p3[c] = pd.to_numeric(p3[c], errors="coerce").where(lambda s: (s >= 0) & (s <= 4))
p3["rigidity_lower"] = p3[["NP3RIGRL", "NP3RIGLL"]].mean(axis=1)

bj = at_baseline(C._find("Nao_Motor_Cognicao_Sono", "Benton_Judgement_of_Line_Orientation_*.csv"),
                 ["JLO_TOTRAW", "DVS_JLO_MSSA", "DVS_JLO_MSSAE"])
col = next((c for c in ("JLO_TOTRAW", "DVS_JLO_MSSA", "DVS_JLO_MSSAE") if c in bj.columns), None)
if col is None:
    raise SystemExit("no Benton score column found; inspect the raw table")
bj["visuospatial"] = pd.to_numeric(bj[col], errors="coerce")
print("Benton score taken from column %s" % col)

mb = ip(pd.read_parquet(os.path.join(C.PROC, "master_baseline.parquet")))
mb = mb[["PATNO", "comp_axial", "comp_tremor"]]

M = pd.read_parquet(os.path.join(C.PROC, "candidate_matrix.parquet"))
if "ageonset" not in M.columns and "ageonset_x" in M.columns:
    M = M.rename(columns={"ageonset_x": "ageonset"})
M = ip(M)[["PATNO", "exit", "NP2FREZ", "BMI", "ageonset"]]

d = (M.merge(mb, on="PATNO", how="left")
      .merge(p3[["PATNO", "rigidity_lower"]], on="PATNO", how="left")
      .merge(bj[["PATNO", "visuospatial"]], on="PATNO", how="left"))

LOO = ["comp_axial", "NP2FREZ", "rigidity_lower", "comp_tremor", "BMI", "ageonset", "visuospatial"]
elig = d[d.exit > 0]
print("\ncompleteness among the %d eligible participants:" % len(elig))
for c in LOO:
    print("   %-16s %4d recorded (%5.1f%%)" % (c, elig[c].notna().sum(), 100 * elig[c].notna().mean()))
print("   %-16s %4d (%5.1f%%)" % ("all seven", elig[LOO].notna().all(axis=1).sum(),
                                  100 * elig[LOO].notna().all(axis=1).mean()))

d[["PATNO"] + [c for c in LOO if c not in ("NP2FREZ", "BMI", "ageonset")]].to_parquet(OUT, index=False)
print("\nsaved %s" % OUT)
