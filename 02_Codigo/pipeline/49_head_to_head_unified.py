# 49_head_to_head_unified.py
# Produces: Supplementary Table S15 and Supplementary Figure S2, from a single analysis.
#
# Why this script exists. The head-to-head comparison used to be split across two scripts
# with two different protocols: one enumerated subsets to match degrees of freedom, the
# other bootstrapped for confidence intervals. They used different complete-case samples
# and different cross-validation, so the same six published predictor sets received two
# different C-indices in the same supplement, differing by up to 0.011. This script does
# both jobs once, in one sample, under the cross-validation protocol used everywhere else
# in the paper: cohort.out_of_fold_c, twenty repetitions of stratified five-fold, seed
# 100 + r, standardisation fitted inside each training fold.
#
# The sample is every eligible participant complete on the twenty-candidate pool, which is
# the eighteen clinical candidates plus disease duration and trait anxiety, the two extra
# variables the published sets need. Using one sample for the subset distributions and for
# the published sets is what makes the percentiles meaningful.
#
# Modes:
#   sets    the published predictor sets, with paired bootstrap confidence intervals
#   dist K  the out-of-fold C of every subset of size K (2 and 3 exhaustive, 6 sampled)
#   summarise   percentiles and the two output tables
#
# Usage:
#   python 49_head_to_head_unified.py sets
#   python 49_head_to_head_unified.py dist 2
#   python 49_head_to_head_unified.py dist 3
#   python 49_head_to_head_unified.py dist 6
#   python 49_head_to_head_unified.py summarise

import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
BOOT = 500
BOOT_SEED = 42

# The two extra candidates the published sets need beyond the 18-variable pool.
EXTRA = ["duration_yrs", "stai", "updrs3_score", "updrs2_score", "pigd"]

# Published predictor sets, refitted in these data so that what is compared is the choice
# of variables and not the coefficients of the original papers. Loo 2024 reports seven
# factors rather than a fitted equation, and how each is operationalised here is set out
# in 49b_loo_variables.py, which builds the two that the candidate matrix does not carry.
PUBLISHED = {
    "Total-6 (this study)": ["updrs_totscore", "ageonset", "SEX", "BMI", "td_pigd_ratio", "NP2FREZ"],
    "Zhao 2025": ["ageonset", "td_pigd_ratio", "updrs3_score"],
    "Olanow / STRIDE-PD": ["ageonset", "SEX", "BMI"],
    "Santos-Lobato 2020": ["ageonset", "duration_yrs", "updrs2_score"],
    "Chen 2021": ["ageonset", "duration_yrs"],
    "Eusebi 2018": ["SEX", "pigd", "stai"],
}
LOO_FILE = os.path.join(C.PROC, "loo2024_variables.parquet")
LOO_SET = ["comp_axial", "NP2FREZ", "rigidity_lower", "comp_tremor", "BMI", "ageonset", "visuospatial"]


def sample():
    """Eligible participants complete on the twenty-candidate pool, plus Loo's variables."""
    M = pd.read_parquet(os.path.join(C.PROC, "candidate_matrix.parquet"))
    if "ageonset" not in M.columns and "ageonset_x" in M.columns:
        M = M.rename(columns={"ageonset_x": "ageonset"})
    M["PATNO"] = pd.to_numeric(M.PATNO, errors="coerce")
    cand = [c for c in sorted(set(C.POOL) | set(EXTRA)) if c in M.columns]
    d = M[M.exit > 0].dropna(subset=cand)
    models = dict(PUBLISHED)
    if os.path.exists(LOO_FILE):
        loo = pd.read_parquet(LOO_FILE)
        loo["PATNO"] = pd.to_numeric(loo.PATNO, errors="coerce")
        d = d.merge(loo, on="PATNO", how="left")
        if d[LOO_SET].notna().all(axis=1).mean() > 0.9:
            d = d.dropna(subset=LOO_SET)
            models["Loo 2024"] = list(LOO_SET)
    return d.reset_index(drop=True), cand, models


D, CAND, MODELS = sample()
E = D.event.astype(bool).values
T = D.exit.values
Y = Surv.from_arrays(E, T)
SPLITS = [list(StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.CV_SEED + r).split(D, D.event))
          for r in range(C.CV_REPEATS)]


def out_of_fold(cols):
    """Mean out-of-fold C over the repetitions, and the averaged out-of-fold risk score.

    The mean of the per-repetition C-indices is the quantity reported everywhere else in
    the paper, and is what cohort.out_of_fold_c returns. The averaged score is kept as
    well, because a paired bootstrap needs one score per participant.
    """
    x = D[cols].values
    cs, acc = [], np.zeros(len(D))
    for folds in SPLITS:
        pred = np.zeros(len(D))
        for tr, te in folds:
            sc = StandardScaler().fit(x[tr])
            m = CoxPHSurvivalAnalysis(alpha=1.0).fit(sc.transform(x[tr]), Y[tr])
            pred[te] = m.predict(sc.transform(x[te]))
        cs.append(concordance_index_censored(E, T, pred)[0])
        acc += (pred - pred.mean()) / pred.std()
    return float(np.mean(cs)), acc / len(SPLITS)


def c_from_score(e, t, s):
    return float(concordance_index_censored(e, t, s)[0])


def subsets(k):
    if k <= 3:
        return [list(c) for c in itertools.combinations(range(len(CAND)), k)]
    rng = np.random.default_rng(11)
    return [sorted(rng.choice(len(CAND), k, replace=False).tolist()) for _ in range(1000)]


# Windows spawns worker processes by re-importing this module, so the command-line
# dispatch below must run only in the parent. Everything above this line is what a worker
# needs and is deliberately left at module level.
mode = (sys.argv[1] if len(sys.argv) > 1 else "summarise") if __name__ == "__main__" else None

if mode == "meta":
    print(json.dumps(dict(n=len(D), events=int(E.sum()), candidates=CAND,
                          models=list(MODELS), has_loo="Loo 2024" in MODELS), indent=1))

elif mode == "sets":
    # a sanity check that the local loop reproduces cohort.out_of_fold_c exactly
    mine, _ = out_of_fold(C.TOTAL6)
    theirs = C.out_of_fold_c(D, C.TOTAL6)
    assert abs(mine - theirs) < 1e-9, (mine, theirs)
    print("local loop reproduces cohort.out_of_fold_c: %.6f\n" % mine, flush=True)

    names = list(MODELS)
    ref = names[0]
    pts, scores = {}, {}
    for n in names:
        pts[n], scores[n] = out_of_fold(MODELS[n])
        print("  %-22s k=%d  out-of-fold C %.4f" % (n, len(MODELS[n]), pts[n]), flush=True)

    cscore = {n: c_from_score(E, T, scores[n]) for n in names}
    rng = np.random.default_rng(BOOT_SEED)
    idx = np.arange(len(D))
    bootC = {n: [] for n in names}
    bootD = {n: [] for n in names if n != ref}
    ok = 0
    for _ in range(BOOT):
        bs = rng.choice(idx, len(D), replace=True)
        eb, tb = E[bs], T[bs]
        if eb.sum() < 10:
            continue
        cs = {n: c_from_score(eb, tb, scores[n][bs]) for n in names}
        if any(np.isnan(v) for v in cs.values()):
            continue
        ok += 1
        for n in names:
            bootC[n].append(cs[n])
            if n != ref:
                bootD[n].append(cs[ref] - cs[n])

    rows = []
    for n in names:
        lo, hi = np.percentile(bootC[n], [2.5, 97.5])
        r = dict(predictor_set=n, k=len(MODELS[n]), C=round(pts[n], 4),
                 C_score=round(cscore[n], 4), C_lo=round(lo, 4), C_hi=round(hi, 4))
        if n == ref:
            r.update(dC=0.0, d_lo=np.nan, d_hi=np.nan, p=np.nan)
        else:
            dv = np.array(bootD[n])
            r.update(dC=round(pts[ref] - pts[n], 4),
                     d_lo=round(float(np.percentile(dv, 2.5)), 4),
                     d_hi=round(float(np.percentile(dv, 97.5)), 4),
                     p=round(max(2 * min((dv <= 0).mean(), (dv >= 0).mean()), 1.0 / ok), 4))
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "tab49_sets_unified.csv"), index=False)
    print("\nbootstrap replicates kept: %d\n" % ok)
    print(out.to_string(index=False))

elif mode == "dist":
    # The enumeration is the expensive half: 2,330 subsets, each cross-validated twenty
    # times over five folds. It is spread across cores, and each worker writes nothing, so
    # the result does not depend on how many cores are used. Subsets already recorded are
    # skipped, which makes the run resumable.
    from joblib import Parallel, delayed

    k = int(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, (os.cpu_count() or 2) - 2)
    f = os.path.join(OUT, "tab49_dist_k%d.csv" % k)
    done = set()
    if os.path.exists(f):
        done = set(pd.read_csv(f, header=None, names=["idx", "C"]).idx.astype(str))
    todo = [c for c in subsets(k) if "|".join(map(str, c)) not in done]
    print("k=%d: %d subsets, %d already done, %d to run on %d workers"
          % (k, len(subsets(k)), len(done), len(todo), jobs), flush=True)

    def one(c):
        return "|".join(map(str, c)), out_of_fold([CAND[j] for j in c])[0]

    CHUNK = max(jobs * 4, 40)
    with open(f, "a") as fh:
        for a in range(0, len(todo), CHUNK):
            part = todo[a:a + CHUNK]
            for key, v in Parallel(n_jobs=jobs, backend="loky")(delayed(one)(c) for c in part):
                fh.write("%s,%.6f\n" % (key, v))
            fh.flush()
            print("   %d/%d" % (min(a + CHUNK, len(todo)), len(todo)), flush=True)
    print("k=%d done" % k, flush=True)

elif mode == "summarise":
    S = pd.read_csv(os.path.join(OUT, "tab49_sets_unified.csv"))
    dist = {}
    for k in sorted(S.k.unique()):
        f = os.path.join(OUT, "tab49_dist_k%d.csv" % k)
        if os.path.exists(f):
            dist[int(k)] = pd.read_csv(f, header=None, names=["idx", "C"]).C.values
    t6 = float(S.loc[S.predictor_set.str.contains("Total-6"), "C"].iloc[0])
    rows = []
    for _, r in S.sort_values("C", ascending=False).iterrows():
        v = dist.get(int(r.k))
        rows.append(dict(predictor_set=r.predictor_set, k=int(r.k), C_out_of_fold=r.C,
                         C_lo=r.C_lo, C_hi=r.C_hi, dC_vs_total6=r.dC, d_lo=r.d_lo,
                         d_hi=r.d_hi, p=r.p,
                         median_C_for_k=round(float(np.median(v)), 4) if v is not None else np.nan,
                         p90_C_for_k=round(float(np.percentile(v, 90)), 4) if v is not None else np.nan,
                         max_C_for_k=round(float(v.max()), 4) if v is not None else np.nan,
                         percentile_within_k=round(float((v < r.C).mean() * 100), 1) if v is not None else np.nan,
                         reaching_total6=("%d of %d" % ((v >= t6).sum(), len(v))) if v is not None else ""))
    md = pd.DataFrame(rows)
    md.to_csv(os.path.join(OUT, "tab49_matched_df_unified.csv"), index=False)
    print(md.to_string(index=False))
    summary = dict(n=len(D), events=int(E.sum()), total6_C=t6,
                   median_by_k={str(k): round(float(np.median(v)), 4) for k, v in dist.items()},
                   gain_2_to_6=round(float(np.median(dist[6]) - np.median(dist[2])), 4),
                   reaching_total6={str(k): int((v >= t6).sum()) for k, v in dist.items()},
                   n_subsets={str(k): int(len(v)) for k, v in dist.items()})
    json.dump(summary, open(os.path.join(OUT, "tab49_head_to_head_unified.json"), "w"), indent=1)
    print("\n" + json.dumps(summary, indent=1))
