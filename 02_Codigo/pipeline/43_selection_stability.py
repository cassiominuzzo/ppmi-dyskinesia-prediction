# 43_selection_stability.py
# Produces: the selection frequency of each candidate predictor across bootstrap
#           resamples.
#
# The six predictors were chosen by forward selection maximising out-of-fold
# concordance within an 18-variable candidate pool, using the whole development
# sample. A reader is entitled to ask whether that particular set is a property of
# the data or of one accident of sampling. The standard answer is to repeat the
# entire selection procedure inside bootstrap resamples and report how often each
# variable is chosen, and how often the published set is recovered exactly.
#
# A variable selected in most resamples is stable. A variable selected in about
# half is interchangeable with a correlated neighbour, which is not the same as
# being unimportant, and the distinction should be stated rather than hidden.
#
# Repeating the whole procedure is expensive: each resample costs 93 model fits
# times five folds. The run is therefore resumable. Results are appended one line
# per resample to tab43_resamples.csv, so an interrupted run loses nothing.
#
#   python3 43_selection_stability.py            # run to completion, resuming
#   python3 43_selection_stability.py --limit 20 # do at most 20 more, then stop
#   python3 43_selection_stability.py --report   # aggregate what is on disk

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
RESAMPLES = os.path.join(OUT, "tab43_resamples.csv")
BOOTSTRAP = 100
TARGET_SIZE = len(C.TOTAL6)
SEED = 4242


def out_of_fold(data, columns, seed):
    y = Surv.from_arrays(data.event.astype(bool).values, data.exit.values)
    x = data[columns].values
    pred = np.zeros(len(data))
    for train, test in StratifiedKFold(C.CV_FOLDS, shuffle=True,
                                       random_state=seed).split(x, data.event):
        scaler = StandardScaler().fit(x[train])
        try:
            model = CoxPHSurvivalAnalysis(alpha=1.0).fit(scaler.transform(x[train]), y[train])
        except Exception:
            return np.nan
        pred[test] = model.predict(scaler.transform(x[test]))
    return concordance_index_censored(data.event.astype(bool).values, data.exit.values, pred)[0]


def forward_selection(data, pool, size, seed):
    chosen, remaining = [], list(pool)
    for _ in range(size):
        best, best_score = None, -np.inf
        for candidate in remaining:
            score = out_of_fold(data, chosen + [candidate], seed)
            if not np.isnan(score) and score > best_score:
                best, best_score = candidate, score
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)
    return chosen


def report(df):
    done = pd.read_csv(RESAMPLES) if os.path.exists(RESAMPLES) else pd.DataFrame()
    if done.empty:
        raise SystemExit("no resamples on disk yet")
    frequency = Counter()
    exact = 0
    for _, row in done.iterrows():
        selected = str(row["selected"]).split("|")
        frequency.update(selected)
        if set(selected) == set(C.TOTAL6):
            exact += 1
    runs = len(done)
    print(f"\nselection frequency across {runs} resamples:")
    print(f"{'variable':22s}{'frequency':>11s}{'in Total-6':>12s}")
    for variable in sorted(C.POOL, key=lambda v: -frequency.get(v, 0)):
        print(f"{variable:22s}{100 * frequency.get(variable, 0) / runs:10.0f}%"
              f"{'yes' if variable in C.TOTAL6 else '':>12s}")
    print(f"\nthe exact published set was recovered in {exact} of {runs} resamples "
          f"({100 * exact / runs:.0f}%)")
    core = [v for v in C.TOTAL6 if 100 * frequency.get(v, 0) / runs >= 80]
    print(f"selected in at least 80% of resamples: {len(core)} of the six "
          f"({', '.join(core) if core else 'none'})")
    print("Recovering an identical six-variable set is a demanding criterion when several")
    print("candidates measure overlapping constructs; the per-variable frequency is the")
    print("more informative summary.")

    original = forward_selection(df, C.POOL, TARGET_SIZE, C.CV_SEED)
    summary = {
        "bootstrap": runs, "target_size": TARGET_SIZE,
        "selection_full_sample": original,
        "identical_to_total6": bool(set(original) == set(C.TOTAL6)),
        "frequency": {v: 100 * frequency.get(v, 0) / runs for v in C.POOL},
        "exact_recovery_pct": 100 * exact / runs,
        "stable_at_80pct": core,
    }
    with open(os.path.join(OUT, "tab43_selection_stability.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    pd.DataFrame([{"variable": v, "frequency_pct": 100 * frequency.get(v, 0) / runs,
                   "in_total6": v in C.TOTAL6} for v in C.POOL]).sort_values(
        "frequency_pct", ascending=False).to_csv(
        os.path.join(OUT, "tab43_selection_stability.csv"), index=False)

    # ---------------------------------------------------------------------------------
    # What the instability costs, and how far each resample lands from the published set.
    # Added in September 2026: panel B and panel C of Supplementary Table S6 were built
    # from a file that no script wrote, so the two most quoted numbers of that table, the
    # 0.669 of the six most frequent candidates and the median overlap of four, could not
    # be regenerated. They are computed here, from the same resamples.
    overlap = Counter(len(set(str(r["selected"]).split("|")) & set(C.TOTAL6))
                      for _, r in done.iterrows())
    top6 = [v for v, _ in sorted(frequency.items(), key=lambda kv: -kv[1])[:TARGET_SIZE]]
    ALTERNATIVES = {
        "Total-6 publicado": list(C.TOTAL6),
        "Seis mais frequentes": top6,
        "Total-6 trocando MDS-UPDRS por MSEADLG":
            [v if v != "updrs_totscore" else "MSEADLG" for v in C.TOTAL6],
        "Total-6 trocando MDS-UPDRS por Parte III":
            [v if v != "updrs_totscore" else "updrs3_score" for v in C.TOTAL6],
        "So idade de inicio (a mais estavel)": ["ageonset"],
    }
    print("\nout-of-fold discrimination of alternative sets:")
    alt = {}
    for name, cols in ALTERNATIVES.items():
        alt[name] = C.out_of_fold_c(df, cols)
        print(f"  {name:44s} C {alt[name]:.4f}  ({', '.join(cols)})")
    counts = {str(k): int(overlap.get(k, 0)) for k in range(TARGET_SIZE + 1)}
    med = int(np.median([len(set(str(r['selected']).split('|')) & set(C.TOTAL6))
                         for _, r in done.iterrows()]))
    with open(os.path.join(OUT, "tab43_alternative_sets.json"), "w") as fh:
        json.dump({"overlap": counts, "median_overlap": med,
                   "at_least_4": int(sum(v for k, v in counts.items() if int(k) >= 4)),
                   "top6": top6, "alternatives": alt}, fh, indent=1)
    print(f"\noverlap with the published set: {counts}, median {med}")
    print(f"\nwritten to {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=BOOTSTRAP)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    df = C.build_cohort()
    if args.report:
        report(df)
        return

    done = set()
    if os.path.exists(RESAMPLES):
        done = set(pd.read_csv(RESAMPLES).index_seed.tolist())
    todo = [i for i in range(BOOTSTRAP) if i not in done][: args.limit]
    print(f"cohort: n={len(df)}, events={int(df.event.sum())}, pool of {len(C.POOL)} candidates")
    print(f"{len(done)} resamples already on disk, running {len(todo)} more "
          f"of a planned {BOOTSTRAP}", flush=True)

    for i in todo:
        local = np.random.default_rng(SEED + i)
        resample = df.iloc[local.integers(0, len(df), len(df))].reset_index(drop=True)
        if resample.event.sum() < 40:
            continue
        selected = forward_selection(resample, C.POOL, TARGET_SIZE, C.CV_SEED)
        line = pd.DataFrame([{"index_seed": i, "selected": "|".join(selected)}])
        line.to_csv(RESAMPLES, mode="a", header=not os.path.exists(RESAMPLES), index=False)
        print(f"  {i}: {', '.join(selected)}", flush=True)

    total = len(pd.read_csv(RESAMPLES)) if os.path.exists(RESAMPLES) else 0
    print(f"\n{total} of {BOOTSTRAP} resamples complete")
    if total >= BOOTSTRAP:
        report(df)
    else:
        print("run again to continue, or use --report to aggregate what is on disk")


if __name__ == "__main__":
    main()
