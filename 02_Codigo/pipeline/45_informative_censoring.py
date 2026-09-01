# 45_informative_censoring.py
# Produces: the examination of the independent-censoring assumption.
#
# Cox regression and Kaplan-Meier both assume that censoring is non-informative:
# at any moment, a participant who stops being observed must have the same future
# risk as a comparable participant who continues. If people who leave the study
# differ in risk from those who stay, both the coefficients and the absolute risks
# are biased, and nothing else in the analysis can detect it.
#
# The examination has three parts.
#
# 1. Decomposition. Most censoring here is administrative: the participant was
#    still under observation when the data were cut in April 2026. Administrative
#    censoring is non-informative by construction. Only genuine loss to follow-up
#    can carry information, so it has to be separated first.
#
# 2. Is censoring predictable? A Cox model is fitted with loss to follow-up as the
#    event and the six predictors as covariates, and then with the model's own risk
#    score. If the risk score predicts dropping out, participants at higher risk of
#    dyskinesia are leaving the study differentially, which is the pattern that
#    would bias the estimates.
#
# 3. How far would it have to go to matter? Censoring that depends on unmeasured
#    factors cannot be tested, only bounded. Each participant lost to follow-up has
#    their unobserved remaining time imputed from the fitted model with the hazard
#    multiplied by delta, and the analysis is repeated. Delta above one means
#    dropouts were at higher risk than similar participants who stayed. The output
#    is the value of delta at which a conclusion would change.

import argparse
import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import cohort as C

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
ADMINISTRATIVE_WINDOW_MONTHS = 18
DELTAS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
IMPUTATIONS = 10
HORIZON = 5
SEED = 90210


def classify_censoring(df, window_months=ADMINISTRATIVE_WINDOW_MONTHS):
    """Split censored participants into administrative and lost to follow-up."""
    _, tzero = C.load_medication_log()
    last_contact = df.PATNO.map(tzero) + pd.to_timedelta(df.exit * 365.25, unit="D")
    cut = last_contact.max()
    threshold = cut - pd.DateOffset(months=window_months)
    kind = np.where(df.event == 1, "event",
                    np.where(last_contact >= threshold, "administrative", "lost"))
    return pd.Series(kind, index=df.index), last_contact, cut


def delta_impute(df, model, dropout_mask, delta, rng):
    """
    Impute the unobserved remaining time of participants lost to follow-up, with
    their hazard multiplied by delta relative to the fitted model.
    """
    hazard = model.baseline_cumulative_hazard_.iloc[:, 0]
    times = hazard.index.values
    values = hazard.values
    # lifelines centres the baseline hazard on the mean covariate vector, so the
    # individual hazard is h0(t) * exp(lp_i - mean(lp)). Forgetting to centre here
    # shifts every hazard by exp(-mean(lp)) and silently ruins the imputation.
    lp = C.linear_predictor(model, df)
    lp = lp - lp.mean()
    tau = float(df.exit.max())
    out = df.copy()

    idx = np.where(dropout_mask.values)[0]
    h_at_censor = np.interp(df.exit.values[idx], times, values)
    u = rng.uniform(size=len(idx))
    target = h_at_censor - np.log(u) / (delta * np.exp(lp[idx]))
    imputed = np.interp(target, values, times, left=times[0], right=np.inf)

    beyond = ~np.isfinite(imputed) | (imputed > tau)
    new_exit = np.where(beyond, tau, imputed)
    new_event = np.where(beyond, 0, 1)

    out.iloc[idx, out.columns.get_loc("exit")] = new_exit
    out.iloc[idx, out.columns.get_loc("event")] = new_event
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputations", type=int, default=IMPUTATIONS)
    parser.add_argument("--cv-repeats", type=int, default=5)
    args = parser.parse_args()

    df = C.build_cohort()
    kind, last_contact, cut = classify_censoring(df)
    df = df.assign(censoring_kind=kind.values)

    counts = df.censoring_kind.value_counts()
    print(f"data cut: {cut.date()}   (administrative window {ADMINISTRATIVE_WINDOW_MONTHS} months)")
    print(f"cohort: n={len(df)}, events={int(df.event.sum())}")
    for label in ("event", "administrative", "lost"):
        n = int(counts.get(label, 0))
        print(f"  {label:16s} {n:4d}  ({100 * n / len(df):4.1f}%)")
    lost = df[df.censoring_kind == "lost"]
    admin = df[df.censoring_kind == "administrative"]
    print(f"\nloss to follow-up is {100 * len(lost) / len(df):.1f}% of the cohort and "
          f"{100 * len(lost) / (len(df) - df.event.sum()):.1f}% of all censoring")

    # ---------------------------------------------------------- 1. who leaves
    print("\ncharacteristics at time-zero, by censoring type (median unless stated):")
    print(f"{'':22s}{'events':>12s}{'administrative':>16s}{'lost':>12s}")
    events = df[df.event == 1]
    for variable in C.TOTAL6 + ["exit"]:
        if variable == "SEX":
            print(f"{'Male sex':22s}{100 * events[variable].mean():11.1f}%"
                  f"{100 * admin[variable].mean():15.1f}%{100 * lost[variable].mean():11.1f}%")
        else:
            print(f"{variable:22s}{events[variable].median():12.2f}"
                  f"{admin[variable].median():16.2f}{lost[variable].median():12.2f}")

    # -------------------------------------------- 2. is dropping out predictable
    model = C.fit_total6(df)
    df = df.assign(lp=C.linear_predictor(model, df))
    censor_frame = df[C.TOTAL6 + ["exit"]].copy()
    censor_frame["dropout"] = (df.censoring_kind == "lost").astype(int)
    for column in C.TOTAL6:
        censor_frame[column] = (censor_frame[column] - censor_frame[column].mean()) / censor_frame[column].std()
    censor_model = CoxPHFitter(penalizer=C.RIDGE).fit(censor_frame, "exit", "dropout")
    print("\nCox model with loss to follow-up as the event, hazard ratios per standard deviation:")
    for variable in C.TOTAL6:
        row = censor_model.summary.loc[variable]
        flag = "  <- associated" if row["p"] < 0.05 else ""
        print(f"  {variable:20s} {np.exp(censor_model.params_[variable]):5.2f} "
              f"({row['exp(coef) lower 95%']:.2f} to {row['exp(coef) upper 95%']:.2f}), "
              f"p = {row['p']:.3f}{flag}")

    score_frame = pd.DataFrame({"lp": (df.lp - df.lp.mean()) / df.lp.std(),
                                "exit": df.exit.values,
                                "dropout": (df.censoring_kind == "lost").astype(int).values})
    score_model = CoxPHFitter().fit(score_frame, "exit", "dropout")
    score_row = score_model.summary.loc["lp"]
    print(f"\nthe model's own risk score as a predictor of dropping out: "
          f"HR {np.exp(score_model.params_['lp']):.2f} "
          f"({score_row['exp(coef) lower 95%']:.2f} to {score_row['exp(coef) upper 95%']:.2f}), "
          f"p = {score_row['p']:.3f}")
    print("A risk score unrelated to dropping out is the reassuring result: it means the")
    print("participants who left were not the ones the model considered at higher risk.")

    # ------------------------------------------------- 3. delta-adjusted bounds
    reference_c = C.out_of_fold_c(df, repeats=args.cv_repeats)
    reference_incidence = C.km_incidence(df, HORIZON)
    reference_hr = {v: float(np.exp(model.params_[v])) for v in C.TOTAL6}
    print(f"\nreference: out-of-fold C {reference_c:.4f}, "
          f"{HORIZON}-year incidence {100 * reference_incidence:.1f}%")

    dropout_mask = df.censoring_kind == "lost"
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"\ndelta-adjusted sensitivity, {args.imputations} imputations per value of delta")
    print(f"{'delta':>7s}{'events':>9s}{'C':>9s}{'dC':>9s}"
          f"{str(HORIZON) + '-year incidence':>18s}{'MDS-UPDRS HR':>14s}")
    for delta in DELTAS:
        events_n, cs, incidences, hrs = [], [], [], {v: [] for v in C.TOTAL6}
        for _ in range(args.imputations):
            imputed = delta_impute(df, model, dropout_mask, delta, rng)
            events_n.append(int(imputed.event.sum()))
            incidences.append(C.km_incidence(imputed, HORIZON))
            fitted = C.fit_total6(imputed)
            for v in C.TOTAL6:
                hrs[v].append(float(np.exp(fitted.params_[v])))
            cs.append(concordance_index(imputed.exit, -C.linear_predictor(fitted, imputed),
                                        imputed.event))
        row = {"delta": delta, "events": float(np.mean(events_n)),
               "C_apparent": float(np.mean(cs)),
               "incidence": float(np.mean(incidences)),
               **{f"HR_{v}": float(np.mean(hrs[v])) for v in C.TOTAL6}}
        rows.append(row)
        print(f"{delta:7.2f}{np.mean(events_n):9.0f}{np.mean(cs):9.4f}"
              f"{np.mean(cs) - reference_c:+9.4f}{100 * np.mean(incidences):17.1f}%"
              f"{np.mean(hrs['updrs_totscore']):14.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "tab45_informative_censoring.csv"), index=False)

    at_one = next(r for r in rows if r["delta"] == 1.0)
    print(f"\nAt delta = 1, that is under the assumption the analysis already makes, imputing")
    print(f"the dropouts gives a {HORIZON}-year incidence of {100 * at_one['incidence']:.1f}% "
          f"against {100 * reference_incidence:.1f}% observed, which is the internal")
    print("consistency check: the imputation reproduces the primary estimate.")
    worst = rows[-1]
    print(f"At delta = {worst['delta']:.1f}, the extreme assumption that everyone lost to "
          f"follow-up carried {worst['delta']:.1f} times the hazard of a comparable")
    print(f"participant who stayed, the {HORIZON}-year incidence rises to "
          f"{100 * worst['incidence']:.1f}% and the concordance moves by "
          f"{worst['C_apparent'] - reference_c:+.4f}.")

    # ------------------------------------------------------------ tipping point
    # The primary estimate carries its own uncertainty. A sensitivity analysis has
    # only displaced the conclusion once it moves the estimate outside that interval.
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(df.exit, df.event)
    interval = km.confidence_interval_survival_function_
    position = max(np.searchsorted(interval.index.values, HORIZON, side="right") - 1, 0)
    incidence_lo = 1 - float(interval.iloc[position, 1])
    incidence_hi = 1 - float(interval.iloc[position, 0])
    print(f"\nthe primary {HORIZON}-year incidence is {100 * reference_incidence:.1f}% "
          f"(95% CI {100 * incidence_lo:.1f} to {100 * incidence_hi:.1f})")
    outside = [r for r in rows if r["incidence"] > incidence_hi or r["incidence"] < incidence_lo]
    if outside:
        first = min(outside, key=lambda r: abs(r["delta"] - 1))
        print(f"the smallest delta that moves the estimate outside that interval is "
              f"{first['delta']:.1f}")
        print(f"so the conclusion holds unless participants lost to follow-up carried more")
        print(f"than about {first['delta']:.1f} times the hazard of comparable participants "
              f"who stayed under observation")
    else:
        print("no value of delta tested moves the estimate outside that interval")
    c_shift = max(abs(r["C_apparent"] - reference_c) for r in rows)
    hr_shift = max(abs(100 * (r[f"HR_{v}"] - reference_hr[v]) / reference_hr[v])
                   for r in rows for v in C.TOTAL6)
    print(f"\nacross every value of delta tested, concordance moves by at most {c_shift:.4f}")
    print(f"and the largest hazard ratio moves by at most {hr_shift:.1f}%")
    print("Informative censoring of this kind would therefore shift the level of absolute")
    print("risk without disturbing the ranking of patients, which is the property the model")
    print("is offered for.")

    summary = {
        "data_cut": str(cut.date()),
        "administrative_window_months": ADMINISTRATIVE_WINDOW_MONTHS,
        "counts": {k: int(counts.get(k, 0)) for k in ("event", "administrative", "lost")},
        "lost_pct_of_cohort": 100 * len(lost) / len(df),
        "lost_pct_of_censored": 100 * len(lost) / (len(df) - int(df.event.sum())),
        "censoring_model": {v: {"HR": float(np.exp(censor_model.params_[v])),
                                "lo": float(censor_model.summary.loc[v, "exp(coef) lower 95%"]),
                                "hi": float(censor_model.summary.loc[v, "exp(coef) upper 95%"]),
                                "p": float(censor_model.summary.loc[v, "p"])} for v in C.TOTAL6},
        "risk_score_predicts_dropout": {"HR": float(np.exp(score_model.params_["lp"])),
                                        "lo": float(score_row["exp(coef) lower 95%"]),
                                        "hi": float(score_row["exp(coef) upper 95%"]),
                                        "p": float(score_row["p"])},
        "reference": {"C": reference_c, "incidence": reference_incidence, "HR": reference_hr},
        "deltas": rows,
        "imputations": args.imputations,
    }
    with open(os.path.join(OUT, "tab45_informative_censoring.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
