# 46_calculator_artifacts.py
# Produces: 05_Calculadora/calculator_artifacts.json, the frozen model behind the
# public risk calculator and behind the baseline-survival block of Table 2.
#
# Why this script exists
# ----------------------
# Until August 2026 no script in this pipeline generated the calculator artifact.
# Script 05 writes a different, older schema built on a different predictor set, and
# script 06 only appends the 'modelo_dinamico' block to a file that already exists.
# The artifact was therefore unauditable, in the same category as tab68_slopes.csv,
# even though Table 2 of the manuscript reports its baseline survival.
#
# The three-year discrepancy
# --------------------------
# The frozen artifact reports S0(3) = 0.8878 while 32_internal_performance.py reports
# 0.8927. Both are correct computations of different quantities:
#
#   step    baseline_survival_ read at the last event time at or before the horizon,
#           which is cohort.baseline_survival_at and the standard definition of the
#           baseline survival function of a Cox model at time t. Gives 0.8927.
#   interp  predict_survival_function(times=[t]), which linearly interpolates the
#           cumulative hazard between adjacent index times. Gives 0.8878.
#
# They coincide at 5, 7 and 10 years because an event time falls at those horizons,
# and differ at 3 years because the nearest events are at 2.9185 and 3.0007 years.
#
# The default here is 'step', for three reasons: it is the definition of S0(t); it is
# what cohort.baseline_survival_at implements, which is the project's single source of
# truth; and it is what script 32 uses to produce the calibration reported in the
# paper, so the alternative leaves Table 2 and the paper's own calibration using
# different values of S0 at the same horizon. Pass --convention interp to reproduce
# the frozen file instead.
#
# Safety
# ------
# Writes to calculator_artifacts_regenerated.json unless --overwrite is passed, and
# always prints a field-by-field comparison against the current frozen file first.
# The calculator is publicly deployed, so overwriting is deliberate, not incidental.
#
# Usage
#   PYTHONPATH=.. python3 46_calculator_artifacts.py                  # dry run, step
#   PYTHONPATH=.. python3 46_calculator_artifacts.py --convention interp
#   PYTHONPATH=.. python3 46_calculator_artifacts.py --overwrite

import argparse
import json
import os

import numpy as np
import pandas as pd

import cohort as C

CALC = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "05_Calculadora")
)
FROZEN = os.path.join(CALC, "calculator_artifacts.json")
TABLES = os.path.join(C.RESULTS, "Tabelas")

HORIZONS = [3, 5, 7, 10]
GRID = [round(0.5 * k, 1) for k in range(1, 27)]  # 0.5 to 13.0 years

# Blocks that are not derived here and are carried over unchanged from the frozen
# file: the input widget definitions, and the dynamic model written by script 06.
CARRY_OVER = ["inputs", "modelo_dinamico"]


def baseline_step(model, times):
    """S0 read off the step function, via cohort.baseline_survival_at."""
    return [float(C.baseline_survival_at(model, t)) for t in times]


def baseline_interp(model, cohort, times):
    """S0 from predict_survival_function, which interpolates the cumulative hazard."""
    means = cohort[C.TOTAL6].mean()
    at_mean = pd.DataFrame([means.to_dict()])
    surv = model.predict_survival_function(at_mean, times=list(times))
    return [float(surv.loc[t].iloc[0]) for t in times]


def competing_risk_factor():
    """c(t) = Aalen-Johansen incidence / naive Kaplan-Meier incidence, from script 12."""
    path = os.path.join(TABLES, "tab45_risco_competitivo.csv")
    tab = pd.read_csv(path).set_index("horizonte_anos")
    return {
        str(h): round(float(tab.loc[h, "Aalen_Johansen"] / tab.loc[h, "KM_ingenuo"]), 4)
        for h in HORIZONS
    }


def performance():
    """Internal performance, read from the output of script 32."""
    with open(os.path.join(TABLES, "tab32_internal_performance.json"), encoding="utf-8") as fh:
        p = json.load(fh)
    oe = {h: p["calibration"][h]["OE"] for h in ("3", "5", "7")}
    # The three fields added in September 2026 were in the frozen file and absent here, so
    # regenerating the artifacts silently dropped them. They are all in the output of
    # script 32; the omission was in this function, not in the data.
    return {
        "C_aparente": round(p["C_apparent"], 3),
        "C_corrigido": round(p["C_corrected"], 3),
        "C_corrigido_IC95": [round(p["C_corrected_lo"], 3), round(p["C_corrected_hi"], 3)],
        "C_fora_da_amostra": round(p["C_out_of_fold"], 3),
        "LOSO_IIQ": [round(p["loso"]["q1"], 3), round(p["loso"]["q3"], 3)],
        "LOSO_mediana": round(p["loso"]["median"], 3),
        "LOSO_ponderada_por_eventos": round(p["loso"]["weighted_mean"], 3),
        "LOSO_sitios_avaliaveis": int(p["loso"]["sites"]),
        "calibracao": "observed-to-expected %.2f, %.2f and %.2f at 3, 5 and 7 years"
        % (oe["3"], oe["5"], oe["7"]),
    }


def build(convention):
    cohort = C.build_cohort(with_metadata=False)
    model = C.fit_total6(cohort)

    coef = {v: float(model.params_[v]) for v in C.TOTAL6}
    means = {v: float(cohort[v].mean()) for v in C.TOTAL6}

    if convention == "step":
        hor = baseline_step(model, HORIZONS)
        grid = baseline_step(model, GRID)
    else:
        hor = baseline_interp(model, cohort, HORIZONS)
        grid = baseline_interp(model, cohort, GRID)

    return {
        "modelo": "Total-6 (Cox PH)",
        "desfecho": "Discinesia problemática (MDS-UPDRS 4.1>=2 OU 4.2>=2)",
        "horizontes": HORIZONS,
        "n": int(len(cohort)),
        "eventos": int(cohort.event.sum()),
        "desempenho": performance(),
        "coeficientes": coef,
        "baseline_survival_hor": {str(h): s for h, s in zip(HORIZONS, hor)},
        "means": means,
        "baseline_survival_grid": {"t": GRID, "S0": [round(s, 5) for s in grid]},
        "competing_risk_factor": competing_risk_factor(),
        "nota_competitivo": (
            "7- and 10-year risk adjusted for the competing risk of death "
            "(Aalen-Johansen); 3 and 5 years practically unchanged."
        ),
        "convencao_sobrevida_basal": convention,
    }


def compare(new, old):
    """Print every scalar that differs between the regenerated and the frozen file."""
    print("\ncomparison with the frozen file")
    print("-" * 78)
    rows = []

    for v in C.TOTAL6:
        a = new["coeficientes"][v]
        b = old.get("coeficientes", {}).get(v)
        rows.append(("coeficiente %s" % v, b, a))
        rows.append(("media %s" % v, old.get("means", {}).get(v), new["means"][v]))

    for h in HORIZONS:
        rows.append(
            (
                "S0(%d)" % h,
                old.get("baseline_survival_hor", {}).get(str(h)),
                new["baseline_survival_hor"][str(h)],
            )
        )
        rows.append(
            (
                "c(%d)" % h,
                old.get("competing_risk_factor", {}).get(str(h)),
                new["competing_risk_factor"][str(h)],
            )
        )

    rows.append(("n", old.get("n"), new["n"]))
    rows.append(("eventos", old.get("eventos"), new["eventos"]))

    changed = 0
    for label, before, after in rows:
        if before is None:
            print("  %-28s  ausente no congelado  ->  %s" % (label, after))
            changed += 1
        elif isinstance(before, float) and abs(before - after) > 5e-7:
            print("  %-28s  %.10f  ->  %.10f   (%+.2e)" % (label, before, after, after - before))
            changed += 1
        elif not isinstance(before, float) and before != after:
            print("  %-28s  %s  ->  %s" % (label, before, after))
            changed += 1

    old_perf, new_perf = old.get("desempenho", {}), new["desempenho"]
    if old_perf != new_perf:
        print("  desempenho (bloco inteiro)")
        print("      antes: %s" % json.dumps(old_perf, ensure_ascii=False, sort_keys=True))
        print("      depois: %s" % json.dumps(new_perf, ensure_ascii=False, sort_keys=True))
        changed += 1

    if not changed:
        print("  nenhuma diferenca")
    print("-" * 78)
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--convention", choices=["step", "interp"], default="step")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the frozen calculator_artifacts.json in place")
    args = ap.parse_args()

    with open(FROZEN, encoding="utf-8") as fh:
        old = json.load(fh)

    new = build(args.convention)
    for key in CARRY_OVER:
        if key in old:
            new[key] = old[key]
        else:
            print("warning: block '%s' absent from the frozen file, not carried over" % key)

    print("convention: %s" % args.convention)
    print("cohort: n = %d, events = %d" % (new["n"], new["eventos"]))
    print("S0 at 3, 5, 7, 10 years: " +
          ", ".join("%.10f" % new["baseline_survival_hor"][str(h)] for h in HORIZONS))

    other = "interp" if args.convention == "step" else "step"
    cohort = C.build_cohort(with_metadata=False)
    model = C.fit_total6(cohort)
    alt = baseline_step(model, HORIZONS) if other == "step" else baseline_interp(model, cohort, HORIZONS)
    print("the same under '%s': " % other + ", ".join("%.10f" % s for s in alt))

    compare(new, old)

    target = FROZEN if args.overwrite else os.path.join(
        CALC, "calculator_artifacts_regenerated.json")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=2)
    print("\nwritten to %s" % target)
    if not args.overwrite:
        print("the frozen file was NOT touched; pass --overwrite to replace it, and "
              "remember that the calculator is publicly deployed")


if __name__ == "__main__":
    main()
