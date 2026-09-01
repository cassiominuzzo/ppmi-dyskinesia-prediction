"""
Risk calculator for problematic levodopa-induced dyskinesia.
PPMI cohort. Cox; closed-form (numpy). Absolute risk adjusted for the competing risk of death (Aalen-Johansen).
Two modes: BASELINE (Total-6, levodopa initiation) and DYNAMIC (Total-6 + responsiveness, at follow-up).
Robust layout: inputs in the sidebar, results in the main area (no nested columns).
RESEARCH TOOL - does not replace clinical judgement.
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "calculator_artifacts.json"), encoding="utf-8"))
HOR = A["horizontes"]
BETA, MEANS = A["coeficientes"], A["means"]
S0_HOR = {int(k): v for k, v in A["baseline_survival_hor"].items()}
GRID_T = A["baseline_survival_grid"]["t"]
GRID_S0 = A["baseline_survival_grid"]["S0"]
VARS = [i["var"] for i in A["inputs"]]
DYN = A.get("modelo_dinamico")
CRF = {int(k): v for k, v in A.get("competing_risk_factor", {}).items()}
# The competing-risk factor is anchored at c(0) = 1, because no one has died at
# time-zero. Without that anchor the factor stayed flat at 1 up to three years and
# then stepped down, which put a visible discontinuity in the plotted curve at t = 3.
CR_T = ([0.0] + sorted(CRF)) if CRF else [0.0, 3.0, 5.0, 7.0, 10.0]
CR_V = ([1.0] + [CRF[t] for t in sorted(CRF)]) if CRF else [1.0, 1.0, 1.0, 1.0, 1.0]

def cr_factor(t):
    return float(np.interp(t, CR_T, CR_V, left=1.0, right=CR_V[-1]))

def lp_generic(x, beta, means, keys):
    return sum(beta[k] * (float(x[k]) - means[k]) for k in keys)

def risk_at(lp, s0, t):
    return float((1 - s0 ** np.exp(lp)) * cr_factor(t))

st.set_page_config(page_title="Dyskinesia risk calculator (PPMI)", page_icon=":brain:", layout="wide")

# ------------------ SIDEBAR: inputs ------------------
st.sidebar.header("Patient data (single visit)")
x = {}
for it in A["inputs"]:
    v, lab, tp, hlp = it["var"], it["label"], it["tipo"], it.get("help", "")
    if tp == "slider":
        x[v] = st.sidebar.slider(lab, float(it["min"]), float(it["max"]), float(it["default"]), step=float(it["step"]), help=hlp)
    elif tp == "number":
        x[v] = st.sidebar.number_input(lab, float(it["min"]), float(it["max"]), float(it["default"]), step=float(it["step"]), help=hlp)
    elif tp == "select":
        opts = it["options"]
        choice = st.sidebar.selectbox(lab, list(opts.keys()), index=list(opts.keys()).index(it["default"]), help=hlp)
        x[v] = opts[choice]

st.sidebar.markdown("---")
st.sidebar.markdown("**Levodopa responsiveness** *(optional - follow-up only)*")
tem_resp = "No"
resp_val = None
if DYN:
    tem_resp = st.sidebar.radio(
        "Do you already know this patient's levodopa responsiveness?",
        ["No", "Yes"],
        help="Responsiveness = improvement of MDS-UPDRS III from OFF to ON. It only exists after the patient has taken levodopa.",
    )
    if tem_resp == "Yes":
        ri = DYN["resp_input"]
        resp_val = st.sidebar.number_input(
            "Motor responsiveness (% improvement of MDS-UPDRS III: OFF -> ON)",
            float(ri["min"]), float(ri["max"]), float(ri["default"]), step=float(ri["step"]),
            help="(UPDRS-III_OFF - UPDRS-III_ON) / UPDRS-III_OFF x 100. E.g., OFF 40, ON 16 -> 60%.",
        )
        if resp_val > 50:
            st.sidebar.caption("Strong responder (>50%) - associated with HIGHER dyskinesia risk.")
        elif resp_val >= 30:
            st.sidebar.caption("Moderate responder.")
        else:
            st.sidebar.caption("Low responsiveness - associated with LOWER risk.")

modo_dinamico = bool(DYN) and tem_resp == "Yes" and resp_val is not None

# ------------------ MAIN AREA ------------------
st.title("Risk of problematic levodopa-induced dyskinesia")
st.caption(
    "Prognostic model (PPMI cohort) - outcome: problematic dyskinesia "
    "(MDS-UPDRS 4.1>=2 OR 4.2>=2) - time-zero = levodopa initiation - absolute risk adjusted for the competing risk of death."
)
st.warning(
    "RESEARCH tool. Estimates the probability of *problematic* dyskinesia, not of any dyskinesia. "
    "Does not replace clinical judgement. Internally well calibrated (observed-to-expected 1.02 at 3, 5 and 7 years) and validated in a later PPMI recruitment wave (C 0.74); absolute risk may require local recalibration. Not validated outside PPMI."
)
st.info("Fill in the patient data in the **sidebar** (left). If it is collapsed, click the `>` arrow at the top left.")

try:
    if modo_dinamico:
        keys = list(DYN["coeficientes"].keys())
        beta, means = DYN["coeficientes"], DYN["means"]
        s0_hor = {int(k): v for k, v in DYN["baseline_survival_hor"].items()}
        grid_t, grid_s0 = DYN["baseline_survival_grid"]["t"], DYN["baseline_survival_grid"]["S0"]
        xf = dict(x); xf["resp"] = resp_val
        lp = lp_generic(xf, beta, means, keys)
        st.subheader("Estimated probability - DYNAMIC mode (with responsiveness)")
    else:
        s0_hor, grid_t, grid_s0 = S0_HOR, GRID_T, GRID_S0
        lp = lp_generic(x, BETA, MEANS, VARS)
        st.subheader("Estimated probability - BASELINE mode (levodopa initiation)")

    probs = {h: risk_at(lp, s0_hor[h], h) for h in HOR}
    # Patients still at risk at each horizon in the PPMI development cohort (n=813).
    # Shown next to each estimate so that the uncertainty travels with the number.
    AT_RISK = {3: 325, 5: 185, 7: 113, 10: 41}
    mc = st.columns(len(HOR))
    for c, h in zip(mc, HOR):
        c.metric(str(h) + " years", str(round(probs[h] * 100)) + "%")
        n_risk = AT_RISK.get(h)
        if n_risk is not None:
            if n_risk < 150:
                c.caption(":warning: based on only " + str(n_risk) + " patients still at risk")
            else:
                c.caption("based on " + str(n_risk) + " patients still at risk")

    # Observed five-year cumulative incidence in the development cohort (Kaplan-Meier,
    # n = 813, 165 events). Used only as a reference point, not as a decision threshold:
    # the model imposes none, and the study reports none.
    INCIDENCE_5Y = 0.240
    r5 = probs[5]
    msg = ("5-year probability: **" + str(round(r5 * 100)) + "%**, against a "
           + str(round(INCIDENCE_5Y * 100)) + "% five-year incidence in the development cohort")
    if r5 < INCIDENCE_5Y / 2:
        st.success("Well below the cohort incidence - " + msg)
    elif r5 < INCIDENCE_5Y:
        st.info("Below the cohort incidence - " + msg)
    else:
        st.warning("Above the cohort incidence - " + msg)
    st.caption(
        "These bands say where the estimate sits relative to the development cohort. "
        "They are **not** decision thresholds: the model imposes none, and the risk above which "
        "acting is worthwhile depends on what the result will be used for. In the internal "
        "decision-curve analysis net benefit was positive from about 6% to 50%."
    )

    curve = pd.DataFrame({
        "Years since levodopa": grid_t,
        "Cumulative risk (%)": [(1 - s0 ** np.exp(lp)) * cr_factor(t) * 100 for t, s0 in zip(grid_t, grid_s0)],
    }).set_index("Years since levodopa")
    st.line_chart(curve, height=280)
    st.caption("Hover over the line to read the year-by-year risk. Follow-up in the development cohort thins after 5 years "
               "(325 patients at risk at 3 years, 185 at 5, 113 at 7 and 41 at 10), so estimates at the longer horizons "
               "are imprecise and should be read as broad indications rather than exact probabilities. "
               "At the top of the range the model also saturates: once the predicted risk approaches 90%, two patients "
               "with clearly different risk scores receive almost the same number, so the ordering of patients stays "
               "informative there while the exact percentage stops being so.")
except Exception as e:
    st.error("An error occurred while computing. Details below (please send this text to the developer):")
    st.exception(e)

st.divider()
st.subheader("Which calculator to use? Concepts")
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown(
        "**BASELINE calculator (6 variables) - main model**\n\n"
        "Use **at the moment of starting levodopa** (or before prescribing). Estimates the risk of "
        "problematic dyskinesia from **a single visit**: total MDS-UPDRS (I+II+III), age at onset, sex, "
        "BMI, TD/PIGD ratio and freezing of gait (item 2.13). "
        "This is the study's internally validated model: optimism-corrected C 0.70 (95% CI 0.66 to 0.75), "
        "out-of-fold C 0.70, and observed-to-expected 1.02 at 3, 5 and 7 years."
    )
with cc2:
    st.markdown(
        "**DYNAMIC calculator (6 variables + responsiveness)**\n\n"
        "Use **at follow-up**, after the patient **has responded to levodopa** and you have an "
        "**OFF/ON** MDS-UPDRS III pair. It adds **motor responsiveness** [(OFF-ON)/OFF x100]: "
        "the **greater the improvement** with levodopa, the **higher** the dyskinesia risk (the classic good "
        "responder who develops dyskinesia). It **refines** the prognosis with information that only "
        "treatment reveals - it does not replace the baseline model."
    )

st.markdown("**When to use the dynamic model (and when NOT):**")
st.markdown(
    "- **Use it** when the patient is already on levodopa and the motor response is known (OFF/ON challenge "
    "or documented ON vs OFF assessment).\n"
    "- **Do not use it** on the first day of levodopa - responsiveness does not exist yet; use the **baseline** model instead."
)

lim_n = str(DYN["n"]) if DYN else "-"
lim_dc = str(DYN["desempenho"]["dC_apparent"]) if DYN else "-"
st.warning(
    "**Limitations of the dynamic calculator (exploratory):**\n\n"
    "1. **No measurable discrimination gain.** The apparent delta-C over the baseline model in the same "
    "patients is +" + lim_dc + ", but out of fold it is **+0.0001**, that is, nothing. Responsiveness is "
    "associated with the outcome (hazard ratio 1.19 per standard deviation, p = 0.045) without improving "
    "how well patients are ranked. Treat this mode as a way of thinking about the association, not as a "
    "better predictor.\n"
    "2. **Smaller subgroup** (n=" + lim_n + " vs 813), since it requires MDS-UPDRS III measured OFF and ON.\n"
    "3. Responsiveness is a **post-baseline marker** (measured during follow-up): interpret it as a risk "
    "**update**, not as a time-zero causal factor.\n"
    "4. It depends on the **quality/definition of the OFF/ON challenge** (standardization of the OFF state, ON peak timing).\n"
    "5. **Temporal validation** (later PPMI recruitment wave, n=421, 20 events, 15 sites absent from the earlier wave): **discrimination** held (C 0.74, 95% CI 0.64 to 0.83) and the calibration slope was 1.00, but absolute risk was **over-predicted** by about a quarter in that lower-risk wave, so the percentages should be read as risk **stratification**, not exact probability. The model has **not** been validated outside PPMI."
)

st.divider()
perf = A["desempenho"]
st.markdown("**Baseline variables (6):** total MDS-UPDRS (I+II+III), age at onset, sex, BMI, "
            "TD/PIGD ratio and freezing of gait (item 2.13) - all from a single visit, no ancillary tests.")
ci = perf["C_corrigido_IC95"]
iqr = perf["LOSO_IIQ"]
st.markdown(
    "**Baseline performance (PPMI, n=" + str(A["n"]) + ", " + str(A["eventos"]) + " events):** "
    "apparent C-index " + str(perf["C_aparente"]) + " - optimism-corrected **" + str(perf["C_corrigido"]) +
    "** (95% CI " + str(ci[0]) + " to " + str(ci[1]) + ") - out-of-fold " + str(perf["C_fora_da_amostra"]) +
    " - calibration " + perf["calibracao"] + "."
)
st.caption(
    "Leave-one-site-out cross-validation, restricted to the " + str(perf["LOSO_sitios_avaliaveis"]) +
    " of 50 sites with at least ten participants and two events, gave a median site-level C of " +
    str(perf["LOSO_mediana"]) + " (IQR " + str(iqr[0]) + " to " + str(iqr[1]) + ") and an event-weighted mean of " +
    str(perf["LOSO_ponderada_por_eventos"]) + ". The spread reflects a median of five events per site rather than "
    "genuine heterogeneity in performance."
)
st.caption("Data: PPMI (cut 29-Apr-2026). TRIPOD reporting. 7/10-year risk adjusted for the competing risk of death (Aalen-Johansen). "
           "Genetics, neuroimaging and biomarkers were tested and did not add discrimination.")
