# 23_supplementary_figure_S1.py
# Produces: Supplementary Figure S1 (internal calibration of the development model at 5 years)
# Out-of-fold predicted risk from repeated 5-fold cross-validation, observed risk by
# Kaplan-Meier within quintiles of predicted risk. Reports the out-of-fold O:E and
# calibration slope quoted in the Results and in the Abstract.

import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import KFold
from config import PROJECT_ROOT

matplotlib.rcParams["svg.fonttype"] = "none"
np.random.seed(42)
PR = os.path.join(PROJECT_ROOT, "02_Dados_Processados")
OUT = os.path.join(PROJECT_ROOT, "04_Resultados", "Figuras")
os.makedirs(OUT, exist_ok=True)

TOTAL6 = ["updrs_totscore", "ageonset", "SEX", "BMI", "td_pigd_ratio", "NP2FREZ"]
HORIZON, N_REP, N_FOLD, PENALIZER = 5.0, 5, 5, 0.05

d = pickle.load(open(os.path.join(PR, "sel_data.pkl"), "rb"))["cc"].reset_index(drop=True)
cols = TOTAL6 + ["exit", "event"]
print("development sample: n=%d, events=%d" % (len(d), int(d.event.sum())))


def risk_at(cph, X, t):
    sf = cph.predict_survival_function(X, times=[t])
    return 1.0 - sf.values[0]


acc = np.zeros(len(d)); acc_lp = np.zeros(len(d))
for rep in range(N_REP):
    kf = KFold(N_FOLD, shuffle=True, random_state=100 + rep)
    for tr, te in kf.split(d):
        m = CoxPHFitter(penalizer=PENALIZER).fit(d.iloc[tr][cols], "exit", "event")
        acc[te] += risk_at(m, d.iloc[te][TOTAL6], HORIZON)
        acc_lp[te] += np.log(m.predict_partial_hazard(d.iloc[te][TOTAL6]).values)
d["pred"] = acc / N_REP
d["lp"] = acc_lp / N_REP

# observed risk by quintile of out-of-fold predicted risk
d["q"] = pd.qcut(d["pred"], 5, labels=False)
rows = []
for q, g in d.groupby("q"):
    km = KaplanMeierFitter().fit(g["exit"], g["event"])
    obs = float(1 - km.predict(HORIZON))
    ci = km.confidence_interval_survival_function_
    j = ci.index.searchsorted(HORIZON, side="right") - 1
    lo, hi = (1 - ci.iloc[j, 1], 1 - ci.iloc[j, 0]) if j >= 0 else (np.nan, np.nan)
    rows.append(dict(q=int(q) + 1, pred=g["pred"].mean(), obs=obs, lo=lo, hi=hi, n=len(g)))
cal = pd.DataFrame(rows)

# Calibration slope: coefficient of the out-of-fold linear predictor in a Cox model
# refitted on the same data. This is the standard definition; a least-squares line
# through the aggregated quintiles is a different and less stable estimator.
oe = cal["obs"].mean() / cal["pred"].mean()
cs = CoxPHFitter().fit(d[["lp", "exit", "event"]], "exit", "event")
slope = cs.params_["lp"]
slope_ci = cs.confidence_intervals_.loc["lp"].values
print("out-of-fold  O:E = %.2f | calibration slope = %.2f (95%% CI %.2f to %.2f)"
      % (oe, slope, slope_ci[0], slope_ci[1]))

fig, ax = plt.subplots(figsize=(6.0, 6.0))
lim = max(cal["hi"].max(), cal["pred"].max()) * 1.15
ax.plot([0, lim], [0, lim], "--", color="#BFBFBF", lw=1.4)
ax.text(lim * 0.97, lim * 0.99, "perfect\ncalibration", ha="right", va="top", fontsize=9,
        color="#8A8A8A", style="italic")
ax.errorbar(cal["pred"], cal["obs"], yerr=[cal["obs"] - cal["lo"], cal["hi"] - cal["obs"]],
            fmt="o-", color="#3C6E9F", ecolor="#3C6E9F", capsize=0, lw=1.6, ms=8, zorder=3)
for _, r in cal.iterrows():
    ax.annotate("Q%d" % r["q"], (r["pred"], r["obs"]), textcoords="offset points",
                xytext=(9, -11), fontsize=9, color="#6B6B6B")
ax.text(0.03, 0.965, "Observed-to-expected ratio  %.2f" % oe, transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#1F4E79")
ax.text(0.03, 0.915, "Calibration slope  %.2f" % slope, transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#1F4E79")
ax.text(0.03, 0.868, "n = %d, %d events" % (len(d), int(d.event.sum())), transform=ax.transAxes,
        fontsize=10, color="#6B6B6B")
ax.set_xlabel("Predicted 5-year risk (out-of-fold)")
ax.set_ylabel("Observed 5-year risk (Kaplan-Meier)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(OUT, "FigureS1_internal_calibration." + ext), dpi=200, bbox_inches="tight")
plt.close()
cal.to_csv(os.path.join(PROJECT_ROOT, "04_Resultados", "Tabelas", "figS1_internal_calibration.csv"), index=False)
print("saved FigureS1_internal_calibration.{png,svg,pdf} to", OUT)
