# 25_figure2_km_tertiles.py
# Produces: Figure 2 (cumulative incidence by tertile of model-predicted risk, PPMI)
# Includes the numbers-at-risk table below the curves. Tertiles are defined by the
# model fitted in these same data, so the figure illustrates the risk gradient; the
# out-of-fold evidence is in Supplementary Figure S1.

import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from config import PROJECT_ROOT

matplotlib.rcParams["svg.fonttype"] = "none"
PR = os.path.join(PROJECT_ROOT, "02_Dados_Processados")
OUT = os.path.join(PROJECT_ROOT, "04_Resultados", "Figuras")
os.makedirs(OUT, exist_ok=True)

TOTAL6 = ["updrs_totscore", "ageonset", "SEX", "BMI", "td_pigd_ratio", "NP2FREZ"]
TMAX, TICKS = 10.0, [0, 2, 4, 6, 8, 10]
NAMES = ["Low risk", "Intermediate risk", "High risk"]
COLS = ["#1B7A6B", "#B08D2E", "#B23A38"]

d = pickle.load(open(os.path.join(PR, "sel_data.pkl"), "rb"))["cc"].reset_index(drop=True)
cph = CoxPHFitter(penalizer=0.05).fit(d[TOTAL6 + ["exit", "event"]], "exit", "event")
d["lp"] = cph.predict_partial_hazard(d[TOTAL6]).values
d["grp"] = pd.qcut(d["lp"], 3, labels=False)
print("n per tertile:", d.groupby("grp").size().tolist())

lr = multivariate_logrank_test(d["exit"], d["grp"], d["event"])
p = lr.p_value

fig = plt.figure(figsize=(9.6, 8.4))
gs = fig.add_gridspec(2, 1, height_ratios=[3.05, 1.0], hspace=0.30)
ax, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

at_risk = {}
for g in range(3):
    s = d[d["grp"] == g]
    km = KaplanMeierFitter().fit(s["exit"], s["event"], label=NAMES[g])
    t = km.survival_function_.index.values
    y = (1 - km.survival_function_.values.ravel()) * 100
    m = t <= TMAX
    ax.step(np.r_[0, t[m]], np.r_[0, y[m]] , where="post", color=COLS[g], lw=2.4, label="%s (n=%d)" % (NAMES[g], len(s)))
    at_risk[g] = [int((s["exit"] >= tt).sum()) for tt in TICKS]

ax.set_xlim(0, TMAX); ax.set_ylim(0, None)
ax.set_xticks(TICKS)
ax.set_xlabel("Years since levodopa initiation", fontsize=13)
ax.set_ylabel("Cumulative incidence of\nproblematic dyskinesia (%)", fontsize=13)
ax.legend(loc="upper left", fontsize=12, frameon=False)
ax.text(0.985, 0.05, "log-rank p < 0.001" if p < 0.001 else "log-rank p = %.3f" % p,
        transform=ax.transAxes, ha="right", fontsize=12, color="#6B6B6B", style="italic")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

axt.axis("off")
axt.set_xlim(0, TMAX)
axt.text(0, 0.86, "Patients at risk", fontsize=12.5, fontweight="bold", color="#1A1A1A",
         transform=axt.get_yaxis_transform() if False else axt.transAxes)
for g in range(3):
    yy = 0.58 - g * 0.22
    axt.text(-0.005, yy, NAMES[g], transform=axt.transAxes, fontsize=11.5, color=COLS[g], ha="left", va="center")
    for tt, v in zip(TICKS, at_risk[g]):
        axt.text(tt / TMAX * 0.845 + 0.155, yy, str(v), transform=axt.transAxes,
                 fontsize=11.5, color="#1A1A1A", ha="center", va="center")

fig.tight_layout()
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(OUT, "Figure2_km_tertiles." + ext), dpi=200, bbox_inches="tight")
plt.close()
print("at risk at %s: %s" % (TICKS, {NAMES[g]: at_risk[g] for g in range(3)}))
print("saved Figure2_km_tertiles.{png,svg,pdf} to", OUT)
