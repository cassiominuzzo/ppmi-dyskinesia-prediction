# 24_supplementary_figure_S2.py
# Produces: Supplementary Figure S2 (head-to-head vs previously published predictor sets)
# Reads the table written by pipeline/14_head_to_head_with_CI.py.

import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PROJECT_ROOT

matplotlib.rcParams["svg.fonttype"] = "none"
TAB = os.path.join(PROJECT_ROOT, "04_Resultados", "Tabelas")
OUT = os.path.join(PROJECT_ROOT, "04_Resultados", "Figuras")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(TAB, "tab67_head_to_head_CI.csv"))
ref = df[df["model"].str.contains("Total-6")].iloc[0]
others = df[~df["model"].str.contains("Total-6")].sort_values("C", ascending=False)
df = pd.concat([pd.DataFrame([ref]), others], ignore_index=True)
n = len(df)
ys = np.arange(n)[::-1]

fig, ax = plt.subplots(figsize=(13.6, 0.92 * n + 2.0))
for y, (_, r) in zip(ys, df.iterrows()):
    is_ref = "Total-6" in r["model"]
    c = "#2E6DA4" if is_ref else "#8A9BA8"
    ax.plot([r["C_lo"], r["C_hi"]], [y, y], color=c, lw=2.6 if is_ref else 2.0, solid_capstyle="round")
    ax.plot(r["C"], y, "o", color=c, ms=11 if is_ref else 8, zorder=4)

ax.axvline(ref["C"], color="#2E6DA4", ls=":", lw=1.4, alpha=0.75)
ax.axvline(0.50, color="#BFBFBF", ls="--", lw=1.2)

labels = ["%s%s  (%d)" % (r["model"], " *" if "Total-6" in r["model"] else "", r["k"])
          for _, r in df.iterrows()]
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=12)
for t, (_, r) in zip(ax.get_yticklabels(), df.iterrows()):
    if "Total-6" in r["model"]:
        t.set_fontweight("bold"); t.set_color("#2E6DA4")

xr = ax.get_xlim()[1]
ax.text(1.02, 1.0, "ΔC vs Total-6\n(95% CI)", transform=ax.transAxes, fontsize=12,
        fontweight="bold", ha="center", va="bottom")
ax.text(1.35, 1.0, "p", transform=ax.transAxes, fontsize=12, fontweight="bold", ha="center", va="bottom")
for y, (_, r) in zip(ys, df.iterrows()):
    if "Total-6" in r["model"]:
        ax.text(1.02, y, "reference", transform=ax.get_yaxis_transform(), fontsize=11.5,
                ha="center", va="center", color="#2E6DA4", style="italic")
        continue
    sig = r["p"] < 0.05
    ax.text(1.02, y, "%+.3f (%+.3f to %+.3f)" % (r["dC"], r["d_lo"], r["d_hi"]),
            transform=ax.get_yaxis_transform(), fontsize=11.5, ha="center", va="center",
            color="#1A1A1A" if sig else "#8A8A8A")
    ax.text(1.35, y, "%.3f" % r["p"], transform=ax.get_yaxis_transform(), fontsize=11.5,
            ha="center", va="center", fontweight="bold" if sig else "normal",
            color="#1A1A1A" if sig else "#8A8A8A")

ax.set_xlabel("Out-of-fold C-index (95% CI)", fontsize=13)
ax.set_ylim(-0.8, n - 0.2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.text(0.02, 0.015,
         "* this study. Numbers in parentheses are the number of predictors. All predictor sets were refitted on the same PPMI sample\n"
         "(n = 814, 165 events), so the comparison reflects the choice of variables rather than the originally published coefficients.",
         fontsize=10, color="#7A7A7A", style="italic")
fig.tight_layout(rect=[0, 0.06, 0.72, 1])
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(OUT, "FigureS2_head_to_head." + ext), dpi=200, bbox_inches="tight")
plt.close()
print("saved FigureS2_head_to_head.{png,svg,pdf} to", OUT)
