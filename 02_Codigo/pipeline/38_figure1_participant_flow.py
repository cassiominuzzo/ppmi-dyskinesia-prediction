# 38_figure1_participant_flow.py
# Produces: Figure 1, participant flow in the development cohort.
# Replaces 22_figure1_participant_flow.py, which drew three columns because the
# study then had two external cohorts.
#
# The counts come from the analysis base itself, so the figure cannot drift from the analysis.
#
# Corrected in November 2026. This script used to recount from the raw Part IV table with a
# tolerance of one month before time-zero, and from a medication log that dates levodopa
# slightly differently. That recount also reached 1,447, by coincidence: it included one
# participant whose last assessment is 29 days before time-zero (PATNO 101492) and missed one
# who is in the analysis (PATNO 41299). So it listed seven exclusions, and the figure showed
# 1,447 minus 7 arriving at 1,441. The analysis base has six, and they add up: five with no
# assessment after time-zero (3514, 3559, 101513, 259658, 383233) and one who already met the
# outcome at time-zero (58783), who does have later assessments.

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import cohort as C

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "DejaVu Sans"

OUT = C.ensure_dir(os.path.join(C.RESULTS, "Figuras"))

base = pd.read_parquet(os.path.join(C.PROC, "analytic_base.parquet"))
n_identified = len(base)
no_time = base[base.exit <= 0]
without_followup = no_time[no_time.event == 0]
prevalent = no_time[no_time.event == 1]
n_excluded_first = len(no_time)

matrix = pd.read_parquet(os.path.join(C.PROC, "candidate_matrix.parquet"))
if "ageonset" not in matrix.columns and "ageonset_x" in matrix.columns:
    matrix = matrix.rename(columns={"ageonset_x": "ageonset"})
eligible = matrix[matrix.exit > 0]
n_eligible = len(eligible)
events_eligible = int(eligible.event.sum())

development = C.build_cohort()
n_final = len(development)
events_final = int(development.event.sum())
n_incomplete = n_eligible - n_final

print(f"identified with a Part IV assessment at or after levodopa initiation: {n_identified}")
print(f"  excluded, no follow-up beyond time-zero: {len(without_followup)}")
print(f"  excluded, outcome already present at time-zero: {len(prevalent)}")
print(f"eligible: {n_eligible} ({events_eligible} events)")
print(f"  excluded, incomplete on the 18 candidate clinical predictors: {n_incomplete}")
print(f"development sample: {n_final} ({events_final} events)")

assert n_identified - n_excluded_first == n_eligible, "the flow no longer adds up"
assert set(pd.to_numeric(base[base.exit > 0].PATNO)) == set(pd.to_numeric(eligible.PATNO))

NAVY = "#1F4E79"
GREY = "#6B6B6B"

boxes = [
    ("Patients with Parkinson's disease in PPMI with an MDS-UPDRS\n"
     "Part IV assessment at or after levodopa initiation",
     f"n = {n_identified:,}"),
    ("Eligible participants",
     f"n = {n_eligible:,}   ({events_eligible} events)"),
    ("Development sample: complete data on all\n18 candidate clinical predictors",
     f"n = {n_final:,}   ({events_final} events)"),
]
exclusions = [
    (f"Excluded (n = {n_excluded_first})\n"
     f"no follow-up beyond time-zero ({len(without_followup)})\n"
     f"outcome already present at time-zero ({len(prevalent)})"),
    (f"Excluded (n = {n_incomplete})\n"
     f"incomplete on at least one of the\n18 candidate clinical predictors"),
]

fig, ax = plt.subplots(figsize=(8.2, 6.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

x_main, w_main = 0.04, 0.56
x_excl, w_excl = 0.64, 0.34
tops = [0.94, 0.58, 0.22]
height = 0.16

for i, (title, count) in enumerate(boxes):
    y = tops[i]
    ax.add_patch(FancyBboxPatch((x_main, y - height), w_main, height,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=1.4, edgecolor=NAVY, facecolor="#EEF3FA"))
    ax.text(x_main + w_main / 2, y - 0.055, title, ha="center", va="center",
            fontsize=9.0, color="#12263A")
    ax.text(x_main + w_main / 2, y - height + 0.036, count, ha="center", va="center",
            fontsize=10.4, color=NAVY, fontweight="bold")

for i, text in enumerate(exclusions):
    y = tops[i] - height - 0.030
    box_h = 0.125
    ax.add_patch(FancyBboxPatch((x_excl, y - box_h), w_excl, box_h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=1.0, edgecolor=GREY, facecolor="#F5F5F5"))
    ax.text(x_excl + w_excl / 2, y - box_h / 2, text, ha="center", va="center",
            fontsize=8.0, color="#3A3A3A")
    ax.annotate("", xy=(x_excl, y - box_h / 2),
                xytext=(x_main + w_main / 2, y - box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=GREY, linewidth=1.0))

for i in range(len(boxes) - 1):
    ax.annotate("", xy=(x_main + w_main / 2, tops[i + 1]),
                xytext=(x_main + w_main / 2, tops[i] - height),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, linewidth=1.4))

fig.tight_layout()
for extension in ("png", "svg", "pdf"):
    path = os.path.join(OUT, f"Figure1_participant_flow.{extension}")
    fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
    print(f"written: {path}")
plt.close(fig)
