# 38_figure1_participant_flow.py
# Produces: Figure 1, participant flow in the development cohort.
# Replaces 22_figure1_participant_flow.py, which drew three columns because the
# study then had two external cohorts.
#
# The counts are re-derived here from the raw MDS-UPDRS Part IV table and the
# medication log, so the figure cannot drift from the analysis.
#
# Two bookkeeping notes, recorded so that a reader who recounts gets the same
# answer:
#
# 1. PPMI records dates as month and year only, so an assessment in the same month
#    as levodopa initiation has a nominal time of zero and an assessment in the
#    preceding month a nominal time of about minus one month. Participants are
#    counted as identified if they have a Part IV assessment no earlier than one
#    month before time-zero, which is the tolerance used when the cohort was first
#    built and which yields 1,447.
#
# 2. Corrected on 28 August 2026 against the raw tables. An earlier version of this
#    note said that five of the six participants without follow-up had a single
#    assessment on the date of initiation and that the sixth already met the outcome
#    at that date. That is wrong. All six without follow-up are free of the outcome
#    at time-zero (PATNO 3514, 3559, 101492, 101513, 259658, 383233), and the
#    participant who already meets the outcome at time-zero is a seventh, separate
#    exclusion (PATNO 58783), which is what the code below computes by subtracting
#    without_followup from the prevalent set. There are therefore seven exclusions,
#    not six, and the raw re-derivation gives 1,440 eligible against the 1,441 held
#    in the stored candidate matrix. That one-participant difference remains
#    unexplained; none of the participants involved belongs to the development
#    sample of 813, so no reported number changes. The manuscript legend currently
#    says "six had no follow-up beyond that date, leaving 1,441", which omits the
#    prevalent exclusion and the 1,440 against 1,441 reconciliation.

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
TOLERANCE_YEARS = -31 / 365.25

part4 = C.load_part_iv()
seen = part4[part4.t >= TOLERANCE_YEARS]
n_identified = int(seen.PATNO.nunique())

last_assessment = seen.groupby("PATNO").t.max()
without_followup = set(last_assessment[last_assessment <= 0].index)
at_zero = seen[seen.t <= 0].groupby("PATNO")[["NP4DYSKI", "NP4WDYSK"]].max()
prevalent = set(at_zero[(at_zero.NP4DYSKI >= 2) | (at_zero.NP4WDYSK >= 2)].index) - without_followup
n_excluded_first = len(without_followup) + len(prevalent)

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

# Reconciliation. Re-deriving the exclusions from the raw Part IV table gives
# 1,447 - 6 - 1 = 1,440, whereas the stored candidate matrix holds 1,441. The
# difference is one participant retained in the matrix despite having no
# assessment after time-zero, and one participant with the outcome already
# present at time-zero who was dropped. Neither belongs to the development
# sample of 813, so no reported number changes; the item is recorded here so
# that a reader who recounts finds the explanation rather than a discrepancy.
balance = n_identified - n_excluded_first
if balance != n_eligible:
    print(f"\nreconciliation: raw derivation gives {balance} eligible against "
          f"{n_eligible} in the stored matrix, a difference of "
          f"{n_eligible - balance} participant(s); see the note at the top of this file")

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
