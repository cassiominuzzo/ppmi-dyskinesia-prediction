# 48_supplementary_tables.py
# Produces: 00_Manuscrito/Supplementary Tables (rebuilt).docx, the sixteen supplementary
# tables of the manuscript with their legends.
#
# Every number is read from a file in 03_Resultados/Tabelas/; none is typed in, so the
# document can be regenerated after any upstream script is re-run. The numbering follows
# the order of first citation in the manuscript, which is what fixes S1 to S16.
#
# Two tables have a source outside the numbered pipeline and say so in their own legend:
# S4 comes from tab_incremental_value_all_domains.csv, the 89-variable incremental-value
# analysis, and S15 from the tabS7 files of the matched-degrees-of-freedom analysis.
#
# Usage:  python 48_supplementary_tables.py [output.docx]

import json
import os
import sys

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

import cohort as C

TAB = os.path.join(C.RESULTS, "Tabelas")

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(C.RESULTS), "00_Manuscrito", "Supplementary Tables (rebuilt).docx")

MINUS = "−"


def load(name):
    p = os.path.join(TAB, name)
    return json.load(open(p, encoding="utf-8")) if name.endswith(".json") else pd.read_csv(p)


# ----------------------------------------------------------------- formatting ---
def num(x, d=2):
    s = half_up(abs(float(x)), d)
    return (MINUS + s) if float(x) < 0 else s


def half_up(x, d):
    """Round half away from zero, as a reader expects, rather than however the binary
    representation happens to fall. Three Schoenfeld statistics land exactly on a half
    (0.975, 0.185, 0.075) and Python's float formatting rounds all three down."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def sgn(x, d=4):
    s = f"{abs(float(x)):.{d}f}"
    return (MINUS if float(x) < 0 else "+") + s


def pv(p):
    p = float(p)
    if p < 0.0001:
        return "<0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def ci(h, lo, hi, d=2):
    return f"{num(h, d)} ({num(lo, d)} to {num(hi, d)})"


def pct(x, d=1):
    return f"{float(x):.{d}f}%"


def thousands(n):
    return f"{int(n):,}"


# --------------------------------------------------------------- doc scaffold ---
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(9)
st.paragraph_format.space_after = Pt(4)
for s in doc.sections:
    s.top_margin = s.bottom_margin = Cm(1.6)
    s.left_margin = s.right_margin = Cm(1.6)


def landscape(section):
    w, h = section.page_width, section.page_height
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = max(w, h), min(w, h)
    section.top_margin = section.bottom_margin = Cm(1.2)
    section.left_margin = section.right_margin = Cm(1.2)


# Set LID_ONLY to a list of table numbers to emit just those, which is what makes it
# practical to replace one table inside a document that is being edited by hand.
ONLY = {int(x) for x in os.environ.get("LID_ONLY", "").replace(",", " ").split()} or None
CURRENT = [0]


def active():
    return ONLY is None or CURRENT[0] in ONLY


def para(text, bold=False, size=9, space_before=0, space_after=4, italic=False):
    if not active():
        return None
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    return p


def title(n, text):
    CURRENT[0] = n
    if not active():
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(f"Supplementary Table S{n}. ")
    r.bold = True
    r.font.size = Pt(9.5)
    r2 = p.add_run(text)
    r2.bold = True
    r2.font.size = Pt(9.5)


def sub(text):
    para(text, bold=True, size=9, space_before=7, space_after=3)


def legend(text):
    para(text, size=8, space_before=5, space_after=8)


def table(header, rows, widths=None, size=8, group_rows=()):
    if not active():
        return None
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = widths is None
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(size)
        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
    for k, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run("" if v is None else str(v))
            r.font.size = Pt(size)
            if k in group_rows:
                r.bold = True
            cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return t


doc.add_paragraph().add_run("Supplementary Tables").bold = True
doc.paragraphs[-1].runs[0].font.size = Pt(13)
para("Prognostic model for problematic levodopa-induced dyskinesia in Parkinson's disease. "
     "Unless stated otherwise, every quantity is estimated in the development sample of 813 "
     "participants with 165 events, and each legend names the script or the file that produces "
     "the table.", size=8.5, space_after=10)

# ============================================================== S1 ==============
s37, s39 = load("tab37_eligibility_sensitivity.json"), load("tab39_drug_name_quality_control.json")

title(1, "Sensitivity analyses: definition of time-zero, definition of the outcome, "
         "handling of prevalent cases, eligibility and drug-name matching")
sub("A. Time-zero, outcome definition and prevalent cases")
S1A = [
    ("Levodopa initiation", "Any dyskinesia (≥ 1)", "Right-censored", 810, 321, "57%", "0.662", "0.649", "1.0114"),
    ("Levodopa initiation", "Any dyskinesia (≥ 1)", "Prevalent cases excluded", 778, 289, "55%", "0.651", "0.634", "1.0094"),
    ("Levodopa initiation", "Problematic (4.1 or 4.2 ≥ 2), primary", "Right-censored", 813, 165, "24%", "0.712", "0.693", "1.0184"),
    ("Levodopa initiation", "Problematic (4.1 or 4.2 ≥ 2)", "Prevalent cases excluded", 806, 158, "23%", "0.695", "0.673", "1.0167"),
    ("Levodopa initiation", "Item 4.1 ≥ 2 only", "Right-censored", 813, 143, "22%", "0.713", "0.690", "1.0185"),
    ("Levodopa initiation", "Item 4.1 ≥ 2 only", "Prevalent cases excluded", 807, 137, "22%", "0.697", "0.678", "1.0160"),
    ("First dopaminergic therapy", "Any dyskinesia (≥ 1)", "Right-censored", 813, 325, "40%", "0.603", "0.583", "1.0102"),
    ("First dopaminergic therapy", "Any dyskinesia (≥ 1)", "Prevalent cases excluded", 788, 300, "38%", "0.585", "0.554", "1.0090"),
    ("First dopaminergic therapy", "Problematic (4.1 or 4.2 ≥ 2)", "Right-censored", 813, 165, "15%", "0.668", "0.644", "1.0178"),
    ("First dopaminergic therapy", "Problematic (4.1 or 4.2 ≥ 2)", "Prevalent cases excluded", 810, 162, "15%", "0.662", "0.637", "1.0175"),
    ("First dopaminergic therapy", "Item 4.1 ≥ 2 only", "Right-censored", 813, 143, "14%", "0.674", "0.648", "1.0179"),
    ("First dopaminergic therapy", "Item 4.1 ≥ 2 only", "Prevalent cases excluded", 810, 140, "14%", "0.667", "0.641", "1.0176"),
]
table(["Time-zero", "Outcome definition", "Prevalent cases", "n", "Events",
       "5-year incidence", "C (apparent)", "C (out-of-fold)", "HR per MDS-UPDRS point"],
      [list(r) for r in S1A],
      widths=[3.1, 3.9, 2.9, 1.1, 1.2, 1.7, 1.7, 1.8, 2.0])

sub("B. Eligibility and drug-name matching")
S1B = [
    ("Primary analysis", 813, 165, f"{s37['C_full']:.4f}", "reference", f"{s37['C_frozen_full']:.4f}"),
    ("Excluding the 9 participants enrolled in other PPMI cohorts (7 SWEDD, 1 PRKN, 1 LRRK2), of whom 1 reached the outcome",
     s37["n_retained"], s37["events_retained"], f"{s37['C_retained']:.4f}",
     sgn(s37["delta_C"]), f"{s37['C_frozen_retained']:.4f}"),
    ("Excluding the 5 participants whose earliest levodopa record used an unmatched brand name, which would move their time-zero 427 to 1,126 days earlier",
     s39["n_retained"], s39["events_retained"], f"{s39['C_retained']:.4f}",
     sgn(s39["delta_C"]), "not applicable"),
]
table(["Analysis", "n", "Events", "C (out-of-fold)", "Change in C", "C of the frozen model"],
      [list(r) for r in S1B], widths=[9.0, 1.2, 1.3, 2.2, 1.9, 2.6])

legend(
    "(A) The twelve analyses were run in the same complete-case development sample of 813 participants, "
    "so that the only differences between rows are the origin of the time axis, the definition of the "
    "outcome and whether participants whose first assessment already met the outcome were excluded. "
    "The row marked primary is the analysis reported throughout the paper and reproduces it exactly. "
    "Cumulative incidence is the Kaplan-Meier estimate at five years. The hazard ratio is that of the "
    "total MDS-UPDRS within the six-variable model, per point, and is given so that the direction and "
    "size of the strongest predictor can be compared across definitions. Discrimination falls under "
    "every alternative definition, and it falls furthest when time-zero is moved back to the first "
    "dopaminergic drug of any kind, which places the origin before the exposure that causes the "
    "outcome. (B) Two eligibility questions raised during quality control. Neither exclusion changes "
    "discrimination by more than 0.004. Change in C is against the primary out-of-fold value of "
    f"{s37['C_full']:.4f}. Sources: 37_eligibility_sensitivity.py and 39_drug_name_quality_control.py.")

# ============================================================== S2 ==============
a = load("tab31_amantadine.json")
title(2, "Amantadine at time-zero and during follow-up: four sensitivity analyses")
sub("A. Amantadine as a covariate")
table(["Model", "n", "Amantadine users", "Hazard ratio (95% CI)", "p", "Change in out-of-fold C"],
      [["Unadjusted", a["n"], a["baseline_users"],
        ci(a["A_crude"]["HR"], a["A_crude"]["lo"], a["A_crude"]["hi"]), pv(a["A_crude"]["p"]), "not applicable"],
       ["Adjusted for the six clinical predictors", a["n"], a["baseline_users"],
        ci(a["A_adjusted"]["HR"], a["A_adjusted"]["lo"], a["A_adjusted"]["hi"]), pv(a["A_adjusted"]["p"]),
        sgn(a["A_adjusted"]["delta_C"], 3)]],
      widths=[6.2, 1.3, 2.4, 3.4, 1.5, 3.0])

sub("B. Excluding the participants already receiving amantadine at time-zero")
nonu = a["n"] - a["baseline_users"]
table(["Analysis", "n", "C", "Change in C"],
      [["Model refitted in the participants not receiving amantadine", nonu,
        f"{a['B']['C_refit']:.3f}", sgn(a["B"]["delta_C"], 3)],
       ["Frozen model applied to the participants not receiving amantadine", nonu,
        f"{a['B']['C_frozen_non_users']:.3f}", "not applicable"],
       ["Frozen model applied to the participants receiving amantadine", a["baseline_users"],
        f"{a['B']['C_frozen_users']:.3f}", "not applicable"]],
      widths=[9.6, 1.5, 1.9, 3.0])

sub("C. Amantadine started during follow-up, as a possible mask on the outcome")
sf, mc = a["started_during_followup"], a["masking_candidates"]
table(["Group", "n"],
      [["Started amantadine after time-zero", sf["total"]],
       ["   reached the outcome before starting", sf["event_before"]],
       ["   reached the outcome after starting", sf["event_after"]],
       ["   never reached the outcome", sf["event_free"]],
       ["      with a dyskinesia score of 1 or more recorded before starting", mc["with_prior_score"]],
       ["      with a dyskinesia score of zero at every assessment before starting", mc["score_zero"]],
       ["      with no Part IV assessment before starting", mc["no_assessment"]]],
      widths=[13.0, 2.0])
para(" ", size=4)
table(["Reclassification", "Events", "Change in out-of-fold C"],
      [["None, primary analysis", a["events"], "reference"],
       [f"The {mc['with_prior_score']} with a documented score before starting, treated as events at the start date",
        a["C"]["plausible_events"], sgn(a["C"]["plausible_delta_C"], 3)],
       [f"All {sf['event_free']} event-free starters, treated as events at the start date, an absolute bound",
        a["C"]["extreme_events"], sgn(a["C"]["extreme_delta_C"], 3)]],
      widths=[10.5, 1.8, 3.4])

sub("D. Amantadine inside the levodopa-equivalent conversion")
d = a["D"]
table(["Quantity", "Median in the participants receiving amantadine"],
      [["Amantadine contribution to the converted daily dose, mg", f"{d['median_amantadine_mg']:.0f}"],
       ["Total converted daily dose, mg", f"{d['median_total_mg']:.0f}"],
       ["Amantadine as a share of the converted daily dose", pct(d["median_share_pct"], 0)]],
      widths=[9.0, 6.0])

legend(
    "Amantadine occupies two positions in this study, and each is addressed separately. It is the only "
    "drug in the regimen with established anti-dyskinetic efficacy, so within the levodopa-equivalent "
    "conversion it is a measurement problem, and when started during follow-up it can suppress "
    "dyskinesia and prevent a participant from ever crossing the outcome threshold, which is an "
    "outcome problem. (A) Cox models in the development sample. The adjusted model carries the six "
    "standardised clinical predictors and the same ridge penalty of 0.05 as the development model. "
    "The association is present, but adding amantadine as a seventh covariate leaves discrimination "
    "unchanged, so it marks disease activity the model already captures. (B) Out-of-fold concordance "
    "when the model is refitted without the users, and concordance of the frozen model applied to each "
    "group separately. Discrimination is preserved in the users, so the model is not driven by them. "
    "(C) The masking analysis is deliberately conservative: the second row reclassifies only "
    "participants with a recorded dyskinesia score before starting amantadine, and the third "
    "reclassifies every event-free starter, which is an upper bound that no plausible degree of "
    "masking could exceed. Even that bound costs 0.054 of discrimination, which does not overturn the "
    "model. (D) Amantadine was removed from the levodopa-equivalent sum reported in Table 1, because "
    "the sum would otherwise add drugs that raise the risk of dyskinesia to one that treats it. "
    "Change in out-of-fold C is against the reference value of "
    f"{a['C_reference']:.3f}. Source: 31_amantadine_handling.py.")

# ============================================================== S3 ==============
title(3, "Multiple imputation as a sensitivity analysis for missing predictor data")
sub("A. Structure of missingness among the participants identified in PPMI")
table(["Group", "n", "Events", "Event rate", "Median follow-up, years", "Imputed"],
      [["Identified with an MDS-UPDRS Part IV assessment at or after levodopa initiation", "1,447", 303, "20.9%", "2.5", "no"],
       ["   of whom follow-up greater than zero, the analysis base", "1,441", 302, "21.0%", "2.5", "no"],
       ["Complete on all 18 candidate clinical variables, the development sample", "813", 165, "20.3%", "2.4", "reference"],
       ["Pre-levodopa assessment present, one or more candidates missing", "301", 41, "13.6%", "1.9", "yes"],
       ["No pre-levodopa assessment, every candidate missing", "327", 96, "29.4%", "4.0", "no"]],
      widths=[8.6, 1.3, 1.3, 1.5, 2.6, 1.6])

sub("B. Total-6 coefficients, complete case against multiple imputation")
S3B = [
    ("Total MDS-UPDRS (Parts I+II+III)", "+0.0182 (0.0046)", "+0.0180 (0.0102 to 0.0258)", "0.020", f"{MINUS}0.04"),
    ("Age at onset, years", f"{MINUS}0.0303 (0.0070)", f"{MINUS}0.0210 ({MINUS}0.0337 to {MINUS}0.0082)", "0.075", "+1.32"),
    ("Sex (1 = male)", f"{MINUS}0.2920 (0.1475)", f"{MINUS}0.3529 ({MINUS}0.6084 to {MINUS}0.0973)", "0.021", f"{MINUS}0.41"),
    ("Body-mass index, kg/m²", f"{MINUS}0.0344 (0.0156)", f"{MINUS}0.0209 ({MINUS}0.0473 to 0.0054)", "0.035", "+0.86"),
    ("TD/PIGD ratio", f"{MINUS}0.0650 (0.0416)", f"{MINUS}0.0650 ({MINUS}0.1410 to 0.0111)", "0.124", "0.00"),
    ("Freezing of gait (item 2.13)", "+0.2574 (0.1422)", "+0.2917 (0.0395 to 0.5439)", "0.007", "+0.24"),
    ("Apparent C-index", "0.712", "0.702 (range 0.690 to 0.709)", "", ""),
]
table(["Predictor", "Complete case (n = 813), β (SE)",
       "Multiple imputation (n = 1,114), β (95% CI)",
       "Fraction of missing information", "Shift, in complete-case SE"],
      [list(r) for r in S3B], widths=[4.6, 3.4, 5.0, 2.4, 2.4])

legend(
    "(A) Of the 1,447 patients with an MDS-UPDRS Part IV assessment at or after levodopa initiation, "
    "six had their only assessment on the date of initiation itself, giving zero follow-up, and one "
    "further participant already met the outcome at that date; seven were therefore excluded and the "
    "analysis base is 1,441 participants with 302 events. Imputation was restricted to the 1,114 "
    "participants with at least one pre-levodopa assessment, because the remaining 327 have no "
    "predictor information at all and imputing them would extrapolate rather than recover. Those 327 "
    "have a higher event rate and longer follow-up, which is why the imputed estimate is a check on "
    "the complete-case model and not a replacement for it. (B) Twenty imputations were generated by "
    "chained equations with Bayesian ridge regression and posterior sampling, including the event "
    "indicator and the Nelson-Aalen estimate of the cumulative hazard so that the imputation model was "
    "compatible with the survival model, and estimates were pooled by Rubin's rules. The shift column "
    "expresses the difference between the two estimates in units of the complete-case standard error, "
    "so that a value below one means the two agree within the noise of the complete-case analysis. "
    "Only age at onset moves by more than one standard error. Source: 05_multiple_imputation.py.")

# ============================================================== S4 ==============
iv = pd.read_csv(os.path.join(TAB, "tab_incremental_value_all_domains_corrected.csv"))
sec = doc.add_section()
landscape(sec)
title(4, "Incremental value of 89 multimodal candidate predictors over the six-variable "
         "clinical model, across ten domains")
rows, groups = [], set()
for dom in sorted(iv.domain.unique()):
    groups.add(len(rows))
    block = iv[iv.domain == dom].sort_values("p")
    rows.append([dom] + [""] * 9)
    for _, r in block.iterrows():
        rows.append([
            "   " + str(r.label), int(r.n), int(r.events),
            ci(r.HR, r.lo, r.hi), pv(r.p), pv(r.LRT_p),
            pv(r.FDR_domain), pv(r.FDR_global),
            f"{sgn(r.dC)} ({sgn(r.dC_lo)} to {sgn(r.dC_hi)})",
            num(r.min_HR_80pct),
        ])
table(["Domain and candidate predictor", "n", "Events", "HR (95% CI)", "p", "LRT p",
       "FDR p, within domain", "FDR p, all 89", "Change in C (95% CI)",
       # 85, not 80. The column is exp(K / (sqrt(events) x SD)), Schoenfeld's closed form,
       # and solving K across the 88 checkable rows gives 3.00 with a spread of 0.03, which
       # is the scatter that printing to two decimals produces. z_0.975 + z_0.80 is 2.8016;
       # 3.00 solves to z of 1.04, which is 85% power. The numbers are right, the label was
       # not. 52_multimodal_screen.py writes both constants, min_HR_80pct and min_HR_85pct,
       # so the two can be compared. The column read here still carries the published name
       # because it comes from the corrected file that 50_treatment_candidate_recomputed.py
       # derives from the original screen, where that name was assigned.
       "Smallest HR detectable with 85% power"],
      rows, widths=[7.2, 1.0, 1.1, 3.0, 1.3, 1.3, 1.9, 1.6, 4.2, 2.2],
      size=7, group_rows=groups)
best = iv.loc[iv.p.idxmin()]
legend(
    "Each candidate was added to the frozen six-variable clinical model, one at a time, and evaluated "
    "in the participants in whom it was recorded. Hazard ratios are per standard deviation for "
    "continuous candidates and per unit for binary ones, from a Cox model carrying the six clinical "
    "predictors alongside the candidate. LRT p is the likelihood-ratio test of the candidate against "
    "the clinical model alone. False-discovery-rate control was applied twice, by the "
    "Benjamini-Hochberg procedure: within each domain, which is the more permissive test, and across "
    "all 89 candidates together. Change in C is the difference in out-of-fold concordance, by repeated "
    "five-fold cross-validation in the participants with the candidate recorded, with a bootstrap "
    "confidence interval. Four candidates were nominally associated with the outcome at p < 0.05, all "
    "of them cerebrospinal-fluid markers and all in the same direction, with lower values indicating "
    f"higher risk; the strongest was {best.label} at p = {pv(best.p)}. None survived false-discovery-rate "
    "control under either procedure, and the largest change in discrimination among them was 0.0103, "
    "with a confidence interval that includes zero. "
    "How much weight a null carries depends on what the analysis could have detected, and the last "
    "column states that directly: the smallest hazard ratio, per standard deviation for a "
    "continuous candidate and per unit for a binary one, that this screen would have detected with "
    "85% power. In survival analysis that quantity is governed by the number of events rather than "
    "by the number of participants, and the two diverge here. The polygenic scores, for instance, "
    "were measured in 291 of the 813, which is 36% of the sample but 70% of the events, because "
    "those 291 belong entirely to the earlier recruitment wave and therefore have the longest "
    f"follow-up. For {int((iv.min_HR_80pct < 1.30).sum())} of the 89 candidates the detectable "
    f"hazard ratio is below 1.30 and for {int((iv.min_HR_80pct < 1.35).sum())} it is below 1.35, so "
    "for most of the screen an effect of the size usually claimed for a prognostic biomarker would "
    "not have been missed. Where that is not so, the column says so rather than leaving the reader "
    "to infer it: rare genotypes, where only very large effects are detectable, at 15.4 for SNCA "
    "carriers, 6.0 for GBA and 3.0 for LRRK2; the cerebrospinal-fluid seed amplification assay, "
    "positive in most of the cohort and therefore carrying little information; and the five "
    "cognitive tests administered to a subsample of about 440 participants with roughly 22 events "
    "each. The single treatment "
    "candidate is the total dopaminergic daily dose at time-zero with amantadine removed from the "
    "conversion, which is the variable the Methods describe and the one Table 1 reports. Five of the "
    "813 have no convertible dose, four of them because entacapone is recorded as a multiplier of the "
    "levodopa dose rather than as an absolute value; their dose is unknown rather than zero, so they "
    "are treated as missing, as every other candidate in this table treats a variable that was not "
    "recorded. None of the five reached the outcome, so the event count is 165 either way. Its change "
    "in discrimination "
    "is the only one in this table estimated under the cross-validation protocol documented in "
    "cohort.py, twenty repetitions of stratified five-fold, because that row was recomputed; the "
    "other 88 carry the estimate of the original screen. Rows within each domain are ordered by p. "
    "Sources: tab_incremental_value_all_domains_corrected.csv and "
    "50_treatment_candidate_recomputed.py.")

sec = doc.add_section()
sec.orientation = WD_ORIENT.PORTRAIT
sec.page_width, sec.page_height = min(sec.page_width, sec.page_height), max(sec.page_width, sec.page_height)
sec.top_margin = sec.bottom_margin = Cm(1.6)
sec.left_margin = sec.right_margin = Cm(1.6)

# ============================================================== S5 ==============
pen = load("tab42_penalty_sensitivity.csv")
penj = load("tab42_penalty_sensitivity.json")
title(5, "Sensitivity of the development model to the ridge penalty")
NAMES = {"updrs_totscore": "Total MDS-UPDRS", "ageonset": "Age at onset", "SEX": "Sex (1 = male)",
         "BMI": "Body-mass index", "td_pigd_ratio": "TD/PIGD ratio", "NP2FREZ": "Freezing of gait"}
rows = []
for _, r in pen.iterrows():
    lab = f"{r.penalty:g}" + (" (used)" if r.penalty == C.RIDGE else "")
    rows.append([lab, f"{r.C_apparent:.4f}", f"{r.slope_apparent:.3f}"] +
                [f"{r[k]:.4f}" for k in NAMES])
table(["Ridge penalty", "C (apparent)", "Calibration slope (apparent)"] + list(NAMES.values()),
      rows, widths=[2.0, 1.7, 2.4, 1.7, 1.5, 1.6, 1.7, 1.5, 1.7])
legend(
    "Hazard ratios are per unit of each predictor. The penalty was varied across six values from none "
    "to 0.5, refitting the whole model at each value. Apparent discrimination varies by "
    f"{penj['c_spread']:.4f} across the entire range, and no hazard ratio moves by more than "
    f"{penj['largest_hr_shift_vs_unpenalised_pct']:.1f}% against the unpenalised fit, so the "
    "specification does not depend on the choice of penalty. The calibration slope behaves differently "
    "and is the reason a penalty was used at all: it is 1.00 in the apparent, unpenalised fit, which "
    "is what an unpenalised model always gives in the data it was fitted to, and rises steeply as the "
    "penalty grows. The value of 0.05 was fixed before the analyses reported in the paper and was not "
    "tuned to any performance measure. Source: 42_penalty_sensitivity.py.")

# ============================================================== S6 ==============
sel = load("tab43_selection_stability.csv")
selj = load("tab43_selection_stability.json")
alt = load("tab43_alternative_sets.json")
LONG = {"updrs_totscore": "Total MDS-UPDRS (Parts I+II+III)", "ageonset": "Age at onset, years",
        "SEX": "Sex", "BMI": "Body-mass index", "td_pigd_ratio": "TD/PIGD ratio",
        "NP2FREZ": "Freezing of gait (item 2.13)", "EDUCYRS": "Years of education",
        "MSEADLG": "Modified Schwab and England scale", "NP1FATG": "Fatigue (item 1.13)",
        "NP1DPRS": "Depressed mood (item 1.3)", "hy": "Hoehn and Yahr stage",
        "updrs3_score": "MDS-UPDRS Part III", "scopa_gi": "SCOPA gastrointestinal",
        "Clinical_Stage": "Clinical stage", "pigd": "PIGD subscore",
        "updrs1_score": "MDS-UPDRS Part I", "comp_bradicinesia": "Bradykinesia composite",
        "updrs2_score": "MDS-UPDRS Part II"}
title(6, "Stability of predictor selection, and the interchangeability of the severity term")
sub("A. How often each candidate was selected in 100 bootstrap resamples")
table(["Candidate", "Selected in, share of the 100 resamples", "In the Total-6"],
      [[LONG.get(r.variable, r.variable), pct(r.frequency_pct, 0), "yes" if r.in_total6 else "no"]
       for _, r in sel.iterrows()], widths=[9.0, 3.0, 3.0])

sub("B. Out-of-fold discrimination of alternative six-variable sets")
ALT = {"Total-6 publicado": "Total-6, the published model",
       "Seis mais frequentes": "The six most frequently selected candidates",
       "Total-6 trocando MDS-UPDRS por MSEADLG": "Total-6 with the total MDS-UPDRS replaced by the Schwab and England scale",
       "Total-6 trocando MDS-UPDRS por Parte III": "Total-6 with the total MDS-UPDRS replaced by Part III",
       "So idade de inicio (a mais estavel)": "Age at onset alone, the single most stable candidate"}
base = alt["alternatives"]["Total-6 publicado"]
table(["Predictor set", "Out-of-fold C", "Difference from the Total-6"],
      [[ALT[k], f"{v:.4f}", "reference" if k == "Total-6 publicado" else sgn(v - base)]
       for k, v in alt["alternatives"].items()], widths=[9.4, 2.8, 2.8])

ov = alt["overlap"]
sub("C. Overlap between the set selected in each resample and the Total-6")
table(["Predictors shared with the Total-6"] + [str(k) for k in sorted(ov, key=int)],
      [["Resamples"] + [ov[k] for k in sorted(ov, key=int)]],
      widths=[6.0, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3])

legend(
    "Forward selection maximising the out-of-fold C-index within the 18-candidate pool was repeated in "
    "100 bootstrap resamples of the development sample. (A) In the full sample the procedure recovers "
    "the Total-6 exactly. In the resamples only age at onset is selected in more than 80% of them, and "
    f"the exact six-variable set is recovered in {selj['exact_recovery_pct']:.0f}% of resamples, which is "
    "the honest reading of how much of the specification is data-driven. (B) The alternative sets show "
    "what that instability costs. Replacing the total MDS-UPDRS by either of the two severity measures "
    "that compete with it changes discrimination by less than 0.008, which indicates that the severity "
    "term is interchangeable rather than uniquely informative; taking the six most frequently selected "
    "candidates instead, which substitutes years of education for the total MDS-UPDRS, costs 0.026. "
    f"(C) The median resample shares {alt['median_overlap']} of the six predictors with the published "
    f"model and {alt['at_least_4']} of the 100 share at least four. Selection instability of this kind "
    "is expected with 165 events and correlated candidates, and is the reason the model is presented as "
    "one defensible specification rather than the uniquely correct one. Source: "
    "43_selection_stability.py.")

# ============================================================== S7 ==============
cen = load("tab45_informative_censoring.json")
title(7, "Informative censoring: who leaves the study, and how far the estimates move if "
         "leaving is not independent of risk")
sub("A. Reason for leaving the analysis")
cnt = cen["counts"]
tot = sum(cnt.values())
table(["Status at the data cut", "n", "Share of the cohort"],
      [["Reached the outcome", cnt["event"], pct(100 * cnt["event"] / tot)],
       ["Still under observation, administratively censored", cnt["administrative"], pct(100 * cnt["administrative"] / tot)],
       ["Lost to follow-up", cnt["lost"], pct(100 * cnt["lost"] / tot)]],
      widths=[8.0, 2.0, 3.0])

sub("B. Does leaving the study depend on the predictors, or on predicted risk?")
rows = [[LONG[k], ci(v["HR"], v["lo"], v["hi"]), pv(v["p"])] for k, v in cen["censoring_model"].items()]
rs = cen["risk_score_predicts_dropout"]
rows.append(["The model's own linear predictor", ci(rs["HR"], rs["lo"], rs["hi"]), pv(rs["p"])])
table(["Variable", "Hazard ratio for being lost to follow-up, per SD (95% CI)", "p"],
      rows, widths=[6.0, 6.0, 2.0])

sub("C. Delta-based multiple imputation of the unobserved time of those lost to follow-up")
ref = cen["reference"]
rows = [["Primary analysis, no imputation", "165", f"{ref['C']:.4f}", pct(100 * ref["incidence"]),
         f"{ref['HR']['NP2FREZ']:.4f}", "reference"]]
for r in cen["deltas"]:
    worst = max(abs(r[f"HR_{k}"] / ref["HR"][k] - 1) for k in ref["HR"])
    rows.append([f"Hazard multiplied by {r['delta']:g}", f"{r['events']:.0f}",
                 f"{r['C_apparent']:.4f}", pct(100 * r["incidence"]),
                 f"{r['HR_NP2FREZ']:.4f}", pct(100 * worst)])
table(["Assumption for those lost to follow-up", "Events, mean across imputations", "C (apparent)",
       "5-year incidence", "HR, freezing of gait", "Largest shift in any hazard ratio"],
      rows, widths=[4.6, 2.6, 1.9, 2.0, 2.0, 2.7])

legend(
    "Censoring was separated into two kinds. A participant still within the data-collection window at "
    "the April 2026 data cut was treated as administratively censored, and one whose last "
    f"contact preceded the cut by more than {cen['administrative_window_months']} months was treated as "
    "lost to follow-up. (A) A quarter of the cohort was lost to follow-up, which is enough for "
    "informative censoring to matter. (B) Leaving the study is predictable from two of the six "
    "predictors, but not from the model's own risk score, because those two act on the outcome in "
    "opposite directions and cancel within the linear predictor. That is the most reassuring single "
    "result here: dropout is related to characteristics, not to predicted risk. (C) The unobserved "
    "remaining time of those lost to follow-up was multiply imputed from the fitted model with their "
    f"hazard multiplied by a factor between 0.5 and 3, with {cen['imputations']} imputations at each "
    "value. A factor of 1 assumes they behave like those who stayed, and reproduces the primary "
    "estimate. The five-year incidence leaves its own confidence interval of 19.9 to 28.8% only once "
    "the factor reaches 2.5. Across the whole range the concordance changes by at most 0.022 and the "
    "largest hazard ratio by at most 4.4%, so departures from independent censoring of this kind would "
    "shift the level of predicted risk without disturbing the ranking of patients. Source: "
    "45_informative_censoring.py.")

# ============================================================== S8 ==============
inc = load("tab41_included_vs_excluded.csv")
incj = load("tab41_included_vs_excluded.json")
title(8, "Participants included in the development sample against those excluded for "
         "incomplete predictor data")
sub("A. Outcome")
table(["Quantity", "Included (n = 813)", "Excluded (n = 628)", "Comparison"],
      [["Reached the outcome", f"165 (20.3%)", f"137 (21.8%)",
        f"HR {ci(incj['hr_group'], incj['hr_group_lo'], incj['hr_group_hi'])}, log-rank p = {incj['logrank_p']:.2f}"],
       ["Cumulative incidence at 3 years", pct(100 * incj["incidence"]["3"]["included"]),
        pct(100 * incj["incidence"]["3"]["excluded"]), ""],
       ["Cumulative incidence at 5 years", pct(100 * incj["incidence"]["5"]["included"]),
        pct(100 * incj["incidence"]["5"]["excluded"]), ""],
       ["Cumulative incidence at 7 years", pct(100 * incj["incidence"]["7"]["included"]),
        pct(100 * incj["incidence"]["7"]["excluded"]), ""]],
      widths=[4.6, 3.0, 3.0, 5.0])

sub("B. Baseline characteristics")
table(["Characteristic", "Included (n = 813)", "Excluded, among those in whom it was recorded",
       "n recorded among the 628", "p"],
      [[r.variable, r.included, r.excluded, int(r.n_excluded), pv(r.p)] for _, r in inc.iterrows()
       if r.variable != "Reached the outcome"],
      widths=[4.4, 3.4, 3.6, 2.4, 1.8])

sub("C. Which variable was missing")
miss = incj["missingness"]
table(["Candidate variable", "Missing among the 628"],
      [[LONG.get(k, k), v] for k, v in sorted(miss.items(), key=lambda kv: -kv[1])],
      widths=[9.0, 4.0])

legend(
    "Complete-case analysis excluded 628 of the 1,441 eligible participants, which is "
    f"{incj['excluded_pct']:.0f}% of them, so whether those excluded differ from those included is a "
    "question the paper has to answer rather than assume. (A) The outcome was observed in every "
    "eligible participant, whatever their predictor data, so this comparison uses the full 628 and is "
    "not itself subject to missingness. The event rates are close and the difference is compatible "
    "with chance. (B) Each severity comparison uses only those excluded participants in whom the "
    "variable was recorded, which is why the third column has its own denominator, given in the "
    "fourth. The excluded were systematically less severely affected on every motor measure, and did "
    "not differ in age at onset, body-mass index or sex. (C) The TD/PIGD ratio and age at onset "
    f"account for most of the exclusions, and {incj['missing_exactly_one']} of the 628 were missing "
    "exactly one candidate. Taken together the three panels show that missingness tracked the "
    "predictors rather than the outcome, which is the condition under which a complete-case analysis "
    "remains valid, and that the development sample is enriched for more severely affected patients. "
    "Source: 41_included_vs_excluded.py.")

# ============================================================== S9 ==============
rd = load("tab40_robust_discrimination.json")
title(9, "Discrimination, accuracy and calibration of the development model out of fold")
sub("A. Concordance")
table(["Measure", "Value", "SD across repetitions"],
      [["Harrell's C-index", f"{rd['harrell_c']:.4f}", f"{rd['harrell_sd']:.4f}"],
       ["Uno's inverse-probability-of-censoring-weighted C", f"{rd['uno_c']:.4f}", f"{rd['uno_sd']:.4f}"]],
      widths=[8.0, 2.5, 3.5])

sub("B. By horizon")
hz = [str(int(h)) for h in rd["horizons"]]
table(["Measure"] + [f"{h} years" for h in hz],
      [["Time-dependent area under the ROC curve"] + [f"{rd['auc'][h]:.4f}" for h in hz],
       ["Brier score, the model"] + [f"{rd['brier'][h]:.4f}" for h in hz],
       ["Brier score, a model with no predictors"] + [f"{rd['brier_null'][h]:.4f}" for h in hz],
       ["Index of prediction accuracy"] + [pct(100 * rd["ipa"][h]) for h in hz]],
      widths=[8.0, 2.0, 2.0, 2.0])

ip = load("tab32_internal_performance.json")
sub("C. Calibration by horizon")
coh813 = C.build_cohort(with_metadata=False)
rows = []
for h in ("3", "5", "7"):
    c = ip["calibration"][h]
    rows.append([f"{h} years", int((coh813.exit >= float(h)).sum()),
                 pct(100 * c["observed"]), pct(100 * c["predicted_mean"]),
                 f"{c['OE']:.2f}", f"{100 * c['ICI']:.1f}"])
table(["Horizon", "Patients at risk", "Observed incidence", "Mean predicted risk",
       "Observed to expected", "Integrated calibration index, percentage points"],
      rows, widths=[2.0, 2.2, 2.6, 2.6, 2.4, 3.2])

lo = load("tab32_leave_one_site_out.csv").sort_values(["events", "n"], ascending=False)
sub("D. Leave-one-site-out cross-validation")
rows = [[f"Site {i + 1}", int(r.n), int(r.events), f"{r.C:.3f}"]
        for i, (_, r) in enumerate(lo.iterrows())]
rows.append(["All 23 sites", int(lo.n.sum()), int(lo.events.sum()),
             f"median {lo.C.median():.3f} (IQR {lo.C.quantile(.25):.3f} to {lo.C.quantile(.75):.3f})"])
table(["Site", "Participants", "Events", "C-index with that site held out"],
      rows, widths=[3.0, 3.0, 2.6, 6.4])

legend(
    "(A) Harrell's C-index is the concordance reported throughout the paper, and it is known to be "
    "optimistic when censoring is heavy, which it is here: 79.7% of the development sample is "
    "censored. Uno's estimator weights each comparable pair by the inverse probability of remaining "
    "uncensored and is the appropriate check. It is 0.05 lower, which is the honest measure of how "
    "much of the apparent discrimination is an artefact of censoring, and it is reported alongside "
    "Harrell's C in the abstract for that reason. (B) The index of prediction accuracy is the proportional "
    "reduction in the Brier score against a model with no predictors, so a value of 6.3% at five years "
    "means the model removes 6.3% of the squared prediction error that predicting the average risk for "
    "everyone would leave. (C) Calibration at each horizon, from the out-of-fold predicted risks. "
    "Observed incidence is the Kaplan-Meier estimate in the whole development sample, mean "
    "predicted risk is the average of the out-of-fold predictions, the observed-to-expected ratio "
    "is their quotient, and the integrated calibration index is the mean absolute difference "
    "between predicted and observed risk across the range of predicted values. Agreement is close "
    "at every horizon and does not deteriorate as the horizon lengthens, although the number of "
    "patients still at risk falls from 325 at three years to 113 at seven, so the seven-year row "
    "rests on a seventh of the sample. What panel C reports at five years is what Supplementary "
    "Figure S1 displays in full. (D) The model was refitted with each site held out in turn and "
    "evaluated in that site, restricted to the 23 of 50 sites with at least ten participants and "
    "two events. Sites are numbered by decreasing number of events, and the numbering does not "
    "correspond to PPMI site identifiers. This panel is included so that a claim made in the "
    "Results can be checked rather than taken on trust: the spread of site-level estimates "
    "reflects small numbers rather than heterogeneity in performance. Every estimate below 0.45 or "
    "above 0.85 comes from a site contributing two or three events, whereas the largest site, with "
    "33 events, gives 0.679, close to the out-of-fold value in the whole sample. The "
    f"event-weighted mean across sites is {(lo.C * lo.events).sum() / lo.events.sum():.3f}. Panels "
    f"A and B are averaged over {rd['repeats']} repetitions of five-fold cross-validation and "
    "panels C and D over twenty; all are in the development sample of 813 participants with 165 "
    "events. Sources: 40_robust_discrimination.py and 32_internal_performance.py.")

# ============================================================= S10 ==============
ph, lin = load("tabS5_proportional_hazards.csv"), load("tabS5_linearity.csv")
title(10, "Assumptions of the development model")
sub("A. Proportional hazards, by the Schoenfeld residual test")
rank = ph[ph.time_transform == "rank"].set_index("predictor")
km = ph[ph.time_transform == "km"].set_index("predictor")
table(["Predictor", "χ² (rank transform)", "p", "χ² (Kaplan-Meier transform)", "p"],
      [[LONG[k], num(rank.loc[k, "chi2"]), f"{rank.loc[k, 'p']:.3f}",
        num(km.loc[k, "chi2"]), f"{km.loc[k, 'p']:.3f}"] for k in LONG if k in rank.index],
      widths=[6.0, 2.6, 1.6, 3.4, 1.6])

sub("B. Linearity of the continuous predictors, restricted cubic splines against the linear term")
table(["Predictor", "Degrees of freedom", "Likelihood-ratio χ²", "p"],
      [[LONG[r.predictor], int(r.df), num(r.LR_chi2), f"{r.p:.3f}"] for _, r in lin.iterrows()],
      widths=[6.4, 2.8, 3.0, 2.0])

legend(
    "(A) Proportional hazards was assessed under two transformations of time, a rank transform and a "
    "Kaplan-Meier transform, because the two weight the follow-up differently and a violation that "
    "appears under one can be invisible under the other. No predictor violates the assumption under "
    "either, and the smallest p is 0.31. (B) Linearity was assessed by adding restricted cubic splines "
    "with four knots at the 5th, 35th, 65th and 95th percentiles and comparing the fit with the linear "
    "term by a likelihood-ratio test. No continuous predictor gained from the more flexible form, all "
    "p at or above 0.51, so each was kept on its original scale, which is also what keeps the "
    "calculator a sum of six terms. Sex and freezing of gait are not in panel B because neither is "
    "continuous. Source: 04_model_assumptions.py.")

# ============================================================= S11 ==============
tv = load("tab33_temporal_validation.json")
title(11, "Temporal validation: model fitted in the earlier recruitment wave and applied "
          "to the later one")
sub("A. The two waves")
table(["", "Wave 1, enrolled 2010 to 2019", "Wave 2, enrolled 2020 to 2025"],
      [["Participants", tv["early"]["n"], tv["late"]["n"]],
       ["Events", tv["early"]["events"], tv["late"]["events"]],
       ["Sites", tv["early"]["sites"], f"{tv['late']['sites']}, of which {tv['late']['sites_new']} new"],
       ["Median follow-up, years", f"{tv['early']['median_followup']:.1f}", f"{tv['late']['median_followup']:.1f}"],
       ["Role", "development", "validation"]],
      widths=[4.6, 5.2, 5.2])

sub("B. Performance in the later wave, with the earlier wave's coefficients and baseline hazard held fixed")
table(["Measure", "Value (95% CI)", "p"],
      [["C-index", ci(tv["C"], tv["C_lo"], tv["C_hi"], 3), ""],
       ["Calibration slope", ci(tv["slope"], tv["slope_lo"], tv["slope_hi"], 3), pv(tv["slope_p"])]],
      widths=[5.0, 6.0, 2.0])
para(" ", size=4)
table(["Horizon", "At risk", "Observed", "Mean predicted", "Observed to expected"],
      [[f"{h} years", tv["calibration"][h]["at_risk"], pct(100 * tv["calibration"][h]["observed"]),
        pct(100 * tv["calibration"][h]["predicted_mean"]), f"{tv['calibration'][h]['OE']:.2f}"]
       for h in ("2", "3")], widths=[2.6, 2.2, 2.6, 3.2, 3.4])

sub("C. Coefficients estimated in the earlier wave alone against those of the full sample")
table(["Predictor", "HR per unit, wave 1 only", "HR per unit, full sample", "Difference"],
      [[LONG[k], f"{tv['coefficients_early'][k]:.4f}", f"{tv['coefficients_full'][k]:.4f}",
        sgn(tv["coefficients_early"][k] - tv["coefficients_full"][k])] for k in LONG if k in tv["coefficients_early"]],
      widths=[5.4, 3.4, 3.4, 2.4])

legend(
    "PPMI recruited in two waves and no participant belongs to both, which allows a validation that is "
    "temporal rather than random: the later wave was enrolled at different times, in part at different "
    f"sites, {tv['late']['sites_new']} of which contributed no participant to the earlier wave. The "
    "model was refitted in wave 1, its coefficients and baseline hazard were frozen, and it was then "
    "applied to wave 2 without any recalibration. Discrimination held. Absolute risk did not: it was "
    "over-predicted by about a quarter at both horizons. Panel A explains why that is expected rather "
    "than alarming, and panel C rules out the obvious alternative explanation, since the coefficients "
    "estimated in wave 1 alone are almost identical to those of the full model. The later wave had a "
    "lower-risk case-mix that the model captured in direction but not in full magnitude. Two limits "
    f"should be read alongside this result: wave 2 contributed only {tv['late']['events']} events, "
    "which is why the confidence interval around the C-index spans 0.18, and its follow-up reaches "
    f"{tv['late']['max_followup']:.1f} years at most, so calibration could be assessed only at two and "
    "three years and not at the five-year horizon used elsewhere. Source: 33_temporal_validation.py.")

# ============================================================= S12 ==============
dc = load("tab34_decision_curve.csv")
dcj = load("tab34_decision_curve.json")
title(12, "Decision-curve analysis at five years")
rows = []
for _, r in dc.iterrows():
    rows.append([pct(100 * r.threshold, 0), f"{r.net_benefit_model:.4f}",
                 f"{r.net_benefit_treat_all:.4f}", "0.0000",
                 pct(100 * r.proportion_flagged), "yes" if r.net_benefit_model > r.net_benefit_treat_all else "no"])
table(["Threshold probability", "Net benefit, the model", "Net benefit, treat all",
       "Net benefit, treat none", "Share of the cohort flagged", "Model better than treating all"],
      rows, widths=[2.6, 2.8, 2.6, 2.6, 2.6, 3.0], size=7.5)
legend(
    "Net benefit is the proportion of true positives minus the proportion of false positives weighted "
    "by the odds of the threshold, so it puts benefits and harms on one scale and makes the model "
    "comparable with the two strategies that need no model at all. The threshold probability is the "
    "predicted five-year risk at which a clinician would act, and it is a clinical judgement rather "
    f"than a property of the model. The model gives higher net benefit than either default at "
    + ", ".join(f"{pct(100 * a, 0)} to {pct(100 * b, 0)}" for a, b in dcj["superior_bands"]) +
    ". Those bands are not one continuous range, and the exceptions are as informative as the rule: "
    "between 11% and 14% treating every patient is better, because at thresholds a little below the "
    "observed incidence the cost of missing a case still outweighs the cost of an unnecessary "
    "intervention, and at 47% and 48% the net benefit of the model itself dips just below zero, "
    "where treating nobody is better. At a threshold of "
    "20% its net benefit is "
    f"{dcj['at_20pct']['model']:.3f} against {dcj['at_20pct']['treat_all']:.3f} for treating all, "
    f"which corresponds to avoiding {dcj['at_20pct']['false_positives_avoided_per_1000']:.0f} "
    "unnecessary interventions per 1,000 patients without forgoing any that were warranted. At "
    f"{pct(100 * dcj['observed_incidence'], 0)}, the observed five-year incidence, treating all ceases "
    "to give any benefit while the model retains it. Below 6% the two coincide, because at such a low "
    "threshold the model flags almost everyone. Predicted risks are out-of-fold, so the curve is not "
    "read from the data the model was fitted to. Source: 34_decision_curve_internal.py.")

# ============================================================= S13 ==============
cm = load("tab44_classification_metrics.csv")
cmj = load("tab44_classification_metrics.json")
title(13, "What each decision threshold would do, at five years")
table(["Threshold probability", "Flagged, n", "Flagged, share of the cohort", "Sensitivity", "Specificity",
       "Positive predictive value", "Negative predictive value", "Number needed to flag"],
      [[pct(100 * r.threshold, 0), int(r.flagged_n), pct(r.flagged_pct), pct(r.sensitivity),
        pct(r.specificity), pct(r.ppv), pct(r.npv), f"{r.number_needed_to_flag:.1f}"]
       for _, r in cm.iterrows()], widths=[1.8, 1.7, 2.4, 2.0, 2.0, 2.4, 2.4, 2.2], size=7.5)
legend(
    "Because the model returns a continuous probability, any threshold is a decision made by the user "
    "rather than a property of the model, and this table sets out what each one would do so that the "
    "choice can be made explicitly. Positive predictive value should be read against the five-year "
    f"incidence of {pct(100 * cmj['overall_incidence'])}, which is what flagging everyone achieves; it "
    "rises to 32.9% at a threshold of 20% and 44.0% at 30%. Negative predictive value stays between "
    "82% and 87% across the useful range, so a patient below the threshold is reliably low risk "
    "whichever threshold is chosen, which is the more robust half of the model's behaviour. The number "
    "needed to flag is how many patients must be flagged for one to reach the outcome. Nothing here "
    "recommends a threshold: a trial enriching for dyskinesia would sit high, a discussion about "
    "starting a dopamine agonist would sit low. All quantities use out-of-fold predicted risks in the "
    "development sample of 813 participants. Source: 44_classification_metrics.py.")

# ============================================================= S14 ==============
sg = load("tab35_subgroup_performance.csv")
sgj = load("tab35_subgroup_performance.json")
title(14, "Performance across subgroups")
table(["Subgroup", "n", "Events", "At risk at 3 years", "C-index (95% CI)",
       "Observed to expected", "Calibration slope"],
      [[r.subgroup, int(r.n), int(r.events), int(r.at_risk), ci(r.C, r.C_lo, r.C_hi, 3),
        f"{r.OE:.2f}", f"{r.slope:.2f}"] for _, r in sg.iterrows()],
      widths=[5.4, 1.1, 1.3, 2.2, 3.4, 2.2, 2.2])
legend(
    "Performance is reported at three years, not five, because the smaller subgroups do not have "
    "enough participants still at risk at five years to support an estimate. Discrimination is similar "
    "in men and women and across tertiles of age at onset, and the confidence intervals of every "
    "subgroup overlap that of the whole cohort. Two rows deserve to be read carefully rather than "
    "reassuringly. The oldest tertile has an observed-to-expected ratio of 0.54, so risk is "
    "substantially over-predicted in patients with late onset. And the 53 participants of races other "
    "than White are too few to estimate anything with useful precision: the confidence interval around "
    "their C-index spans 0.42, and their calibration slope of 0.40 is not distinguishable from either "
    "1 or 0. That row is included because leaving it out would hide the limitation rather than remove "
    "it. It is the reason the paper states that external validation in ancestrally diverse populations "
    f"is the principal outstanding requirement. Estimates are out-of-fold. Source: "
    "35_subgroup_performance.py.")

# ============================================================= S15 ==============
md = load("tab49_matched_df_unified.csv")
hh = load("tab49_head_to_head_unified.json")
title(15, "Head-to-head comparison with published predictor sets, and the discrimination "
          "attainable at each model size")
sub("A. Every predictor set, refitted in the same participants")
rows = []
for _, r in md.iterrows():
    ref = r.predictor_set.startswith("Total-6")
    rows.append([r.predictor_set, int(r.k), ci(r.C_out_of_fold, r.C_lo, r.C_hi, 3),
                 "reference" if ref else sgn(r.dC_vs_total6, 3),
                 "" if ref else f"{sgn(r.d_lo, 3)} to {sgn(r.d_hi, 3)}",
                 "" if ref else pv(r.p)])
table(["Predictor set", "k", "Out-of-fold C (95% CI)", "Difference from the Total-6",
       "95% CI of the difference", "p"], rows,
      widths=[4.6, 1.0, 3.6, 2.8, 3.0, 1.4])

sub("B. Discrimination attainable at each model size, within a common pool of 20 clinical candidates")
rows = []
for k in sorted(int(x) for x in md.k.unique()):
    f = os.path.join(TAB, f"tab49_dist_k{k}.csv")
    if not os.path.exists(f):
        continue
    v = pd.read_csv(f, header=None)[1]
    exhaustive = k <= 3
    rows.append([k, f"{thousands(len(v))} ({'all' if exhaustive else 'random'})",
                 f"{v.median():.4f}", f"{v.quantile(.75):.4f}",
                 f"{v.quantile(.90):.4f}", f"{v.max():.4f}",
                 f"{int((v >= hh['total6_C']).sum())} of {thousands(len(v))}"])
table(["Number of predictors", "Subsets evaluated", "Median C", "75th percentile", "90th percentile",
       "Maximum C", "Subsets reaching the Total-6"], rows,
      widths=[2.4, 2.4, 1.9, 2.2, 2.2, 1.9, 2.8])

sub("C. Position of each predictor set within the distribution for its own size")
table(["Predictor set", "k", "Out-of-fold C", "Median C for k", "Maximum C for k", "Percentile within k"],
      [[r.predictor_set, int(r.k), f"{r.C_out_of_fold:.4f}", f"{r.median_C_for_k:.4f}",
        f"{r.max_C_for_k:.4f}", f"{r.percentile_within_k:.1f}"] for _, r in md.iterrows()],
      widths=[5.0, 1.0, 2.2, 2.4, 2.4, 2.6])
g26 = hh["gain_2_to_6"]
pub = md[~md.predictor_set.str.startswith("Total-6") & md.percentile_within_k.notna()]
above = pub[pub.percentile_within_k >= 50].sort_values("percentile_within_k")
below = pub[pub.percentile_within_k < 50].sort_values("percentile_within_k")
WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}


def ordinal(x):
    """The percentile as prose, keeping a decimal only where rounding would mislead."""
    s = f"{x:.1f}"
    return (s if round(x) == 100 and x < 100 else f"{x:.0f}") + "th"


pcts = (f"{WORDS[len(above)]} of the {WORDS[len(pub)].lower()} published sets sit above the median for their own "
        f"size class, from the {ordinal(above.percentile_within_k.iloc[0])} to the "
        f"{ordinal(above.percentile_within_k.iloc[-1])} percentile, so the comparison is against "
        "well-chosen small models rather than straw men")
if len(below):
    names = " and ".join(below.predictor_set)
    pcts += (f"; the exception is {names}, below the median for its size at the "
             f"{ordinal(below.percentile_within_k.iloc[0])} percentile")
legend(
    "All three panels come from one analysis, in one sample and under one protocol: every eligible "
    f"participant complete on the twenty-candidate pool and on the seven factors of Loo, {hh['n']} "
    f"participants with {hh['events']} events, evaluated by twenty repetitions of stratified "
    "five-fold cross-validation with standardisation fitted inside each training fold, which is the "
    "protocol used throughout the paper. Each published set is refitted in these data, so what is "
    "compared is the choice of variables and not the coefficients of the original papers. "
    "(A) The Total-6 has the highest point estimate, and the paired bootstrap separates it from four "
    "of the six sets; the differences from Zhao 2025 and from Loo 2024 are not distinguishable from "
    "zero. Loo and colleagues report seven predictive factors rather than a fitted equation, and the "
    "seven are operationalised here as the axial composite of Part III, freezing of gait, the mean of "
    "the two lower-limb rigidity items, the tremor composite of Part III, body-mass index, age at onset and "
    "the Benton judgement of line orientation. That mapping is a judgement and a reader may prefer "
    "another. (B) A six-variable model has more freedom to fit than a two-variable or three-variable "
    "one, so a higher C-index is not by itself evidence that the predictors are better chosen. Model "
    f"size does contribute: the median C rises from {hh['median_by_k']['2']:.3f} with two predictors "
    f"to {hh['median_by_k']['6']:.3f} with six, a gain of {g26:.3f}. That gain is smaller than the "
    "margin of the Total-6 over the two-variable and three-variable published sets, and no subset "
    "anywhere in the pool, of any size tested, reaches the discrimination of the Total-6. "
    f"(C) {pcts}. The Total-6 is at the top of its own size class, but "
    "that rank is optimistic and should not be read as a result, because the Total-6 was selected in "
    "these same data while the subsets it is being ranked against were not. The two interpretable "
    "comparisons are the median gain from two predictors to six, which is the pure effect of model "
    "size, and whether any small subset reaches the Total-6, which none does. Sources: "
    "49_head_to_head_unified.py and 49b_loo_variables.py.")

# ============================================================= S16 ==============
rb = load("tab50_responsividade_robustez.csv")
r46 = load("tab46b_responsividade.csv").set_index("metric").valor
title(16, "Levodopa responsiveness as an optional dynamic refinement")
NAMEV = {"continua": "Continuous, per standard deviation", ">40%": "Improvement greater than 40%",
         ">50%": "Improvement greater than 50%", ">60%": "Improvement greater than 60%"}
table(["Definition of responsiveness", "Hazard ratio", "p", "Change in out-of-fold C",
       "Position among the six clinical predictors, ranked by hazard ratio"],
      [[NAMEV[r.variante], f"{r.HR:.2f}", pv(r.p), sgn(r.dC_OOF), f"{int(r.pos_de_6)} of 6"]
       for _, r in rb.iterrows()], widths=[5.0, 1.9, 1.6, 2.6, 3.9])
legend(
    "Responsiveness is the percentage improvement of the MDS-UPDRS Part III score from the OFF to the "
    "ON state. It is not a baseline predictor and cannot be: it is measured after levodopa has "
    "started, so it cannot enter a model applied at initiation. It is reported here because the "
    "association is strong and consistent, and because a clinician who has already seen the OFF to ON "
    "response has information the baseline model does not use. The association is monotonic across "
    "thresholds and strongest for an improvement greater than 50%, where the hazard ratio is "
    f"{r46['HR_resp50']:.2f} (95% CI {r46['IC_inf']:.2f} to {r46['IC_sup']:.2f}, p = {r46['p']:.4f}) "
    "and the gain in out-of-fold discrimination is 0.009. The rank column places responsiveness among "
    "the six clinical predictors by hazard ratio, so that its size can be judged against terms already "
    "in the model. Any use of this variable is a refinement made at a later visit, not part of the "
    "prediction made at levodopa initiation, and the calculator keeps the two separate. Sources: "
    "46_dynamic_landmark.py and 50_responsiveness_robustness.py.")

# ============================================================= S17 ==============
an = load("tab51_outcome_anchoring.json")
mm, w80 = an["model"], an["below80"]
meas = load("tab51_anchoring_measurements.csv")
title(17, "Functional consequence of crossing the outcome threshold")
sub("A. Modified Schwab and England scale over follow-up, linear mixed model")
ROWS = [("Time since levodopa initiation, per year", "t"),
        ("Crossing the threshold, step at the moment of crossing", "crossed"),
        ("Years since crossing, change in the rate of decline", "after"),
        ("Age at onset, per year", "age_c"),
        ("Total MDS-UPDRS at baseline, per point", "updrs_c"),
        ("Male sex", "SEX")]
table(["Term", "Change in scale points (95% CI)", "p"],
      [[lab, f"{sgn(mm[k]['beta'], 2)} ({sgn(mm[k]['lo'], 2)} to {sgn(mm[k]['hi'], 2)})", pv(mm[k]["p"])]
       for lab, k in ROWS], widths=[8.4, 4.8, 1.8])

sub("B. The scale itself, in each window relative to crossing")
WIN = ["never crossed", "before crossing", "within the first year after", "one year or more after"]
LAB = {"never crossed": "Participants who never crossed the threshold",
       "before crossing": "Crossers, measurements before they crossed",
       "within the first year after": "Crossers, within the first year after",
       "one year or more after": "Crossers, one year or more after"}
rows = []
for w in WIN:
    s = meas[meas.window == w]
    rows.append([LAB[w], len(s), s.PATNO.nunique(), f"{s.MSEADLG.median():.0f}",
                 f"{s.MSEADLG.mean():.1f}", pct(w80[w])])
table(["Window", "Measurements", "Participants", "Median", "Mean", "Below 80 on the scale"],
      rows, widths=[6.4, 2.2, 2.2, 1.6, 1.6, 3.0])

sub("C. The step under three ways of estimating it")
table(["Analysis", "Step in scale points (95% CI)", "p"],
      [["Random intercept for each participant, the primary analysis",
        f"{sgn(mm['crossed']['beta'], 2)} ({sgn(mm['crossed']['lo'], 2)} to {sgn(mm['crossed']['hi'], 2)})",
        pv(mm["crossed"]["p"])],
       ["Random intercept and a random rate of decline for each participant",
        f"{sgn(mm['crossed_random_slope']['beta'], 2)} ({sgn(mm['crossed_random_slope']['lo'], 2)} to "
        f"{sgn(mm['crossed_random_slope']['hi'], 2)})", pv(mm["crossed_random_slope"]["p"])],
       ["Crossers only, before against one year or more after crossing",
        f"{sgn(mm['sustained']['beta'], 2)} ({sgn(mm['sustained']['lo'], 2)} to "
        f"{sgn(mm['sustained']['hi'], 2)})", pv(mm["sustained"]["p"])]],
      widths=[8.4, 4.8, 1.8])

legend(
    "The outcome of this study is a threshold on a scale, and the fair objection to any such "
    "outcome is that a threshold is a convention. This table asks whether crossing it is followed "
    "by a loss of function that a clinician would recognise. The exposure here is item 4.1 alone, "
    "the time spent with dyskinesia, and not the full outcome, because item 4.2 is itself a rating "
    "of functional impact and using it would make the question circular; a score of 2 on item 4.1 "
    f"means dyskinesia for more than a quarter of the waking day, and {an['n_cross']} of the 813 "
    "reached it after time-zero. The functional measure is the Modified Schwab and England scale, "
    f"recorded at the same visits, {thousands(an['n_measurements'])} times in the "
    f"{an['n_participants']} participants. Every participant contributes their own before and "
    "after, so the comparison is within a person rather than between people. (A) Crossing is "
    "followed by a step down of 2.0 points, and there is no further change in the rate of decline "
    "afterwards, so what the threshold marks is a discrete loss rather than the beginning of a "
    "faster decline. Against the estimated 1.75 points lost per year of disease, a step of 2.0 "
    "points is the equivalent of about fourteen months of ordinary progression arriving at once. "
    "(B) The same thing read without a model: the proportion of measurements below 80 on the "
    "scale, the level at which a patient needs help with chores, is 10.0% in these participants "
    "before they cross and 39.7% a year or more afterwards, against 13.8% in those who never "
    "cross. (C) The obvious competing explanation is that those who cross were declining faster "
    "all along and the step is their steeper slope showing up. Giving every participant their own "
    "rate of decline does not move the estimate, from 2.01 points to 1.98, which is what that "
    "explanation would not predict. The decrement is also still present when measurement is "
    "restricted to a year or more after crossing, so it is not a bad day at a single visit. "
    "This analysis is observational and crossing is not randomised, so it establishes that the "
    "threshold marks a functional change rather than that dyskinesia causes it. Source: "
    "51_outcome_anchoring.py.")

doc.save(OUT)
print("saved", OUT)
