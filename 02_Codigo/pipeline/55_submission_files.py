# 55_submission_files.py
# Produces: the submission files of the November 2026 revision, in
#           07_Versao_STROBE/Revisao Daniel 2026-11/:
#             Tables.xlsx, Supplementary Tables.xlsx    one sheet per table; Table 3 is new
#             Supplementary Methods.docx                text, with the supplementary figure legends
#             Figure legends.docx                       legends of Figures 1 and 2
#             Figures/                                  each figure as PNG at 300 dpi and SVG
#             STROBE checklist (cohort) - completed.docx  locations moved to the revised text
#
# The co-author asked for tables in Excel and figures as separate files rather than inside
# Word documents, and for fewer sensitivity analyses. This script changes no number: it reads
# the tables, legends and methods from the Word files already delivered and checked, drops the
# analyses that left the paper, renumbers what remains and writes the new formats.
#
# What leaves the supplement: S14 ridge penalty, S15 stability of selection, S16 informative
# censoring and S18 levodopa responsiveness of the October 2026 version, together with the
# Supplementary Methods paragraphs on informative censoring and responsiveness and the two
# sentences on the ridge sensitivity analysis and the bootstrap repetition of the selection.
#
# How what remains is numbered: in the order the revised manuscript first cites it, which
# the shorter text changed. Every cross-reference is remapped in one pass from the October
# numbering, never chained, so a table cannot be renumbered twice:
#
#   October   S1  S2  S3  S4  S5  S6  S7  S8  S9  S10 S11 S12 S13 S17
#   November  S1  S3  S4  S6  S7  S8  S9  S10 S11 S12 S14 S5  S2  S13
#
# Supplementary figures: S1 stays S1; the decision curve, Figure 3 of the October version,
# becomes S2; the head-to-head comparison, S2 in October, becomes S3.
import copy, os, re, shutil, sys
import docx
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
import cohort as C

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(C.RESULTS)
SRC = os.path.join(ROOT, "07_Versao_STROBE", "Revisao Daniel 2026-10")
DST = C.ensure_dir(os.path.join(ROOT, "07_Versao_STROBE", "Revisao Daniel 2026-11"))
FIGS = os.path.join(C.RESULTS, "Figuras")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

REMOVED = {"S14", "S15", "S16", "S18"}
TABLE_MAP = {"S1": "S1", "S2": "S3", "S3": "S4", "S4": "S6", "S5": "S7", "S6": "S8", "S7": "S9",
             "S8": "S10", "S9": "S11", "S10": "S12", "S11": "S14", "S12": "S5", "S13": "S2",
             "S17": "S13"}
FIGURE_MAP = {"S1": "S1", "S2": "S3"}      # and the main Figure 3 becomes Supplementary Figure S2
N_TABLES, N_FIGURES = 14, 3
assert sorted(TABLE_MAP.values(), key=lambda s: int(s[1:])) == ["S%d" % i for i in range(1, N_TABLES + 1)]

REF = re.compile(r"(Supplementary (?:Table|Figure)s?) (S\d+(?:(?:, | and | to |–)S\d+)*)\b"
                 r"|(?<!Supplementary )\bFigure 3\b")


def renum(s):
    """Every reference to a supplementary table or figure, from the October to the November
    numbering, in one pass. Stops if anything still cites a table that left the supplement."""
    def one(m):
        if m.group(1) is None:
            return "Supplementary Figure S2"
        mp = TABLE_MAP if "Table" in m.group(1) else FIGURE_MAP

        def num(x):
            assert x.group(0) in mp, ("cites something no longer in the supplement", m.group(0), s[:90])
            return mp[x.group(0)]
        return m.group(1) + " " + re.sub(r"S\d+", num, m.group(2))
    return REF.sub(one, s)


def check_refs(s, where):
    """After renumbering, every reference must point inside the new supplement."""
    for kind, nums in re.findall(r"Supplementary (Table|Figure)s? (S\d+(?:(?:, | and | to |–)S\d+)*)", s):
        top = N_TABLES if kind == "Table" else N_FIGURES
        for n in re.findall(r"S(\d+)", nums):
            assert 1 <= int(n) <= top, (where, kind, n)
    assert not re.search(r"(?<!Supplementary )\bFigure [3-9]\b", s), (where, "main figure above 2")


def text(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))


def _set_span(ts, i, j, new):
    """Write new over characters i to j of the runs ts, keeping each run's formatting."""
    pos, first = 0, True
    for t in ts:
        s = t.text or ""
        a, b = pos, pos + len(s); pos = b
        if b <= i or a >= j:
            continue
        if first:
            t.text = s[:i - a] + new + (s[j - a:] if j <= b else ""); first = False
        else:
            t.text = s[j - a:] if j < b else ""
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def replace_in(el, old, new):
    """Replace one occurrence of old by new inside a paragraph, across runs, keeping formatting."""
    full = text(el)
    assert full.count(old) == 1, (full.count(old), old[:60])
    i = full.find(old)
    _set_span(list(el.iter(W + "t")), i, i + len(old), new)
    assert text(el) == full.replace(old, new)


def renum_in(el):
    """renum() applied inside a paragraph, across runs, keeping formatting."""
    full = text(el)
    for m in reversed(list(REF.finditer(full))):
        new = renum(m.group(0))
        if new != m.group(0):
            _set_span(list(el.iter(W + "t")), m.start(), m.end(), new)
    assert text(el) == renum(full)


def table_rows(tbl):
    """Rows of a Word table as lists of strings; a merged cell is written once."""
    out = []
    for tr in tbl.findall(W + "tr"):
        row, seen = [], set()
        for tc in tr.findall(W + "tc"):
            span = tc.find(W + "tcPr/" + W + "gridSpan")
            n = int(span.get(W + "val")) if span is not None else 1
            row += [text(tc).strip()] + [""] * (n - 1)
        out.append(row)
    return out


def blocks(document, start_marker):
    """Paragraphs and tables after the paragraph whose text is start_marker, in body order."""
    items, on = [], False
    for el in document.element.body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "p" and text(el).strip() == start_marker:
            on = True; continue
        if on and tag in ("p", "tbl"):
            items.append((tag, el))
    return items


def group(items, title_re):
    """Split blocks into units that start at a title paragraph (Table 1., Supplementary Table S3.)."""
    units, cur, preamble = [], None, []
    for k, (tag, el) in enumerate(items):
        t = text(el).strip() if tag == "p" else None
        m = re.match(title_re, t) if t else None
        if m:
            cur = {"id": m.group(1), "title": t, "parts": []}; units.append(cur); continue
        if cur is None:
            if t:
                preamble.append(t)
            continue
        if tag == "tbl":
            cur["parts"].append(("table", table_rows(el)))
        elif t:
            nxt = items[k + 1][0] if k + 1 < len(items) else None
            cur["parts"].append(("panel" if nxt == "tbl" and len(t) < 220 else "note", t))
    return preamble, units


# Every table defines its own abbreviations in a closing line, as journals ask, so that it can
# be read without the text. The line lists only the abbreviations the table actually uses.
# Gene symbols, Roman numerals, software and PPMI field names are left as they are.
ABBREVIATIONS = [
    (r"C-index", "C-index", "concordance index"),
    (r"CI", "CI", "confidence interval"),
    (r"CSF", "CSF", "cerebrospinal fluid"),
    (r"FAS", "FAS", "letter fluency with F, A and S"),
    (r"FDR", "FDR", "false-discovery rate"),
    (r"HR", "HR", "hazard ratio"),
    (r"HVLT-R", "HVLT-R", "Hopkins Verbal Learning Test, Revised"),
    (r"ICV", "ICV", "intracranial volume"),
    (r"IQR", "IQR", "interquartile range"),
    (r"JLO", "JLO", "judgement of line orientation"),
    (r"LP", "LP", "linear predictor"),
    (r"LRT", "LRT", "likelihood-ratio test"),
    (r"MDS-UPDRS", "MDS-UPDRS", "Movement Disorder Society Unified Parkinson's Disease Rating Scale"),
    (r"MoCA", "MoCA", "Montreal Cognitive Assessment"),
    (r"MRI", "MRI", "magnetic resonance imaging"),
    (r"PD", "PD", "Parkinson's disease"),
    (r"PIGD", "PIGD", "postural-instability gait-difficulty"),
    (r"PPMI", "PPMI", "Parkinson's Progression Markers Initiative"),
    (r"QUIP", "QUIP", "Questionnaire for Impulsive-Compulsive Disorders in Parkinson's Disease"),
    (r"RBD", "RBD", "rapid-eye-movement sleep behaviour disorder"),
    (r"ROC", "ROC", "receiver operating characteristic"),
    (r"SCOPA(?:-AUT)?", "SCOPA-AUT", "Scales for Outcomes in Parkinson's Disease, Autonomic"),
    (r"SD", "SD", "standard deviation"),
    (r"SE", "SE", "standard error"),
    (r"STAI", "STAI", "State-Trait Anxiety Inventory"),
    (r"STRIDE-PD", "STRIDE-PD", "Stalevo Reduction in Dyskinesia Evaluation in Parkinson's Disease trial"),
    (r"SWEDD", "SWEDD", "scans without evidence of dopaminergic deficit"),
    (r"TD/PIGD", "TD/PIGD", "tremor-dominant to postural-instability gait-difficulty ratio"),
    (r"UPSIT", "UPSIT", "University of Pennsylvania Smell Identification Test"),
]


def with_abbreviations(title, parts):
    text = title + "\n" + "\n".join(v if k != "table" else "\n".join("\t".join(r) for r in v) for k, v in parts)
    used = []
    for pattern, short, full in ABBREVIATIONS:
        # PIGD on its own, not only inside TD/PIGD; PD on its own, not inside STRIDE-PD
        if re.search(r"(?<![\w/-])%s(?![\w/])" % pattern, text):
            used.append("%s, %s" % (short, full))
    return parts + ([("note", "Abbreviations: " + "; ".join(used) + ".")] if used else [])


def renum_parts(parts):
    return [(k, renum(v) if k != "table" else [[renum(c) for c in row] for row in v]) for k, v in parts]


def blob(title, parts):
    return "\n".join([title] + [v if k != "table" else "\n".join("\t".join(r) for r in v) for k, v in parts])


BOLD, PANEL = Font(bold=True, size=11), Font(bold=True, italic=True, size=10)
HEAD = PatternFill("solid", fgColor="E8E8E8")
WRAP = Alignment(wrap_text=True, vertical="top")


def write_sheet(wb, name, title, parts, split_title=False):
    ws = wb.create_sheet(name)
    ncol = max([len(r) for kind, rows in parts if kind == "table" for r in rows] + [2])
    notes = []
    if split_title and ". " in title:
        head, rest = title.split(". ", 1)
        head2, rest2 = rest.split(". ", 1) if ". " in rest else (rest, "")
        title, notes = head + ". " + head2 + ".", ([rest2] if rest2 else [])
    r = 1

    def long_row(r, value, font=None):
        ws.cell(row=r, column=1, value=value).alignment = WRAP
        if font:
            ws.cell(row=r, column=1).font = font
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=ncol)
        ws.row_dimensions[r].height = max(15, 15 * (1 + len(value) // (12 * ncol)))

    long_row(r, title, BOLD); r += 2
    widths = [10] * ncol
    for kind, payload in parts:
        if kind == "panel":
            long_row(r, payload, PANEL); r += 1
        elif kind == "table":
            for i, row in enumerate(payload):
                for c, v in enumerate(row, 1):
                    cell = ws.cell(row=r, column=c, value=v)
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                    if i == 0:
                        cell.font = Font(bold=True); cell.fill = HEAD
                    widths[c - 1] = min(48, max(widths[c - 1], len(v) * 0.95 + 2))
                r += 1
            r += 1
        else:
            notes.append(payload)
    for n in notes:
        long_row(r, n); r += 1
    for c, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    return ws


# ------------------------------------------------------------- main tables -------------
tables = docx.Document(os.path.join(SRC, "Tables.docx"))
_, units = group(blocks(tables, "Tables:"), r"^Table (\d+)\.")
assert [u["id"] for u in units] == ["1", "2"], [u["id"] for u in units]
# Table 1 gains the two recruitment waves, at the co-author's request: the temporal
# validation fits the model in one wave and tests it in the other, so the reader needs the
# case-mix of each. The columns come from 04b_table1.py --write; the overall column must
# still equal the delivered table cell by cell, so nothing in it can drift.
T1 = pd.read_csv(os.path.join(C.RESULTS, "Tabelas", "Table1_development_sample.csv"), dtype=str,
                 keep_default_na=False)


def table1_by_wave(u):
    k = [i for i, (kind, _) in enumerate(u["parts"]) if kind == "table"]
    assert len(k) == 1
    rows = u["parts"][k[0]][1]
    cols = list(T1.columns)
    assert len(cols) == 5 and len(rows) - 1 == len(T1) and rows[0][1] == cols[1], (cols, len(rows))
    for r, (_, t) in zip(rows[1:], T1.iterrows()):
        assert r[1] == t[cols[1]], (r[0], r[1], t[cols[1]])
    n = [re.search(r"n = (\d+)", c).group(1) for c in cols[1:4]]
    head = [rows[0][0], cols[1], "Earlier wave, 2010 to 2019, n = %s" % n[1],
            "Later wave, 2020 to 2025, n = %s" % n[2], "Standardised difference"]
    new = [head] + [[r[0], r[1], t[cols[2]], t[cols[3]], t[cols[4]]] for r, (_, t) in zip(rows[1:], T1.iterrows())]
    title = u["title"]
    for old, rep in [
        ("Table 1. Baseline characteristics of the development sample.",
         "Table 1. Baseline characteristics of the development sample, overall and by recruitment wave."),
        # dose availability by wave, as 04b_table1.py prints it
        ("were available for 805 and 808 of the 813 participants respectively.",
         "were available for 805 and 808 of the 813 participants respectively, for 389 and 392 of the "
         "earlier wave and for 416 and 416 of the later."),
        ("PPMI development sample, n = 813, 165 events.",
         "The model was fitted in the earlier recruitment wave and applied unchanged to the later wave for "
         "temporal validation. The standardised difference is the absolute difference between the waves in "
         "means or proportions, divided by the pooled standard deviation; values below 0.1 are usually taken "
         "as negligible. It is not given for the events and the follow-up, which describe what happened "
         "after time-zero rather than who was enrolled. PPMI development sample, n = 813, 165 events; "
         "earlier wave, n = %s, 145 events; later wave, n = %s, 20 events." % (n[1], n[2]))]:
        assert title.count(old) == 1, old[:50]
        title = title.replace(old, rep)
    u["title"] = title
    u["parts"][k[0]] = ("table", new)


# Table 2 gave only the confidence interval of the cumulative incidence, while the text quotes
# the point estimates (12.2%, 24.0% and 37.7%) and cites Table 2 for them. The point estimate
# is added to each cell from the Kaplan-Meier fit of the development sample, and the interval
# recomputed in the same fit must equal the delivered one, so nothing else can move.
def table2_incidence(u):
    from lifelines import KaplanMeierFitter
    d = C.build_cohort(with_metadata=False)
    assert len(d) == 813 and int(d.event.sum()) == 165
    km = KaplanMeierFitter().fit(d.exit, d.event)
    band = km.confidence_interval_survival_function_
    hits = 0
    for i, (kind, rows) in enumerate(u["parts"]):
        if kind != "table":
            continue
        for r in rows:
            if r[0] != "Cumulative incidence, 95% CI":
                continue
            years = [int(re.match(r"(\d+) years", h).group(1)) for h in rows[0][1:] if h]
            for j, h in enumerate(years, 1):
                s = float(km.survival_function_at_times(h).iloc[0])
                lo_s, hi_s = band[band.index <= h].iloc[-1]
                assert r[j] == "%.1f to %.1f%%" % (100 * (1 - hi_s), 100 * (1 - lo_s)), (h, r[j])
                r[j] = "%.1f%% (%s)" % (100 * (1 - s), r[j])
            r[0] = "Cumulative incidence (95% CI)"
            hits += 1
    assert hits == 1


wb = Workbook(); wb.remove(wb.active)
for u in units:
    if u["id"] == "1":
        table1_by_wave(u)
    if u["id"] == "2":
        table2_incidence(u)
    title, parts = renum(u["title"]), renum_parts(u["parts"])
    check_refs(blob(title, parts), "Table " + u["id"])
    write_sheet(wb, "Table " + u["id"], title, with_abbreviations(title, parts), split_title=True)

# Table 3, new in November 2026: the multimodal screen by domain, from
# 57_table3_multimodal_by_domain.py, which reads the file behind Supplementary Table S10. It
# replaces a figure with one row per candidate that filled a page.
t3 = pd.read_csv(os.path.join(C.RESULTS, "Tabelas", "tab57_multimodal_by_domain.csv"))
assert len(t3) == 11 and t3.domain.iloc[-1] == "All candidates" and int(t3.candidates.iloc[-1]) == 89


def dc(v):
    s = ("%+.3f" % v).replace("-", "−")
    return "0.000" if s in ("+0.000", "−0.000") else s


def span(a, b):
    return "%d" % a if a == b else "%d to %d" % (a, b)


rows3 = [["Domain", "Candidates", "Participants", "Events", "p < 0.05", "Lowest FDR-adjusted p",
          "Change in C-index, median (range)"]]
for _, r in t3.iterrows():
    change = dc(r.dC_median) if r.candidates == 1 else "%s (%s to %s)" % (dc(r.dC_median), dc(r.dC_min), dc(r.dC_max))
    rows3.append([r.domain, "%d" % r.candidates, span(r.n_min, r.n_max), span(r.events_min, r.events_max),
                  "%d" % r.nominal_p05, "%.2f" % r.lowest_fdr_p, change])
iv = pd.read_csv(os.path.join(C.RESULTS, "Tabelas", "tab_incremental_value_all_domains_corrected.csv"))
nominal = iv[iv.p < 0.05]
assert len(nominal) == 4 and nominal.label.str.startswith("CSF").all()
assert ("%.3f" % nominal.dC.min(), "%.3f" % nominal.dC.max()) == ("0.004", "0.010")
T3_TITLE = ("Table 3. Incremental value of 89 multimodal candidate predictors over the six clinical "
            "predictors, by domain")
T3_LEGEND = (
    "Each candidate was added, one at a time, to the six-variable clinical model in the participants in whom "
    "it was recorded; participants and events give the range across the candidates of each domain. The "
    "column p < 0.05 counts the candidates whose hazard ratio in that model was nominally significant, and "
    "the false-discovery rate was controlled across all 89 candidates by the Benjamini-Hochberg procedure. "
    "Change in C-index is the difference in out-of-fold C-index with and without the candidate, by repeated "
    "five-fold cross-validation; a negative value means that the candidate lowered discrimination. No "
    "candidate survived false-discovery-rate control. The four nominal associations were cerebrospinal-fluid "
    "alpha-synuclein, total tau, phosphorylated tau and their composite, which changed the C-index by 0.004 "
    "to 0.010. Supplementary Table S10 gives every candidate with its hazard ratio, confidence intervals and "
    "power.")
check_refs(T3_TITLE + T3_LEGEND, "Table 3")
write_sheet(wb, "Table 3", T3_TITLE, with_abbreviations(T3_TITLE, [("table", rows3), ("note", T3_LEGEND)]))
assert wb.sheetnames == ["Table 1", "Table 2", "Table 3"], wb.sheetnames
wb.save(os.path.join(DST, "Tables.xlsx"))
print("Tables.xlsx:", wb.sheetnames, "| a Tabela 3 vem do 57")

# ------------------------------------------------------- supplementary tables ----------
sup = docx.Document(os.path.join(SRC, "Supplementary Tables.docx"))
pre, units = group(blocks(sup, "Supplementary Tables:"), r"^Supplementary Table (S\d+)\.")
assert [u["id"] for u in units] == ["S%d" % i for i in range(1, 19)], [u["id"] for u in units]
assert set(TABLE_MAP) | REMOVED == {u["id"] for u in units} and not set(TABLE_MAP) & REMOVED
wb = Workbook(); ws = wb.active; ws.title = "Notes"
ws.cell(row=1, column=1, value="Supplementary Tables").font = BOLD
for i, p in enumerate(pre, 3):
    ws.cell(row=i, column=1, value=renum(p)).alignment = WRAP
check_refs("\n".join(renum(p) for p in pre), "Notes")
ws.column_dimensions["A"].width = 120
# Supplementary Table S4 (S3 in October) now shows all 18 clinical candidates, not only the
# six selected, from 56_candidate_hazard_ratios.py: a reader sees what the selection chose
# from and why the rest stayed out. Its six rows must still read exactly as delivered.
def s4_all_candidates(u):
    t = pd.read_csv(os.path.join(C.RESULTS, "Tabelas", "tab56_candidate_hazard_ratios.csv"))
    assert len(t) == 18 and int(t.selected.sum()) == 6

    def hr(a, lo, hi):
        return "%.2f (%.2f to %.2f)" % (a, lo, hi)

    def pv(p):
        return "<0.001" if p < 0.001 else "%.3f" % p

    head = ["Candidate", "Hazard ratio per", "Unadjusted hazard ratio (95% CI)", "p",
            "Hazard ratio adjusted for the six predictors (95% CI)", "p",
            "Change in C-index when added to the six"]
    rows = {True: [head], False: [head]}
    for _, r in t.iterrows():
        rows[bool(r.selected)].append([
            r.candidate, r.increment.replace("kg/m2", "kg/m²"),
            hr(r.unadjusted_HR, r.unadjusted_lo, r.unadjusted_hi), pv(r.unadjusted_p),
            hr(r.adjusted_HR, r.adjusted_lo, r.adjusted_hi), pv(r.adjusted_p),
            "In the model" if r.selected else ("%+.4f" % r.delta_c).replace("-", "−")])
    # the delivered six-row table: predictor, increment, unadjusted HR, p, adjusted HR
    old = [p for k, p in u["parts"] if k == "table"]
    assert len(old) == 1 and len(old[0]) == 7
    for a, b in zip(old[0][1:], rows[True][1:]):
        assert (a[2], a[3], a[4]) == (b[2], b[3], b[4]), (a, b)
    assert all(r[6].startswith("−") for r in rows[False][1:]), "every candidate left out lowers the C-index"
    title = ("Supplementary Table S4. Hazard ratios of the 18 routine clinical candidates, unadjusted and "
             "adjusted for the six predictors of the final model")
    legend = (
        "Unadjusted hazard ratios come from a Cox model with each candidate alone, without penalty, in the "
        "development sample of 813 participants with 165 events, all of whom were complete on the 18 "
        "candidates. Adjusted hazard ratios carry the ridge penalty of the final model: for the six "
        "predictors they are the mutually adjusted estimates of Table 2, and for each of the other twelve "
        "they come from the six predictors with that candidate added as a seventh term. Change in C-index "
        "is the difference in out-of-fold C-index between the six predictors with and without the "
        "candidate, by twenty repetitions of five-fold cross-validation on the same folds; the six alone "
        "reach 0.696. The predictors were chosen on cross-validated discrimination together with clinical "
        "relevance, not on p values (Methods). Every one of the twelve candidates not retained lowered the "
        "cross-validated C-index when added, including the four that kept a nominal association after "
        "adjustment: the PPMI clinical stage, the Modified Schwab and England scale, fatigue and years of "
        "education. Candidates measuring motor severity or phenotype, such as Parts II and III of the "
        "MDS-UPDRS and the PIGD subscore, lost most of their association once the total MDS-UPDRS and the "
        "TD/PIGD ratio were in the model. Source: 56_candidate_hazard_ratios.py.")
    # the sentence above names the four nominal ones: check it against the table
    nominal = sorted(t[(~t.selected) & (t.adjusted_p < 0.05)].variable)
    assert nominal == sorted(["Clinical_Stage", "MSEADLG", "NP1FATG", "EDUCYRS"]), nominal
    return title, [("panel", "A. The six predictors of the final model"), ("table", rows[True]),
                   ("panel", "B. The twelve candidates not retained"), ("table", rows[False]),
                   ("note", legend)]


def replace_once(parts, old, new_text):
    hits = [i for i, (k, v) in enumerate(parts) if k == "note" and old in v]
    assert len(hits) == 1, (len(hits), old[:60])
    i = hits[0]
    parts[i] = ("note", parts[i][1].replace(old, new_text))


# Supplementary Table S6 (S4 in October) gains the C-index the abstract and the Results quote,
# corrected for optimism by bootstrap, beside the apparent and out-of-fold values and the
# calibration slope, read from 32_internal_performance.py. Until November 2026 the text cited
# this table for numbers it did not carry. Two sentences of its legend referred to passages
# that the revision removed from the abstract and the Results, and are rewritten.
def s6_add_performance(title, parts):
    import json
    j = json.load(open(os.path.join(C.RESULTS, "Tabelas", "tab32_internal_performance.json"), encoding="utf-8"))
    k = [i for i, (kind, _) in enumerate(parts) if kind == "table"][0]
    rows = parts[k][1]
    assert rows[0] == ["Measure", "Value", "SD across repetitions"] and rows[1][0] == "Harrell's C-index", rows[:2]
    assert rows[2][0].startswith("Uno") and len(rows) == 3
    corrected = "%.3f (%.3f to %.3f)" % (j["C_corrected"], j["C_corrected_lo"], j["C_corrected_hi"])
    assert (corrected, "%.3f" % j["C_apparent"], "%.3f" % j["C_out_of_fold"], "%.2f" % j["slope_out_of_fold"]) == \
        ("0.701 (0.656 to 0.750)", "0.712", "0.696", "1.07"), "the values the text quotes"
    parts[k] = ("table", [rows[0],
                          ["Harrell's C-index, apparent", "%.3f" % j["C_apparent"], ""],
                          ["Harrell's C-index, corrected for optimism over %d bootstrap resamples (95%% CI)"
                           % j["bootstrap"], corrected, ""],
                          ["Harrell's C-index, out of fold, twenty repetitions", "%.3f" % j["C_out_of_fold"], ""],
                          ["Harrell's C-index, out of fold, ten repetitions", rows[1][1], rows[1][2]],
                          rows[2],
                          ["Calibration slope, out of fold", "%.2f" % j["slope_out_of_fold"], ""]])
    replace_once(parts,
        "(A) Harrell's C-index is the concordance reported throughout the paper, and it is known to be optimistic "
        "when censoring is heavy, which it is here: 79.7% of the development sample is censored. Uno's estimator "
        "weights each comparable pair by the inverse probability of remaining uncensored and is the appropriate "
        "check. It is 0.05 lower, which is the honest measure of how much of the apparent discrimination is an "
        "artefact of censoring, and it is reported alongside Harrell's C in the abstract for that reason.",
        "(A) Harrell's C-index is the concordance reported throughout the paper. The first three rows give it in "
        "the data used for fitting, corrected for optimism by bootstrap, which is the estimate in the abstract and "
        "the Results, and out of fold by twenty repetitions of five-fold cross-validation, the protocol behind "
        "every comparison of discrimination. The next two rows share the same ten repetitions, so that Harrell's "
        "and Uno's estimates are computed on the same folds. Harrell's C is optimistic when censoring is heavy, "
        "which it is here: 79.7% of the development sample is censored. Uno's estimator weights each comparable "
        "pair by the inverse probability of remaining uncensored and is the appropriate check. It is 0.05 lower, "
        "which is the honest measure of how much of the apparent discrimination is an artefact of censoring. The "
        "out-of-fold calibration slope exceeds one because of the deliberate shrinkage of the ridge penalty.")
    replace_once(parts,
        "This panel is included so that a claim made in the Results can be checked rather than taken on trust: "
        "the spread of site-level estimates reflects small numbers rather than heterogeneity in performance.",
        "The spread of site-level estimates reflects small numbers rather than heterogeneity in performance.")
    replace_once(parts,
        "Panels A and B are averaged over 10 repetitions of five-fold cross-validation and panels C and D over "
        "twenty;",
        "In panel A, the bootstrap row uses 500 resamples and the out-of-fold rows the repetitions shown; panel B "
        "is averaged over ten repetitions of five-fold cross-validation and panels C and D over twenty;")
    old = "Supplementary Table S6. Discrimination, accuracy and calibration of the development model out of fold"
    assert title == old, title
    return "Supplementary Table S6. Discrimination, accuracy and calibration of the development model", parts


# Supplementary Table S11 (S9 in October) gains the dopamine-agonist rows the Results quote
# beside amantadine, from 58_dopamine_agonist.py, modelled as amantadine is in panel A.
def s11_add_agonist(title, parts):
    import json
    j = json.load(open(os.path.join(C.RESULTS, "Tabelas", "tab58_dopamine_agonist.json"), encoding="utf-8"))
    old = "Supplementary Table S11. Amantadine at time-zero and during follow-up: four sensitivity analyses"
    assert title == old, title
    title = ("Supplementary Table S11. Amantadine and dopamine agonists at time-zero, and amantadine during "
             "follow-up")
    hr = lambda a, lo, hi: "%.2f (%.2f to %.2f)" % (a, lo, hi)
    dc3 = ("%+.3f" % j["delta_C"]).replace("-", "−")
    rows = [["Model", "n", "Agonist users", "Hazard ratio (95% CI)", "p", "Change in out-of-fold C"],
            ["Unadjusted", "%d" % j["n"], "%d" % j["users"], hr(j["unadjusted_HR"], j["unadjusted_lo"], j["unadjusted_hi"]),
             "%.3f" % j["unadjusted_p"], "not applicable"],
            ["Adjusted for the six clinical predictors", "%d" % j["n"], "%d" % j["users"],
             hr(j["adjusted_HR"], j["adjusted_lo"], j["adjusted_hi"]), "%.3f" % j["adjusted_p"], dc3]]
    assert rows[2][3] == "0.96 (0.71 to 1.29)" and rows[1][3] == "1.09 (0.80 to 1.49)", "the values the text quotes"
    note = max(i for i, (k, _) in enumerate(parts) if k == "note")
    parts = parts[:note] + [("panel", "E. Dopamine agonist at time-zero"), ("table", rows)] + parts[note:]
    replace_once(parts,
        "Change in out-of-fold C is against the reference value of 0.696. Source: 31_amantadine_handling.py.",
        "(E) Dopamine-agonist use at time-zero, taken from the same treatment record and modelled as amantadine "
        "is in panel A, was not associated with the outcome and did not improve discrimination. Change in "
        "out-of-fold C is against the reference value of 0.696. Sources: 31_amantadine_handling.py and "
        "58_dopamine_agonist.py.")
    return title, parts


kept = sorted([u for u in units if u["id"] not in REMOVED], key=lambda u: int(TABLE_MAP[u["id"]][1:]))
for u in kept:
    new = TABLE_MAP[u["id"]]
    title, parts = renum(u["title"]), renum_parts(u["parts"])
    if new == "S4":
        title, parts = s4_all_candidates(u)
    if new == "S6":
        title, parts = s6_add_performance(title, parts)
    if new == "S11":
        title, parts = s11_add_agonist(title, parts)
    parts = with_abbreviations(title, parts)
    assert title.startswith("Supplementary Table %s. " % new), (u["id"], title[:40])
    check_refs(blob(title, parts), new)
    write_sheet(wb, "Table " + new, title, parts)
assert wb.sheetnames == ["Notes"] + ["Table S%d" % i for i in range(1, N_TABLES + 1)], wb.sheetnames
wb.save(os.path.join(DST, "Supplementary Tables.xlsx"))
print("Supplementary Tables.xlsx:", ", ".join(
    "%s (antiga %s, %d)" % (TABLE_MAP[u["id"]], u["id"], sum(k == "table" for k, _ in u["parts"]))
    if TABLE_MAP[u["id"]] != u["id"] else "%s (%d)" % (u["id"], sum(k == "table" for k, _ in u["parts"]))
    for u in kept))

# ----------------------------------------------------- supplementary methods -----------
meth = docx.Document(os.path.join(SRC, "Supplementary Tables.docx"))
body = meth.element.body
on_tables = False
for el in list(body.iterchildren()):
    tag = el.tag.split("}")[1]
    if tag == "sectPr":
        continue
    t = text(el).strip() if tag == "p" else ""
    if t == "Supplementary Tables:":
        on_tables = True
    if on_tables or tag == "tbl" or t.startswith(("Informative censoring.", "Levodopa responsiveness.")):
        body.remove(el)
val = [p for p in meth.paragraphs if p.text.startswith("Validation.")]
assert len(val) == 1
replace_in(val[0]._p, " The ridge penalty of 0.05 was fixed before the analyses and varied across six values "
           "between 0 and 0.5 in a sensitivity analysis (Supplementary Table S14). The whole selection "
           "procedure, forward selection maximising out-of-fold concordance within the 18-candidate pool, "
           "was repeated in 100 bootstrap resamples (Supplementary Table S15).",
           " The ridge penalty of 0.05 was fixed before the analyses.")
# supplementary figure legends, in their new order: calibration, decision curve, head-to-head
sfig = docx.Document(os.path.join(SRC, "Supplementary Figures.docx"))
fig = docx.Document(os.path.join(SRC, "Figures.docx"))
cal = [p for p in sfig.paragraphs if p.text.startswith("Supplementary Figure S1.")]
h2h = [p for p in sfig.paragraphs if p.text.startswith("Supplementary Figure S2.")]
dca = [p for p in fig.paragraphs if p.text.startswith("Figure 3.")]
assert len(cal) == len(h2h) == len(dca) == 1
anchor = body.find(W + "sectPr")
head = copy.deepcopy([p for p in sfig.paragraphs if p.text.strip() == "Supplementary Figures:"][0]._p)
anchor.addprevious(head)
def calibration_second_quintile():
    """The second quintile of out-of-fold five-year risk exactly as figuras_suplementares.R draws
    it: mean predicted risk, and Kaplan-Meier observed risk with its log-log interval, in %.
    The delivered legend gave 8.5% and 3.8%, rounded from values already rounded to four
    decimals (0.0845 and 0.0375); at full precision they are 8.4% and 3.7%."""
    import json
    import numpy as np
    from lifelines import KaplanMeierFitter
    tab = os.path.join(C.RESULTS, "Tabelas")
    perf = json.load(open(os.path.join(tab, "tab32_internal_performance.json"), encoding="utf-8"))
    d = C.build_cohort(with_metadata=False)
    beta = C.fit_total6(d).params_[C.TOTAL6].values
    d = d.merge(pd.read_csv(os.path.join(tab, "tab32_out_of_fold_linear_predictor.csv")), on="PATNO")
    assert len(d) == 813
    centre = float((d[C.TOTAL6].mean().values * beta).sum())
    d["risk"] = 1 - perf["calibration"]["5"]["S0"] ** np.exp(d.lp_out_of_fold - centre)
    assert abs(d.risk.mean() - perf["calibration"]["5"]["predicted_mean"]) < 1e-6
    s = d[pd.qcut(d.risk, 5, labels=False) == 1]
    km = KaplanMeierFitter().fit(s.exit, s.event)
    band = km.confidence_interval_survival_function_
    lo_s, hi_s = band[band.index <= 5].iloc[-1]
    return (100 * (1 - float(km.survival_function_at_times(5).iloc[0])), 100 * s.risk.mean(),
            100 * (1 - hi_s), 100 * (1 - lo_s))


for p in cal + dca + h2h:
    e = copy.deepcopy(p._p)
    for d in list(e.iter(W + "drawing")):
        d.getparent().remove(d)
    if p is cal[0]:
        replace_in(e, "an observed risk of 8.5% against 16.4% predicted, although its confidence interval, "
                      "3.8% to 18.5%,",
                   "an observed risk of %.1f%% against %.1f%% predicted, although its confidence interval, "
                   "%.1f%% to %.1f%%," % calibration_second_quintile())
    anchor.addprevious(e)
for p in body.iter(W + "p"):           # one pass over the whole document, legends included
    renum_in(p)
alltext = "\n".join(p.text for p in meth.paragraphs)
check_refs(alltext, "Supplementary Methods")
legs = [p.text[:60] for p in meth.paragraphs if p.text.startswith("Supplementary Figure S")]
assert [l[:23] for l in legs] == ["Supplementary Figure S1", "Supplementary Figure S2", "Supplementary Figure S3"], legs
assert "Decision-curve" in legs[1] and "Head-to-head" in legs[2], legs
meth.save(os.path.join(DST, "Supplementary Methods.docx"))
print("Supplementary Methods.docx:", sum(1 for p in meth.paragraphs if p.text.strip()), "paragrafos | legendas:",
      " / ".join(l.split(".")[0] + ": " + l.split(". ")[1][:22] for l in legs))

# ------------------------------------------------------------ figure legends -----------
for p in fig.paragraphs:
    has_img = p._p.find(".//" + W + "drawing") is not None
    if has_img or p.text.startswith("Figure 3.") or (not p.text.strip() and p._p.getnext() is not None):
        p._p.getparent().remove(p._p)
heads = [p for p in fig.paragraphs if p.text.strip() == "Figures:"]
assert len(heads) == 1
replace_in(heads[0]._p, "Figures:", "Figure legends:")
for p in fig.element.body.iter(W + "p"):
    renum_in(p)
check_refs("\n".join(p.text for p in fig.paragraphs), "Figure legends")
assert [p.text[:9] for p in fig.paragraphs if re.match(r"Figure \d", p.text)] == ["Figure 1.", "Figure 2."]
fig.save(os.path.join(DST, "Figure legends.docx"))
print("Figure legends.docx: Figure 1, Figure 2")

# ------------------------------------------------------------------ figures ------------
out = C.ensure_dir(os.path.join(DST, "Figures"))
NAMES = {"Figure1_participant_flow": "Figure 1", "Figure2_km_tertiles": "Figure 2",
         "FigureS1_internal_calibration": "Supplementary Figure S1",
         "FigureS2_decision_curve": "Supplementary Figure S2",
         "FigureS3_head_to_head": "Supplementary Figure S3"}
# the folder is this script's own output: a figure that left the paper must not linger in it
wanted = {"%s.%s" % (n, e) for n in NAMES.values() for e in ("png", "svg")}
for f in os.listdir(out):
    if f not in wanted:
        os.remove(os.path.join(out, f))
        print("removed from Figures/: %s" % f)
from PIL import Image
for stem, name in NAMES.items():
    dpi = Image.open(os.path.join(FIGS, stem + ".png")).info.get("dpi", (0, 0))
    assert round(dpi[0]) == 300, (stem, dpi)
    for ext in ("png", "svg"):
        shutil.copy2(os.path.join(FIGS, "%s.%s" % (stem, ext)), os.path.join(out, "%s.%s" % (name, ext)))
assert sorted(os.listdir(out)) == sorted("%s.%s" % (n, e) for n in NAMES.values() for e in ("png", "svg")), os.listdir(out)
print("Figures:", ", ".join(NAMES.values()), "| PNG a 300 dpi e SVG")

# ------------------------------------------------------------ STROBE checklist ---------
# The October checklist pointed at the October text. The revision moved or removed several
# of the passages it cites, so each changed location is rewritten here against the revised
# manuscript, with the new supplementary numbers written out rather than remapped. The Note
# column now records what this revision changed; the October notes described the previous one.
chk = docx.Document(os.path.join(SRC, "STROBE checklist (cohort) - completed.docx"))
intro = [p for p in chk.paragraphs if p.text.startswith("The study is reported according to TRIPOD+AI")]
assert len(intro) == 1
replace_in(intro[0]._p, "the revised manuscript of October 2026", "the revised manuscript of November 2026")
replace_in(intro[0]._p, "by group in Supplementary Table S9.", "by group in Supplementary Table S11.")
WHERE = {
    "3": ("Introduction, paragraph 3 (two aims). The outcome definition was prespecified (Methods, Outcome "
          "and predictors).", "Aims reduced to two in this revision"),
    "6a": ("Methods, Study design and setting (PPMI enrols recently diagnosed, untreated patients; motor "
           "complications assessed every 6 to 12 months after levodopa initiation) and Participants and study "
           "size (eligibility, exclusions).", "Assessment schedule given as the protocol interval in this revision"),
    "7": ("Methods, Outcome and predictors: outcome (item 4.1 or 4.2 of 2 or more), event date and censoring; "
          "the 18 clinical candidates and the 89 multimodal candidates by domain; treatment at time-zero, "
          "adjusted for the six predictors. The six selected predictors are given in Results, Model performance "
          "and validation, and in Table 2, and their timing in the Discussion, paragraph 4. Diagnosis follows "
          "the PPMI protocol (reference 9).", "Selected predictors moved to the Results in this revision"),
    "9": ("Methods, Statistical analysis, paragraph 3 (bias from excluding participants with incomplete data); "
          "assessors unaware of the model (Outcome and predictors); Discussion, paragraph 4 (limitations).",
          "Informative-censoring analysis removed in this revision"),
    "10": ("Methods, Participants and study size: minimum sample size criteria of Riley and colleagues; Results, "
           "Participants and outcome: 630 participants required, 813 available.",
           "Calculation moved to the Results in this revision"),
    "11": ("Methods, Statistical analysis: continuous predictors on their original scale, linearity by restricted "
           "cubic splines; tertiles of predicted risk used for display only (boundaries in the Figure 2 legend) "
           "and tertiles of age at onset in Supplementary Table S13.", ""),
    "12a": (None, ""),
    "14a": ("Table 1, overall and by recruitment wave, including treatment at time-zero.",
            "Columns for the two recruitment waves added in this revision"),
    "12b": ("Methods, Statistical analysis: performance by sex and by reported race; Supplementary Table S13 also "
            "gives tertiles of age at onset and recruitment wave.", "Reduced to sex and race in the text in this revision"),
    "12d": ("Methods, Outcome and predictors: participants without the event censored at their last Part IV "
            "assessment; Results, Participants and outcome: 196 lost to follow-up.",
            "Informative-censoring analysis removed in this revision"),
    "12e": ("Methods, Statistical analysis, paragraph 3: multiple imputation, definition of time-zero and of the "
            "outcome, proportional hazards and linearity.", "Sensitivity analyses reduced in this revision"),
    "1b": (None, ""), "5": (None, ""), "15": (None, ""), "22": (None, ""),
    "16a": ("Results, Model performance and validation (mutually adjusted hazard ratios with 95% CI); Table 2; "
            "Supplementary Table S4 (unadjusted); treatment adjusted for the six predictors in the Results, and "
            "unadjusted and adjusted in Supplementary Table S11.", ""),
    "16b": ("Figure 2 legend (16.9% and 25.5% five-year risk); Supplementary Table S13 (age at onset tertiles).", ""),
    "16c": ("Table 2 (absolute risk formula and baseline survival at 3, 5, 7 and 10 years); Discussion, final "
            "paragraph (two worked patients); Supplementary Figure S2.",
            "The decision curve moved to the supplement in this revision"),
    "17": ("Results, Subgroup analyses (Supplementary Table S13); multiple imputation (Results, Participants and "
           "outcome; Supplementary Table S2); proportional hazards and linearity (Results, Model performance and "
           "validation; Supplementary Table S5); definition of time-zero and of the outcome (Discussion, paragraph "
           "2; Supplementary Table S14).", "Sensitivity analyses reduced in this revision"),
    "19": ("Discussion, paragraph 4 (strengths and limitations): single cohort, complete-case exclusions with their "
           "event rates, timing and ON state of the predictor examination, outcome from history, and the direction "
           "of the resulting measurement error.", ""),
    "21": ("Discussion, paragraph 4 (single cohort, 92.7% White) and final paragraph (confirmation and "
           "recalibration before use outside PPMI).", ""),
}


def set_cell(cell, new, like):
    """Write new as the whole text of a one-paragraph cell, keeping the formatting of `like`."""
    assert len(cell.paragraphs) == 1, len(cell.paragraphs)
    p = cell.paragraphs[0]
    if p.text:
        replace_in(p._p, p.text, new) if new else [r._r.getparent().remove(r._r) for r in p.runs]
    elif new:
        run = p.add_run(new)
        if like.runs and like.runs[0]._r.rPr is not None:
            run._r.insert(0, copy.deepcopy(like.runs[0]._r.rPr))
    assert p.text == new, (p.text[:40], new[:40])


done = set()
for row in chk.tables[0].rows[1:]:
    item = row.cells[1].text.strip()
    if item.rstrip("*") in WHERE:
        where, note = WHERE[item.rstrip("*")]
        if where is not None:
            set_cell(row.cells[3], where, row.cells[3].paragraphs[0])
        set_cell(row.cells[4], note, row.cells[3].paragraphs[0])
        done.add(item.rstrip("*"))
assert done == set(WHERE), set(WHERE) - done
cells = "\n".join(c.text for r in chk.tables[0].rows for c in r.cells) + "\n" + "\n".join(p.text for p in chk.paragraphs)
check_refs(cells, "STROBE checklist")
assert "October 2026" not in cells and "Sensitivity and subgroup analyses" not in cells and "S16" not in cells
chk.save(os.path.join(DST, "STROBE checklist (cohort) - completed.docx"))
print("STROBE checklist: %d itens com local reescrito ou nota da revisao" % len(done))
