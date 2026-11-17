# 59_jmd_package.py
# Produces: the tables and supplementary files of the Journal of Movement Disorders package,
#           in Pacote JMD/ at the project root:
#             Tables.docx                    Tables 1 to 3 as Word tables, one per page
#             Supplementary Tables.xlsx      the base package's, with the figures renumbered
#             Supplementary Methods.docx     the base package's, with the figures renumbered
#           The figures come from 59b_jmd_figures.py, which needs a different environment,
#           and the manuscript file from 59c_jmd_manuscript.py.
#
# In the JMD package the calibration plot (Supplementary Figure S1 of the base package) is
# Figure 2 of the main text, so its legend leaves the Supplementary Methods for the
# manuscript, a reference to it in the supplement points to Figure 2, and the decision curve
# and the head-to-head comparison move up to Supplementary Figures S1 and S2. Every
# reference is renumbered in one pass, so no label is renumbered twice.
#
# The base package (55_submission_files.py) delivers the main tables in Excel, as the
# co-author asked. The JMD asks for each table "as a Word table in order to ensure correct
# column alignment", typed "with a title, on a separate sheet", with footnotes marked by
# *, †, ‡ and so on. This script reads the three sheets of the base Tables.xlsx, so the two
# packages cannot disagree, and writes each as a Word table on its own landscape page: the
# title above, the table, and the notes below. Notes that refer to particular rows or columns
# are moved into footnotes marked with the journal's symbols, and every cell is checked
# against the sheet it came from.
#
# Usage:
#   python 59_jmd_package.py

import os
import re
import sys

import docx
import openpyxl
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

import cohort as C

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(C.RESULTS)
BASE = os.path.join(ROOT, "07_Versao_STROBE", "Revisao Daniel 2026-11")
OUT = C.ensure_dir(os.path.join(ROOT, "Pacote JMD"))

# Notes that refer to particular rows or columns: (sheet, start of the sentence in the
# legend, the row labels or column header that carry the symbol, the symbol)
MARKED = {
    "Table 1": [
        ("Levodopa daily dose and total dopaminergic daily dose were available",
         ["Levodopa daily dose at time-zero, mg",
          "Total dopaminergic daily dose at time-zero, amantadine excluded, mg"], "*"),
        ("The standardised difference is the absolute difference", ["Standardised difference"], "†"),
    ],
}


# base package -> JMD package
SUPP_FIGURE_MAP = {"1": "Figure 2", "2": "Supplementary Figure S1", "3": "Supplementary Figure S2"}
SUPP_FIGURE = re.compile(r"Supplementary Figure S([1-3])\b")
# a main figure cited in the supplement would need renumbering too (Figure 2 becomes 3)
MAIN_FIGURE = re.compile(r"(?<!Supplementary )Figures? [2-9]\b")


def renumber(text):
    return SUPP_FIGURE.sub(lambda m: SUPP_FIGURE_MAP[m.group(1)], text)


def supplementary_methods():
    d = docx.Document(os.path.join(BASE, "Supplementary Methods.docx"))
    legend = [p for p in d.paragraphs if p.text.startswith("Supplementary Figure S1. Internal calibration")]
    assert len(legend) == 1
    legend[0]._p.getparent().remove(legend[0]._p)
    assert not any(MAIN_FIGURE.search(p.text) for p in d.paragraphs), "figura principal citada"
    expected = [renumber(p.text) for p in d.paragraphs]
    for t in d.element.body.iter(qn("w:t")):
        if t.text and SUPP_FIGURE.search(t.text):
            t.text = renumber(t.text)
    # a label split across two runs would escape the replacement: the paragraphs must match
    assert [p.text for p in d.paragraphs] == expected
    labels = [p.text.split(".")[0] for p in d.paragraphs if p.text.startswith("Supplementary Figure S")]
    assert labels == ["Supplementary Figure S1", "Supplementary Figure S2"], labels
    d.save(os.path.join(OUT, "Supplementary Methods.docx"))
    print("Supplementary Methods.docx: legenda da calibracao levada ao texto; legendas", labels)


def supplementary_tables():
    wb = openpyxl.load_workbook(os.path.join(BASE, "Supplementary Tables.xlsx"))
    changed = []
    for ws in wb:
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str):
                    assert not MAIN_FIGURE.search(c.value), (ws.title, c.coordinate)
                    new = renumber(c.value)
                    if new != c.value:
                        changed.append("%s!%s" % (ws.title, c.coordinate))
                        c.value = new
    assert changed, "nenhuma figura suplementar citada nas tabelas"
    wb.save(os.path.join(OUT, "Supplementary Tables.xlsx"))
    print("Supplementary Tables.xlsx: figuras renumeradas em", ", ".join(changed))


def read_sheet(ws):
    """The sheet as 55_submission_files.py wrote it: title, then panels, tables and notes."""
    merged = {r.min_row for r in ws.merged_cells.ranges if r.min_col == 1 and r.max_col == ws.max_column}
    title, parts, block = None, [], []
    for r in range(1, ws.max_row + 1):
        cells = [ws.cell(r, c) for c in range(1, ws.max_column + 1)]
        values = ["" if c.value is None else str(c.value) for c in cells]
        if r in merged or (ws.max_column == 1 and values[0]):
            if block:
                parts.append(("table", block)); block = []
            text, font = values[0], cells[0].font
            if r == 1:
                title = text
            elif text.startswith("Abbreviations:"):
                parts.append(("abbreviations", text))
            elif font.b and font.i:
                parts.append(("panel", text))
            elif text:
                parts.append(("note", text))
        elif any(values):
            block.append(values)
        elif block:
            parts.append(("table", block)); block = []
    if block:
        parts.append(("table", block))
    # the sheet is as wide as its widest table: drop the columns a narrower table leaves empty
    trimmed = []
    for kind, value in parts:
        if kind == "table":
            width = max(max((c + 1 for c, v in enumerate(r) if v), default=0) for r in value)
            value = [r[:width] for r in value]
        trimmed.append((kind, value))
    return title, trimmed


def split_sentences(text):
    return re.split(r"(?<=[.;])\s+(?=[A-Z(])", text)


def mark(sheet, parts):
    """Move the notes that refer to particular rows or columns into marked footnotes."""
    rules = MARKED.get(sheet, [])
    if not rules:
        return parts, []
    footnotes = []
    for start, targets, symbol in rules:
        k = [i for i, (kind, v) in enumerate(parts) if kind == "note" and start in v]
        assert len(k) == 1, (sheet, start)
        sentences = split_sentences(parts[k[0]][1])
        i = [j for j, s in enumerate(sentences) if s.startswith(start)]
        assert len(i) == 1, (sheet, start)
        taken = [sentences.pop(i[0])]
        # a following sentence that continues the same point travels with it
        while i[0] < len(sentences) and sentences[i[0]].startswith(("It is not given", "values below")):
            taken.append(sentences.pop(i[0]))
        parts[k[0]] = ("note", " ".join(sentences))
        footnotes.append(symbol + " ".join(taken))
        hit = 0
        for kind, block in parts:
            if kind != "table":
                continue
            for row in block:
                for c, v in enumerate(row):
                    if v in targets:
                        row[c] = v + symbol; hit += 1
        assert hit == len(targets), (sheet, targets, hit)
    return parts, footnotes


def set_cell(cell, text, bold=False, size=9):
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    run.bold = bold; run.font.size = Pt(size)
    cell.paragraphs[0].paragraph_format.space_after = Pt(0)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)


def paragraph(document, text, bold=False, italic=False, size=10, after=4):
    p = document.add_paragraph()
    run = p.add_run(text); run.bold = bold; run.italic = italic; run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(after)
    return p


def main():
    wb = openpyxl.load_workbook(os.path.join(BASE, "Tables.xlsx"))
    assert wb.sheetnames == ["Table 1", "Table 2", "Table 3"], wb.sheetnames
    document = docx.Document()
    style = document.styles["Normal"]; style.font.name = "Arial"; style.font.size = Pt(10)
    for n, sheet in enumerate(wb.sheetnames):
        section = document.sections[0] if n == 0 else document.add_section()
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = Cm(29.7), Cm(21.0)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Cm(2.0))
        title, parts = read_sheet(wb[sheet])
        assert title and title.startswith(sheet + "."), (sheet, title)
        source = [[list(r) for r in b] for k, b in parts if k == "table"]
        parts, footnotes = mark(sheet, parts)
        paragraph(document, title, bold=True, size=12, after=8)
        written = []
        for kind, value in parts:
            if kind == "panel":
                paragraph(document, value, bold=True, italic=True, size=10)
            elif kind == "table":
                ncol = max(len(r) for r in value)
                t = document.add_table(rows=len(value), cols=ncol)
                t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
                t.autofit = False
                # widths follow the content: the longest cell of each column, capped so that one
                # long label cannot starve the others, over the 25.7 cm of a landscape page
                weight = [max(min(len(r[c]) if c < len(r) else 0, 40) for r in value) + 6 for c in range(ncol)]
                widths = [Cm(25.7 * w / sum(weight)) for w in weight]
                for row in t.rows:
                    for c, cell in enumerate(row.cells):
                        cell.width = widths[c]
                for i, row in enumerate(value):
                    for c in range(ncol):
                        set_cell(t.cell(i, c), row[c] if c < len(row) else "", bold=(i == 0))
                repeat_header(t.rows[0])
                written.append(value)
                document.add_paragraph().paragraph_format.space_after = Pt(2)
        for f in footnotes:
            paragraph(document, f, size=9)
        for kind, value in parts:
            if kind == "note" and value.strip():
                paragraph(document, value, size=9)
        for kind, value in parts:
            if kind == "abbreviations":
                paragraph(document, value, size=9)
        # every cell of every table must equal the sheet, apart from the added symbols
        strip = lambda v: re.sub(r"[*†‡]$", "", v)
        assert [[[strip(v) for v in r] for r in b] for b in written] == source, sheet
        print("%s: %d tabela(s), %d nota(s) marcada(s) com simbolo" % (sheet, len(written), len(footnotes)))
    document.save(os.path.join(OUT, "Tables.docx"))
    supplementary_methods()
    supplementary_tables()
    print("Pacote JMD:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
