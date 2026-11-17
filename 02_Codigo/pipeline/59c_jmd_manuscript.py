# 59c_jmd_manuscript.py
# Produces: Pacote JMD/Manuscript.docx, the blinded manuscript file of the Journal of
#           Movement Disorders package.
#
# The text is not edited here. It is written in the Google Doc of the JMD copy, where every
# change was applied by hand and checked against its HTML export; this script takes the Word
# export of that document and only lays it out as the journal asks:
#   - a title page with the title, the running title and the counts the journal asks for
#     (characters of the title, words of the abstract and of the main text, tables, figures),
#     all computed from the document itself and from the package, never typed;
#   - the abstract and its keywords on page 2, the main text from page 3, the references and
#     the figure legends each starting on a new page;
#   - Arial 12, double spacing, A4 with 2.5 cm margins, the page number centred at the foot
#     of every page from the title page onwards;
#   - the Paperpile links unwrapped into plain superscript numbers and the author fields of
#     the file cleared, so that nothing in the file identifies the authors.
# The legends come from the base package: Figure 1 unchanged, Figure 2 (the calibration) from
# the legend of Supplementary Figure S1 without its source line, and Figure 3 (the tertile
# curves) from the legend of the base Figure 2, renumbered as 59b_jmd_figures.py numbers the
# files. The script stops if any paragraph of the export is lost or altered, if a name or link
# of the authors appears, or if the text does not cite Figures 1 to 3 in that order and every
# supplementary figure in the package.
#
# Usage:
#   python 59c_jmd_manuscript.py <Word export of the JMD Google Doc> [output.docx]

import copy
import os
import re
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.run import Run

import cohort as C

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(C.RESULTS)
BASE = os.path.join(ROOT, "07_Versao_STROBE", "Revisao Daniel 2026-11")
OUT = os.path.join(ROOT, "Pacote JMD")
RUNNING = "Prediction of problematic dyskinesia in PD"
HEADINGS = {"Introduction:", "Materials & Methods:", "Results:", "Discussion:"}
# names and links that would unblind the review; the reference list may cite the authors
FORBIDDEN = ["Minuzzo", "Teixeira", "Schumacher", "cassiominuzzo", "github", "streamlit",
             "Hospital de Cl", "HCPA", "UFCSPA", "UFRGS", "Porto Alegre"]


def ptext(p):
    return re.sub(r"\s+", " ", "".join(t.text or "" for t in p.iter(qn("w:t")))).strip()


def words(s):
    return len(re.findall(r"\S+", s))


def para_after(anchor, text, bold=False):
    """A new paragraph after anchor, in the Normal style."""
    p = OxmlElement("w:p")
    anchor.addnext(p)
    if text:
        r = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        p.append(r)
        Run(r, None).bold = bold or None
    return p


def replace_in_paragraph(p, old, new):
    """Replace text that occurs once in the paragraph, even when it spans several runs: the new
    text goes into the first run the old one touches, and the rest of the old one is cut."""
    nodes = [t for t in p.iter(qn("w:t"))]
    full = "".join(t.text or "" for t in nodes)
    assert full.count(old) == 1, (old, full.count(old))
    start, end = full.index(old), full.index(old) + len(old)
    pos = 0
    placed = False
    for t in nodes:
        s, e = pos, pos + len(t.text or "")
        pos = e
        if e <= start or s >= end:
            continue
        a, b = max(start, s) - s, min(end, e) - s
        t.text = t.text[:a] + ("" if placed else new) + t.text[b:]
        t.set(qn("xml:space"), "preserve")
        placed = True


def bold_label_only(p):
    """In a legend, only the label "Figure n." is bold, as in the base package: a bold first
    run that carries the title as well is split in two."""
    runs = p.findall(qn("w:r"))
    m = re.match(r"(Figure \d+\.)(.+)$", "".join(t.text or "" for t in runs[0].iter(qn("w:t"))) if runs else "")
    if not m or not Run(runs[0], None).bold:
        return
    rest = copy.deepcopy(runs[0])
    runs[0].addnext(rest)
    for r, text in ((runs[0], m.group(1)), (rest, m.group(2))):
        ts = list(r.iter(qn("w:t")))
        ts[0].text = text
        ts[0].set(qn("xml:space"), "preserve")
        for t in ts[1:]:
            t.getparent().remove(t)
    Run(rest, None).bold = False


def page_break_before(p):
    pPr = p.get_or_add_pPr()
    el = OxmlElement("w:pageBreakBefore")
    pPr.insert(0, el)


def format_paragraph(p, before=0):
    pPr = p.get_or_add_pPr()
    for old in pPr.findall(qn("w:spacing")):
        pPr.remove(old)
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), str(before * 20))
    sp.set(qn("w:after"), "0")
    sp.set(qn("w:line"), "480")
    sp.set(qn("w:lineRule"), "auto")
    # w:spacing comes after w:pStyle, w:keepNext, w:keepLines, w:pageBreakBefore and w:numPr
    order = [qn("w:pStyle"), qn("w:keepNext"), qn("w:keepLines"), qn("w:pageBreakBefore"),
             qn("w:framePr"), qn("w:widowControl"), qn("w:numPr"), qn("w:suppressLineNumbers"),
             qn("w:pBdr"), qn("w:shd"), qn("w:tabs"), qn("w:suppressAutoHyphens")]
    k = 0
    for i, child in enumerate(pPr):
        if child.tag in order:
            k = i + 1
    pPr.insert(k, sp)
    for r in p.iter(qn("w:r")):
        font = Run(r, None).font
        font.name = "Arial"
        font.size = Pt(12)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python 59c_jmd_manuscript.py <export.docx> [output.docx]")
    d = docx.Document(sys.argv[1])
    body = d.element.body
    original = [t for t in (ptext(p) for p in body.iter(qn("w:p"))) if t]

    # Paperpile links -> plain runs, and their relationships dropped
    rids = []
    for h in list(body.iter(qn("w:hyperlink"))):
        rids.append(h.get(qn("r:id")))
        for child in list(h):
            h.addprevious(child)
        h.getparent().remove(h)
    for rid in set(filter(None, rids)):
        d.part.drop_rel(rid)

    paras = list(body.iter(qn("w:p")))
    texts = [ptext(p) for p in paras]
    first = lambda s: paras[texts.index(s)]
    title_p = paras[0]
    title = texts[0]
    abstract_p, intro_p, refs_p = first("Abstract:"), first("Introduction:"), first("References:")

    # the counts, from the document
    i_abs, i_key = texts.index("Abstract:"), [k for k, t in enumerate(texts) if t.startswith("Keywords:")][0]
    n_abstract = sum(words(t) for t in texts[i_abs + 1:i_key])
    i_int, i_data = texts.index("Introduction:"), texts.index("Data availability:")
    n_main = sum(words(t) for t in texts[i_int:i_data] if t and t not in HEADINGS and len(t.split()) > 6)
    tables = docx.Document(os.path.join(OUT, "Tables.docx"))
    n_tables = sum(1 for p in tables.paragraphs if re.match(r"Table \d+\. ", p.text))
    n_figures = len([f for f in os.listdir(os.path.join(OUT, "Figures")) if re.match(r"Figure \d+\.tif$", f)])
    assert n_abstract <= 250 and n_main < 4000 and len(RUNNING) < 50, (n_abstract, n_main)
    assert n_tables + n_figures <= 7

    # title page
    lines = [("Running title: " + RUNNING, False),
             ("Number of characters in the title (including spaces): %d" % len(title), False),
             ("Number of words in the abstract: %d" % n_abstract, False),
             ("Number of words in the main text (Introduction to Discussion, excluding references and "
              "figure legends): {:,}".format(n_main), False),
             ("Number of tables: %d" % n_tables, False),
             ("Number of figures: %d" % n_figures, False)]
    anchor = title_p
    for text, bold in lines:
        anchor = para_after(anchor, text, bold)
    page_break_before(abstract_p)

    # the empty paragraph that carried a page break before the Introduction
    between = paras[i_key + 1:i_int]
    for p in between:
        assert not ptext(p), "texto entre Keywords e Introduction"
        p.getparent().remove(p)
    page_break_before(intro_p)
    page_break_before(refs_p)

    # figure legends, on a page of their own at the end: Figure 1 and the tertile curves from
    # the base package's legends, the calibration from the base Supplementary Methods, each
    # renumbered as 59b_jmd_figures.py numbers the files
    base_legends = [lp._p for lp in docx.Document(os.path.join(BASE, "Figure legends.docx")).paragraphs
                    if lp.text.strip()]
    assert [ptext(p)[:9] for p in base_legends] == ["Figure le", "Figure 1.", "Figure 2."]
    calibration = [p._p for p in docx.Document(os.path.join(BASE, "Supplementary Methods.docx")).paragraphs
                   if p.text.startswith("Supplementary Figure S1. Internal calibration")]
    assert len(calibration) == 1
    LEGENDS = [
        (base_legends[0], []),
        (base_legends[1], []),
        (calibration[0], [("Supplementary Figure S1.", "Figure 2."),
                          ("counted in the last bin of the lower panel", "counted in the last bar of the lower panel"),
                          (" Source: figuras_R/figuras_suplementares.R, from the out-of-fold linear predictor "
                           "stored by 32_internal_performance.py.", "")]),
        (base_legends[2], [("Figure 2.", "Figure 3."), ("Supplementary Figure S1", "Figure 2")]),
    ]
    sectPr = body.find(qn("w:sectPr"))
    first_legend = True
    added_legends = []
    for source, changes in LEGENDS:
        p = copy.deepcopy(source)
        for old, new in changes:
            replace_in_paragraph(p, old, new)
        bold_label_only(p)
        added_legends.append(ptext(p))
        pPr = p.find(qn("w:pPr"))
        if pPr is not None:
            for st in pPr.findall(qn("w:pStyle")):
                pPr.remove(st)
        # a manual line break that closes a justified paragraph spreads its last line
        for br in list(p.iter(qn("w:br"))):
            r = br.getparent()
            if r.getnext() is None and br.getnext() is None:
                r.remove(br)
        if first_legend:
            for r in p.iter(qn("w:r")):
                Run(r, None).bold = True
                Run(r, None).underline = False
            page_break_before(p)
            first_legend = False
        sectPr.addprevious(p)

    # layout: every paragraph, the styles and the defaults
    for p in body.iter(qn("w:p")):
        style = p.find(qn("w:pPr") + "/" + qn("w:pStyle"))
        heading = style is not None and style.get(qn("w:val")) in ("Heading1", "Title")
        format_paragraph(p, before=12 if heading and ptext(p) not in ("Abstract:",) and p is not title_p else 0)
    for name in ("Normal", "Title", "Heading 1"):
        st = d.styles[name]
        st.font.name = "Arial"
        st.font.size = Pt(12)
    title_p.find(qn("w:pPr")).insert(0, OxmlElement("w:keepNext"))
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.5))
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    run.font.name, run.font.size = "Arial", Pt(12)
    for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), kind)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
        run._r.append(e)
    cp = d.core_properties
    cp.author = cp.last_modified_by = cp.title = cp.comments = cp.keywords = cp.subject = ""

    # checks: every paragraph of the export, in order and unchanged; nothing that unblinds
    final = [t for t in (ptext(p) for p in body.iter(qn("w:p"))) if t]
    added = [t for t, _ in lines] + added_legends
    assert [t for t in final if t not in added] == original, "um paragrafo da exportacao mudou"
    refs_at = final.index("References:")
    # figures cited in the order of their numbers, each with a file and a legend
    cited = "\n".join(final[:refs_at])
    order = []
    for m in re.finditer(r"(?<!Supplementary )Figures? (\d+)", cited):
        if int(m.group(1)) not in order:
            order.append(int(m.group(1)))
    assert order == list(range(1, n_figures + 1)), "figuras citadas no texto: %s" % order
    assert [int(re.match(r"Figure (\d+)\. ", t).group(1)) for t in added_legends[1:]] == order
    supp_files = sorted(int(m.group(1)) for m in (re.match(r"Supplementary Figure S(\d+)\.tif$", f)
                        for f in os.listdir(os.path.join(OUT, "Figures"))) if m)
    supp_cited = sorted({int(x) for x in re.findall(r"Supplementary Figures? S(\d+)", cited)})
    assert supp_cited == supp_files, (supp_cited, supp_files)
    before_refs = "\n".join(final[:refs_at] + final[final.index("Figure legends:"):])
    for name in FORBIDDEN:
        assert name.lower() not in before_refs.lower(), name
    assert all("paperpile" not in (r.target_ref or "") for r in d.part.rels.values()), "link do Paperpile"
    assert "—" not in "\n".join(added)
    sup = sum(1 for r in body.iter(qn("w:r")) if Run(r, None).font.superscript)
    path = sys.argv[2] if len(sys.argv) == 3 else os.path.join(OUT, "Manuscript.docx")
    d.save(path)
    print("Manuscript.docx: titulo %d caracteres | resumo %d palavras | texto %d palavras | %d tabelas | "
          "%d figuras | %d paragrafos | %d trechos sobrescritos"
          % (len(title), n_abstract, n_main, n_tables, n_figures, len(final), sup))


if __name__ == "__main__":
    main()
