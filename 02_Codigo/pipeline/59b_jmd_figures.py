# 59b_jmd_figures.py
# Produces: the figures of the Journal of Movement Disorders package, in Pacote JMD/Figures/,
#           as TIFF in CMYK at the resolution the journal asks for: 1,200 dpi for line art
#           (Figure 1, the participant flow) and 600 dpi for every other figure.
#
# The figures are not redrawn. Each is rasterised from the vector PDF that figuras.R,
# figuras_suplementares.R and 38_figure1_participant_flow.py already write, so the TIFF shows
# exactly the PNG and SVG of the base package, only at a higher resolution, and is converted
# from sRGB to CMYK with the SWOP profile that Windows ships (RSWOP.icm), with perceptual
# rendering, rather than by a naive channel split. LZW compression keeps the files small.
#
# Needs pillow and pypdfium2 only, so it runs without the analysis environment.
#
# Usage:
#   python 59b_jmd_figures.py

import os
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageCms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FIGS = os.path.join(ROOT, "03_Resultados", "Figuras")
OUT = os.path.join(ROOT, "Pacote JMD", "Figures")
CMYK_PROFILE = r"C:\Windows\System32\spool\drivers\color\RSWOP.icm"

# file in 03_Resultados/Figuras -> name in the package, resolution in dpi.
# In the JMD package the calibration leaves the supplement for the main text. The text cites
# it before the tertile curves, and the journal numbers figures in the order of citation, so
# it is Figure 2 there and the tertile curves become Figure 3; the decision curve and the
# head-to-head comparison move up to Supplementary Figures S1 and S2. The base package keeps
# its own numbering.
FIGURES = [
    ("Figure1_participant_flow", "Figure 1", 1200),
    ("JMD_Figure2_calibration", "Figure 2", 600),
    ("Figure2_km_tertiles", "Figure 3", 600),
    ("FigureS2_decision_curve", "Supplementary Figure S1", 600),
    ("FigureS3_head_to_head", "Supplementary Figure S2", 600),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    # a figure the package no longer has must not stay behind under its old number
    keep = {name + ".tif" for _, name, _ in FIGURES}
    for f in os.listdir(OUT):
        if f.endswith(".tif") and f not in keep:
            os.remove(os.path.join(OUT, f))
            print("removed", f)
    srgb = ImageCms.createProfile("sRGB")
    cmyk = ImageCms.getOpenProfile(CMYK_PROFILE)
    to_cmyk = ImageCms.buildTransform(srgb, cmyk, "RGB", "CMYK",
                                      renderingIntent=ImageCms.Intent.PERCEPTUAL)
    for stem, name, dpi in FIGURES:
        pdf = pdfium.PdfDocument(os.path.join(FIGS, stem + ".pdf"))
        assert len(pdf) == 1, (stem, len(pdf))
        page = pdf[0]
        w_in, h_in = page.get_width() / 72, page.get_height() / 72
        rgb = page.render(scale=dpi / 72, fill_color=(255, 255, 255, 255)).to_pil().convert("RGB")
        out = ImageCms.applyTransform(rgb, to_cmyk)
        assert out.mode == "CMYK"
        path = os.path.join(OUT, name + ".tif")
        out.save(path, compression="tiff_lzw", dpi=(dpi, dpi))
        check = Image.open(path)
        assert check.mode == "CMYK" and round(check.info["dpi"][0]) == dpi
        print("%-26s %5.2f x %5.2f in  %4d dpi  %5d x %5d px  %6.1f MB"
              % (name, w_in, h_in, dpi, out.width, out.height, os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    sys.exit(main())
