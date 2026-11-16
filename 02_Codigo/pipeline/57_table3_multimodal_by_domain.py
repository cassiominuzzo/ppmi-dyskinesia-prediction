# 57_table3_multimodal_by_domain.py
# Produces: tab57_multimodal_by_domain.csv, Table 3 of the November 2026 revision: the
#           multimodal screen summarised by domain, one row per domain and one for all 89.
#
# Why this script exists. The screen of 89 candidates is the central negative finding of the
# paper, but its full table (Supplementary Table S10) is too long for the main text, and a
# figure with one row per candidate filled a page. This table carries what a reader needs to
# see that nothing beyond routine clinical data helped: how many candidates each domain had,
# in how many participants and events they were tested, how many were nominally associated,
# the lowest false-discovery-rate-adjusted p value, and the median and range of the change in
# out-of-fold C-index when each was added to the six clinical predictors. Every number is read
# from the file behind Supplementary Table S10, so the two tables cannot disagree.
#
# Usage:
#   python 57_table3_multimodal_by_domain.py

import os

import pandas as pd

import cohort as C

TAB = os.path.join(C.RESULTS, "Tabelas")
SOURCE = os.path.join(TAB, "tab_incremental_value_all_domains_corrected.csv")
OUT = os.path.join(TAB, "tab57_multimodal_by_domain.csv")

# the order and names of Supplementary Table S10, without its letters
NAMES = {
    "B. Cognitive and neuropsychological": "Cognitive and neuropsychological tests",
    "C. Non-motor scales": "Non-motor scales",
    "D. Genetic": "Genetic",
    "E. Cerebrospinal fluid and blood": "Cerebrospinal fluid and blood",
    "F. Dopamine-transporter imaging": "Dopamine-transporter imaging, striatal",
    "G. Treatment at time-zero": "Treatment at time-zero",
    "H. Structural MRI (FreeSurfer, baseline)": "Structural magnetic resonance imaging",
    "I. Dopamine-transporter imaging, subregional": "Dopamine-transporter imaging, subregional",
    "J. Lysosomal and other biospecimen markers": "Lysosomal and other biospecimen markers",
    "K. Composite of correlated CSF markers": "Composite of cerebrospinal-fluid markers",
}


def summarise(label, g):
    return {"domain": label, "candidates": len(g),
            "n_min": int(g.n.min()), "n_max": int(g.n.max()),
            "events_min": int(g.events.min()), "events_max": int(g.events.max()),
            "nominal_p05": int((g.p < 0.05).sum()), "lowest_fdr_p": float(g.FDR_global.min()),
            "dC_median": float(g.dC.median()), "dC_min": float(g.dC.min()), "dC_max": float(g.dC.max())}


def main():
    iv = pd.read_csv(SOURCE)
    assert len(iv) == 89 and set(iv.domain) == set(NAMES), "the screen must have its 89 candidates in ten domains"
    rows = [summarise(NAMES[d], iv[iv.domain == d]) for d in sorted(NAMES)]
    rows.append(summarise("All candidates", iv))
    t = pd.DataFrame(rows)
    assert t.candidates[:-1].sum() == 89 and t.nominal_p05[:-1].sum() == 4
    assert round(t.lowest_fdr_p.iloc[-1], 2) == 0.77, "no candidate survives false-discovery-rate control"
    for _, r in t.iterrows():
        print("%-44s %2d  n %3d to %3d  events %3d to %3d  p<0.05 %d  lowest FDR p %.2f  change in C %+.3f (%+.3f to %+.3f)"
              % (r.domain, r.candidates, r.n_min, r.n_max, r.events_min, r.events_max, r.nominal_p05,
                 r.lowest_fdr_p, r.dC_median, r.dC_min, r.dC_max))
    t.to_csv(OUT, index=False)
    print("\nwritten %s" % OUT)


if __name__ == "__main__":
    main()
