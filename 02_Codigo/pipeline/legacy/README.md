# Superseded scripts

These nine scripts produced results that later analyses replaced. They are kept because
they document how the analysis got to where it is, and because their output files are still
on disk under `03_Resultados/Tabelas`, where a reader could otherwise mistake a stale file
for a current one. **Nothing in the manuscript or in the supplement comes from any of
them.** None is on the run order in the main README.

| Script | What it did | What replaced it |
| --- | --- | --- |
| `07_incremental_value_prs.py` | Incremental value of the two polygenic scores | `52_multimodal_screen.py`, which screens all three scores inside the 89-candidate screen under one protocol |
| `08_incremental_value_imaging_csf.py` | DaTSCAN laterality and cerebrospinal fluid markers | `52_multimodal_screen.py`, domains E and F |
| `09_fdr_variants.py` | False-discovery-rate variants for the screen | `52_multimodal_screen.py`, which computes the within-domain and global adjustment over the same 89 p values it produced |
| `10_supplementary_table_S1.py` | An earlier multimodal added-value table | `52_multimodal_screen.py` |
| `14_head_to_head_with_CI.py` | Comparison against published predictor sets | `49_head_to_head_unified.py`, which fits every set in one complete-case sample so the differences are paired |
| `23_supplementary_figure_S1.py` | Supplementary Figure S1 in matplotlib | `figuras_R/figuras_suplementares.R` |
| `24_supplementary_figure_S2.py` | Supplementary Figure S2 in matplotlib | `figuras_R/figuras_suplementares.R`, drawn from `tab49_sets_unified.csv` |
| `25_figure2_km_tertiles.py` | Figure 2 in matplotlib | `figuras_R/figuras.R` |
| `30_head_to_head_matched_df.py` | Head-to-head on a matched frame | `49_head_to_head_unified.py` |

Two of their outputs contradict the current results and are the reason this folder exists
rather than a plain deletion:

- `tab67_head_to_head_CI.csv` (from `14`) carries a different complete-case sample from the
  one Supplementary Table S15 and Supplementary Figure S2 report. The unified analysis in
  `49` fits all seven sets in the same 801 participants; the older script let each set use
  its own sample, so the concordances were not comparable with each other.
- `tabS_multimodal_added_value.csv` (from `10`) carries seven candidates where the screen
  now carries 89, and its false-discovery adjustment was computed over those seven alone.

If either file is ever needed again, regenerate it by running the script from this folder
with the pipeline directory on `PYTHONPATH`; do not read the stored copy.
