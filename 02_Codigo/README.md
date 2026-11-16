# Analysis code

Code for *Development and internal validation of a single-visit clinical model to
predict problematic levodopa-induced dyskinesia in Parkinson's disease*.

Development cohort: PPMI. There is no external cohort in this version of the study.
The frozen coefficients and the interactive calculator are in `../05_Calculadora/`.

## Data availability

PPMI is controlled-access and is **not** redistributed here.
Request access at https://www.ppmi-info.org (data use agreement required).

Set the data root before running anything:

```bash
export LID_PROJECT_ROOT=/path/to/your/data
pip install -r requirements.txt
```

Expected layout under `LID_PROJECT_ROOT`:

```
01_Dados_Brutos/
  Curated_Data_Cut/PPMI_Curated_Data_Cut_Public_<date>.xlsx
  Historia_Medica_e_Medicacao/LEDD_Concomitant_Medication_Log_<date>.csv
  MDS_UPDRS_e_Motor/MDS-UPDRS_Part_IV__Motor_Complications_<date>.csv
  ... (remaining raw domains, see 00_data_inventory.py)
```

## `pipeline/` — reproduces the manuscript

Run in order. Each script states at the top which manuscript element it produces.

| Script | Produces |
|---|---|
| `00_data_inventory.py` | Inventory of all raw tables |
| `00b_curated_cache.py` | `curated_cache.parquet`, the Curated Data Cut read once and stored as parquet. Six scripts read that file and none wrote it; `--write` rebuilds it, and the default run checks all 181 columns against the stored copy |
| `01_cohort_and_time_zero.py` | Cohort and time-zero; numbers in Figure 1 |
| `02_horizon_support.py` | Follow-up supporting each horizon; at-risk row of Table 2 |
| `03_baseline_predictor_matrix.py` | Baseline predictor matrix, all candidates pre-levodopa |
| `03b_candidate_matrix.py` | `candidate_matrix.parquet`, the analysis matrix, rebuilt from the raw tables. The default run checks all 39 columns against the stored file and stops if any diverges |
| `04_table1_and_incidence.py` | The eligible cohort described, and cumulative incidence. Note that this describes the 1,447 eligible, not the development sample |
| `04b_table1.py` | Table 1 of the manuscript: the 813 of the development sample, with the treatment rows from script 27 and the four rows STROBE item 14a asks for (age at levodopa initiation, disease duration, Hoehn and Yahr stage, race). Since November 2026 also by recruitment wave, the split of the temporal validation, with the standardised difference between the waves. Prints each cell beside the published one |
| `05_final_model_internal_validation.py` | Table 2; optimism correction; leave-one-site-out |
| `06_dynamic_model_refit.py` | Dynamic mode of the calculator |
| `11_supplementary_table_S3.py` | Supplementary Table S14, panel A: the twelve analyses varying time-zero, outcome definition and prevalent cases, with the out-of-fold concordance of each. The file name keeps an older numbering |
| `12_competing_risk.py` | Aalen-Johansen adjustment, factor c(t) in Table 2 |
| `13_supplementary_table_S4.py` | Levodopa responsiveness across thresholds. It was Supplementary Table S18 until the November 2026 revision removed it from the supplement, because responsiveness is measured after levodopa is started. The file name keeps an older numbering |
| `21_full_audit.py` | Independent re-derivation of every key number from the raw tables |
| `27_treatments_at_time_zero.py` | Treatment rows of Table 1, the amantadine-free dose included. Rewritten in August 2026 to import the audited drug searches from `cohort.py` |
| `28_model_assumptions.py` | Proportional hazards and linearity checks |
| `29_multiple_imputation.py` | Multiple imputation sensitivity analysis |
| `31_amantadine_handling.py` | Amantadine sensitivity analyses and corrected dose variable |
| `32_internal_performance.py` | Apparent and optimism-corrected C, calibration slope, calibration at 3, 5 and 7 years, leave-one-site-out |
| `33_temporal_validation.py` | Temporal validation across PPMI recruitment waves |
| `34_decision_curve_internal.py` | Internal decision-curve analysis |
| `35_subgroup_performance.py` | Performance by sex, age at onset, recruitment wave and race |
| `36_sample_size_riley.py` | Riley minimum sample size criteria |
| `37_eligibility_sensitivity.py` | Exclusion of participants outside the PPMI Parkinson's disease cohort |
| `38_figure1_participant_flow.py` | Figure 1, PPMI only |
| `39_drug_name_quality_control.py` | Drug-name search audit and the time-zero sensitivity analysis |
| `40_robust_discrimination.py` | Uno's C, time-dependent AUC, Brier score and index of prediction accuracy |
| `41_included_vs_excluded.py` | The 813 included against the 628 excluded for incomplete data |
| `42_penalty_sensitivity.py` | Hazard ratios and C across six values of the ridge penalty. Out of the supplement since the November 2026 revision |
| `43_selection_stability.py` | Selection frequency over 100 bootstrap resamples; resumable with `--limit`. Out of the supplement since the November 2026 revision |
| `44_classification_metrics.py` | Sensitivity, specificity and predictive values by risk threshold |
| `45_informative_censoring.py` | Censoring decomposition, dropout model and delta-based imputation. Out of the supplement since the November 2026 revision |
| `46_calculator_artifacts.py` | Regenerates `05_Calculadora/calculator_artifacts.json` from the raw data |
| `48_supplementary_tables.py` | Builds the eighteen supplementary tables of the October 2026 version with their legends, reading every number from `03_Resultados/Tabelas/`. `55_submission_files.py` keeps fourteen of them and renumbers them |
| `49_head_to_head_unified.py` | Supplementary Table S12 and Supplementary Figure S3 from one analysis: the published predictor sets with paired bootstrap intervals, and the discrimination attainable at each model size. Replaces the two earlier scripts, which used different samples and different cross-validation |
| `49b_loo_variables.py` | Builds the seven baseline factors reported by Loo 2024, including the two the candidate matrix does not carry |
| `50_treatment_candidate_recomputed.py` | Recomputes the treatment row of Supplementary Table S10 with the amantadine-free dose, after reproducing the protocol of the multimodal screen |
| `52_multimodal_screen.py` | Supplementary Table S10: all 89 multimodal candidates rebuilt from the raw downloads and rescreened. `--probe` reports the sample recovered for each candidate against the published one |
| `51_outcome_anchoring.py` | Supplementary Table S3: whether crossing the outcome threshold is followed by a loss of function on the Modified Schwab and England scale |
| `54_biospecimen_candidates.py` | The rows of the lysosomal assays in Supplementary Table S10, each read from one specimen and one reported quantity, with the false-discovery columns recomputed around them. Also runs every assay in every specimen. Run after 50 and after `52_multimodal_screen.py --write` |
| `53_strobe_additions.py` | Supplementary Table S4, the unadjusted hazard ratios, the p values of Table 2, and the descriptive numbers STROBE asks for: person-time and incidence rate, the period of levodopa initiation, the assessment schedule, the timing of the two sources of predictors, the examinations recorded in the ON state, the participants with no pre-levodopa assessment, the boundaries of the risk tertiles in Figure 2 and the two patients of the Discussion |
| `55_submission_files.py` | The submission files of the November 2026 revision: tables and supplementary tables in Excel, the Supplementary Methods and figure legends in Word, and each figure as PNG at 300 dpi and SVG. Changes no number: it reads the delivered Word files, adds the recruitment waves to Table 1 from `04b_table1.py --write`, builds Supplementary Table S4 from `56_candidate_hazard_ratios.py` and Table 3 from `57_table3_multimodal_by_domain.py`, adds the optimism-corrected C-index and the calibration slope to Supplementary Table S6 and the agonist rows of `58_dopamine_agonist.py` to S11, closes every table with its abbreviations, drops Supplementary Tables S14, S15, S16 and S18 of the October version, and renumbers the other tables and the supplementary figures in the order the revised manuscript first cites them, remapping every cross-reference in one pass; the header gives the mapping. Run after the figure scripts |
| `56_candidate_hazard_ratios.py` | Supplementary Table S4 of the November 2026 revision: the unadjusted hazard ratio of each of the 18 clinical candidates, its hazard ratio adjusted for the six predictors, and for the twelve not retained the change in out-of-fold C-index when added to the six. Its six selected rows reproduce `tab53_unadjusted_hazard_ratios.csv` and the script stops if they do not |
| `57_table3_multimodal_by_domain.py` | Table 3 of the November 2026 revision: the 89-candidate multimodal screen summarised by domain (candidates, participants, events, nominal associations, lowest false-discovery-rate-adjusted p, and the median and range of the change in out-of-fold C-index), read from the file behind Supplementary Table S10 |
| `58_dopamine_agonist.py` | The dopamine-agonist rows of Supplementary Table S11: agonist use at time-zero, unadjusted and adjusted for the six predictors as `31_amantadine_handling.py` does for amantadine, with the change in out-of-fold C-index. Reproduces the published 1.09 (0.80 to 1.49) and 0.96 (0.71 to 1.29), which the August 2026 audit had computed without a script, and stops if they move |

### Scripts removed with the external validation

Numbers 15 to 20 and 26 built and validated the frozen five-variable model in
LARGE-PD and in the non-PPMI AMP-PD cohorts, and drew the external calibration and
decision-curve figures. They were removed when external validation was dropped from
the study. The numbering of the surviving scripts was deliberately left unchanged so
that earlier drafts and the audit notebook remain traceable; the gaps are not
missing files. Script `22_figure1_participant_flow.py` was replaced by
`38_figure1_participant_flow.py`, which draws the development cohort alone.

### `legacy/` — superseded by later analyses

Nine scripts whose results were replaced: the earlier polygenic, imaging and
cerebrospinal-fluid screens, now inside the 89-candidate screen of
`52_multimodal_screen.py`; the two earlier head-to-head comparisons, now inside
`49_head_to_head_unified.py`; and the three matplotlib figure scripts, now drawn in R.
They are kept because two of their output files are still on disk and contradict the
current results, so a reader needs to be told which is which. `legacy/README.md` names
each one and its replacement. Nothing in the paper comes from that folder.

Every figure and table in the manuscript and in the supplement is produced by a script
in this folder. Two qualifications, both of which are stated in the scripts themselves and
in the legend of Supplementary Table S10. The levodopa-equivalent dose that the published
screen used cannot be recovered, because no dose column survives in the processed data:
`50_treatment_candidate_recomputed.py` establishes this and replaces that one row with the
amantadine-free dose the Methods describe. And the change in out-of-fold concordance of
the other 85 candidates carries the estimate of the original screen, whose cross-validation
seeding was never recorded: `52_multimodal_screen.py` reproduces their hazard ratios and p
values but not that column.

Figures are written to `../03_Resultados/Figuras/` in PNG, SVG and PDF.

### `pipeline/figuras_R/` — R

| Script | O que faz |
|---|---|
| `figuras.R` | Figure 2 (incidência cumulativa por tercil) e Supplementary Figure S2 (curva de decisão em dois painéis), em PNG a 300 dpi, SVG e PDF. A Figure 1 é desenhada pelo `38_figure1_participant_flow.py`. Substitui os scripts 25 e 26, que apontavam para o layout antigo. A Figura S2 confere a redução por 1.000 contra o `tab34_decision_curve.json` antes de desenhar |
| `dose_sem_amantadina.R` | Linhas de tratamento da Table 1, com a dose dopaminérgica recalculada sem a amantadina |
| `amantadina_pkmerz.R` | Reproduz o script 31 com e sem o PK-Merz na busca de fármacos, e imprime a conferência contra o `tab31_amantadine.json` |
| `figuras_suplementares.R` | Supplementary Figure S1 (calibração interna em 5 anos) e S3 (comparação direta), em PNG a 300 dpi, SVG e PDF. Substitui os scripts 23 e 24, que apontavam para o layout antigo e vinham de análises que já não batem com o artigo. Confere a calibração contra o `tab32_internal_performance.json` e para com erro se divergir |

Rodar com `Rscript`, de qualquer diretório. A variável de ambiente `LID_RAW` aponta
para `01_Dados_Brutos`; `LID_PROJECT` sobrescreve a raiz do projeto.

## `exploratory/` — superseded

Model-selection history: the univariate screens, the hierarchical models and the
earlier candidate models that preceded the six-variable specification. Retained so
that the predictor-selection process is inspectable.

**These scripts do not reproduce the manuscript.** They were run on earlier outcome
definitions and earlier predictor sets, and the numbers they print differ from the
published ones by design. Use `pipeline/` for anything reported in the paper.

## Environment

Python 3.10.12 originally; the analyses were re-run on 3.12.10 in August 2026 with the
same pinned versions and reproduced exactly. Exact package versions in
`requirements.txt`. Random seeds are fixed for every bootstrap and cross-validation, so
results reproduce on re-running. One estimator did not respect that until September 2026:
lifelines breaks tied event times in the Aalen-Johansen estimator with unseeded noise, so
`12_competing_risk.py` returned a slightly different competing-risk factor on every run,
moving the third decimal of the four constants the calculator applies. It now averages the
estimate over 400 seeded tie-breaks and reports their spread, which is under 0.05
percentage points at every horizon.

Two environment variables locate the data, which is controlled-access and lives outside
this repository. `LID_PROJECT_ROOT` points at the directory holding `01_Dados_Brutos`.
`LID_PROC_DIR` overrides the processed-data directory, which otherwise resolves to
`../../04_Dados_processados`. Scripts import `cohort` and `config`, so both this folder
and `pipeline/` have to be on `PYTHONPATH`, and scripts are run from `pipeline/`.

The R scripts in `pipeline/figuras_R/` take `LID_RAW` for the raw directory and
`LID_PROJECT` for the project root; both have sensible defaults and are usually not
needed.
