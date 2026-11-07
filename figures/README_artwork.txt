ARTWORK PACKAGE — Vaccine (Elsevier) submission
Manuscript: Machine-learning prediction of foot-and-mouth disease virus
serotype A vaccine matching from VP1 sequence, with focus on A/ASIA/Iran-05
Generated: 2026-10-27

-------------------------------------------------------------------------------
1. FILES
-------------------------------------------------------------------------------
Fig1.tif  Fig1.pdf   Boruta-selected VP1 features
Fig2.tif  Fig2.pdf   Classifier performance on the held-out test set
Fig3.tif  Fig3.pdf   SHAP summary (beeswarm)
Fig4.tif  Fig4.pdf   Distribution of predictive signal along VP1
Fig5.tif  Fig5.pdf   A/ASIA/Iran-05 case study

src/      the Python scripts that generate each figure directly from the
          step-5 to step-8 result files, for reproducibility.

Upload EITHER the .tif OR the .pdf set — not both. Elsevier accepts TIFF, EPS,
PDF and MS Office artwork. TIFF is the safest for Editorial Manager; the PDFs
are fully vector and will reproduce at any size.

-------------------------------------------------------------------------------
2. ELSEVIER SPECIFICATION — HOW EACH REQUIREMENT IS MET
-------------------------------------------------------------------------------
Requirement                           Status in this package
------------------------------------  -----------------------------------------
Format: TIFF/EPS/PDF/MS Office        TIFF (LZW) + vector PDF
Resolution: 300 dpi halftone,         900 dpi native; >=750 dpi effective even
500 dpi combination, 1000 dpi line    after scaling to full column width
Colour mode RGB                       RGB, alpha channel flattened onto white
Fonts embedded                        TrueType (fonttype 42) in the PDFs;
                                      Arial / Liberation Sans throughout
Minimum font size                     6.0 pt (annotations), 7.0-7.5 pt (labels)
No caption/figure number in artwork    none present; captions supplied as text
No author names or title in artwork    none present
File named with the figure number      Fig1 ... Fig5
Numbered in order of citation          Fig. 1 - Fig. 5, matching the Results
Captions supplied separately           see the "Figure captions" section of the
                                      manuscript file

Printed widths (the typesetter scales to the column grid):
  Fig. 1  1.5 column (140 mm)
  Fig. 2  double column (190 mm)
  Fig. 3  1.5 column (140 mm)
  Fig. 4  double column (190 mm)
  Fig. 5  double column (190 mm)

Colour is free both online and in print at Vaccine; every series is also
separable in greyscale (hatching in Figs 1 and 5, marker shape in Fig. 2,
position in Figs 3 and 4).

-------------------------------------------------------------------------------
3. DATA PROVENANCE
-------------------------------------------------------------------------------
Fig. 1  boruta_feature_selection.csv, feature_matrix_selected.csv
        Panel A shows the Boruta ranking and decision for all 96 candidates.
        A classical Boruta box plot of real-vs-shadow importances could not be
        regenerated: boruta_feature_selection.csv stores only the final ranking
        and decision, not the per-iteration importance distributions.
Fig. 2  rf_test_predictions.csv (ROC/PR recomputed; AUC-ROC reproduces
        0.842857 exactly, AUC-PR 0.8097 as average precision)
Fig. 3  shap_test_values.csv + feature_matrix_selected.csv,
        row order from shap_global_importance.csv (all 190 pairs)
Fig. 4  shap_global_importance.csv, serotype_comparison_table.csv,
        alignment_to_pdb_mapping.csv; signed effects taken verbatim from
        shap_interpretation_report.txt
Fig. 5  shap_test_values.csv, rf_test_predictions.csv,
        shap_case_study_iran05.csv; the waterfall is additive
        (0.4995 + sum of 23 SHAP values = 0.58494 = model output)

Checks that passed while building the figures:
  - test-set AUC-ROC recomputed from the stored probabilities = 0.842857
  - sensitivity/specificity at both thresholds reproduce the step-7 summary
    (0.50: TP 7, FP 2, FN 3, TN 26; 0.23: TP 7, FP 5, FN 3, TN 23)
  - shap_case_study_iran05.csv reproduces exactly from the 10 Iran-05 test
    pairs (3 match, 7 no match)
  - 18 confirmed VP1 site features, 13 with a negative and 5 with a positive
    mean effect, consistent with shap_interpretation_report.txt
