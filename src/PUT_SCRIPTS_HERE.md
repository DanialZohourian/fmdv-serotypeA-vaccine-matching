# Scripts to add to this folder

Copy the analysis pipeline from the machine where it was run, and rename the files so the
execution order is obvious. The README documents this order:

- `01_curate_dataset.py`      WRLFMD record extraction, inclusion/exclusion, master_pairs.csv
- `02_align_sequences.py`     MAFFT L-INS-i alignment, trimming, PDB coordinate mapping
- `03_feature_engineering.py` pairwise distance, site differences, N-glycosylation features
- `04_boruta_selection.py`    Boruta on the SMOTE-balanced training set
- `05_train_model.py`         hyperparameter search, final model, test-set metrics, baselines
- `06_shap_analysis.py`       SHAP values, global importance, Iran-05 case study

Keep the file names if they differ, but update the README so the two match.
Delete this file once the scripts are in place.
