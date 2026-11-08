# Machine learning prediction of FMDV serotype A vaccine matching from VP1 sequences

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24053304.svg)](https://doi.org/10.5281/zenodo.24053304)

Data, analysis pipeline, trained model and web application for:

> Norouzi S, Zohourian D. *Machine learning prediction of vaccine matching for foot-and-mouth
> disease virus serotype A from VP1 amino acid sequences, with application to the A/ASIA/Iran-05
> lineage.* Manuscript submitted, 2026.

A Boruta-selected Random Forest predicts binary vaccine matching (r1 ≥ 0.3, the WOAH threshold)
for foot-and-mouth disease virus serotype A directly from VP1 amino acid sequences. On an
independent held-out test set (n = 38) it reaches **AUC-ROC 0.843** and **AUC-PR 0.810**, and it
separates matched from unmatched pairs perfectly within the A/ASIA/Iran-05 subgroup (n = 10).

> **This is a pre-screening aid, not a replacement for the two-dimensional virus neutralisation
> test (2D-VNT). Confirm any vaccine strain decision serologically.**

## Web application

**https://fmdv-serotypea-vaccine-matching.streamlit.app**

Paste a VP1 amino acid sequence (208–220 aa) and choose a reference antiserum. The app returns a
match probability, a binary call at the pre-specified threshold of 0.23, and a SHAP waterfall
chart showing which positions drove the prediction.

## Dataset

190 unique virus–antiserum pairs from 65 field viruses and 15 reference antisera, spanning
14 countries and 22 years (2001–2022), curated from WRLFMD Vaccine Matching Reports.
Labels follow the WOAH threshold: r1 ≥ 0.3 = match (n = 48, 25.3%); r1 < 0.3 = no match
(n = 142, 74.7%).

Of 661 candidate records screened, 413 were excluded — 187 with non-numeric r1 values, 171 with
no publicly retrievable sequence, 66 tested by an assay other than 2D-VNT, and 38 with censored
reporting. A further 58 pairs involved antisera whose sequence could not be retrieved.
`dataset/master_excluded.csv` records every exclusion and its reason.

All sequences were retrieved from NCBI GenBank; accession numbers are in
`step2_sequences/05_final/sequence_registry.csv`. No new sequences were generated.

## Model

| | |
|---|---|
| Algorithm | RandomForestClassifier (scikit-learn 1.9.1) |
| Features | 23, Boruta-selected from 96 candidates |
| Hyperparameters | n_estimators = 300, max_depth = 10, max_features = "sqrt", min_samples_leaf = 2, class_weight = "balanced" |
| Class balancing | SMOTE (k = 5), applied inside cross-validation folds only |
| Decision threshold | 0.23, maximising F1 on out-of-fold training predictions |
| Random seed | 42 |
| Cross-validated AUC | 0.887 ± 0.081 |

Features are the pairwise Poisson-corrected amino acid distance, binary site-difference indicators
at 18 alignment positions, and four N-glycosylation sequon features. Positions follow the
A22/IRQ/24/64 reference and map onto PDB entry 4GH4 chain A.

The strongest predictor is **VP1 position 149** in the G-H loop. None of the seven positions
reported as most informative for serotype O was confirmed for serotype A, so the determinants of
cross-neutralisation are serotype-specific and models should not be transferred between serotypes
without retraining.

## Repository layout

```
src/                      the analysis pipeline, numbered in execution order
dataset/                  curated pairs, exclusions, search logs, audit trail
step2_sequences/          GenBank downloads, extracted VP1, translated protein, QC, registry
step3_alignment/          MAFFT L-INS-i alignment, trimmed alignment, alignment report
step4_features/           pairwise features, site differences, full and ML feature matrices
step5_boruta/             Boruta output, selected features, SMOTE summary, train/test split
step6_model/              cross-validation, final metrics, predictions, baselines, figures
step8_shap/               SHAP values, global importance, Iran-05 case study, PDB mapping
app/                      Streamlit application, utilities and model artifacts
docs/provenance/          report URLs, OCR output and extracted text behind the dataset
```

Each `src/` script writes into the matching `stepN_*` directory, so the outputs of the published
run sit beside the code that produced them and can be compared against a fresh run.

## Reproducing the analysis

```bash
git clone https://github.com/DanialZohourian/fmdv-serotypeA-vaccine-matching.git
cd fmdv-serotypeA-vaccine-matching
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

MAFFT must be installed separately (https://mafft.cbrc.jp/alignment/software/). The scripts find
it on the PATH; set `MAFFT_PATH` to override, or `MAFFT_BASH` when using the Windows mafft-win
all-in-one build.

Then run, in order:

```bash
python src/01_build_master_table.py        # curated r1 master table -> dataset/
python src/02_build_sequences.py           # VP1 extraction, translation, QC, registry
python src/03_align_vp1.py                 # MAFFT L-INS-i alignment and trimming
python src/04a_features_distance.py        # pairwise Poisson-corrected distance
python src/04b_features_sites.py           # site-difference indicators
python src/04c_features_glycosylation.py   # N-glycosylation sequon features
python src/04d_features_assemble.py        # assemble the pair feature table
python src/04e_features_matrix.py          # full and ML-ready feature matrices
python src/04f_features_report.py          # feature engineering report
python src/05_boruta_selection.py          # stratified split, SMOTE on train only, Boruta
python src/06a_hyperparameter_search.py    # cross-validated hyperparameter search
python src/06b_train_final_model.py        # final model, test metrics, baselines
python src/06c_all96_comparison.py         # 96-feature model for DeLong's test
python src/07_shap_analysis.py             # SHAP values and interpretation
python src/08_pdb_mapping.py               # alignment-to-structure mapping (fetches 4GH4)
python src/09_export_artifacts.py          # export artifacts into app/model_artifacts
python src/10_verify_artifacts.py          # confirm artifacts reproduce the reported metrics
```

Every step is seeded (`random_state = 42`) and reproduces the reported metrics exactly.
The upstream collection utilities used to locate and download the source reports and GenBank
records (web crawling, PDF text extraction, OCR of scanned reports, GenBank searches) were
one-off exploratory scripts and are not included; their curated outputs are in `dataset/` and
`step2_sequences/`, with the search audit trail in `dataset/genbank_search_log.csv` and
`dataset/master_audit.txt`. Exploratory model comparisons (LightGBM, XGBoost) are likewise not
included; their cross-validation results are retained in `step6_model/` for transparency.

## Running the application locally

```bash
streamlit run app/app.py
```

The app loads `app/model_artifacts/`. On Streamlit Community Cloud, set the main file path to
`app/app.py` and **choose Python 3.12 under Advanced settings before deploying** — several
dependencies have no wheels for newer Python releases, and the version cannot be changed after
deployment without deleting and redeploying the app. `packages.txt` installs MAFFT and the root
`requirements.txt` installs the Python dependencies.

## Licence

Code is released under the MIT Licence (`LICENSE`); data under CC BY 4.0 (`LICENSE-data`).

## Citation

Please cite both the paper and the archived release. See `CITATION.cff` for machine-readable
metadata.

**Archive:** Norouzi S, Zohourian D. *FMDV serotype A vaccine matching: machine learning
prediction from VP1 sequences.* Zenodo; 2026. https://doi.org/10.5281/zenodo.24053304

The DOI above resolves to the latest version. To cite this exact snapshot, use
https://doi.org/10.5281/zenodo.24053305 (v1).

## Contact

Silena Norouzi, DVM — silenanorouzi@gmail.com
Pasouk Knowledge-Based Company, Tehran, Iran
