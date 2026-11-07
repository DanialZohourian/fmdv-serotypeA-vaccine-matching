# Machine learning prediction of FMDV serotype A vaccine matching from VP1 sequences

Code, data and trained model for the study:

> Norouzi S, Zohourian D. *Machine learning prediction of vaccine matching for foot-and-mouth disease virus serotype A from VP1 amino acid sequences, with application to the A/ASIA/Iran-05 lineage.* Manuscript submitted, 2026.

A Boruta-selected Random Forest predicts binary vaccine matching (r1 >= 0.3, WOAH threshold) for
foot-and-mouth disease virus serotype A directly from VP1 amino acid sequences. On an independent
held-out test set (n = 38) the model reaches **AUC-ROC 0.843** and **AUC-PR 0.810**, and it separates
matched from unmatched pairs perfectly within the A/ASIA/Iran-05 lineage subgroup (n = 10).

**This tool is a pre-screening aid. It does not replace two-dimensional virus neutralisation testing
(2D-VNT). Confirm any vaccine strain decision serologically.**

## Web application

<!-- Replace with the deployed URL once the app is live -->
A freely accessible web application is available at: **[APP URL]**

Paste a VP1 amino acid sequence (208-220 aa) and a reference antiserum strain; the app returns a
calibrated match probability, a binary call at the pre-specified threshold of 0.23, and a SHAP
waterfall chart showing which positions drove the prediction.

## Dataset

190 unique virus-antiserum pairs derived from 65 field viruses and 15 reference antisera, spanning
14 countries and 22 years (2001-2022), curated from WRLFMD Vaccine Matching Reports. Labels follow
the WOAH threshold: r1 >= 0.3 = match (n = 48, 25.3%); r1 < 0.3 = no match (n = 142, 74.7%).

Of 661 candidate records screened, 413 were excluded (187 non-numeric r1; 171 no public sequence;
66 non-2D-VNT assay; 38 censored reporting) and a further 58 pairs involved antisera with no
retrievable sequence. `data/raw/master_excluded.csv` records every exclusion and its reason.

## Model

| | |
|---|---|
| Algorithm | RandomForestClassifier (scikit-learn) |
| Features | 23, Boruta-selected from 96 candidates |
| Hyperparameters | n_estimators = 300, max_depth = 10, max_features = "sqrt", min_samples_leaf = 2, class_weight = "balanced" |
| Class balancing | SMOTE (k = 5), applied within cross-validation folds only |
| Decision threshold | 0.23, chosen by maximising F1 on out-of-fold training predictions |
| Random seed | 42 |
| Cross-validated AUC | 0.887 +/- 0.081 |

Features comprise pairwise Poisson-corrected amino acid distance, binary site-difference indicators
at 18 alignment positions, and four N-glycosylation sequon features. Alignment positions follow the
A22/IRQ/24/64 reference and map one-to-one onto PDB entry 4GH4 chain A.

The strongest predictor is **VP1 position 149** in the G-H loop. None of the seven positions
reported as most informative for serotype O was confirmed for serotype A, so the sequence
determinants of cross-neutralisation are serotype-specific and models should not be transferred
between serotypes without retraining.

## Repository layout

```
data/raw/          curated pairs, exclusions, sequence registry, QC report, VP1 FASTA
data/processed/    alignments, PDB coordinate mapping, feature matrices, Boruta output, split
data/results/      model metrics, predictions, baselines, SHAP values, serotype comparison
src/               analysis pipeline (curation -> alignment -> features -> training -> SHAP)
model/             trained model, SHAP explainer, metadata, selected feature list
app/               Streamlit web application
figures/           scripts reproducing Figures 1-5 and the PyMOL colouring script
docs/              SHAP interpretation report
```

## Reproducing the analysis

```bash
git clone https://github.com/[USERNAME]/fmdv-serotypeA-vaccine-matching.git
cd fmdv-serotypeA-vaccine-matching
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

MAFFT must be installed separately and available on the PATH (https://mafft.cbrc.jp/alignment/software/).

Run the pipeline in order:

```bash
python src/01_curate_dataset.py
python src/02_align_sequences.py
python src/03_feature_engineering.py
python src/04_boruta_selection.py
python src/05_train_model.py
python src/06_shap_analysis.py
```

Reproduce the figures:

```bash
python figures/fig1_boruta.py
python figures/fig2_performance.py
python figures/fig3_shap_beeswarm.py
python figures/fig4_vp1_map.py
python figures/fig5_iran05.py
```

Run the application locally:

```bash
streamlit run app/app.py
```

All steps are seeded (`random_state = 42`) and reproduce the reported metrics exactly.

## Sequence data

All VP1 sequences were retrieved from NCBI GenBank. Accession numbers for every field virus and
reference antiserum are listed in `data/raw/sequence_registry.csv`. No new sequences were generated
in this study.

## Licence

Code (`src/`, `app/`, `figures/`) is released under the MIT Licence — see `LICENSE`.
Data (`data/`) is released under CC BY 4.0 — see `LICENSE-data`.

## Citation

If you use this code or dataset, please cite the paper and the archived release. See `CITATION.cff`.

## Contact

Silena Norouzi, DVM — silenanorouzi@gmail.com
Pasouk Knowledge-Based Company, Tehran, Iran
