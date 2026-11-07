# Files to add to this folder

Copy these from the machine where the model was trained, then delete this file:

- `rf_model.joblib`        trained RandomForestClassifier
- `shap_explainer.joblib`  fitted SHAP TreeExplainer (optional; can be rebuilt from the model)

`model_meta.json`, `selected_features.txt` and `rf_model_summary.txt` are already here.

If a file is larger than 100 MB, GitHub will reject it. A Random Forest of 300 trees should be
well under that; if it is not, reduce it with joblib compression:

    joblib.dump(model, "rf_model.joblib", compress=3)
