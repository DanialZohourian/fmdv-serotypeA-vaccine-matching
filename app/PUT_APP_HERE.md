# Application files to add to this folder

- `app.py`  the Streamlit application

The app loads `../model/rf_model.joblib`, validates an input sequence of 208-220 amino acids,
aligns it to the A22/IRQ/24/64 reference with MAFFT, computes the 23 features listed in
`../model/selected_features.txt`, and returns the match probability, the binary call at
threshold 0.23, and a SHAP waterfall chart.

Streamlit Community Cloud needs the main file path set to `app/app.py` at deployment, and
installs from the repository-root `requirements.txt`.

Delete this file once the app is in place.
