# -*- coding: utf-8 -*-
"""PredImmune-FMD: VP1 match prediction for serotype A."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import pandas as pd
import streamlit as st

from utils.sequence_processing import (
    alignment_cache,
    extract_features,
    features_to_array,
    match_shap,
    place_sequences,
    read_fasta,
    validate_sequence,
)

APP_DIR = Path(__file__).resolve().parent
ART = APP_DIR / "model_artifacts"
EXAMPLES = [
    {
        "key": "match",
        "button": "Likely match",
        "virus": "A/IRN/10/2018",
        "serum": "A22/IRQ/24/64",
        "r1": 0.45,
        "detail": "Held-out pair. Laboratory r1 = 0.45.",
    },
    {
        "key": "nomatch",
        "button": "Likely no-match",
        "virus": "A/IRN/5/2006",
        "serum": "A/IRN/1996",
        "r1": 0.09,
        "detail": "Held-out pair. Laboratory r1 = 0.09.",
    },
    {
        "key": "false_match",
        "button": "Known model miss",
        "virus": "A/IRN/23/2018",
        "serum": "A/Iran-05",
        "r1": 0.19,
        "detail": "Laboratory r1 = 0.19. The model called this pair a match on the test set.",
    },
]


def pretty_feature(name):
    if name == "pairwise_aa_distance":
        return "VP1 amino-acid distance"
    if name.startswith("site_"):
        return f"Site {int(name.split('_')[1])} differs"
    return {
        "nglyc_count_diff": "N-glycosylation count difference",
        "nglyc_shared": "Shared N-glycosylation sites",
        "nglyc_virus_only": "Virus-only N-glycosylation sites",
        "nglyc_serum_only": "Serum-only N-glycosylation sites",
    }.get(name, name)


def format_value(name, value):
    if name == "pairwise_aa_distance":
        return f"{value:.4f}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def residue_count(text):
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith(">")]
    return sum(character.isalpha() for character in "".join(lines).upper() if character != "X")


def load_explainer(model):
    """Return a SHAP TreeExplainer for the model.

    The explainer is rebuilt from the model rather than unpickled. Rebuilding costs a fraction
    of a second for a 300-tree forest and avoids depending on a pickle that only loads under the
    exact shap and Python versions it was written with. shap_explainer.pkl is kept in
    model_artifacts/ for reference and is used only if rebuilding fails.
    """
    try:
        import shap

        return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    except Exception:
        return joblib.load(ART / "shap_explainer.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ART / "rf_model.pkl")
    explainer = load_explainer(model)
    meta = json.loads((ART / "model_meta.json").read_text(encoding="utf-8"))
    selected = [line.strip() for line in (ART / "selected_features.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping = pd.read_csv(ART / "alignment_to_pdb_mapping.csv", encoding="utf-8-sig")
    records = read_fasta(ART / "reference_alignment.fasta")
    by_name = {header.split("|")[0]: seq for header, seq in records}
    examples = []
    for spec in EXAMPLES:
        item = dict(spec)
        item["virus_seq"] = by_name[spec["virus"]].replace("-", "")
        item["serum_seq"] = by_name[spec["serum"]].replace("-", "")
        examples.append(item)
    return {
        "model": model,
        "explainer": explainer,
        "meta": meta,
        "selected": selected,
        "mapping": mapping,
        "reference": by_name["A22/IRQ/24/64"],
        "profile": [seq for _header, seq in records],
        "cache": alignment_cache(records),
        "examples": examples,
    }


def load_example(example):
    st.session_state["virus_seq"] = example["virus_seq"]
    st.session_state["serum_seq"] = example["serum_seq"]
    st.session_state["example_note"] = f"{example['virus']}  ×  {example['serum']}. {example['detail']}"
    st.session_state["predict_now"] = True


def clear_sequences():
    st.session_state["virus_seq"] = ""
    st.session_state["serum_seq"] = ""
    st.session_state["example_note"] = ""
    st.session_state["predict_now"] = False


st.set_page_config(page_title="PredImmune-FMD", page_icon="🐄", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px;}
    textarea {font-family: ui-monospace, SFMono-Regular, Consolas, monospace !important; font-size: 0.82rem !important; line-height: 1.45 !important;}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e4dfd4; border-radius: 12px; padding: 0.7rem 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

bundle = load_artifacts()
model = bundle["model"]
explainer = bundle["explainer"]
meta = bundle["meta"]
selected = bundle["selected"]
threshold = float(meta["decision_threshold"])

with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
        Paste two **VP1 protein** sequences: the field virus and the vaccine or antiserum.

        Amino acids only, 208–220 residues. A FASTA header is fine. Nucleotide sequences are not accepted.
        """
    )
    st.divider()
    st.header("Model")
    st.metric("Decision threshold", f"{threshold:.2f}")
    st.caption("A score at or above this value is called a match. The threshold was chosen on the training data.")
    side_left, side_right = st.columns(2)
    side_left.metric("Test AUC", f"{meta['test_auc']:.3f}")
    side_right.metric("Test F1", f"{meta['test_f1_opt']:.3f}")
    side_left.metric("Sensitivity", f"{meta['test_sensitivity']:.2f}")
    side_right.metric("Specificity", f"{meta['test_specificity']:.2f}")
    pdb_name = bundle["mapping"]["pdb_id"].iloc[0] if "pdb_id" in bundle["mapping"].columns else "4GH4"
    st.caption(
        f"{meta['model_type']} · serotype {meta['serotype']} · trained {meta['date_trained']}. "
        f"Alignment reference {meta['reference_strain']}. Structure {pdb_name}, chain A. "
        f"{meta['paper_reference']}."
    )

st.title("PredImmune-FMD")
st.markdown(
    "Predict whether a **serotype A** field virus is likely to match a vaccine strain, using only the VP1 protein. "
    "A match here means the model expects a virus-neutralization r1 of at least 0.3."
)

st.subheader("Try a real pair")
example_columns = st.columns(3)
for column, example in zip(example_columns, bundle["examples"]):
    with column:
        with st.container(border=True):
            st.markdown(f"**{example['button']}**")
            st.caption(f"{example['virus']} against {example['serum']}")
            st.caption(f"Laboratory r1 = {example['r1']:.2f}")
            st.button("Use this pair", key=f"ex_{example['key']}", width="stretch", on_click=load_example, args=(example,))

if st.session_state.get("example_note"):
    st.info(st.session_state["example_note"])

st.subheader("Sequences")
virus_col, serum_col = st.columns(2)
with virus_col:
    virus_input = st.text_area(
        "Field virus VP1",
        height=190,
        key="virus_seq",
        placeholder="Paste the amino-acid sequence, with or without a FASTA header.",
    )
    st.caption(f"{residue_count(virus_input)} amino acids, excluding X")
with serum_col:
    serum_input = st.text_area(
        "Vaccine or antiserum VP1",
        height=190,
        key="serum_seq",
        placeholder="Paste the vaccine strain or the serum strain.",
    )
    st.caption(f"{residue_count(serum_input)} amino acids, excluding X")

predict_col, clear_col = st.columns([3, 1])
predict_clicked = predict_col.button("Predict match", type="primary", width="stretch")
clear_col.button("Clear", width="stretch", on_click=clear_sequences)
run_prediction = predict_clicked or st.session_state.pop("predict_now", False)

if run_prediction:
    virus_ok, virus_seq = validate_sequence(virus_input)
    serum_ok, serum_seq = validate_sequence(serum_input)
    if not virus_ok:
        st.error(f"Field virus: {virus_seq}")
        st.stop()
    if not serum_ok:
        st.error(f"Vaccine or antiserum: {serum_seq}")
        st.stop()

    with st.spinner("Aligning to the reference VP1 and scoring the pair. A new sequence can take about half a minute."):
        try:
            aligned, align_note = place_sequences(
                [virus_seq, serum_seq], bundle["reference"], bundle["cache"], bundle["profile"]
            )
            features = extract_features(aligned[0], aligned[1], selected)
            matrix = features_to_array(features, selected)
            probability = float(model.predict_proba(matrix)[0, 1])
            shap_row, base_value = match_shap(explainer, matrix)
        except (ValueError, KeyError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

    is_match = probability >= threshold
    if align_note:
        st.info(align_note)

    st.subheader("Result")
    with st.container(border=True):
        if is_match:
            st.markdown("### Likely match")
            st.markdown("The score is at or above the decision threshold. The model expects r1 ≥ 0.3.")
        else:
            st.markdown("### Likely no-match")
            st.markdown("The score is below the decision threshold. The model expects r1 < 0.3.")
        score_col, call_col, distance_col = st.columns(3)
        score_col.metric("Match probability", f"{probability:.3f}")
        call_col.metric("Call", "Match" if is_match else "No-match")
        distance_col.metric("VP1 distance", f"{features['pairwise_aa_distance']:.4f}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"The marker runs from 0 to 1. The decision threshold is {threshold:.2f}.")
        st.caption("This is a research prediction. It does not replace a virus-neutralization test.")

    st.subheader("Why this call")
    st.caption(f"Starting score {base_value:.3f}. Red raises the match probability. Blue lowers it.")
    ranking = pd.DataFrame({
        "feature": selected,
        "label": [pretty_feature(feature) for feature in selected],
        "shap": shap_row,
        "value": [features[feature] for feature in selected],
    }).sort_values("shap", key=abs, ascending=False).head(12)
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")
    bar_colors = ["#B23A48" if value > 0 else "#1F4E79" for value in ranking["shap"]]
    axis.barh(range(len(ranking)), ranking["shap"].iloc[::-1], color=bar_colors[::-1], height=0.72)
    axis.set_yticks(range(len(ranking)))
    axis.set_yticklabels(
        [f"{row.label}  ({format_value(row.feature, row.value)})" for row in ranking.iloc[::-1].itertuples()],
        fontsize=9,
    )
    axis.axvline(0, color="#1C2833", linewidth=0.8)
    axis.set_xlabel("Change in match probability")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(handles=[
        mpatches.Patch(color="#B23A48", label="Toward match"),
        mpatches.Patch(color="#1F4E79", label="Toward no-match"),
    ], frameon=False, fontsize=9)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    with st.expander("All 23 features"):
        detail = pd.DataFrame([
            {
                "Feature": pretty_feature(feature),
                "Code": feature,
                "Value": round(features[feature], 4),
                "SHAP": round(float(shap_row[index]), 5),
                "Effect": "Toward match" if shap_row[index] > 0 else "Toward no-match",
            }
            for index, feature in enumerate(selected)
        ]).sort_values("SHAP", key=abs, ascending=False)
        st.dataframe(detail, width="stretch", hide_index=True)

    result = pd.DataFrame([{
        "virus_length": len(virus_seq.replace("X", "")),
        "serum_length": len(serum_seq.replace("X", "")),
        "prediction": "Match" if is_match else "No-match",
        "probability_match": round(probability, 4),
        "threshold": threshold,
        **{feature: round(features[feature], 5) for feature in selected},
    }])
    st.download_button(
        "Download this result (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="predimmune_fmd_result.csv",
        mime="text/csv",
    )

    with st.expander("Training set behind this score"):
        st.markdown(
            f"The forest was fit on **{meta['n_train_pairs']}** pairs "
            f"({meta['train_match']} match, {meta['train_nomatch']} no-match) before SMOTE. "
            f"The full computable set has **{meta['n_dataset_pairs']}** pairs "
            f"({meta['dataset_match']} match, {meta['dataset_nomatch']} no-match). "
            f"Held-out test AUC-ROC is **{meta['test_auc']:.4f}** "
            f"(precision-recall AUC {meta['test_pr_auc']:.4f})."
        )
