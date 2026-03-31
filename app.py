"""
Full-Stack Inferencing Application
====================================================
Streamlit web app showcasing the full ML pipeline:
EDA findings, feature engineering, model results, live predictions,
SHAP explainability, and LLM-powered narrative insights.

Usage:
    streamlit run app.py
"""

import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from autogluon.tabular import TabularPredictor
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from predict import is_raw_data, preprocess_raw_data

# ============================================================
# Constants
# ============================================================
TARGET = "readmitted"
CLASS_LABELS = ["<30", ">30", "NO"]
MODEL_PATH = "ag_models/"
FAST_MODEL = "LightGBMLarge_BAG_L1"
CHARTS = os.path.join(BASE_DIR, "charts")

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA Findings", "Feature Engineering",
     "Model Results", "Live Predictions", "SHAP Explainability",
     "AI Narrative Report"],
)
st.sidebar.divider()
_env_key = os.getenv("OPENAI_API_KEY", "")
if _env_key:
    api_key = _env_key
    st.sidebar.success("OpenAI API Key loaded from .env")
else:
    api_key = st.sidebar.text_input(
        "OpenAI API Key", type="password",
        help="Create a .env file with OPENAI_API_KEY=sk-... or enter here",
    )


# ============================================================
# Cached loaders
# ============================================================
@st.cache_resource
def load_autogluon_model():
    return TabularPredictor.load(MODEL_PATH)


@st.cache_resource
def train_shap_proxy():
    train_df = pd.read_csv("train_data.csv")
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]
    X_enc = X.copy()
    encoders = {}
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encoders[col] = le
    target_le = LabelEncoder()
    target_le.fit(CLASS_LABELS)
    y_enc = target_le.transform(y)
    lgb = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=128,
        feature_fraction=0.9, min_data_in_leaf=3,
        class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1,
    )
    lgb.fit(X_enc, y_enc)
    explainer = shap.TreeExplainer(lgb)
    return lgb, explainer, encoders, target_le, cat_cols


# ============================================================
# LLM helper
# ============================================================
def call_llm(prompt, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    system_msg = (
        "You are a clinical data scientist. Follow these rules strictly:\n\n"
        "CLASS DEFINITIONS (never confuse):\n"
        "- <30 = readmitted WITHIN 30 days (high risk, minority class 11.2%)\n"
        "- >30 = readmitted AFTER 30 days (moderate risk, 34.9%)\n"
        "- NO = NOT readmitted (majority class, 53.9%)\n\n"
        "CLINICAL INTERPRETATION (critical):\n"
        "- Higher num_prior_encounters = HIGHER readmission risk (sicker patients "
        "with repeated hospitalizations), NOT better care. EDA confirmed: <30 patients "
        "avg 3.9 encounters vs NO patients avg 1.4.\n"
        "- Higher number_inpatient = HIGHER readmission risk (more severe cases), "
        "NOT better monitoring.\n"
        "- A high SHAP value for the NO class means the feature strongly pushes "
        "predictions TOWARD non-readmission when the feature value is LOW.\n"
        "- Never claim that more encounters or more inpatient visits reduce readmission risk."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1500,
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PAGE: Overview
# ============================================================
if page == "Overview":
    readme_path = os.path.join(BASE_DIR, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            readme_content = f.read()
        st.markdown(readme_content, unsafe_allow_html=True)
    else:
        st.error("README.md not found.")


# ============================================================
# PAGE: EDA Findings
# ============================================================
elif page == "EDA Findings":
    st.title("Exploratory Data Analysis")
    st.markdown("Key findings from the raw dataset (101,766 encounters x 50 columns) that shaped our preprocessing pipeline.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Target Distribution", "Missing Values", "Patient Encounters",
        "Diagnosis Codes", "Medications"
    ])

    with tab1:
        st.image(os.path.join(CHARTS, "target_distribution.png"), use_container_width=True)
        st.markdown("""
        **Finding:** Moderate class imbalance — NO (53.9%), >30 (34.9%), <30 (11.2%).
        The 4.8:1 majority-to-minority ratio means accuracy alone is misleading.

        **Action:** Applied inverse-frequency sample weighting + optimized F1 Macro metric.
        """)

    with tab2:
        st.image(os.path.join(CHARTS, "missing_values.png"), use_container_width=True)
        st.markdown("""
        **Finding:** `weight` (97%), `max_glu_serum` (95%), `A1Cresult` (83%) have extreme missingness.

        **Action:**
        - Dropped `weight` and `payer_code` (no usable signal)
        - Kept `A1Cresult` and `max_glu_serum` — "Not Tested" itself is clinically informative (Strack 2014)
        """)

    with tab3:
        st.image(os.path.join(CHARTS, "patient_encounters.png"), use_container_width=True)
        st.markdown("""
        **Finding (Strongest Signal):** Patients readmitted within 30 days averaged **3.9 prior encounters**
        vs 1.4 for non-readmitted patients. Nearly 3x difference.

        **Action:** Engineered `num_prior_encounters` from `patient_nbr`.
        Later confirmed as #1 predictor by both permutation importance and SHAP.
        """)

    with tab4:
        st.image(os.path.join(CHARTS, "diagnosis_grouped.png"), use_container_width=True)
        st.markdown("""
        **Finding:** 700+ unique ICD-9 codes create extreme cardinality.

        **Action:** Grouped into 9 clinical categories per Strack et al. (2014):
        Diabetes, Circulatory, Respiratory, Digestive, Genitourinary, Injury, Musculoskeletal, Neoplasms, Other.
        """)

    with tab5:
        st.image(os.path.join(CHARTS, "medication_no_pct.png"), use_container_width=True)
        st.markdown("""
        **Finding:** 15 of 23 medication columns exceed 99% "No". Only `insulin` (~46%) and `metformin` (~80%) show meaningful variation.

        **Action:** Aggregated 23 columns into 3 summary features: `num_active_meds`, `num_med_changes`, `num_steady_meds`. Kept `insulin` individually.
        """)

    st.divider()
    st.subheader("Advanced EDA")

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(CHARTS, "pca_analysis.png"), use_container_width=True)
        st.caption("PCA: Classes overlap significantly — confirming this is an inherently difficult problem.")
    with col2:
        st.image(os.path.join(CHARTS, "hypothesis_test_effects.png"), use_container_width=True)
        st.caption("Kruskal-Wallis: All features statistically significant (p < 0.05). number_inpatient has the largest effect size.")


# ============================================================
# PAGE: Feature Engineering
# ============================================================
elif page == "Feature Engineering":
    st.title("Feature Engineering Pipeline")
    st.markdown("Every preprocessing decision is grounded in EDA findings and published literature.")

    st.markdown("""
    ```
    Original Data (101,766 x 50)
      ├── ID mapping (admission_type, discharge_disposition, admission_source)
      ├── Remove deceased/hospice patients (~2,500 rows)
      ├── Clean: '?' → NaN, remove invalid gender (3 rows)
      ├── Feature Engineering (see below)
      ├── Stratified 5% unseen sample → unseen_data.csv
      └── 80/20 stratified split → train / test
    Final Data (~94,000 x 31)
    ```
    """)

    st.markdown("### EDA Finding → Engineering Decision")
    st.markdown("""
    | # | EDA Finding | Engineering Decision |
    |---|-------------|---------------------|
    | 1 | `weight` (97% missing), `payer_code` (40% missing) | **Drop** — no usable signal |
    | 2 | `A1Cresult` (83% missing), `max_glu_serum` (95% missing) | **Keep** — "Not Tested" is clinically informative |
    | 3 | `diag_1/2/3`: 700+ unique ICD-9 codes | **Group** into 9 clinical categories (Strack 2014) |
    | 4 | 23 medication columns, 15 exceed 99% "No" | **Aggregate** into 3 summary features; keep `insulin` |
    | 5 | Patients with multiple encounters → higher readmission | **Engineer** `num_prior_encounters` from `patient_nbr` |
    | 6 | 2,423 deceased/hospice patients all labeled "NO" | **Remove** — cannot be readmitted (data leakage) |
    | 7 | Target imbalance: 4.8:1 (NO vs <30) | **Apply** inverse-frequency sample weights + F1 Macro |
    | 8 | `age` stored as "[70-80)" strings | **Convert** to numeric midpoints |
    | 9 | `medical_specialty`: 73 categories | **Group** into top 10 + Other + Unknown |
    """)

    st.markdown("### Interaction Features")
    st.markdown("""
    | Feature | Formula | Rationale |
    |---------|---------|-----------|
    | `total_visits` | outpatient + emergency + inpatient | Composite healthcare utilization |
    | `med_per_day` | num_medications / time_in_hospital | Medication intensity proxy |
    | `inpatient_plus_time` | number_inpatient + time_in_hospital | Severity indicator |
    """)

    st.markdown("### Literature References")
    st.markdown("""
    - **Strack et al. (2014)** — ICD-9 grouping, A1C study, deceased removal
    - **Garcia-Mosquera et al. (2025)** — `num_prior_encounters` as #1 predictor
    - **Emi-Johnson et al. (2025)** — `total_visits` composite feature
    """)


# ============================================================
# PAGE: Model Results
# ============================================================
elif page == "Model Results":
    st.title("Model Training Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("F1 Macro", "0.586")
    col2.metric("Accuracy", "69.9%")
    col3.metric("AUC-ROC", "0.782")

    st.markdown("### Training Configuration")
    st.markdown("""
    | Setting | Value |
    |---------|-------|
    | Framework | AutoGluon TabularPredictor 1.5.0 |
    | Preset | `best_quality` (multi-level stacking + bagging) |
    | Eval Metric | `f1_macro` |
    | Time Limit | 3,600 seconds (1 hour) |
    | Sample Weight | Inverse class frequency (\<30: 2.93x, >30: 0.93x, NO: 0.63x) |
    """)

    st.markdown("### Per-Class Performance")
    st.markdown("""
    | Class | Precision | Recall | F1 Score | Support |
    |-------|-----------|--------|----------|---------|
    | <30 (within 30 days) | 0.354 | 0.303 | 0.327 | 2,150 |
    | >30 (after 30 days) | 0.646 | 0.572 | 0.607 | 6,745 |
    | NO (not readmitted) | 0.786 | 0.871 | 0.826 | 9,980 |
    """)

    st.image(os.path.join(CHARTS, "confusion_matrix.png"), use_container_width=True)

    st.markdown("### Leaderboard (Top 10)")
    try:
        lb = pd.read_csv("leaderboard.csv")
        st.dataframe(lb.head(10), use_container_width=True)
    except FileNotFoundError:
        st.info("leaderboard.csv not found.")


# ============================================================
# PAGE: Live Predictions
# ============================================================
elif page == "Live Predictions":
    st.title("Live Predictions")
    st.info(
        "**For Graders:** Upload `unseen_data.csv` (or click 'Use sample data') to instantly view "
        "predictions with confidence scores. This is an alternative to running `python predict.py` in the terminal."
    )
    st.markdown("Upload patient data (CSV) to get predictions with confidence scores.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload CSV (raw or preprocessed)", type=["csv"])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("Use sample data (unseen_data.csv)")

    data = None
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success(f"Uploaded: {data.shape[0]} rows x {data.shape[1]} columns")
    elif use_sample:
        data = pd.read_csv("unseen_data.csv")
        st.success(f"Loaded unseen_data.csv: {data.shape[0]} rows x {data.shape[1]} columns")

    if data is not None:
        with st.expander("Preview data", expanded=False):
            st.dataframe(data.head(20), use_container_width=True)

        if is_raw_data(data):
            with st.spinner("Raw data detected — applying feature engineering..."):
                data = preprocess_raw_data(data)
            st.info(f"Preprocessed: {data.shape[0]} rows x {data.shape[1]} columns")

        has_target = TARGET in data.columns
        X = data.drop(columns=[TARGET], errors="ignore")
        if "sample_weight" in X.columns:
            X = X.drop(columns=["sample_weight"])

        with st.spinner("Running predictions..."):
            predictor = load_autogluon_model()
            available = predictor.model_names()
            use_model = FAST_MODEL if FAST_MODEL in available else None
            predictions = predictor.predict(X, model=use_model)
            probabilities = predictor.predict_proba(X, model=use_model)

        # Store in session state for other pages
        st.session_state["predictions"] = predictions
        st.session_state["probabilities"] = probabilities
        st.session_state["X"] = X
        st.session_state["has_target"] = has_target
        if has_target:
            st.session_state["y_true"] = data[TARGET]

        results = pd.DataFrame()
        results["Predicted"] = predictions.values
        for col in probabilities.columns:
            results[f"Prob({col})"] = (probabilities[col].values * 100).round(1)
        if has_target:
            results["Actual"] = data[TARGET].values
            results["Correct"] = results["Predicted"] == results["Actual"]

        pred_counts = predictions.value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("NO", f"{pred_counts.get('NO', 0):,}",
                     f"{pred_counts.get('NO', 0)/len(predictions)*100:.1f}%")
        col2.metric(">30", f"{pred_counts.get('>30', 0):,}",
                     f"{pred_counts.get('>30', 0)/len(predictions)*100:.1f}%")
        col3.metric("<30", f"{pred_counts.get('<30', 0):,}",
                     f"{pred_counts.get('<30', 0)/len(predictions)*100:.1f}%")

        if has_target:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(data[TARGET], predictions)
            f1 = f1_score(data[TARGET], predictions, average="macro", zero_division=0)
            st.markdown(f"**Evaluation:** Accuracy = {acc:.4f} | F1 Macro = {f1:.4f}")

        st.dataframe(results.head(100), use_container_width=True)
        st.download_button("Download Predictions (CSV)",
                           results.to_csv(index=False), "predictions.csv", "text/csv")


# ============================================================
# PAGE: SHAP Explainability
# ============================================================
elif page == "SHAP Explainability":
    st.title("SHAP Model Explainability")

    tab1, tab2, tab3 = st.tabs(["Global Importance", "Individual Explanations", "Dependence Plots"])

    with tab1:
        st.markdown("### Global Feature Importance (SHAP)")
        st.image(os.path.join(CHARTS, "shap_class_comparison.png"), use_container_width=True)
        st.markdown("""
        `num_prior_encounters` dominates across all three classes, followed by `number_inpatient`
        and `discharge_disposition_id`. This confirms that **prior healthcare utilization** is the
        strongest predictor of readmission.
        """)

        st.markdown("### Permutation Importance (AutoGluon)")
        st.image(os.path.join(CHARTS, "autogluon_feature_importance.png"), use_container_width=True)

    with tab2:
        st.markdown("### Individual Patient Explanations (Waterfall)")

        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            st.image(os.path.join(CHARTS, "shap_waterfall_lt30.png"), use_container_width=True)
            st.caption("Patient predicted <30: num_prior_encounters (+0.55) is the dominant driver")
        with wcol2:
            st.image(os.path.join(CHARTS, "shap_waterfall_gt30.png"), use_container_width=True)
            st.caption("Patient predicted >30: multiple moderate contributors")
        with wcol3:
            st.image(os.path.join(CHARTS, "shap_waterfall_NO.png"), use_container_width=True)
            st.caption("Patient predicted NO: num_prior_encounters (+1.22) strongly favors non-readmission")

        # Live SHAP if predictions exist
        if "X" in st.session_state:
            st.divider()
            st.markdown("### Live SHAP for Your Data")
            top_n = st.slider("Number of patients to analyze", 1, 20, 5)

            with st.spinner("Computing SHAP values..."):
                lgb, explainer, encoders, target_le, cat_cols = train_shap_proxy()
                X = st.session_state["X"]
                predictions = st.session_state["predictions"]

                X_enc = X.copy()
                for col in cat_cols:
                    if col in X_enc.columns and col in encoders:
                        le = encoders[col]
                        X_enc[col] = X_enc[col].astype(str).map(
                            {v: i for i, v in enumerate(le.classes_)}
                        ).fillna(0).astype(int)

                sample_size = min(top_n, len(X_enc))
                X_shap = X_enc.iloc[:sample_size]
                shap_raw = explainer.shap_values(X_shap)
                if shap_raw.ndim == 3:
                    shap_values = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
                else:
                    shap_values = shap_raw

            patient_idx = st.selectbox(
                "Select patient",
                range(sample_size),
                format_func=lambda i: f"Patient {i} — Predicted: {predictions.iloc[i]}",
            )

            pred_class = predictions.iloc[patient_idx]
            cls_idx = list(target_le.classes_).index(pred_class)

            patient_data = X_enc.iloc[[patient_idx]]
            p_shap_raw = explainer.shap_values(patient_data)
            if p_shap_raw.ndim == 3:
                p_shap = p_shap_raw[0, :, cls_idx]
            else:
                p_shap = p_shap_raw[cls_idx][0]

            ev = explainer.expected_value
            base_val = ev[cls_idx] if isinstance(ev, (list, tuple)) else (ev[cls_idx] if hasattr(ev, "__len__") else ev)

            explanation = shap.Explanation(
                values=p_shap, base_values=base_val,
                data=patient_data.iloc[0].values,
                feature_names=list(X_enc.columns),
            )
            fig_w = plt.figure(figsize=(12, 6))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.title(f'Why predicted as "{pred_class}"', fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_w)
            plt.close()

    with tab3:
        st.markdown("### SHAP Dependence Plots (Top 3 Features for <30 Class)")
        st.image(os.path.join(CHARTS, "shap_dependence_plots.png"), use_container_width=True)
        st.markdown("""
        Dependence plots reveal **non-linear relationships** and **interaction effects**.
        For example, readmission risk may jump sharply after a certain number of prior encounters.
        """)

        st.markdown("### SHAP Beeswarm (All Classes)")
        st.image(os.path.join(CHARTS, "shap_summary_beeswarm.png"), use_container_width=True)


# ============================================================
# PAGE: AI Narrative Report
# ============================================================
elif page == "AI Narrative Report":
    st.title("AI-Generated Narrative Report")
    st.markdown("Use GPT-4o to generate plain-English explanations of model predictions and SHAP analysis.")

    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar.")
    else:
        report_type = st.selectbox("Report type", [
            "Full Analysis Summary",
            "EDA Findings Narrative",
            "SHAP Explanation Narrative",
            "Prediction Results Narrative",
        ])

        if st.button("Generate Report", type="primary"):
            try:
                with st.spinner("GPT-4o is generating the report..."):
                    if report_type == "Full Analysis Summary":
                        prompt = """You are a clinical data scientist. Write a comprehensive narrative report (4-5 paragraphs) about a diabetes hospital readmission prediction project:

## Dataset
- 101,766 encounters from 130 US hospitals (1999-2008), 50 features
- 3-class: NO (53.9%), >30 (34.9%), <30 (11.2%)

## Key EDA Findings
- Patient encounter frequency is the strongest signal: <30 patients avg 3.9 encounters vs 1.4 for NO
- 23 medication columns are 80-99%+ "No" — aggregated into 3 summary features
- 700+ ICD-9 codes grouped into 9 clinical categories (Strack 2014)
- 2,423 deceased/hospice patients removed (data leakage — cannot be readmitted)
- A1Cresult (83% missing) kept as "Not Tested" category (clinically informative)

## Model Performance
- AutoGluon WeightedEnsemble_L3: F1 Macro 0.586, Accuracy 0.699, AUC-ROC 0.781
- Outperforms Garcia-Mosquera Transformer (Acc 46.5%, AUC 0.619) on same dataset

## SHAP Analysis
- Top features: num_prior_encounters (dominant), number_inpatient, discharge_disposition_id
- num_prior_encounters SHAP value for NO class: 1.09 (extremely strong)
- Non-linear effects visible in dependence plots

Write a professional narrative report covering findings, methodology strengths, clinical implications, and limitations."""

                    elif report_type == "EDA Findings Narrative":
                        prompt = """Write a 3-paragraph narrative about EDA findings on a diabetes readmission dataset (101,766 encounters, 50 features):
- Class imbalance: 4.8:1 (NO vs <30)
- weight column 97% missing, A1Cresult 83% missing (but "Not Tested" is clinically meaningful)
- 700+ unique ICD-9 diagnosis codes
- 23 medication columns mostly 99%+ "No"
- Strongest finding: patients readmitted <30 days had avg 3.9 encounters vs 1.4 for NO
- 2,423 deceased/hospice patients all labeled "NO" (data leakage)
- PCA/t-SNE show heavy class overlap — inherently difficult problem

Explain each finding and why it matters for modeling."""

                    elif report_type == "SHAP Explanation Narrative":
                        prompt = """Write a 3-paragraph narrative explaining SHAP analysis results for a diabetes readmission model:
- Proxy LightGBM used (F1 0.575) since AutoGluon ensemble not SHAP-compatible
- Top global features by mean |SHAP|:
  1. num_prior_encounters: <30=0.516, >30=0.292, NO=1.092
  2. number_inpatient: <30=0.098, >30=0.054, NO=0.267
  3. discharge_disposition_id: <30=0.153, >30=0.098, NO=0.042
- Waterfall example: <30 patient driven by num_prior_encounters (+0.551)
- Dependence plots show non-linear threshold effects

Explain what these mean clinically and how they could inform interventions."""

                    else:  # Prediction Results
                        pred_info = "No predictions loaded yet. Describe a hypothetical prediction scenario."
                        if "predictions" in st.session_state:
                            preds = st.session_state["predictions"]
                            pc = preds.value_counts()
                            pred_info = f"Total: {len(preds)}, NO: {pc.get('NO',0)}, >30: {pc.get('>30',0)}, <30: {pc.get('<30',0)}"
                        prompt = f"""Write a 3-paragraph narrative about prediction results for a diabetes readmission model:
Prediction results: {pred_info}
Model: AutoGluon ensemble, F1 Macro 0.586, Accuracy 0.699
Explain the distribution, what it means clinically, and recommended next steps."""

                    narrative = call_llm(prompt, api_key)

                st.markdown(narrative)
                st.download_button("Download Report (.md)", narrative, "ai_narrative_report.md", "text/markdown")

            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================
# Footer
# ============================================================
st.sidebar.divider()
st.sidebar.caption(
    "AutoGluon WeightedEnsemble_L3 | F1 0.586 | Acc 0.699\n\n"
    "SHAP via proxy LightGBM | Narrative via GPT-4o"
)
