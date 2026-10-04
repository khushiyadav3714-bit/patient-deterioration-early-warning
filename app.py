"""
Patient Deterioration Early Warning System
Machine Learning-Based Clinical Risk Screening
------------------------------------------------------------------
Streamlit Application for College ML Assessment & Technical Interview Demonstration
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Import prediction logic
from src.predict import DeteriorationPredictor

# Page Configuration
st.set_page_config(
    page_title="Patient Deterioration Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean clinical dashboard aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3182ce;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 15px;
    }
    .risk-high {
        background-color: #fff5f5;
        border: 2px solid #e53e3e;
        border-radius: 10px;
        padding: 18px;
        color: #9b2c2c;
    }
    .risk-moderate {
        background-color: #fffaf0;
        border: 2px solid #dd6b20;
        border-radius: 10px;
        padding: 18px;
        color: #9c4221;
    }
    .risk-low {
        background-color: #f0fff4;
        border: 2px solid #38a169;
        border-radius: 10px;
        padding: 18px;
        color: #22543d;
    }
    .disclaimer-box {
        background-color: #edf2f7;
        border-left: 4px solid #718096;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #4a5568;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Cache the predictor instance so models are loaded once."""
    return DeteriorationPredictor()


@st.cache_data
def load_results():
    """Load model results from CSV."""
    if os.path.exists("model_results.csv"):
        return pd.read_csv("model_results.csv")
    return None


predictor = load_predictor()
results_df = load_results()

# App Header
st.markdown('<div class="main-title">🏥 Patient Deterioration Early Warning System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning-Based Clinical Risk Screening</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Patient Risk Assessment", "📊 Comparative ML Analysis", "📖 Clinical Context & Dataset"])

# ==============================================================================
# TAB 1: PATIENT RISK ASSESSMENT
# ==============================================================================
with tab1:
    st.markdown("### Enter Patient Vital Signs")
    st.caption("Provide bedside physiological observations to compute early risk of clinical deterioration.")

    # Quick Presets for Demo / Viva presentation
    col_demo1, col_demo2 = st.columns([1, 3])
    with col_demo1:
        preset = st.selectbox(
            "Quick Demo Profiles:",
            ["Custom Input", "Profile A: Stable Ward Patient", "Profile B: Acute Deterioration (SIRS/Sepsis)", "Profile C: Borderline Patient"]
        )

    # Preset defaults
    if preset == "Profile A: Stable Ward Patient":
        def_age, def_hr, def_rr, def_spo2, def_sbp, def_dbp, def_temp = 62.0, 72.0, 15.0, 98.0, 120.0, 78.0, 36.8
    elif preset == "Profile B: Acute Deterioration (SIRS/Sepsis)":
        def_age, def_hr, def_rr, def_spo2, def_sbp, def_dbp, def_temp = 72.0, 118.0, 26.0, 91.0, 86.0, 52.0, 38.6
    elif preset == "Profile C: Borderline Patient":
        def_age, def_hr, def_rr, def_spo2, def_sbp, def_dbp, def_temp = 68.0, 96.0, 21.0, 94.0, 105.0, 64.0, 37.6
    else:
        def_age, def_hr, def_rr, def_spo2, def_sbp, def_dbp, def_temp = 55.0, 80.0, 18.0, 97.0, 122.0, 76.0, 37.0

    st.markdown("---")

    # Form with 2 balanced columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Demographics & Cardiac Function")
        age = st.slider("Patient Age (Years)", min_value=18, max_value=95, value=int(def_age), step=1)
        heart_rate = st.slider("Heart Rate (bpm)", min_value=35, max_value=180, value=int(def_hr), step=1,
                               help="Normal resting rate: 60-100 bpm. >100 bpm indicates tachycardia.")
        systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=50, max_value=230, value=int(def_sbp), step=1,
                                help="Target: 90-130 mmHg. <90 mmHg signifies hypotension/shock.")
        diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=35, max_value=140, value=int(def_dbp), step=1,
                                 help="Target: 60-85 mmHg.")

    with col2:
        st.markdown("##### 🫁 Respiratory & Metabolic Profile")
        respiratory_rate = st.slider("Respiratory Rate (breaths/min)", min_value=8, max_value=55, value=int(def_rr), step=1,
                                     help="Normal rate: 12-20 breaths/min. >22 is a hallmark of acute deterioration.")
        spo2 = st.slider("Oxygen Saturation - SpO2 (%)", min_value=70, max_value=100, value=int(def_spo2), step=1,
                         help="Normal SpO2: 95-100%. <92% indicates hypoxia.")
        temperature = st.slider("Body Temperature (°C)", min_value=32.0, max_value=42.0, value=float(def_temp), step=0.1,
                                help="Normal: 36.5 - 37.5°C. >38.0°C denotes fever; <36.0°C denotes hypothermia.")

    patient_payload = {
        'age': float(age),
        'heart_rate': float(heart_rate),
        'respiratory_rate': float(respiratory_rate),
        'spo2': float(spo2),
        'systolic_bp': float(systolic_bp),
        'diastolic_bp': float(diastolic_bp),
        'temperature': float(temperature)
    }

    st.write("")
    analyze_btn = st.button("🔍 Analyze Patient Risk", type="primary", use_container_width=True)

    if analyze_btn or preset != "Custom Input":
        result = predictor.predict(patient_payload)
        risk_cat = result['risk_category']
        prob = result['deterioration_probability']
        concerning = result['concerning_values']

        st.markdown("---")
        st.markdown("### Clinical Risk Assessment Output")

        # Color-coded result card
        if risk_cat == "HIGH RISK":
            st.markdown(f"""
            <div class="risk-high">
                <h3 style="margin-top:0; color:#c53030;">🚨 HIGH DETERIORATION RISK</h3>
                <p style="font-size: 1.15rem; font-weight: 600;">Estimated Deterioration Probability: {prob}%</p>
                <p><strong>Primary Model:</strong> {result['model_name']} | <strong>Action Recommended:</strong> {result['clinical_recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_cat == "MODERATE RISK":
            st.markdown(f"""
            <div class="risk-moderate">
                <h3 style="margin-top:0; color:#c05621;">⚠️ MODERATE DETERIORATION RISK</h3>
                <p style="font-size: 1.15rem; font-weight: 600;">Estimated Deterioration Probability: {prob}%</p>
                <p><strong>Primary Model:</strong> {result['model_name']} | <strong>Action Recommended:</strong> {result['clinical_recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h3 style="margin-top:0; color:#276749;">✅ LOW DETERIORATION RISK</h3>
                <p style="font-size: 1.15rem; font-weight: 600;">Estimated Deterioration Probability: {prob}%</p>
                <p><strong>Primary Model:</strong> {result['model_name']} | <strong>Action Recommended:</strong> {result['clinical_recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        # Progress bar showing probability
        st.markdown(f"**Risk Probability Scale:**")
        st.progress(prob / 100.0)

        # Concerning Vitals List
        st.markdown("#### 🩺 Potentially Concerning Values:")
        if concerning:
            for item in concerning:
                st.error(f"• {item}")
        else:
            st.success("• No acute physiological vital sign alerts detected. All measured parameters fall within stable ward thresholds.")


# ==============================================================================
# TAB 2: COMPARATIVE ML ANALYSIS
# ==============================================================================
with tab2:
    st.markdown("### Comparative Performance Analysis of 5 ML Algorithms")
    st.markdown("""
    This project evaluated **5 standard Scikit-Learn algorithms** on a held-out clinical test cohort of **318 patients**
    (159 deteriorating vs. 159 stable) from the real-world PhysioNet Challenge 2019 dataset.
    """)

    if results_df is not None:
        st.markdown("#### Model Performance Leaderboard")
        # Format table cleanly
        styled_df = results_df.copy()
        if 'Clinical_Score' in styled_df.columns:
            styled_df = styled_df.drop(columns=['Clinical_Score'])

        # Highlight best model
        st.dataframe(
            styled_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], color='#c6f6d5'),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### Visual Performance Comparisons")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if os.path.exists("reports/figures/roc_curves.png"):
            st.image("reports/figures/roc_curves.png", caption="Combined Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
    with col_g2:
        if os.path.exists("reports/figures/confusion_matrices.png"):
            st.image("reports/figures/confusion_matrices.png", caption="Test Set Confusion Matrices Across All 5 Algorithms", use_container_width=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        if os.path.exists("reports/figures/recall_comparison.png"):
            st.image("reports/figures/recall_comparison.png", caption="Clinical Recall (Sensitivity) Comparison", use_container_width=True)
    with col_g4:
        if os.path.exists("reports/figures/feature_importance.png"):
            st.image("reports/figures/feature_importance.png", caption="Feature Importance for Best Model (Random Forest)", use_container_width=True)

    st.info("""
    **Key Clinical Insight:**
    In healthcare early warning systems, **Recall** is critical. A false negative (classifying a deteriorating septic patient as 'Stable')
    leads to delayed medical intervention and avoidable mortality, whereas a false positive merely triggers a harmless bedside nurse review.
    Random Forest and KNN demonstrated the best capability to identify complex non-linear physiological interactions.
    """)


# ==============================================================================
# TAB 3: CLINICAL CONTEXT & DATASET
# ==============================================================================
with tab3:
    st.markdown("### Clinical Background and Dataset Architecture")
    st.markdown("""
    #### 1. Why Patient Deterioration Prediction Matters
    In inpatient hospital wards and intensive care units, rapid physiological decompensation (often triggered by sepsis, respiratory distress, or shock)
    can lead to cardiac arrest or death within hours. Early automated warning based on routine bedside vital signs provides clinicians a vital window for early fluid resuscitation and antibiotics.

    #### 2. Dataset Overview
    - **Dataset Source:** [PhysioNet / Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data](https://physionet.org/content/challenge-2019/1.0.0/)
    - **Original Database:** 38,809 hourly clinical records from intensive care units.
    - **Features Extracted (7 Vital Signs & Demographics):**
      1. `age`: Patient Age (years)
      2. `heart_rate`: Beats per minute (bpm)
      3. `respiratory_rate`: Breaths per minute (tachypnea marker)
      4. `spo2`: Blood Oxygen Saturation (%)
      5. `systolic_bp`: Systolic Blood Pressure (mmHg)
      6. `diastolic_bp`: Diastolic Blood Pressure (mmHg)
      7. `temperature`: Core Body Temperature (°C)
    - **Target Variable:** `deterioration_risk`
      - `0 = Stable` (Normal ward ICU progression)
      - `1 = Deterioration Risk` (Onset of clinical sepsis / organ dysfunction)
    - **Clinical Cohort Balancing:**
      The original ICU data suffered from extreme ~98% class imbalance. To prevent standard ML models from trivializing to a majority-class predictor (which yields zero recall), a 1:1 matched case-control cohort of **1,586 patients** was created, split 80% for training (1,268 patients) and 20% for testing (318 patients) using stratified sampling.

    #### 3. Limitations of the System
    - Evaluates vital signs only; does not incorporate laboratory blood tests (e.g. lactate, WBC count, arterial blood gases).
    - Single time-point cross-sectional screening rather than continuous temporal recurrent tracking.
    - Trained on adult ICU patients (not validated for pediatric cohorts).
    """)

# Mandatory Medical Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>Medical Disclaimer:</strong> This application is an academic machine-learning project developed for educational and assessment demonstration.
    It is not intended to replace professional medical judgment, diagnosis, or treatment. Never disregard professional clinical advice or delay seeking care because of output generated by this tool.
</div>
""", unsafe_allow_html=True)
