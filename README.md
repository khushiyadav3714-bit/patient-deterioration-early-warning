# Comparative Performance Analysis of Machine Learning Algorithms for Early Prediction of Patient Deterioration

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, reproducible healthcare machine learning project designed for academic assessment and technical interview discussion. This project compares five foundational machine learning algorithms on real-world clinical ICU data to predict early physiological deterioration using routine bedside vital signs.

---

## Table of Contents
1. [Project Overview & Title](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objective](#objective)
4. [Why Patient Deterioration Prediction Matters](#why-patient-deterioration-prediction-matters)
5. [Dataset Details](#dataset-details)
6. [Features Used](#features-used)
7. [Target Variable](#target-variable)
8. [Algorithms Compared](#algorithms-compared)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Project Workflow](#project-workflow)
11. [Project Directory Structure](#project-directory-structure)
12. [Installation Steps](#installation-steps)
13. [How to Run the Project](#how-to-run-the-project)
14. [Model Comparison Results](#model-comparison-results)
15. [Best Model Selection](#best-model-selection)
16. [Screenshots Section](#screenshots-section)
17. [Limitations](#limitations)
18. [Future Enhancements](#future-enhancements)
19. [Medical Disclaimer](#medical-disclaimer)

---

## 1. Project Overview
- **Title:** Comparative Performance Analysis of Machine Learning Algorithms for Early Prediction of Patient Deterioration
- **Domain:** Healthcare AI / Clinical Informatics / Predictive Monitoring
- **Scope:** Binary classification on bedside vital signs using standard `scikit-learn` algorithms and an interactive `Streamlit` clinical decision support dashboard.

---

## 2. Problem Statement
In intensive care units and general hospital wards, inpatients can decompensate rapidly due to acute clinical conditions such as sepsis, respiratory failure, or septic shock. While bedside vital signs are routinely collected, manual clinical surveillance is subjective, prone to alarm fatigue, and can overlook subtle multi-parameter trends until organ damage has already progressed. There is a need for objective, automated machine learning models that can evaluate standard vital signs and alert healthcare teams before catastrophic deterioration occurs.

---

## 3. Objective
1. Extract and preprocess authentic clinical time-series data from the open-access **PhysioNet / Computing in Cardiology Challenge 2019** dataset.
2. Formulate a clean, balanced clinical case-control cohort of stable and deteriorating patients.
3. Train, evaluate, and benchmark **five classical machine learning classification algorithms**:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Machine (SVM with calibrated probabilities)
4. Analyze model trade-offs with emphasis on **Recall**, **F1-Score**, and **ROC-AUC** to prioritize clinical safety.
5. Deploy the best-performing pipeline into an interactive, clinician-friendly **Streamlit** web application.

---

## 4. Why Patient Deterioration Prediction Matters
- **Early Intervention Window:** Sepsis and respiratory collapse carry an hourly mortality increase if untreated. Identifying deterioration even 2 to 4 hours earlier allows timely fluid resuscitation and broad-spectrum antibiotics.
- **Alarm Fatigue Mitigation:** Traditional threshold-based monitors trigger frequent false alarms for isolated spikes. Multivariable machine learning models synthesize interactions among heart rate, respiratory rate, blood pressure, and oxygen saturation simultaneously.
- **Triage Optimization:** Helps nursing and rapid response teams prioritize ward beds requiring immediate bedside reviews.

---

## 5. Dataset Details
- **Source:** [PhysioNet / Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) (*Early Prediction of Sepsis from Clinical Data*)
- **Total Records in Raw Extract:** 38,809 hourly clinical observations across 41 physiological and laboratory columns.
- **Quality Control & Cleansing:**
  - Removed sensor artifacts (physiologically impossible vital sign extremes).
  - Imputed non-measured values using training set medians within scikit-learn pipelines.
  - Filtered records to ensure at least 3 active vital signs were simultaneously monitored.
- **Cohort Balancing:** Created a matched 1:1 case-control cohort of **1,586 patient records** (793 deteriorating cases vs. 793 stable controls) to eliminate majority-class bias.
- **Data Splitting:** 80% Training set (1,268 records) and 20% Test set (318 records), stratified by target to preserve identical class ratios without data leakage.

---

## 6. Features Used
The system selects seven clinically intuitive, standard vital signs and demographic parameters available at any hospital bedside:

| Feature Name | Clinical Parameter | Unit | Physiological Normal Range |
|---|---|---|---|
| `age` | Patient Age | Years | Adult (18 - 90+) |
| `heart_rate` | Cardiac Pulse Rate | beats/min (bpm) | 60 - 100 bpm |
| `respiratory_rate` | Respiratory Frequency | breaths/min | 12 - 20 breaths/min |
| `spo2` | Pulse Oximetry Oxygen Saturation | % | 95% - 100% |
| `systolic_bp` | Systolic Blood Pressure | mmHg | 90 - 120 mmHg |
| `diastolic_bp` | Diastolic Blood Pressure | mmHg | 60 - 80 mmHg |
| `temperature` | Core Body Temperature | °C | 36.5°C - 37.5°C |

---

## 7. Target Variable
- **Variable Name:** `deterioration_risk`
- **Type:** Binary (0 or 1)
  - **`0 = Stable`**: Patients maintaining physiological equilibrium without systemic organ failure or sepsis onset during hospital stay.
  - **`1 = Deterioration Risk`**: Patients experiencing clinical sepsis / acute deterioration requiring urgent escalation and intensive therapy.
- **Clinical Rationale:** Sepsis is the single leading cause of preventable in-hospital death and clinical deterioration worldwide. Predicting its onset from vital signs represents an authentic, life-critical deterioration detection task.

---

## 8. Algorithms Compared
We trained and evaluated five standard Scikit-Learn algorithms using clean, leak-free pipelines:

1. **Logistic Regression:** Linear baseline model with L2 regularization (`solver='lbfgs'`, `C=1.0`). Provides interpretable odds-ratio perspectives.
2. **K-Nearest Neighbors (KNN):** Non-parametric instance-based algorithm (`n_neighbors=7`, `metric='euclidean'`) with distance weighting.
3. **Decision Tree Classifier:** White-box non-linear tree (`criterion='gini'`, `max_depth=5`, `min_samples_leaf=5`) mirroring clinical flowchart rules.
4. **Random Forest Classifier:** Bagged ensemble of 120 decision trees (`n_estimators=120`, `max_depth=7`, `min_samples_leaf=3`) combining variance reduction and non-linear interactions.
5. **Support Vector Machine (SVM):** Radial Basis Function kernel (`kernel='rbf'`, `C=1.0`) with probability calibration via `CalibratedClassifierCV`.

---

## 9. Evaluation Metrics
In medical diagnostic systems, accuracy alone is insufficient. We evaluate all algorithms using:

- **Accuracy:** Overall percentage of correct predictions:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision (Positive Predictive Value):** Proportion of flagged patients who were genuinely deteriorating:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity / True Positive Rate):** Proportion of truly deteriorating patients correctly flagged:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  > **Why Recall is Paramount:** In hospital deterioration, a **False Negative** means a deteriorating patient is misdiagnosed as 'Stable', leading to delayed treatment and potential mortality. Conversely, a **False Positive** merely triggers a nurse to do a 60-second bedside check.
- **F1-Score:** Harmonic mean balancing Precision and Recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve measuring separation capability across all decision thresholds.

---

## 10. Project Workflow

```
[Raw PhysioNet ICU Data]
         │
         ▼
[Preprocessing & Quality Control]  <-- Handle 0s, check physiological ranges
         │
         ▼
[1:1 Case-Control Cohort]          <-- 1,586 patients (793 Stable : 793 Deteriorating)
         │
         ▼
[Stratified 80/20 Train-Test Split] <-- Train: 1,268 | Test: 318
         │
         ▼
[Scikit-Learn Pipelines]           <-- Median Imputation + Feature Scaling
         │
         ▼
[Train 5 ML Algorithms]            <-- LogReg, KNN, Decision Tree, Random Forest, SVM
         │
         ▼
[Comprehensive Evaluation]         <-- Acc, Prec, Recall, F1, ROC-AUC, Confusion Matrix
         │
         ▼
[Best Model Serialization]         <-- Saved to models/best_model.pkl (Random Forest)
         │
         ▼
[Interactive Streamlit Dashboard]  <-- app.py with triage categories & abnormal vitals alert
```

---

## 11. Project Directory Structure

```
ML PROJECT/
│
├── data/
│   ├── raw/
│   │   └── physionet_sepsis_raw.csv         # Raw PhysioNet ICU extract
│   └── processed/
│       ├── patient_deterioration_cleaned.csv# Cleaned 1:1 cohort (1,586 rows)
│       ├── train.csv                        # 80% Training split (1,268 rows)
│       └── test.csv                         # 20% Held-out test split (318 rows)
│
├── models/
│   ├── all_trained_models.pkl               # Dictionary of all 5 fitted pipelines
│   ├── best_model.pkl                       # Best performing serialized pipeline
│   └── model_metadata.json                  # Model hyperparameters and scores
│
├── notebooks/
│   └── analysis.ipynb                       # Comprehensive EDA & verification notebook
│
├── reports/
│   └── figures/
│       ├── accuracy_comparison.png          # Accuracy bar chart
│       ├── confusion_matrices.png           # 2x3 test confusion matrices
│       ├── correlation_heatmap.png          # Pearson correlation matrix
│       ├── f1_comparison.png                # F1-score bar chart
│       ├── feature_importance.png           # Random Forest Gini feature importances
│       ├── recall_comparison.png            # Clinical Recall bar chart
│       ├── roc_curves.png                   # Combined ROC curves (5 models)
│       ├── target_distribution.png          # Target class breakdown
│       ├── vitals_boxplots.png              # Stable vs Deterioration boxplots
│       └── vitals_distribution.png          # Histograms and KDE distributions
│
├── src/
│   ├── preprocess.py                        # Cleansing, outlier filtering & cohort splits
│   ├── train_models.py                      # Trains all 5 Scikit-Learn pipelines
│   ├── evaluate.py                          # Computes metrics, exports CSV & plots
│   ├── predict.py                           # Inference engine with clinical warning flags
│   └── generate_notebook.py                 # Notebook generation utility
│
├── app.py                                   # Streamlit clinical triage application
├── model_results.csv                        # Final comparative metrics table
├── requirements.txt                         # Minimal production dependencies
├── ASSESSMENT_NOTES.md                      # College assessment notes & viva guide
├── VIVA_QUESTIONS.md                        # 25+ viva questions with concise answers
├── REPORT_CONTENT.md                        # Academic project report text (14 sections)
└── .gitignore                               # Git ignore configuration
```

---

## 12. Installation Steps

### Step 1: Clone or navigate to the project directory
```powershell
cd "c:\Users\uday_\OneDrive\Documents\ML PROJECT"
```

### Step 2: Create a virtual environment
```powershell
python -m venv venv
```

### Step 3: Activate the virtual environment
- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install dependencies
```powershell
pip install -r requirements.txt
```

---

## 13. How to Run the Project

### 1. Run Data Preprocessing
```powershell
python src/preprocess.py
```
*Outputs: Cleansed dataset and stratified train/test files in `data/processed/`.*

### 2. Train All 5 Machine Learning Models
```powershell
python src/train_models.py
```
*Outputs: Trained pipeline dictionary saved to `models/all_trained_models.pkl`.*

### 3. Evaluate Models and Generate Plots
```powershell
python src/evaluate.py
```
*Outputs: Generates `model_results.csv`, exports plots to `reports/figures/`, and saves `models/best_model.pkl`.*

### 4. Run Sample Command-Line Prediction
```powershell
python src/predict.py
```
*Outputs: Evaluates both a stable patient and a deteriorating patient profile.*

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
*Access the interactive interface in your browser at `http://localhost:8501`.*

---

## 14. Model Comparison Results

All metrics were computed on the **identical 20% held-out test cohort of 318 patients** (159 deteriorating vs. 159 stable):

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.7044** | **0.7181** | **0.6730** | **0.6948** | **0.7868** |
| **K-Nearest Neighbors** | 0.6572 | 0.6420 | **0.7107** | 0.6746 | 0.7346 |
| **Support Vector Machine** | 0.6415 | 0.6490 | 0.6164 | 0.6323 | 0.7218 |
| **Decision Tree** | 0.6226 | 0.6154 | 0.6541 | 0.6341 | 0.6397 |
| **Logistic Regression** | 0.6006 | 0.6000 | 0.6038 | 0.6019 | 0.6437 |

---

## 15. Best Model Selection

### Selected Algorithm: **Random Forest Classifier**
- **Accuracy:** 70.44%
- **Recall (Sensitivity):** 67.30%
- **Precision:** 71.81%
- **F1-Score:** 69.48%
- **ROC-AUC:** 0.7868 (78.68%)

### Clinical Rationale for Selection:
1. **Superior Discrimination (ROC-AUC 0.7868):** Demonstrates the strongest ability across all operating thresholds to differentiate patients who are deteriorating from those who remain stable.
2. **Balanced Sensitivity (Recall 67.30%) & Precision (71.81%):** Successfully captures approximately two-thirds of acute deterioration events while maintaining a high positive predictive value to limit unnecessary nursing disruptions.
3. **Handling Non-linear Physiological Dynamics:** Vital signs interact synergistically during acute illness (e.g., tachypnea + tachycardia + hypotension indicates septic shock). Linear models (Logistic Regression) fail to model these multi-threshold interactions and achieve only 0.6437 ROC-AUC.
4. **Key Features Identified:** Feature importance analysis revealed that **patient age**, **heart rate**, and **respiratory rate** were the top three drivers of deterioration risk.

---

## 16. Screenshots Section

*(Placeholders for presentation slides or assessment report)*

- **Streamlit Web Application - Risk Assessment:**
  `reports/figures/accuracy_comparison.png`
- **Comparative ROC Curves:**
  `reports/figures/roc_curves.png`
- **Confusion Matrices Across 5 Models:**
  `reports/figures/confusion_matrices.png`
- **Random Forest Feature Importance:**
  `reports/figures/feature_importance.png`

---

## 17. Limitations
1. **Vital Signs Exclusivity:** The model relies solely on basic bedside vital signs and does not include diagnostic laboratory markers (such as serum lactate, arterial blood pH, or white blood cell counts).
2. **Cross-Sectional vs. Time-Series Dynamics:** The current system analyzes a single hourly snapshot rather than modeling longitudinal rate-of-change trajectories (e.g., a drop of 20 mmHg over 2 hours).
3. **ICU Demographic Specificity:** The PhysioNet dataset is collected from hospital adult intensive care units; performance on outpatient or pediatric populations may differ.

---

## 18. Future Enhancements
- Incorporate temporal trend features (e.g., $\Delta \text{HR}_{3h}$, moving averages of MAP).
- Add laboratory values if available in extended electronic health record (EHR) feeds.
- Implement an automated SMS/Pager alerting webhook for rapid response team activation.
- Conduct prospective bedside validation in hospital ward testbeds.

---

## 19. Medical Disclaimer
> **IMPORTANT NOTICE:** This project is an academic machine learning study designed solely for educational evaluation and technical research. It does NOT constitute medical advice, diagnosis, or clinical guidance. It must not be deployed in real patient care settings without formal clinical trials, ethical board approval, and medical device regulatory certification.
