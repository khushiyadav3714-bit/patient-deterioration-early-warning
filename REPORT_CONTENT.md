# Academic Mini-Project Report

## Title
**Comparative Performance Analysis of Machine Learning Algorithms for Early Prediction of Patient Deterioration**

---

### 1. Abstract
Early identification of physiological deterioration in hospitalized patients is critical for preventing avoidable morbidity, unanticipated intensive care admissions, and in-hospital mortality. In current clinical workflows, vital signs are periodically recorded by nursing staff, but subtle early deterioration can be missed due to workload and the complex non-linear interactions among physiological parameters. This project investigates and compares five standard machine learning algorithms—Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Machine (SVM)—for predicting clinical deterioration using routine bedside vital signs from the authentic PhysioNet Challenge 2019 dataset. A matched case-control cohort of 1,586 patient records was created and evaluated on an independent 20% test partition (318 patients). Random Forest demonstrated superior overall discrimination, achieving an Accuracy of 70.44%, an F1-Score of 0.6948, and an Area Under the ROC Curve (ROC-AUC) of 0.7868, while KNN achieved the highest raw sensitivity (Recall: 71.07%). Feature importance analysis revealed that patient age, heart rate, and respiratory rate are the dominant drivers of early deterioration risk. Finally, an interactive Streamlit application was developed to translate model predictions into intuitive clinical triage tiers and specific physiological alerts.

---

### 2. Introduction
In acute hospital care, timely intervention can be the deciding factor between a full recovery and catastrophic organ failure. When patients experience serious underlying complications such as sepsis, pneumonia, or internal bleeding, their body attempts to compensate by adjusting basic physiological processes. For instance, the heart pumps faster to maintain tissue perfusion, and the breathing rate increases to clear carbon dioxide and maintain oxygenation. 

Traditional clinical scoring systems, such as the National Early Warning Score (NEWS2) or Modified Early Warning Score (MEWS), use rigid hand-crafted point tables for individual vital signs. While these systems have helped standardize nursing assessments, they frequently trigger false-positive alerts or fail to detect deterioration until vital signs have already crossed extreme thresholds. Machine learning provides an opportunity to evaluate multiple continuous physiological signals simultaneously, uncovering hidden patterns that human observers or static rule tables might overlook.

---

### 3. Problem Statement
Hospitals continuously monitor patient vital signs, but manually identifying early deterioration can be difficult. Hospital wards are busy environments where nurse-to-patient ratios may be stretched, and deterioration often manifests gradually over several hours before a patient enters respiratory collapse or septic shock. Furthermore, conventional bedside monitoring alarms evaluate parameters in isolation, which causes frequent false alarms and leads to clinical "alarm fatigue." 

There is a clear clinical need for an automated, lightweight, and interpretable machine learning system that utilizes basic vital signs already collected at the bedside to screen for early deterioration risk, allowing medical teams to act proactively rather than reactively.

---

### 4. Objectives
The primary objectives of this mini-project are:
1. To obtain and preprocess an authentic, real-world clinical dataset containing inpatient vital signs from the PhysioNet repository.
2. To isolate a small set of physiologically meaningful and clinically accessible bedside features suitable for an early warning system.
3. To address severe clinical class imbalance by constructing a matched case-control cohort.
4. To implement and train five classical machine learning classification algorithms using scikit-learn pipelines to ensure leak-free evaluation.
5. To perform a comprehensive comparative performance analysis using clinical evaluation metrics, with special emphasis on Recall, F1-Score, and ROC-AUC.
6. To deploy the best-performing model into an easy-to-use Streamlit web application that provides real-time risk stratification and flags concerning vital signs.

---

### 5. Dataset Description
This study utilizes clinical observation data derived from the **PhysioNet / Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data**. The original public archive comprises 38,809 hourly clinical records from intensive care units, containing 41 variables including vital signs, laboratory measurements, and demographics.

For this project, we focused on seven easily obtainable bedside features:
- **`age`**: Patient age in years.
- **`heart_rate`**: Heart rate in beats per minute (bpm).
- **`respiratory_rate`**: Breaths per minute, a sensitive marker of respiratory compromise and metabolic acidosis.
- **`spo2`**: Peripheral capillary oxygen saturation percentage (%).
- **`systolic_bp`**: Systolic blood pressure in mmHg.
- **`diastolic_bp`**: Diastolic blood pressure in mmHg.
- **`temperature`**: Core body temperature in degrees Celsius (°C).

**Target Variable:**
The target variable is `deterioration_risk`, where:
- `0 = Stable`: Patients who maintained clinical stability without acute sepsis onset.
- `1 = Deterioration Risk`: Patients who progressed to acute clinical sepsis and systemic deterioration.

Sepsis was selected as the target outcome because it represents the single largest cause of acute in-hospital physiological deterioration, multiorgan dysfunction syndrome (MODS), and ICU mortality.

---

### 6. Methodology
The project methodology follows standard machine learning and clinical study design principles:
1. **Data Ingestion:** Reading the raw clinical data and identifying variable schema.
2. **Quality Control & Cleaning:** Replacing unrecorded 0.0 values with `NaN` and removing physiological transducer artifacts outside human survival boundaries.
3. **Cohort Construction:** Subsampling stable records to form a balanced 1:1 case-control clinical cohort of 1,586 patients (793 deteriorating and 793 stable).
4. **Data Partitioning:** Splitting into 80% training (1,268 records) and 20% held-out test data (318 records) using stratified random sampling (`random_state=42`).
5. **Pipeline Construction:** Building Scikit-Learn pipelines combining median imputation, standard scaling (where applicable), and model estimators to prevent data leakage.
6. **Model Training & Evaluation:** Training five distinct classifiers and computing Accuracy, Precision, Recall, F1-Score, and ROC-AUC on the test set.
7. **Interpretability & Deployment:** Extracting feature importance and building an interactive Streamlit user interface.

---

### 7. Data Preprocessing
Real-world clinical data is notoriously irregular. The preprocessing pipeline involved:
- **Handling Missing Values:** In clinical monitoring, certain measurements (like core temperature or intermittent blood pressure) are recorded less frequently than continuous ECG heart rate. Unmeasured values in the raw dataset were denoted as 0.0; these were transformed to `NaN`. Records missing more than four vital signs simultaneously were excluded. During modeling, missing values were imputed using the median of the training split via `SimpleImputer(strategy='median')`.
- **Physiological Bounds Checking:** Sensor displacement or movement can create extreme values. Plausibility bounds were enforced (e.g., heart rate between 30 and 250 bpm, SpO2 between 50% and 100%, temperature between 30°C and 44°C).
- **Feature Scaling:** Distance-based algorithms (KNN) and margin-based algorithms (SVM) require features to be on comparable scales. `StandardScaler` was fit only on the training set to standardize features to zero mean and unit variance.
- **Class Imbalance Handling:** In raw ICU monitoring, less than 3% of hourly records represent active deterioration. Training directly on this raw distribution causes algorithms to predict 'Stable' for every case, yielding high accuracy but near-zero recall. We used safe majority-class undersampling to create a balanced 1:1 case-control design.

---

### 8. Algorithms Used
Five standard machine learning algorithms were compared:
1. **Logistic Regression:** A linear parametric model using the sigmoid function to map vital sign linear combinations to posterior deterioration probabilities. It acts as our linear baseline.
2. **K-Nearest Neighbors (KNN):** A non-parametric distance-based method with $k=7$ neighbors and Euclidean distance, classifying patients based on proximity to similar patient states.
3. **Decision Tree Classifier:** A rule-based hierarchical tree with `max_depth=5` and `min_samples_leaf=5`, partitioning feature space based on Gini impurity reduction.
4. **Random Forest Classifier:** A bagging ensemble consisting of 120 decision trees (`max_depth=7`, `min_samples_leaf=3`). It combines individual tree predictions to reduce variance and capture complex multi-vital interactions.
5. **Support Vector Machine (SVM):** A non-linear kernel classifier using the Radial Basis Function (RBF) kernel with probability calibration via `CalibratedClassifierCV`.

---

### 9. Performance Metrics
Models were evaluated across five standard metrics:
- **Accuracy:** The overall proportion of correct classifications out of all predictions.
- **Precision:** The proportion of predicted deterioration alarms that were true cases ($\frac{TP}{TP + FP}$). High precision minimizes alarm fatigue.
- **Recall (Sensitivity):** The proportion of truly deteriorating patients that the model successfully captured ($\frac{TP}{TP + FN}$). In medicine, this is the most critical metric because a missed deteriorating patient can be fatal.
- **F1-Score:** The harmonic mean of Precision and Recall ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$).
- **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve, quantifying the model's ability to rank positive cases higher than negative cases across all operating thresholds.

---

### 10. Results and Comparison
All five algorithms were evaluated on the exact same held-out test cohort of 318 patients (159 stable and 159 deteriorating). The results are summarized below:

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **70.44%** | **71.81%** | **67.30%** | **69.48%** | **0.7868** |
| **K-Nearest Neighbors** | 65.72% | 64.20% | **71.07%** | 67.46% | 0.7346 |
| **Support Vector Machine** | 64.15% | 64.90% | 61.64% | 63.23% | 0.7218 |
| **Decision Tree** | 62.26% | 61.54% | 65.41% | 63.41% | 0.6397 |
| **Logistic Regression** | 60.06% | 60.00% | 60.38% | 60.19% | 0.6437 |

**Analysis of Findings:**
- **Random Forest** achieved the highest overall Accuracy (70.44%), Precision (71.81%), F1-Score (69.48%), and ROC-AUC (0.7868).
- **KNN** achieved the highest raw Recall (71.07%), but had slightly lower precision (64.20%), resulting in more false alarms.
- **Logistic Regression** achieved the lowest ROC-AUC (0.6437). This indicates that physiological deterioration cannot be adequately captured by a simple linear boundary; patients deteriorate through combinations of threshold crossings.

---

### 11. Best Model Selection
**Random Forest Classifier** was selected as the final production model. 

**Clinical Rationale:**
While KNN offered slightly higher sensitivity, Random Forest delivered superior overall discrimination (ROC-AUC of 0.7868 vs. 0.7346) and substantially higher precision (71.81% vs. 64.20%). In a busy hospital, excessive false alarms lead to alarms being muted or ignored. Random Forest strikes the best clinical balance: it catches over two-thirds of acute deterioration cases while ensuring that more than 7 out of 10 alerts represent genuine clinical instability.

**Feature Importance:**
Feature importance extracted from the Random Forest model revealed that **age** (23.2%), **heart rate** (20.3%), and **respiratory rate** (14.8%) were the top contributors to the decision process. This strongly aligns with clinical knowledge, as elevated heart and breathing rates are the hallmark signs of the Systemic Inflammatory Response Syndrome (SIRS).

---

### 12. Conclusion
This project successfully implemented an end-to-end machine learning system for early patient deterioration prediction. By benchmarking five diverse classification algorithms on real-world ICU data, we demonstrated that:
1. Standard, non-invasive bedside vital signs possess significant predictive power for detecting acute clinical decompensation.
2. Ensemble tree-based methods (Random Forest) outperform linear models due to their ability to capture non-linear interactions among heart rate, respiratory rate, and blood pressure.
3. The model was successfully deployed into an interactive Streamlit application that provides rapid triage categorization and highlights specific physiological warning signs.

---

### 13. Limitations
1. **Vital Signs Only:** The model does not include laboratory results such as white blood cell count, serum creatinine, or lactate, which would enhance diagnostic precision.
2. **Point-in-Time Assessment:** The model evaluates a static snapshot rather than continuous temporal trends (e.g., whether blood pressure has dropped steadily over the last two hours).
3. **ICU Setting Bias:** The dataset originates from adult intensive care units; further validation is required before deploying this model in pediatric wards or outpatient clinics.

---

### 14. Future Scope
1. **Longitudinal Time-Series Modeling:** Integrating recurrent models (such as LSTMs or GRUs) to process multi-hour trajectories of vital sign changes.
2. **Laboratory Data Integration:** Incorporating standard blood panel results when available in hospital Electronic Health Records (EHR).
3. **Prospective Clinical Trials:** Conducting shadow testing in hospital wards to compare machine learning alerts against standard NEWS2 scores in real time.
4. **Bedside Explainability:** Implementing SHAP (SHapley Additive exPlanations) in the user interface to show clinicians the exact mathematical contribution of each vital sign for an individual patient.
