# Machine Learning Assessment Notes

## Project Title
**Comparative Performance Analysis of Machine Learning Algorithms for Early Prediction of Patient Deterioration**

---

## 1. Problem Statement
Hospitals continuously monitor patient vital signs, but manually identifying early deterioration can be difficult. Patients in wards and intensive care units can decompensate rapidly due to systemic infections, respiratory compromise, or septic shock. Traditional threshold alarms often sound too late or trigger false alarms. This project compares machine learning algorithms for predicting patient deterioration risk using basic clinical measurements available at the bedside.

---

## 2. Objective
To compare multiple standard machine learning classification algorithms and determine which performs best for early detection of patient deterioration, balancing overall accuracy with clinical safety (high recall to avoid missing critically ill patients).

---

## 3. Dataset Summary
- **Source:** PhysioNet / Computing in Cardiology Challenge 2019 (*Early Prediction of Sepsis from Clinical Data*).
- **Size:** 38,809 raw ICU observations filtered into a clean 1:1 matched clinical case-control cohort of **1,586 patients** (793 deteriorating vs. 793 stable).
- **Features Used (7):**
  1. `age` (Years)
  2. `heart_rate` (Beats per minute)
  3. `respiratory_rate` (Breaths per minute)
  4. `spo2` (Oxygen saturation %)
  5. `systolic_bp` (Systolic blood pressure mmHg)
  6. `diastolic_bp` (Diastolic blood pressure mmHg)
  7. `temperature` (Body temperature °C)
- **Target Variable:** Binary `deterioration_risk` (0 = Stable, 1 = Deterioration / Sepsis Risk).

---

## 4. Algorithms Evaluated
1. **Logistic Regression:**
   - A linear statistical classification model that uses a logistic (sigmoid) function to model the probability of deterioration.
   - *Advantage:* Fast, simple, highly interpretable.
   - *Limitation:* Assumes a linear boundary, cannot naturally capture complex physiological thresholds.

2. **K-Nearest Neighbors (KNN):**
   - An instance-based (lazy learning) algorithm that classifies a patient based on the majority class of their $k$ nearest neighbors in feature space ($k=7$).
   - *Advantage:* Non-parametric, intuitive.
   - *Limitation:* Computationally slower at test time, sensitive to irrelevant features, strictly requires feature scaling.

3. **Decision Tree Classifier:**
   - A flowchart-like tree structure where internal nodes test specific vital sign values (e.g., $HR > 95$) and leaf nodes output risk decisions.
   - *Advantage:* Mimics how doctors follow clinical checklists.
   - *Limitation:* Prone to overfitting on small data variations if depth is unconstrained.

4. **Random Forest Classifier (Best Model):**
   - An ensemble bagging algorithm combining 120 randomized decision trees and aggregating their votes.
   - *Advantage:* High accuracy, robust to noise, models non-linear vital sign interactions, provides feature importances.
   - *Limitation:* More complex than a single tree, acts more like a gray-box model.

5. **Support Vector Machine (SVM):**
   - A maximum-margin classifier using a Radial Basis Function (RBF) kernel to map vital signs into higher dimensions and find the optimal separating hyperplane.
   - *Advantage:* Effective in finding non-linear decision boundaries.
   - *Limitation:* Requires probability calibration (`CalibratedClassifierCV`) and careful feature scaling.

---

## 5. Evaluation Metrics Explained in Simple Language

| Metric | What it Measures | Simple Explanation | Formula |
|---|---|---|---|
| **Accuracy** | Overall correctness | Out of all patients, how many were correctly classified? | $\frac{TP + TN}{\text{Total}}$ |
| **Precision** | Reliability of positive alert | When the model sounds an alarm saying a patient is deteriorating, how often is it actually right? | $\frac{TP}{TP + FP}$ |
| **Recall (Sensitivity)** | Safety / Catch rate | Out of all patients who are genuinely deteriorating, what percentage did the model successfully detect? | $\frac{TP}{TP + FN}$ |
| **F1-Score** | Balance score | The harmonic average of Precision and Recall. High F1 means both false alarms and missed cases are kept low. | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **ROC-AUC** | Separation ability | How well the model can distinguish between a stable patient and a deteriorating patient across all threshold settings (0.5 is coin flip, 1.0 is perfect). | Area under ROC Curve |

> **Why Recall is the Most Critical Metric in Healthcare:**
> - A **False Positive** means the model flags a stable patient as at-risk. The consequence is merely that a bedside nurse visits the patient to take another manual reading.
> - A **False Negative** means the model misses a deteriorating patient and labels them 'Stable'. The consequence is delayed treatment, septic shock, or death.
> Therefore, high recall is vital to ensure clinical safety.

---

## 6. Actual Comparative Results (Test Cohort - 318 Patients)

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **70.44%** | **71.81%** | **67.30%** | **69.48%** | **0.7868** |
| **K-Nearest Neighbors** | 65.72% | 64.20% | **71.07%** | 67.46% | 0.7346 |
| **Support Vector Machine** | 64.15% | 64.90% | 61.64% | 63.23% | 0.7218 |
| **Decision Tree** | 62.26% | 61.54% | 65.41% | 63.41% | 0.6397 |
| **Logistic Regression** | 60.06% | 60.00% | 60.38% | 60.19% | 0.6437 |

---

## 7. Conclusion
- **Random Forest achieved the best overall performance**, obtaining 70.44% accuracy, an F1-score of 0.6948, and an ROC-AUC of 0.7868.
- KNN achieved the highest raw sensitivity (Recall of 71.07%), but Random Forest provided a superior balance between sensitivity and precision, preventing excessive false alerts.
- Tree-based ensemble learning successfully captured the physiological threshold interactions (e.g. combined high respiratory rate and low blood pressure) that linear models missed.
- The project demonstrates that simple bedside vital signs alone contain strong predictive signal for detecting acute clinical deterioration early.

---

## 8. Limitations
1. Uses only 7 routine vital signs; lacks laboratory test results (e.g. serum lactate, WBC, blood gases).
2. Cross-sectional prediction at a single point in time, rather than tracking longitudinal time-series changes over hours.
3. Derived from an adult ICU dataset; may not generalize directly to pediatric or outpatient clinic populations.

---

## 9. Future Scope
1. **Dynamic Trend Modeling:** Incorporate rate-of-change features (e.g., changes in mean arterial pressure over 3-hour windows).
2. **EHR Integration:** Connect to hospital HL7/FHIR feeds for automated, real-time calculation.
3. **Clinical Alerting System:** Integrate with nurse pagers or mobile app push notifications.
4. **Explainable AI:** Deploy SHAP (SHapley Additive exPlanations) values to provide patient-specific bedside explanations for why an alert was triggered.
