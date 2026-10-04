# Viva & Interview Preparation Guide

### Project: Comparative Performance Analysis of Machine Learning Algorithms for Early Prediction of Patient Deterioration

This document contains **28 comprehensive viva and interview questions** with concise, clear, and technically accurate answers.

---

### Q1: What is patient deterioration?
**Answer:** Patient deterioration is the acute physiological decline of an inpatient's health state, often marked by worsening vital signs (e.g., dropping blood pressure, rising heart and respiratory rates). It frequently precedes clinical emergencies such as septic shock, respiratory failure, cardiac arrest, or death.

---

### Q2: Why did you choose this project?
**Answer:** I chose this project because in-hospital deterioration is a major healthcare challenge where early detection directly saves lives. Traditional bedside monitoring relies on individual thresholds that miss complex multi-vital trends. Applying machine learning to basic vital signs creates an automated, non-invasive early warning system suitable for any hospital ward.

---

### Q3: Why is Recall especially important in this healthcare problem?
**Answer:** In healthcare, a **False Negative** means the model diagnoses a critically deteriorating patient as 'Stable', leading to delayed medical treatment and potential death. **Recall measures the percentage of deteriorating patients the system successfully catches**. We prioritize high recall to ensure no dying or septic patient is overlooked.

---

### Q4: What is a False Negative, and what is its consequence here?
**Answer:** A False Negative occurs when a patient is genuinely in a deteriorating clinical state, but the model incorrectly classifies them as '0' (Stable). In a hospital setting, this error can lead to omitted antibiotic administration, missed ICU transfer, and catastrophic patient mortality.

---

### Q5: Why shouldn't we rely solely on Accuracy?
**Answer:** Accuracy only measures the total percentage of correct predictions. In medical datasets where deterioration is rare (e.g., 2% positive cases), a naive model that predicts every patient is 'Stable' achieves 98% accuracy while having a **0% Recall** (failing to save a single patient). Accuracy is misleading under class imbalance.

---

### Q6: What is Precision?
**Answer:** Precision (Positive Predictive Value) is the fraction of patients predicted as 'Deteriorating' who are truly deteriorating ($\frac{TP}{TP + FP}$). In clinical practice, high precision prevents "alarm fatigue" by minimizing false alarms.

---

### Q7: What is the difference between Precision and Recall?
**Answer:**
- **Precision:** Focuses on the reliability of the alarm: *"When the model rings the bell, is the patient truly sick?"*
- **Recall:** Focuses on patient safety: *"Out of all sick patients in the hospital, how many did the model catch?"*

---

### Q8: What is F1-Score?
**Answer:** F1-score is the harmonic mean of Precision and Recall ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$). It provides a single balanced metric, penalizing models that have high precision but poor recall (or vice-versa).

---

### Q9: What is ROC-AUC?
**Answer:** ROC-AUC stands for Area Under the Receiver Operating Characteristic Curve. It plots the True Positive Rate (Recall) against the False Positive Rate across all possible probability thresholds. An AUC of 1.0 represents perfect discrimination, while 0.5 is no better than random guessing. Our best model achieved **0.7868**.

---

### Q10: Why did you use Random Forest?
**Answer:** Random Forest is an ensemble of decision trees trained on random bootstrapped subsets of data and features. It naturally handles non-linear physiological interactions (e.g., high pulse combined with low blood pressure), is resistant to noise and outliers, and provides interpretable feature importance scores.

---

### Q11: What is the difference between a Decision Tree and a Random Forest?
**Answer:**
- A **Decision Tree** builds a single tree, which is prone to high variance and overfitting on small data shifts.
- A **Random Forest** builds multiple decision trees (in our case, 120 trees) using bagging and averages their predictions. This ensemble technique significantly reduces variance and improves generalization.

---

### Q12: Why does K-Nearest Neighbors (KNN) require feature scaling?
**Answer:** KNN calculates Euclidean distance between data points in feature space:
$$d = \sqrt{\sum (x_i - y_i)^2}$$
If features are on different scales (e.g., Systolic BP varies from 50 to 220 mmHg, while Temperature varies from 35 to 40 °C), blood pressure differences will artificially dominate the distance calculation. `StandardScaler` places all features on a zero-mean, unit-variance scale.

---

### Q13: Why does Support Vector Machine (SVM) require feature scaling?
**Answer:** SVM finds a maximum-margin hyperplane that separates the classes. The optimization problem relies on inner products of feature vectors (or distance in RBF kernel space). Unscaled features with larger numeric ranges distort the margin geometry and dominate the support vector calculations.

---

### Q14: What is overfitting, and how did you prevent it?
**Answer:** Overfitting occurs when a model memorizes the training data and noise rather than learning generalizable patterns, resulting in high training performance but poor test performance. We prevented overfitting by:
1. Evaluating strictly on a held-out 20% test set.
2. Restricting tree depth (`max_depth=7` in Random Forest, `max_depth=5` in Decision Tree).
3. Setting minimum sample leaf constraints (`min_samples_leaf=3`).
4. Using cross-validation/bootstrapping inside Random Forest.

---

### Q15: What is train-test split, and why did you use 80-20?
**Answer:** Train-test split divides the dataset into a training subset (used to fit model weights) and an independent test subset (used to evaluate unseen performance). An 80/20 ratio provides sufficient data for model learning (1,268 records) while maintaining an adequately powered test set (318 records) to calculate statistically valid metrics.

---

### Q16: Why did you set `random_state=42`?
**Answer:** `random_state=42` sets the seed for pseudo-random number generators in data splitting and model initialization. It ensures reproducibility so that any researcher, examiner, or reviewer running the code gets the exact same data split and experimental results.

---

### Q17: What is class imbalance, and how did you address it?
**Answer:** Class imbalance occurs when one class heavily outnumbers the other. In the raw PhysioNet data, less than 3% of patients had active deterioration. To prevent models from learning a trivial majority-class shortcut, we created a **1:1 matched clinical case-control cohort** (793 deteriorating vs. 793 stable controls), a standard epidemiological study design.

---

### Q18: Why did you use a real-world dataset instead of synthetic data?
**Answer:** Synthetic medical data often assumes unrealistic Gaussian distributions and misses the messy realities of clinical care (transducer noise, missing hourly readings, correlated physiological shifts). Using authentic **PhysioNet Challenge 2019** data ensures our findings are clinically credible and relevant for real hospital systems.

---

### Q19: What preprocessing steps did you perform?
**Answer:**
1. **Column Mapping:** Standardized raw PhysioNet column headers to intuitive vital signs (`age`, `heart_rate`, `respiratory_rate`, `spo2`, `systolic_bp`, `diastolic_bp`, `temperature`).
2. **Missing Value Replacement:** Converted 0.0 values (which represent unmeasured vitals in PhysioNet) to `NaN`.
3. **Quality Control:** Filtered out physiological sensor artifacts (e.g., HR < 30 or > 250 bpm).
4. **Leak-free Imputation & Scaling:** Used Scikit-Learn `Pipeline` with `SimpleImputer(strategy='median')` and `StandardScaler()`, fit exclusively on the training partition.

---

### Q20: Which model performed best overall?
**Answer:** **Random Forest Classifier** was the best overall model, achieving:
- **Accuracy:** 70.44%
- **Precision:** 71.81%
- **Recall:** 67.30%
- **F1-Score:** 69.48%
- **ROC-AUC:** 0.7868

---

### Q21: Why did Random Forest perform better than Logistic Regression?
**Answer:** Human physiology does not deteriorate in a linear fashion. For example, a heart rate of 120 bpm might be well tolerated if blood pressure is normal, but if accompanied by a low systolic BP of 80 mmHg and high respiratory rate of 28, it indicates septic shock. Random Forest captures these multi-variable non-linear threshold interactions across its trees, whereas Logistic Regression only fits a flat hyper-plane.

---

### Q22: What are the main features driving deterioration risk?
**Answer:** Based on Random Forest's Mean Decrease in Impurity (MDI), the top three features were:
1. **Patient Age:** Advanced age reduces physiological reserve.
2. **Heart Rate:** Tachycardia is an early compensatory response to infection and blood volume depletion.
3. **Respiratory Rate:** Tachypnea (>22 breaths/min) is the earliest and most sensitive marker of clinical deterioration and systemic inflammation.

---

### Q23: What are the limitations of this project?
**Answer:**
1. It relies only on 7 bedside vital signs and does not include laboratory tests (e.g. serum lactate, arterial blood gas, WBC count).
2. It uses a cross-sectional point-in-time assessment rather than a recurrent time-series model (such as an LSTM).
3. The dataset comes from adult ICU patients; it cannot be applied to pediatric patients without retraining.

---

### Q24: Can this machine learning system replace doctors or nurses?
**Answer:** **No, absolutely not.** This system is designed as a **Clinical Decision Support System (CDSS)**. It acts as an automated safety net to flag deteriorating patients who might otherwise be overlooked during busy hospital shifts. Final clinical decisions, diagnosis, and treatment always remain with the licensed medical team.

---

### Q25: How could this project be improved in future work?
**Answer:**
1. Adding temporal trend features (e.g., rate of systolic blood pressure decline over the past 3 hours).
2. Incorporating laboratory blood test markers.
3. Testing gradient boosting algorithms like XGBoost or LightGBM.
4. Implementing patient-specific explainability using SHAP values.

---

### Q26: How could this system be deployed in a real hospital setting?
**Answer:** It could be integrated into the hospital's Electronic Health Record (EHR) or bedside monitor middleware via HL7/FHIR APIs. As nurses record routine observations, the model pipeline calculates a live risk score in the background and sends push notifications to the Rapid Response Team (RRT) if the risk exceeds 60%.

---

### Q27: How does your Streamlit app categorize patient risk?
**Answer:** The Streamlit app takes patient vital signs, executes the pipeline, extracts the deterioration probability, and maps it to three clinical triage categories:
- **LOW RISK (0% – 45%):** Patient vitals are within stable baseline parameters.
- **MODERATE RISK (45% – 60%):** Signs of borderline instability; increased monitoring recommended.
- **HIGH RISK (> 60%):** Critical physiological red flags; immediate bedside review recommended.
It also lists specific abnormal parameters (e.g. tachycardia, hypoxia, hypotension) for quick clinical interpretation.

---

### Q28: What is data leakage, and how did you prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as test set statistics) is inadvertently used during model training, leading to overly optimistic performance estimates. We prevented data leakage by:
1. Performing the train-test split *before* computing any summary statistics.
2. Using Scikit-Learn `Pipeline` objects so that median imputers and standard scalers were `fit` only on `X_train` and merely `transformed` on `X_test`.
