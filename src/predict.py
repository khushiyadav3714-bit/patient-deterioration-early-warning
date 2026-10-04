"""
Patient Deterioration Prediction Module
----------------------------------------
Loads the persisted best model pipeline and provides real-time risk assessment,
deterioration probability estimation, clinical triage categorization,
and physiological warning flags for vital signs.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


class DeteriorationPredictor:
    """Predictor class encapsulating model inference and clinical risk logic."""

    def __init__(self,
                 model_path: str = "models/best_model.pkl",
                 metadata_path: str = "models/model_metadata.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")

        self.model = joblib.load(model_path)
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

        self.model_name = self.metadata.get('best_algorithm', 'Trained Best Model')
        self.feature_names = ['age', 'heart_rate', 'respiratory_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'temperature']

    def check_concerning_vitals(self, data: dict) -> list:
        """
        Flags concerning vital signs based on standard clinical guidelines
        (e.g., NEWS2 / SIRS criteria).
        """
        flags = []

        hr = data.get('heart_rate')
        if hr is not None:
            if hr > 100:
                flags.append(f"Elevated Heart Rate ({hr} bpm, Tachycardia)")
            elif hr < 50:
                flags.append(f"Low Heart Rate ({hr} bpm, Bradycardia)")

        rr = data.get('respiratory_rate')
        if rr is not None:
            if rr > 22:
                flags.append(f"High Respiratory Rate ({rr} breaths/min, Tachypnea - Early Deterioration Sign)")
            elif rr < 10:
                flags.append(f"Low Respiratory Rate ({rr} breaths/min, Bradypnea)")

        spo2 = data.get('spo2')
        if spo2 is not None:
            if spo2 < 92:
                flags.append(f"Critically Low SpO2 ({spo2}%, Severe Hypoxia)")
            elif spo2 < 95:
                flags.append(f"Suboptimal SpO2 ({spo2}%, Mild Hypoxia)")

        sbp = data.get('systolic_bp')
        if sbp is not None:
            if sbp < 90:
                flags.append(f"Low Systolic BP ({sbp} mmHg, Hypotension / Shock Risk)")
            elif sbp > 160:
                flags.append(f"Severely High Systolic BP ({sbp} mmHg, Hypertension)")

        dbp = data.get('diastolic_bp')
        if dbp is not None:
            if dbp < 60:
                flags.append(f"Low Diastolic BP ({dbp} mmHg)")
            elif dbp > 100:
                flags.append(f"High Diastolic BP ({dbp} mmHg)")

        temp = data.get('temperature')
        if temp is not None:
            if temp > 38.0:
                flags.append(f"Elevated Body Temperature ({temp:.1f}°C, Pyrexia / Sepsis Indicator)")
            elif temp < 36.0:
                flags.append(f"Hypothermia ({temp:.1f}°C, Severe Sepsis / Shock Indicator)")

        age = data.get('age')
        if age is not None and age >= 65:
            flags.append(f"Advanced Age Risk Group ({int(age)} years)")

        return flags

    def predict(self, patient_dict: dict) -> dict:
        """
        Predicts deterioration risk for a single patient record.
        """
        # Ensure all required features are present
        row = {f: patient_dict.get(f, np.nan) for f in self.feature_names}
        input_df = pd.DataFrame([row])

        # Prediction and Probability
        pred_class = int(self.model.predict(input_df)[0])
        probabilities = self.model.predict_proba(input_df)[0]
        prob_deterioration = float(probabilities[1])
        prob_percentage = round(prob_deterioration * 100, 1)

        # Risk Stratification:
        # Clinical Risk Tiers tailored for balanced cohort model output:
        # LOW RISK: Below 45% (or below 52% with entirely normal vitals)
        # MODERATE RISK: 45% - 60%
        # HIGH RISK: Above 60%
        concerning_vitals = self.check_concerning_vitals(patient_dict)

        if prob_percentage < 45.0 or (pred_class == 0 and len(concerning_vitals) == 0 and prob_percentage < 52.0):
            risk_category = "LOW RISK"
            badge_color = "green"
            clinical_recommendation = "Patient vitals are currently within stable baseline parameters. Continue routine ward observation."
        elif prob_percentage <= 60.0:
            risk_category = "MODERATE RISK"
            badge_color = "orange"
            clinical_recommendation = "Signs of emerging physiological instability or borderline vitals detected. Increase vital signs monitoring frequency and perform clinical nursing review."
        else:
            risk_category = "HIGH RISK"
            badge_color = "red"
            clinical_recommendation = "Critical deterioration indicators present! Immediate medical evaluation, escalation to rapid response team / ICU assessment strongly recommended."

        return {
            'risk_category': risk_category,
            'predicted_class': pred_class,
            'deterioration_probability': prob_percentage,
            'stable_probability': round((1.0 - prob_deterioration) * 100, 1),
            'model_name': self.model_name,
            'badge_color': badge_color,
            'clinical_recommendation': clinical_recommendation,
            'concerning_values': concerning_vitals,
            'input_data': patient_dict
        }


def sample_test_prediction():
    """Runs a quick CLI test of both a stable patient and a deteriorating patient."""
    predictor = DeteriorationPredictor()

    print("\n" + "="*70)
    print("TESTING PREDICTOR: SAMPLE 1 - NORMAL / STABLE PATIENT")
    print("="*70)
    stable_patient = {
        'age': 45.0,
        'heart_rate': 72.0,
        'respiratory_rate': 16.0,
        'spo2': 98.0,
        'systolic_bp': 120.0,
        'diastolic_bp': 78.0,
        'temperature': 36.8
    }
    res1 = predictor.predict(stable_patient)
    print(f"Risk Category : {res1['risk_category']}")
    print(f"Probability   : {res1['deterioration_probability']}%")
    print(f"Concerning    : {res1['concerning_values'] if res1['concerning_values'] else 'None'}")
    print(f"Advice        : {res1['clinical_recommendation']}")

    print("\n" + "="*70)
    print("TESTING PREDICTOR: SAMPLE 2 - ACUTE DETERIORATION / SEPSIS RISK")
    print("="*70)
    deteriorating_patient = {
        'age': 72.0,
        'heart_rate': 118.0,
        'respiratory_rate': 26.0,
        'spo2': 91.0,
        'systolic_bp': 85.0,
        'diastolic_bp': 52.0,
        'temperature': 38.6
    }
    res2 = predictor.predict(deteriorating_patient)
    print(f"Risk Category : {res2['risk_category']}")
    print(f"Probability   : {res2['deterioration_probability']}%")
    print(f"Concerning    : {res2['concerning_values']}")
    print(f"Advice        : {res2['clinical_recommendation']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    sample_test_prediction()
