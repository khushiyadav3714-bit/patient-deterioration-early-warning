"""
Model Training Module for Patient Deterioration Early Warning System
---------------------------------------------------------------------
Defines standard Scikit-Learn pipelines for 5 machine learning algorithms:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
4. Random Forest Classifier
5. Support Vector Machine (SVM)

Each pipeline encapsulates imputation and optional scaling to completely
prevent data leakage.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


def get_model_pipelines() -> dict:
    """
    Returns a dictionary of Scikit-Learn pipelines for all 5 algorithms.
    Sensible default / lightly-tuned parameters are used for comparative analysis.
    """
    pipelines = {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver='lbfgs',
                random_state=42
            ))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='euclidean'
            ))
        ]),
        'Decision Tree': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                criterion='gini',
                random_state=42
            ))
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', RandomForestClassifier(
                n_estimators=120,
                max_depth=7,
                min_samples_split=4,
                min_samples_leaf=3,
                random_state=42
            ))
        ]),
        'Support Vector Machine': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', CalibratedClassifierCV(
                estimator=SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            ))
        ])
    }
    return pipelines


def train_all_models(train_csv_path: str = "data/processed/train.csv",
                     models_dir: str = "models") -> tuple:
    """
    Trains all 5 pipelines on the processed training set and persists trained models.
    """
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training data not found at {train_csv_path}. Run preprocess.py first.")

    train_df = pd.read_csv(train_csv_path)
    feature_cols = ['age', 'heart_rate', 'respiratory_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'temperature']
    target_col = 'deterioration_risk'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    print(f"[INFO] Training 5 algorithms on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

    pipelines = get_model_pipelines()
    trained_models = {}

    for name, pipe in pipelines.items():
        print(f"  --> Training {name}...")
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

    # Save dictionary of all trained models
    all_models_path = os.path.join(models_dir, "all_trained_models.pkl")
    joblib.dump(trained_models, all_models_path)
    print(f"[INFO] Successfully trained all 5 models and saved to: {all_models_path}")

    return trained_models, feature_cols, target_col


if __name__ == "__main__":
    train_all_models()
