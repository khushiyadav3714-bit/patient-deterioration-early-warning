"""
Data Preprocessing Module for Patient Deterioration Early Warning System
-------------------------------------------------------------------------
This module loads the raw clinical time-series dataset from the PhysioNet
Challenge 2019, cleans physiological outliers, maps columns to standard
vital signs, handles missing values, and creates a balanced clinical cohort
with stratified 80-20 train-test splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects relevant clinical vital signs, converts 0s to NaN (unmeasured vitals),
    cleans physiologically implausible values, and filters records with sufficient vitals.
    """
    # Map raw PhysioNet columns to standard clinical feature names
    column_mapping = {
        'Age': 'age',
        'HR': 'heart_rate',
        'Resp': 'respiratory_rate',
        'O2Sat': 'spo2',
        'SBP': 'systolic_bp',
        'DBP': 'diastolic_bp',
        'Temp': 'temperature',
        'SepsisLabel': 'deterioration_risk'
    }

    # Ensure required columns exist
    missing_cols = [c for c in column_mapping.keys() if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    df_clean = df[list(column_mapping.keys())].copy()
    df_clean.rename(columns=column_mapping, inplace=True)

    vital_cols = ['heart_rate', 'spo2', 'temperature', 'systolic_bp', 'diastolic_bp', 'respiratory_rate']

    # In PhysioNet Challenge 2019 dataset, unmeasured values are stored as 0.0
    for col in vital_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    # Apply clinical physiological plausibility bounds (remove extreme transducer artifacts)
    df_clean.loc[(df_clean['heart_rate'] < 30) | (df_clean['heart_rate'] > 250), 'heart_rate'] = np.nan
    df_clean.loc[(df_clean['spo2'] < 50) | (df_clean['spo2'] > 100), 'spo2'] = np.nan
    df_clean.loc[(df_clean['temperature'] < 30) | (df_clean['temperature'] > 44), 'temperature'] = np.nan
    df_clean.loc[(df_clean['systolic_bp'] < 40) | (df_clean['systolic_bp'] > 260), 'systolic_bp'] = np.nan
    df_clean.loc[(df_clean['diastolic_bp'] < 30) | (df_clean['diastolic_bp'] > 160), 'diastolic_bp'] = np.nan
    df_clean.loc[(df_clean['respiratory_rate'] < 4) | (df_clean['respiratory_rate'] > 60), 'respiratory_rate'] = np.nan

    # Retain observations where at least 3 essential vital signs were recorded
    df_clean['valid_vitals_count'] = df_clean[vital_cols].notna().sum(axis=1)
    df_filtered = df_clean[df_clean['valid_vitals_count'] >= 3].drop(columns=['valid_vitals_count']).copy()

    # Drop duplicate records if any
    df_filtered = df_filtered.drop_duplicates().reset_index(drop=True)

    print(f"[INFO] After filtering and quality control: {df_filtered.shape[0]} rows remaining.")
    print(f"[INFO] Deterioration cases (1): {(df_filtered['deterioration_risk'] == 1).sum()}, "
          f"Stable cases (0): {(df_filtered['deterioration_risk'] == 0).sum()}")

    return df_filtered


def create_balanced_cohort(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Creates a matched 1:1 case-control clinical cohort to address acute class imbalance.
    Prevents standard classification algorithms from biasing toward the 98% majority class.
    """
    positives = df[df['deterioration_risk'] == 1]
    negatives = df[df['deterioration_risk'] == 0]

    n_pos = len(positives)
    # Undersample majority class to equal the positive class
    negatives_sampled = negatives.sample(n=n_pos, random_state=random_state)

    cohort = pd.concat([positives, negatives_sampled], axis=0)
    cohort = cohort.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[INFO] Created 1:1 balanced cohort: {cohort.shape[0]} total cases "
          f"({n_pos} deteriorating, {n_pos} stable).")
    return cohort


def prepare_and_split_data(raw_csv_path: str = "data/raw/physionet_sepsis_raw.csv",
                           output_dir: str = "data/processed",
                           random_state: int = 42):
    """
    Full pipeline: load raw -> clean -> create cohort -> train/test split -> save files.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_raw = load_raw_data(raw_csv_path)
    df_cleaned = clean_and_standardize(df_raw)
    cohort = create_balanced_cohort(df_cleaned, random_state=random_state)

    # Save full cleaned cohort for EDA and visualization
    cleaned_path = os.path.join(output_dir, "patient_deterioration_cleaned.csv")
    cohort.to_csv(cleaned_path, index=False)
    print(f"[INFO] Saved cleaned cohort dataset to: {cleaned_path}")

    # Perform stratified 80/20 train-test split
    features = ['age', 'heart_rate', 'respiratory_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'temperature']
    target = 'deterioration_risk'

    X = cohort[features]
    y = cohort[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] Saved train set ({train_df.shape[0]} rows) to: {train_path}")
    print(f"[INFO] Saved test set  ({test_df.shape[0]} rows) to: {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    prepare_and_split_data()
