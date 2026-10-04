"""
Model Evaluation Module for Patient Deterioration Early Warning System
-----------------------------------------------------------------------
Evaluates all 5 trained models on the held-out test set (318 patients),
computes clinical performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
generates publication-quality comparison charts, saves results to model_results.csv,
and identifies and saves the clinically superior best model.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# Apply clean visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def evaluate_models(test_csv_path: str = "data/processed/test.csv",
                    models_path: str = "models/all_trained_models.pkl",
                    results_csv_path: str = "model_results.csv",
                    figures_dir: str = "reports/figures",
                    best_model_path: str = "models/best_model.pkl"):
    """
    Evaluates all trained models, generates comparison plots, and saves results.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test data not found at {test_csv_path}. Run preprocess.py first.")
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"Trained models not found at {models_path}. Run train_models.py first.")

    test_df = pd.read_csv(test_csv_path)
    feature_cols = ['age', 'heart_rate', 'respiratory_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'temperature']
    target_col = 'deterioration_risk'

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    trained_models = joblib.load(models_path)

    metrics_list = []
    roc_data = {}
    cm_dict = {}

    for name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        cm_dict[name] = confusion_matrix(y_test, y_pred)

        # Clinical composite score giving heavy weight to Recall (minimizing false negatives) and ROC-AUC
        clinical_score = (0.40 * rec) + (0.35 * auc) + (0.25 * f1)

        metrics_list.append({
            'Algorithm': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'Clinical_Score': round(clinical_score, 4)
        })

    results_df = pd.DataFrame(metrics_list)
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n[INFO] Model Evaluation Results saved to: {results_csv_path}")
    print("\n" + "="*80)
    print("COMPARATIVE PERFORMANCE ANALYSIS (TEST SET - 318 PATIENTS)")
    print("="*80)
    print(results_df.drop(columns=['Clinical_Score']).to_string(index=False))
    print("="*80)

    # 1. Plot Combined ROC Curves
    plt.figure(figsize=(8, 6), dpi=300)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=palette[idx % len(palette)], linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    roc_plot_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"[INFO] Saved ROC curve to: {roc_plot_path}")

    # 2. Plot Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={'size': 13, 'weight': 'bold'},
                    xticklabels=['Stable (0)', 'Deterioration (1)'],
                    yticklabels=['Stable (0)', 'Deterioration (1)'])
        ax.set_title(f"{name}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)
    # Hide the 6th unused subplot
    axes[5].axis('off')
    plt.suptitle('Confusion Matrices Comparison (Test Set)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    cm_plot_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"[INFO] Saved confusion matrices to: {cm_plot_path}")

    # 3. Bar Chart Comparing Accuracy
    plt.figure(figsize=(8, 4.5), dpi=300)
    ax = sns.barplot(x='Algorithm', y='Accuracy', hue='Algorithm', data=results_df, palette='Blues_d', legend=False)
    plt.title('Accuracy Comparison Across Machine Learning Algorithms', fontsize=12, fontweight='bold', pad=10)
    plt.ylim([0.0, 1.0])
    plt.ylabel('Accuracy Score', fontsize=10, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=15, ha='right', fontweight='medium')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                    textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    acc_path = os.path.join(figures_dir, "accuracy_comparison.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"[INFO] Saved Accuracy comparison to: {acc_path}")

    # 4. Bar Chart Comparing Recall (Critical for Healthcare)
    plt.figure(figsize=(8, 4.5), dpi=300)
    ax = sns.barplot(x='Algorithm', y='Recall', hue='Algorithm', data=results_df, palette='Reds_d', legend=False)
    plt.title('Clinical Recall Comparison (Sensitivity to Deterioration)', fontsize=12, fontweight='bold', pad=10)
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall Score (True Positive Rate)', fontsize=10, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=15, ha='right', fontweight='medium')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                    textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    rec_path = os.path.join(figures_dir, "recall_comparison.png")
    plt.savefig(rec_path)
    plt.close()
    print(f"[INFO] Saved Recall comparison to: {rec_path}")

    # 5. Bar Chart Comparing F1 Score
    plt.figure(figsize=(8, 4.5), dpi=300)
    ax = sns.barplot(x='Algorithm', y='F1 Score', hue='Algorithm', data=results_df, palette='Greens_d', legend=False)
    plt.title('F1-Score Comparison Across Machine Learning Algorithms', fontsize=12, fontweight='bold', pad=10)
    plt.ylim([0.0, 1.0])
    plt.ylabel('F1-Score (Harmonic Mean of Precision & Recall)', fontsize=10, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=15, ha='right', fontweight='medium')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                    textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    f1_path = os.path.join(figures_dir, "f1_comparison.png")
    plt.savefig(f1_path)
    plt.close()
    print(f"[INFO] Saved F1-score comparison to: {f1_path}")

    # 6. Identify the Best Model based on Clinical Utility
    # We prioritize models that optimize Recall and ROC-AUC while maintaining strong F1-score
    best_row = results_df.sort_values(by='Clinical_Score', ascending=False).iloc[0]
    best_algo_name = best_row['Algorithm']
    best_pipeline = trained_models[best_algo_name]

    print(f"\n[CLINICAL SELECTION] Best Performing Model: {best_algo_name}")
    print(f"  - Accuracy : {best_row['Accuracy']}")
    print(f"  - Recall   : {best_row['Recall']} (Clinical safety: minimizes missed deteriorating patients)")
    print(f"  - F1-Score : {best_row['F1 Score']}")
    print(f"  - ROC-AUC  : {best_row['ROC-AUC']}")

    # Save the best model pipeline
    joblib.dump(best_pipeline, best_model_path)
    print(f"[INFO] Best model pipeline successfully saved to: {best_model_path}")

    # Feature Importance for Tree / Forest or Linear Models
    if hasattr(best_pipeline.named_steps.get('classifier'), 'feature_importances_'):
        rf_clf = best_pipeline.named_steps['classifier']
        importances = rf_clf.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)

        plt.figure(figsize=(7, 4.5), dpi=300)
        plt.barh(fi_df['Feature'], fi_df['Importance'], color='#2b5c8f', edgecolor='black')
        plt.title(f'Feature Importance ({best_algo_name})', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel('Mean Decrease in Impurity (Gini Importance)', fontsize=10, fontweight='bold')
        for idx, val in enumerate(fi_df['Importance']):
            plt.text(val + 0.005, idx, f"{val:.3f}", va='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        fi_path = os.path.join(figures_dir, "feature_importance.png")
        plt.savefig(fi_path)
        plt.close()
        print(f"[INFO] Saved Feature Importance plot to: {fi_path}")

    # Save model metadata
    metadata = {
        'best_algorithm': best_algo_name,
        'features': feature_cols,
        'target': target_col,
        'accuracy': float(best_row['Accuracy']),
        'precision': float(best_row['Precision']),
        'recall': float(best_row['Recall']),
        'f1_score': float(best_row['F1 Score']),
        'roc_auc': float(best_row['ROC-AUC']),
        'clinical_rationale': "Selected for superior ROC-AUC and high recall, protecting against lethal false-negative classifications in clinical deteriorating states."
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return results_df, best_algo_name


if __name__ == "__main__":
    evaluate_models()
