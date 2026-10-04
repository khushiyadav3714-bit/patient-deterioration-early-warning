"""
Generates notebooks/analysis.ipynb with clean cells for EDA and model evaluation.
"""

import json
import os

cells = []

def add_md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.split('\n')]
    })

def add_code(code):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in code.split('\n')]
    })

add_md("# Comparative Performance Analysis of Machine Learning Algorithms for Early Prediction of Patient Deterioration\n\n"
       "**Author:** ML Assessment Team\n"
       "**Dataset:** PhysioNet / Computing in Cardiology Challenge 2019 (ICU Clinical Deterioration Cohort)\n"
       "**Objective:** Build, evaluate, and compare 5 standard Scikit-Learn classification algorithms to detect early physiological signs of acute patient deterioration.")

add_md("## 1. Imports and Environment Setup")
add_code("import os\n"
         "import pandas as pd\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n\n"
         "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
         "print('Environment and libraries initialized successfully.')")

add_md("## 2. Load Processed Clinical Dataset\n"
       "We load the cleaned, matched clinical cohort created by `src/preprocess.py`.")
add_code("data_path = '../data/processed/patient_deterioration_cleaned.csv'\n"
         "df = pd.read_csv(data_path)\n"
         "print('Dataset Shape (Rows, Columns):', df.shape)\n"
         "df.head()")

add_md("## 3. Dataset Information and Missing Value Inspection")
add_code("df.info()\n\n"
         "print('\\nMissing values count per feature:')\n"
         "print(df.isna().sum())")

add_md("## 4. Target Class Distribution\n"
       "Examine the balance between Stable patients (0) and Deterioration Risk patients (1).")
add_code("print('Target counts:')\n"
         "print(df['deterioration_risk'].value_counts())\n\n"
         "print('\\nPercentage distribution:')\n"
         "print(df['deterioration_risk'].value_counts(normalize=True) * 100)")

add_md("## 5. Summary Statistics of Vital Signs")
add_code("df.describe().T")

add_md("## 6. Pearson Correlation Heatmap")
add_code("features = ['age', 'heart_rate', 'respiratory_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'temperature']\n"
         "plt.figure(figsize=(8, 6), dpi=150)\n"
         "corr = df[features + ['deterioration_risk']].corr()\n"
         "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, vmin=-1, vmax=1)\n"
         "plt.title('Correlation Heatmap of Patient Vitals and Deterioration Risk', fontweight='bold', pad=10)\n"
         "plt.tight_layout()\n"
         "plt.show()")

add_md("## 7. Vital Signs Distributions (Histograms and KDE)")
add_code("fig, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=150)\n"
         "axes = axes.flatten()\n"
         "for i, col in enumerate(features):\n"
         "    sns.histplot(df[col], kde=True, ax=axes[i], color='#1f77b4', bins=20)\n"
         "    axes[i].set_title(col.replace('_', ' ').title(), fontweight='bold')\n"
         "axes[7].axis('off')\n"
         "plt.suptitle('Histograms and Density Estimates for Monitored Vital Signs', fontweight='bold', y=0.98)\n"
         "plt.tight_layout()\n"
         "plt.show()")

add_md("## 8. Boxplots Comparing Stable vs Deterioration Risk Patients")
add_code("fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), dpi=150)\n"
         "axes = axes.flatten()\n"
         "for i, col in enumerate(features):\n"
         "    sns.boxplot(x='deterioration_risk', y=col, data=df, ax=axes[i], palette=['#5bc0de', '#d9534f'], hue='deterioration_risk', legend=False)\n"
         "    axes[i].set_xticks([0, 1])\n"
         "    axes[i].set_xticklabels(['Stable (0)', 'Deterioration (1)'])\n"
         "    axes[i].set_title(col.replace('_', ' ').title(), fontweight='bold')\n"
         "axes[7].axis('off')\n"
         "plt.suptitle('Vital Signs Comparison: Stable vs Deterioration Risk', fontweight='bold', y=0.98)\n"
         "plt.tight_layout()\n"
         "plt.show()")

add_md("## 9. Comparative Performance of 5 Machine Learning Algorithms\n"
       "Metrics computed on the 20% held-out test cohort (318 patients).")
add_code("results_df = pd.read_csv('../model_results.csv')\n"
         "results_df")

add_md("## 10. Receiver Operating Characteristic (ROC) Comparison Curve")
add_code("import matplotlib.image as mpimg\n"
         "img = mpimg.imread('../reports/figures/roc_curves.png')\n"
         "plt.figure(figsize=(10, 8), dpi=150)\n"
         "plt.imshow(img)\n"
         "plt.axis('off')\n"
         "plt.title('Comparative ROC Curves Across All 5 ML Models', fontweight='bold', pad=10)\n"
         "plt.show()")

add_md("## 11. Confusion Matrices Across Models")
add_code("cm_img = mpimg.imread('../reports/figures/confusion_matrices.png')\n"
         "plt.figure(figsize=(12, 7), dpi=150)\n"
         "plt.imshow(cm_img)\n"
         "plt.axis('off')\n"
         "plt.title('Test Set Confusion Matrices', fontweight='bold', pad=10)\n"
         "plt.show()")

add_md("## 12. Feature Importance of Best Performing Model (Random Forest)")
add_code("fi_img = mpimg.imread('../reports/figures/feature_importance.png')\n"
         "plt.figure(figsize=(9, 6), dpi=150)\n"
         "plt.imshow(fi_img)\n"
         "plt.axis('off')\n"
         "plt.title('Random Forest Feature Importance (MDI)', fontweight='bold', pad=10)\n"
         "plt.show()")

notebook = {
    'cells': cells,
    'metadata': {
        'language_info': {
            'name': 'python',
            'version': '3.13.3'
        },
        'orig_nbformat': 4
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print('[INFO] notebooks/analysis.ipynb generated successfully!')
