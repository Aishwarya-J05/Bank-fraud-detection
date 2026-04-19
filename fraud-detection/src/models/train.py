# src/models/train.py
"""
Training pipeline: SMOTE oversampling + baseline models + evaluation harness.
All evaluation uses PR-AUC as primary metric, never accuracy.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

from src.data.preprocessor import FraudPreprocessor, load_and_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.1,
    random_state: int = 42
):
    """
    Oversample fraud class to 10% of majority class.
    sampling_strategy=0.1 is intentional — going to 0.5 hurts real-world performance
    because synthetic samples become too dominant and don't reflect actual fraud patterns.
    """
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    logger.info(f"After SMOTE — Total: {len(X_res):,} | "
                f"Fraud: {y_res.sum():,} ({y_res.mean()*100:.1f}%)")
    return X_res, y_res


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = 0.5
) -> dict:
    """
    Full evaluation suite. Returns metrics dict for comparison across models.
    Primary metric: PR-AUC (not ROC-AUC, not accuracy).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "pr_auc": round(pr_auc, 4),
        "recall": round(tp / (tp + fn), 4),        # fraud catch rate
        "precision": round(tp / (tp + fp), 4),     # how many flagged are real fraud
        "fpr": round(fp / (fp + tn), 4),           # false positive rate
        "f1": round(2*tp / (2*tp + fp + fn), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn)
    }

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}  (fraud catch rate — target >0.80)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}  (false alert rate — target <0.05)")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:,}  (legit correctly passed)")
    print(f"  False Positives: {fp:,}  (legit wrongly flagged)")
    print(f"  False Negatives: {fn:,}  (fraud missed) ← most costly")
    print(f"  True Positives:  {tp:,}  (fraud caught)")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    # Plot PR curve
    _plot_pr_curve(y_test, y_proba, model_name, pr_auc)

    return metrics


def _plot_pr_curve(y_test, y_proba, model_name: str, pr_auc: float):
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 5))
    plt.plot(recall_vals, precision_vals, linewidth=2,
             label=f'{model_name} (PR-AUC={pr_auc:.4f})')
    plt.axhline(y=y_test.mean(), color='red', linestyle='--',
                label=f'Baseline (random) = {y_test.mean():.4f}')
    plt.xlabel('Recall (Fraud Catch Rate)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name}')
    plt.legend()
    plt.tight_layout()

    save_path = REPORTS_DIR / f"pr_curve_{model_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    logger.info(f"PR curve saved to {save_path}")


def train_baseline_models(data_path: str) -> dict:
    """
    Full training run: load → preprocess → SMOTE → train → evaluate.
    Returns metrics for all models for comparison.
    """
    # Step 1: Load and split (raw — no preprocessing yet)
    X_train, X_test, y_train, y_test = load_and_split(data_path)

    # Step 2: Fit preprocessor on training data only
    preprocessor = FraudPreprocessor()
    X_train_proc = preprocessor.fit_transform(
        pd.concat([X_train, y_train], axis=1)
    ).drop('Class', axis=1)

    # Transform test data using fitted preprocessor
    X_test_proc = preprocessor.transform(
        pd.concat([X_test, y_test], axis=1)
    ).drop('Class', axis=1)

    # Step 3: SMOTE on training data only
    X_train_res, y_train_res = apply_smote(X_train_proc, y_train)

    # Step 4: Train models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
    }

    all_metrics = []

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_res, y_train_res)

        metrics = evaluate_model(model, X_test_proc, y_test, name)
        all_metrics.append(metrics)

        # Save each model
        model_path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

    # Save preprocessor
    preprocessor.save(str(MODELS_DIR / "preprocessor.pkl"))

    # Print comparison table
    print("\n=== MODEL COMPARISON ===")
    comparison = pd.DataFrame(all_metrics).set_index('model')
    print(comparison[['pr_auc', 'recall', 'precision', 'fpr', 'f1']])

    return {m['model']: m for m in all_metrics}


if __name__ == "__main__":
    results = train_baseline_models("data/raw/creditcard.csv")