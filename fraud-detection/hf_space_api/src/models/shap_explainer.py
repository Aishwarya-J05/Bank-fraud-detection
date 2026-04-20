# src/models/shap_explainer.py
"""
SHAP explainability for XGBoost fraud model.
Required for PSD2/PCI DSS compliance — regulators require explainable decisions.
TreeExplainer is used (not KernelExplainer) because it's 100x faster for tree models.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

import shap
import matplotlib.pyplot as plt

from src.data.preprocessor import FraudPreprocessor, load_and_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def build_explainer(model, X_sample: pd.DataFrame) -> shap.TreeExplainer:
    """
    TreeExplainer computes exact Shapley values for tree-based models.
    Much faster than model-agnostic KernelExplainer which uses sampling approximations.
    """
    explainer = shap.TreeExplainer(model)
    logger.info("TreeExplainer built successfully.")
    return explainer


def plot_global_importance(shap_values: np.ndarray, X_sample: pd.DataFrame):
    """
    Summary plot: shows which features drive fraud predictions globally.
    Each dot = one transaction. Color = feature value. X-axis = SHAP impact.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="dot",
        max_display=20,
        show=False
    )
    plt.title("SHAP Feature Importance — Global (Top 20 Features)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_global_importance.png", dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Global SHAP plot saved.")


def plot_bar_importance(shap_values: np.ndarray, X_sample: pd.DataFrame):
    """
    Bar plot: mean absolute SHAP value per feature.
    Easier to read for non-technical stakeholders.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title("Mean |SHAP Value| — Feature Importance Ranking")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_bar_importance.png", dpi=150, bbox_inches='tight')
    plt.show()


def explain_single_transaction(
    explainer: shap.TreeExplainer,
    transaction: pd.DataFrame,
    transaction_idx: int,
    fraud_prob: float
) -> dict:
    """
    Generates per-transaction explanation for the API response.
    Returns top 5 features driving the prediction — required for compliance.
    """
    shap_vals = explainer.shap_values(transaction)

    feature_impact = pd.Series(
        shap_vals[0],
        index=transaction.columns
    ).sort_values(key=abs, ascending=False)

    top_factors = {}
    for feat, val in feature_impact.head(5).items():
        direction = "increases fraud risk" if val > 0 else "decreases fraud risk"
        top_factors[feat] = {
            "shap_value": round(float(val), 4),
            "direction": direction,
            "feature_value": round(float(transaction[feat].values[0]), 4)
        }

    print(f"\nTransaction #{transaction_idx} — Fraud Probability: {fraud_prob:.4f}")
    print("Top 5 risk factors:")
    for feat, info in top_factors.items():
        print(f"  {feat}: SHAP={info['shap_value']:+.4f} ({info['direction']})")

    return top_factors


def run_shap_analysis(data_path: str, sample_size: int = 500):
    """
    Runs full SHAP analysis on a sample of test data.
    sample_size=500 is enough for stable global importance estimates.
    Using full test set would work but takes longer — no meaningful difference.
    """
    # Load artifacts
    model = joblib.load(MODELS_DIR / "xgb_fraud_model.pkl")
    preprocessor = FraudPreprocessor.load(str(MODELS_DIR / "preprocessor.pkl"))
    optimal_threshold = joblib.load(MODELS_DIR / "optimal_threshold.pkl")

    # Load and preprocess test data
    _, X_test, _, y_test = load_and_split(data_path)
    X_test_proc = preprocessor.transform(
        pd.concat([X_test, y_test], axis=1)
    ).drop('Class', axis=1)

    # Sample for SHAP (representative subset)
    sample_idx = np.random.RandomState(42).choice(len(X_test_proc), sample_size, replace=False)
    X_sample = X_test_proc.iloc[sample_idx].reset_index(drop=True)

    logger.info(f"Computing SHAP values for {sample_size} transactions...")
    explainer = build_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # Global plots
    plot_global_importance(shap_values, X_sample)
    plot_bar_importance(shap_values, X_sample)

    # Save explainer for API use
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
    logger.info("SHAP explainer saved.")

    # Explain a few fraud cases individually
    y_proba = model.predict_proba(X_test_proc)[:, 1]
    fraud_indices = np.where(y_test.values == 1)[0][:3]  # first 3 actual fraud cases

    print("\n=== INDIVIDUAL FRAUD EXPLANATIONS ===")
    for idx in fraud_indices:
        transaction = X_test_proc.iloc[[idx]]
        explain_single_transaction(
            explainer, transaction, idx, float(y_proba[idx])
        )

    print(f"\nSHAP explainer saved to {MODELS_DIR / 'shap_explainer.pkl'}")
    return explainer


if __name__ == "__main__":
    run_shap_analysis("data/raw/creditcard.csv")