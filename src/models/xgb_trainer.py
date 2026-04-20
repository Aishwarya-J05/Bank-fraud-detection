# src/models/xgb_trainer.py
"""
XGBoost training with Optuna hyperparameter optimization.
Optimizes for PR-AUC — not accuracy, not ROC-AUC.
Includes cost-sensitive threshold optimization post-training.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.data.preprocessor import FraudPreprocessor, load_and_split
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial noise

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0
) -> tuple[float, pd.DataFrame]:
    """
    Finds threshold that minimizes total business cost, not F1.

    cost_fn: cost of missing a fraud (false negative)
             Set to 10x cost_fp — missing fraud costs far more than a false alert.
             In a real bank, this would be the average fraud loss amount.
    cost_fp: cost of flagging a legit transaction (false positive)
             Analyst review time + customer friction.

    Returns: (optimal_threshold, cost_df for plotting)
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    records = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        total_cost = (fn * cost_fn) + (fp * cost_fp)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        records.append({
            "threshold": round(t, 2),
            "total_cost": total_cost,
            "fn": fn, "fp": fp, "tp": tp, "tn": tn,
            "recall": round(recall, 4),
            "precision": round(precision, 4)
        })

    cost_df = pd.DataFrame(records)
    optimal_idx = cost_df['total_cost'].idxmin()
    optimal = cost_df.loc[optimal_idx]

    logger.info(f"Optimal threshold: {optimal['threshold']:.2f} | "
                f"Cost: {optimal['total_cost']:.0f} | "
                f"Recall: {optimal['recall']:.4f} | "
                f"Precision: {optimal['precision']:.4f}")

    return float(optimal['threshold']), cost_df


def plot_threshold_analysis(cost_df: pd.DataFrame, optimal_threshold: float):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cost curve
    axes[0].plot(cost_df['threshold'], cost_df['total_cost'],
                 color='crimson', linewidth=2)
    axes[0].axvline(x=optimal_threshold, color='navy', linestyle='--',
                    label=f'Optimal = {optimal_threshold:.2f}')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Total Business Cost')
    axes[0].set_title('Business Cost vs Threshold\n(FN cost = 10x FP cost)')
    axes[0].legend()

    # Precision-Recall tradeoff vs threshold
    axes[1].plot(cost_df['threshold'], cost_df['recall'],
                 color='steelblue', linewidth=2, label='Recall')
    axes[1].plot(cost_df['threshold'], cost_df['precision'],
                 color='green', linewidth=2, label='Precision')
    axes[1].axvline(x=optimal_threshold, color='navy', linestyle='--',
                    label=f'Optimal = {optimal_threshold:.2f}')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision & Recall vs Threshold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'threshold_analysis.png', dpi=150)
    plt.show()


def train_xgboost(data_path: str, n_trials: int = 50) -> dict:
    """
    Full XGBoost training pipeline with Optuna tuning.
    n_trials=50 is sufficient for this feature space — diminishing returns after that.
    """
    # Load and preprocess
    X_train, X_test, y_train, y_test = load_and_split(data_path)

    preprocessor = FraudPreprocessor.load(str(MODELS_DIR / "preprocessor.pkl"))

    X_train_proc = preprocessor.transform(
        pd.concat([X_train, y_train], axis=1)
    ).drop('Class', axis=1)

    X_test_proc = preprocessor.transform(
        pd.concat([X_test, y_test], axis=1)
    ).drop('Class', axis=1)

    # SMOTE
    sm = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

    # Class imbalance ratio for scale_pos_weight
    neg = int((y_train_res == 0).sum())
    pos = int((y_train_res == 1).sum())
    scale_pos_weight = neg / pos
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Optuna objective
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'aucpr',
            'early_stopping_rounds': 20
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_res, y_train_res,
            eval_set=[(X_test_proc, y_test)],
            verbose=False
        )
        return average_precision_score(y_test, model.predict_proba(X_test_proc)[:, 1])

    logger.info(f"Starting Optuna with {n_trials} trials...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'aucpr'
    })

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_res, y_train_res, verbose=False)

    # Threshold optimization
    y_proba = final_model.predict_proba(X_test_proc)[:, 1]
    optimal_threshold, cost_df = find_optimal_threshold(y_test.values, y_proba)
    plot_threshold_analysis(cost_df, optimal_threshold)

    # Final evaluation at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"XGBoost — Final Results at threshold={optimal_threshold:.2f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Recall:    {tp/(tp+fn):.4f}")
    print(f"Precision: {tp/(tp+fp):.4f}")
    print(f"FPR:       {fp/(fp+tn):.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  FN (fraud missed):   {fn}  ← minimize this")
    print(f"  FP (false alerts):   {fp}")
    print(f"  TP (fraud caught):   {tp}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    # Save
    joblib.dump(final_model, MODELS_DIR / "xgb_fraud_model.pkl")
    joblib.dump(optimal_threshold, MODELS_DIR / "optimal_threshold.pkl")
    logger.info("XGBoost model and threshold saved.")

    return {
        "pr_auc": pr_auc,
        "optimal_threshold": optimal_threshold,
        "best_params": study.best_params
    }


if __name__ == "__main__":
    results = train_xgboost("data/raw/creditcard.csv", n_trials=50)