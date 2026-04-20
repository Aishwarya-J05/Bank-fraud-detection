# src/data/preprocessor.py
"""
FraudPreprocessor: Handles all data transformation steps required before model training.

Design decisions:
- RobustScaler for Amount: median/IQR-based, resistant to the heavy outliers in transaction amounts
- StratifiedKFold: preserves fraud ratio in every fold during cross-validation
- fit_transform vs transform separation: prevents data leakage — scaler is fit only on training data
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """
    Stateful preprocessor — fit on training data, transform on any split.
    Persists to disk via save/load for use in the inference API.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns: list = []
        self.is_fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler on this data and transform it.
        Call ONLY on training data.
        """
        df = self._engineer_features(df.copy())
        df['Amount_scaled'] = self.scaler.fit_transform(df[['Amount']])
        df.drop(['Time', 'Amount'], axis=1, inplace=True)

        self.feature_columns = [c for c in df.columns if c != 'Class']
        self.is_fitted = True

        logger.info(f"Preprocessor fitted. Feature count: {len(self.feature_columns)}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using already-fitted scaler.
        Call on validation, test, and inference data.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        df = self._engineer_features(df.copy())
        df['Amount_scaled'] = self.scaler.transform(df[['Amount']])
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering applied identically to train and inference data.
        Add all new features here to keep train/inference consistent.
        """
        # Hour of day — captures night-time fraud spikes
        df['Hour'] = (df['Time'] // 3600) % 24

        # Amount buckets — fraudsters tend to cluster in specific ranges
        df['Amount_log'] = np.log1p(df['Amount'])  # log1p handles zero amounts

        # Is it a round amount? Round numbers are a fraud signal
        df['Is_round_amount'] = (df['Amount'] % 1 == 0).astype(int)

        return df

    def get_feature_names(self) -> list:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        return self.feature_columns

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> 'FraudPreprocessor':
        preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


def load_and_split(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw CSV, removes duplicates, splits into train/test.
    Returns raw splits — preprocessing happens after to prevent leakage.
    
    Returns: X_train, X_test, y_train, y_test
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Remove duplicates found in EDA
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicates. Rows remaining: {len(df)}")

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Stratify ensures fraud ratio is preserved in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    logger.info(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    logger.info(f"Train fraud rate: {y_train.mean()*100:.3f}%")
    logger.info(f"Test fraud rate:  {y_test.mean()*100:.3f}%")

    return X_train, X_test, y_train, y_test