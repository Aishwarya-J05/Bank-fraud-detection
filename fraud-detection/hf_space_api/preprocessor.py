import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._engineer_features(df.copy())
        df['Amount_scaled'] = self.scaler.fit_transform(df[['Amount']])
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        self.feature_columns = [c for c in df.columns if c != 'Class']
        self.is_fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted.")
        df = self._engineer_features(df.copy())
        df['Amount_scaled'] = self.scaler.transform(df[['Amount']])
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Is_round_amount'] = (df['Amount'] % 1 == 0).astype(int)
        return df

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)