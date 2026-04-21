import joblib
import sys
sys.path.insert(0, '.')

from src.data.preprocessor import FraudPreprocessor

# Load existing preprocessor
p = joblib.load('models/preprocessor.pkl')

# Save its internal components separately
joblib.dump(p.scaler, 'models/scaler.pkl')
joblib.dump(p.feature_columns, 'models/feature_columns.pkl')
joblib.dump(p.is_fitted, 'models/is_fitted.pkl')

print("Saved scaler.pkl, feature_columns.pkl, is_fitted.pkl")
print(f"Feature columns: {p.feature_columns}")