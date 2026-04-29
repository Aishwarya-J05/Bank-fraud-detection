import time
import logging
import joblib
import shutil
import sys
import types
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
import preprocessor as hf_preprocessor
from preprocessor import FraudPreprocessor  # noqa: F401 - required for pickle deserialization

src_module = types.ModuleType("src")
data_module = types.ModuleType("src.data")
src_module.data = data_module
data_module.preprocessor = hf_preprocessor

sys.modules["src"] = src_module
sys.modules["src.data"] = data_module
sys.modules["src.data.preprocessor"] = hf_preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPO_ID = "AishwaryaNJ/fraud-detection-models"
artifacts = {}


def download_models():
    MODELS_DIR.mkdir(exist_ok=True)
    for f in ["xgb_fraud_model.pkl", "preprocessor.pkl", "shap_explainer.pkl", "optimal_threshold.pkl"]:
        dest = MODELS_DIR / f
        logger.info(f"Downloading latest {f}...")
        path = hf_hub_download(repo_id=REPO_ID, filename=f, force_download=True)
        shutil.copy(path, dest)
        logger.info(f"Saved latest: {f}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    download_models()
    artifacts["model"] = joblib.load(MODELS_DIR / "xgb_fraud_model.pkl")
    artifacts["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.pkl")
    artifacts["explainer"] = joblib.load(MODELS_DIR / "shap_explainer.pkl")
    artifacts["threshold"] = joblib.load(MODELS_DIR / "optimal_threshold.pkl")
    logger.info(f"Ready. Threshold: {artifacts['threshold']:.2f}")
    yield
    artifacts.clear()


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TransactionRequest(BaseModel):
    Time: float = 0.0
    Amount: float = Field(..., gt=0)
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in artifacts,
        "threshold": artifacts.get("threshold")
    }


@app.post("/predict")
async def predict(transaction: TransactionRequest):
    start = time.perf_counter()
    try:
        features = pd.DataFrame([transaction.model_dump()])
        features_proc = artifacts["preprocessor"].transform(features)
        proba = float(artifacts["model"].predict_proba(features_proc)[0][1])
        threshold = artifacts["threshold"]
        is_fraud = proba >= threshold

        shap_vals = artifacts["explainer"].shap_values(features_proc)
        impact = pd.Series(shap_vals[0], index=features_proc.columns).sort_values(key=abs, ascending=False)

        top_factors = {
            feat: {
                "shap_value": round(float(val), 4),
                "direction": "increases fraud risk" if val > 0 else "decreases fraud risk",
                "feature_value": round(float(features_proc[feat].values[0]), 4)
            }
            for feat, val in impact.head(5).items()
        }

        return {
            "transaction_id": f"txn_{int(time.time()*1000)}",
            "fraud_probability": round(proba, 4),
            "is_fraud": bool(is_fraud),
            "risk_level": "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.4 else "LOW",
            "top_risk_factors": top_factors,
            "inference_latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "threshold_used": round(float(threshold), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(transactions: list[TransactionRequest]):
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 per batch")
    start = time.perf_counter()
    features = pd.DataFrame([t.model_dump() for t in transactions])
    features_proc = artifacts["preprocessor"].transform(features)
    probas = artifacts["model"].predict_proba(features_proc)[:, 1]
    threshold = artifacts["threshold"]
    preds = (probas >= threshold).astype(bool)
    return {
        "total": len(transactions),
        "flagged": int(preds.sum()),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "results": [
            {
                "index": i,
                "fraud_probability": round(float(p), 4),
                "is_fraud": bool(f),
                "risk_level": "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"
            }
            for i, (p, f) in enumerate(zip(probas, preds))
        ]
    }
