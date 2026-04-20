# src/api/main.py
"""
FastAPI inference API for fraud detection.
Serves XGBoost predictions with SHAP explanations per request.
Designed for <100ms latency — artifacts loaded once at startup, not per request.
"""

import time
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")

# Global artifacts — loaded once at startup
artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts at startup. Fail fast if any are missing."""
    logger.info("Loading model artifacts...")
    try:
        artifacts["model"] = joblib.load(MODELS_DIR / "xgb_fraud_model.pkl")
        artifacts["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.pkl")
        artifacts["explainer"] = joblib.load(MODELS_DIR / "shap_explainer.pkl")
        artifacts["threshold"] = joblib.load(MODELS_DIR / "optimal_threshold.pkl")
        logger.info(f"All artifacts loaded. Threshold: {artifacts['threshold']:.2f}")
    except FileNotFoundError as e:
        logger.error(f"Artifact missing: {e}")
        raise RuntimeError(f"Cannot start API — missing artifact: {e}")
    yield
    artifacts.clear()


app = FastAPI(
    title="Bank Fraud Detection API",
    description="Real-time transaction fraud scoring with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ──────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Raw transaction features as received from the bank's transaction system.
    V1-V28 are PCA-transformed features (anonymized for privacy).
    """
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., gt=0, description="Transaction amount in USD")
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


class RiskFactor(BaseModel):
    shap_value: float
    direction: str
    feature_value: float


class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str          # LOW / MEDIUM / HIGH
    top_risk_factors: dict   # SHAP explanation for compliance
    inference_latency_ms: float
    threshold_used: float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Liveness check — confirms model is loaded and ready."""
    return {
        "status": "healthy",
        "model_loaded": "model" in artifacts,
        "threshold": artifacts.get("threshold")
    }


@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Score a single transaction for fraud.
    Returns probability, binary decision, risk level, and SHAP explanation.
    Target latency: <100ms
    """
    start = time.perf_counter()

    try:
        # Build feature DataFrame matching training schema
        features = pd.DataFrame([transaction.model_dump()])

        # Preprocess — uses fitted scaler, adds engineered features
        preprocessor = artifacts["preprocessor"]
        features_processed = preprocessor.transform(features)

        # Inference
        model = artifacts["model"]
        threshold = artifacts["threshold"]
        proba = float(model.predict_proba(features_processed)[0][1])
        is_fraud = proba >= threshold

        # SHAP explanation
        explainer = artifacts["explainer"]
        shap_vals = explainer.shap_values(features_processed)
        feature_impact = pd.Series(
            shap_vals[0],
            index=features_processed.columns
        ).sort_values(key=abs, ascending=False)

        top_risk_factors = {}
        for feat, val in feature_impact.head(5).items():
            top_risk_factors[feat] = {
                "shap_value": round(float(val), 4),
                "direction": "increases fraud risk" if val > 0 else "decreases fraud risk",
                "feature_value": round(float(features_processed[feat].values[0]), 4)
            }

        # Risk bucketing
        if proba >= 0.7:
            risk_level = "HIGH"
        elif proba >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"fraud={is_fraud} | prob={proba:.4f} | "
            f"risk={risk_level} | latency={latency_ms:.1f}ms"
        )

        return FraudResponse(
            transaction_id=f"txn_{int(time.time() * 1000)}",
            fraud_probability=round(proba, 4),
            is_fraud=bool(is_fraud),
            risk_level=risk_level,
            top_risk_factors=top_risk_factors,
            inference_latency_ms=round(latency_ms, 2),
            threshold_used=round(threshold, 2)
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(transactions: list[TransactionRequest]):
    """
    Score multiple transactions in one request.
    More efficient than individual calls for batch processing.
    Capped at 1000 transactions per request.
    """
    if len(transactions) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 1000 transactions per request."
        )

    start = time.perf_counter()

    features = pd.DataFrame([t.model_dump() for t in transactions])
    features_processed = artifacts["preprocessor"].transform(features)

    probas = artifacts["model"].predict_proba(features_processed)[:, 1]
    threshold = artifacts["threshold"]
    predictions = (probas >= threshold).astype(bool)

    latency_ms = (time.perf_counter() - start) * 1000

    results = []
    for i, (proba, is_fraud) in enumerate(zip(probas, predictions)):
        results.append({
            "index": i,
            "fraud_probability": round(float(proba), 4),
            "is_fraud": bool(is_fraud),
            "risk_level": "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.4 else "LOW"
        })

    logger.info(f"Batch={len(transactions)} | "
                f"Flagged={predictions.sum()} | latency={latency_ms:.1f}ms")

    return {
        "total": len(transactions),
        "flagged": int(predictions.sum()),
        "latency_ms": round(latency_ms, 2),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)