# 🔍 Bank Fraud Detection System

> Real-time transaction fraud detection using XGBoost + SHAP explainability, deployed on Hugging Face Spaces.

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-HuggingFace-orange)](https://aishwaryanj-fraud-detection-dashboard.hf.space)
[![API](https://img.shields.io/badge/FastAPI-Live-green)](https://aishwaryanj-fraud-detection-api.hf.space/health)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optuna-red)](https://xgboost.readthedocs.io)

---

## 🎯 Problem Statement

A mid-sized digital bank processing 500,000+ daily transactions needed an AI-powered fraud detection engine to replace its outdated rule-based system. Key requirements:

- Flag suspicious transactions in real time (<100ms latency)
- Handle extreme class imbalance (<0.2% fraud rate)
- Provide explainable decisions for PSD2/PCI DSS compliance
- Minimize false positives while maintaining high fraud recall

---

## 📊 Model Performance

| Metric | Target | Achieved |
|---|---|---|
| Recall (Fraud Catch Rate) | >80% | **80.0%** ✅ |
| False Positive Rate | <5% | **0.02%** ✅ |
| Precision | Maximize | **87.4%** ✅ |
| PR-AUC | Maximize | **0.8192** ✅ |
| Inference Latency | <100ms | **~15ms** ✅ |

---

## 🏗️ Architecture

```
Raw Transaction Data
        ↓
Data Pipeline (Pandas + Feature Engineering)
        ↓
Imbalance Handling (SMOTE — 10% sampling strategy)
        ↓
Model Training
  ├── Logistic Regression (baseline)   PR-AUC: 0.6746
  ├── Random Forest (baseline)         PR-AUC: 0.8071
  └── XGBoost + Optuna (final)         PR-AUC: 0.8192
        ↓
Cost-Sensitive Threshold Optimization (FN cost = 10x FP cost)
        ↓
SHAP Explainability Layer
        ↓
FastAPI Backend (Docker, HuggingFace Spaces)
        ↓
Interactive Dashboard (Static HTML, HuggingFace Spaces)
```

---

## 🚀 Live Demo

| Service | URL |
|---|---|
| 🖥️ Dashboard | https://aishwaryanj-fraud-detection-dashboard.hf.space |
| ⚡ API | https://aishwaryanj-fraud-detection-api.hf.space |
| 📖 API Docs | https://aishwaryanj-fraud-detection-api.hf.space/docs |
| 🤗 Models | https://huggingface.co/AishwaryaNJ/fraud-detection-models |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost, Scikit-learn, Optuna |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | HTML, CSS, JavaScript |
| Deployment | Docker, Hugging Face Spaces |
| Data | Pandas, NumPy |

---

## 📁 Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                    # Place creditcard.csv here
│   └── processed/
├── notebooks/
│   └── eda_01.ipynb            # Exploratory data analysis
├── src/
│   ├── data/
│   │   └── preprocessor.py     # FraudPreprocessor class
│   ├── models/
│   │   ├── train.py            # Baseline model training
│   │   ├── xgb_trainer.py      # XGBoost + Optuna tuning
│   │   └── shap_explainer.py   # SHAP analysis
│   ├── api/
│   │   └── main.py             # FastAPI inference endpoint
│   └── dashboard/
│       └── index.html          # Frontend dashboard
├── scripts/
│   └── upload_models.py        # Upload artifacts to HF Hub
├── models/                     # Saved model artifacts (gitignored)
├── reports/                    # EDA plots and SHAP visualizations
├── hf_space_api/               # HuggingFace Docker Space files
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/Aishwarya-J05/Bank-fraud-detection
cd Bank-fraud-detection/fraud-detection
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`.

### 5. Train models

```bash
# Baseline models
python -m src.models.train

# XGBoost + Optuna (50 trials, ~5-10 min)
python -m src.models.xgb_trainer

# SHAP explainer
python -m src.models.shap_explainer
```

### 6. Start API

```bash
python -m src.api.main
```

API runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### 7. Open dashboard

Open `src/dashboard/index.html` in your browser.

---

## 🔬 Key Design Decisions

**Why PR-AUC over ROC-AUC?**
With 0.17% fraud rate, ROC-AUC is misleadingly high even for bad models. PR-AUC directly measures performance on the minority class.

**Why SMOTE at 10% not 50%?**
Oversampling to 50/50 creates too many synthetic samples that don't reflect real fraud patterns, hurting real-world precision. 10% gives the model enough signal without distorting the distribution.

**Why cost-sensitive threshold optimization?**
Default threshold of 0.5 optimizes F1. But missing fraud costs 10x more than a false alert — so the optimal business threshold is 0.78, not 0.5.

**Why SHAP TreeExplainer?**
100x faster than KernelExplainer for tree-based models. Computes exact Shapley values, required for PSD2 regulatory compliance where black-box decisions are not permitted.

---

## 📡 API Usage

### Single transaction

```bash
curl -X POST https://aishwaryanj-fraud-detection-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"Amount": 2.70, "Time": 406, "V14": -4.28, "V10": -2.77}'
```

### Response

```json
{
  "transaction_id": "txn_1776745464682",
  "fraud_probability": 0.9520,
  "is_fraud": true,
  "risk_level": "HIGH",
  "top_risk_factors": {
    "V14": {"shap_value": -3.42, "direction": "increases fraud risk"},
    "V10": {"shap_value": -2.19, "direction": "increases fraud risk"}
  },
  "inference_latency_ms": 10.4,
  "threshold_used": 0.78
}
```

---

## 📈 Model Progression

| Model | PR-AUC | Recall | Precision | FPR |
|---|---|---|---|---|
| Logistic Regression | 0.6746 | 0.8632 | 0.0554 | 0.0247 |
| Random Forest | 0.8071 | 0.7895 | 0.9259 | 0.0001 |
| **XGBoost + Optuna** | **0.8192** | **0.8000** | **0.8736** | **0.0002** |

---

## 📄 Dataset

- **Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions (283,726 after deduplication)
- **Fraud rate:** 0.172% (492 fraud cases)
- **Features:** 28 PCA-transformed features (V1–V28) + Amount + Time

---

## 👩‍💻 Author

**Aishwarya Joshi**  
BE Electronics & Communication Engineering (AI/ML) — VTU, 2026  
AI/ML Engineer Intern @ GlowLogics Solutions

[![GitHub](https://img.shields.io/badge/GitHub-Aishwarya--J05-black)](https://github.com/Aishwarya-J05)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-AishwaryaNJ-yellow)](https://huggingface.co/AishwaryaNJ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-aishwaryajoshiaiml-blue)](https://linkedin.com/in/aishwaryajoshiaiml)
