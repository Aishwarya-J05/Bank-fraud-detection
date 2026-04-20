# scripts/upload_models.py
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
REPO_ID = "AishwaryaNJ/fraud-detection-models"

models_dir = Path(r"C:\Users\Asus\OneDrive\Desktop\Bank fraud detection\fraud-detection\models")

files = [
    "preprocessor.pkl",  # put this first
    "xgb_fraud_model.pkl",
    "shap_explainer.pkl",
    "optimal_threshold.pkl"
]

for f in files:
    path = models_dir / f
    if not path.exists():
        print(f"MISSING: {f}")
        continue
    print(f"⬆️ Uploading {f}...")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"✅ Done: {f}")

print("\n🚀 Upload complete")
print(f"🔗 https://huggingface.co/{REPO_ID}")