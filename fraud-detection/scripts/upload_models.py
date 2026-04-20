from huggingface_hub import HfApi, create_repo
from pathlib import Path

api = HfApi()
REPO_ID = "AishwaryaNJ/fraud-detection-models"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"   # ✅ FIXED

create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)

files = [
    "xgb_fraud_model.pkl",
    "preprocessor.pkl",
    "shap_explainer.pkl",
    "optimal_threshold.pkl"
]

print(f"Looking for models in: {MODEL_DIR}\n")

for file_name in files:
    path = MODEL_DIR / file_name

    if not path.exists():
        print(f"❌ MISSING: {path}")
        continue

    print(f"⬆️ Uploading {file_name}...")

    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=file_name,
        repo_id=REPO_ID,
        repo_type="model"
    )

    print(f"✅ Done: {file_name}")

print("\n🚀 Upload process finished")
print(f"🔗 https://huggingface.co/{REPO_ID}")