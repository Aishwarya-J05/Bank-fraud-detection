from huggingface_hub import HfApi

api = HfApi()
for f in ['scaler.pkl', 'feature_columns.pkl', 'is_fitted.pkl']:
    print(f'Uploading {f}...')
    api.upload_file(
        path_or_fileobj=f'models/{f}',
        path_in_repo=f,
        repo_id='AishwaryaNJ/fraud-detection-models',
        repo_type='model'
    )
    print(f'Done: {f}')
print('All done!')