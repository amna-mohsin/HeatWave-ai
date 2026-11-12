"""Upload the model + model card to a Hugging Face MODEL repo.
Usage: python scripts/push_to_hf_hub.py YOUR_HF_USERNAME   (needs `huggingface-cli login` first; pip install huggingface_hub)"""
import sys
from pathlib import Path
from huggingface_hub import HfApi
ROOT = Path(__file__).resolve().parents[1]
repo = f"{sys.argv[1]}/heatwave-ai-karachi"
api = HfApi(); api.create_repo(repo, repo_type="model", exist_ok=True)
for local, remote in [("models/heatwave_pipeline.joblib", "heatwave_pipeline.joblib"), ("models/model_metadata.json", "model_metadata.json"),
                      ("docs/model_card.md", "README.md"), ("requirements.txt", "requirements.txt")]:
    api.upload_file(path_or_fileobj=str(ROOT / local), path_in_repo=remote, repo_id=repo)
print(f"Done: https://huggingface.co/{repo}")
