"""Build a ready-to-push Hugging Face Space folder (Docker SDK + Streamlit) in ./hf_space.
Run:  python scripts/prepare_hf_space.py   then see README section 'Deployment 2'."""
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "hf_space"
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir()
for d in ["src", "app", ".streamlit", "models"]:
    shutil.copytree(ROOT / d, OUT / d, ignore=shutil.ignore_patterns("__pycache__", ".gitkeep"))
(OUT / "data" / "processed").mkdir(parents=True)
shutil.copy(ROOT / "data/processed/heatwave_dataset.csv", OUT / "data/processed/")
(OUT / "reports").mkdir()
shutil.copy(ROOT / "reports/metrics.json", OUT / "reports/")
shutil.copytree(ROOT / "reports/figures", OUT / "reports/figures", ignore=shutil.ignore_patterns(".gitkeep"))
shutil.copy(ROOT / "requirements.txt", OUT / "requirements.txt")
shutil.copy(ROOT / "deploy/huggingface/Dockerfile", OUT / "Dockerfile")
shutil.copy(ROOT / "deploy/huggingface/README.md", OUT / "README.md")
print(f"Space folder ready: {OUT}")
