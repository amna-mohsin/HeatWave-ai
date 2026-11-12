"""Write requirements.txt with the EXACT versions installed in your clean venv.
Run after `pip install -r requirements.in`:  python scripts/pin_requirements.py"""
from importlib.metadata import version
from pathlib import Path
pkgs = [l.split(">=")[0].split("[")[0].strip() for l in Path("requirements.in").read_text().splitlines() if l.strip() and not l.startswith("#")]
extras = {"uvicorn": "uvicorn[standard]"}
lines = [f"{extras.get(p, p)}=={version(p)}" for p in pkgs]
Path("requirements.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
