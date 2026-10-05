"""Colab bootstrap copied into the first code cell of each notebook."""
from pathlib import Path
import os
import subprocess
import sys

URL = "https://github.com/Purin1410/Olympic_AI_PTIT_2026_preliminary_round.git"
REF = os.environ.get("AILAAI_RELEASE_REF", "main")
REPO = Path("/content/Olympic_AI_PTIT_2026_preliminary_round")
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, URL, str(REPO)], check=True)
TASK = REPO / "AI_LA_AI"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(TASK / "requirements-colab.txt"), "-e", str(TASK)], check=True)
