"""Path fix for Streamlit multipage app on Colab/Kaggle.
Import this at the top of every page to ensure 'app' and 'src' are importable.
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
