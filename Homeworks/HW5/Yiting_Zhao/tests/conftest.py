import sys
from pathlib import Path

# Make app.py and model_service.py importable when pytest runs from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
