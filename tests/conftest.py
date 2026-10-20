from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for part in ("core", "annealing", "physics", "data", "surrogate", "uncertainty", "optimization", "evaluation"):
    sys.path.insert(0, str(ROOT / "code" / part))
