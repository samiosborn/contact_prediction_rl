"""Repository-local paths for Reorientation reference data and generated candidates."""

from pathlib import Path

# Editable-checkout convention: config/paths.py is six directories below repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[5]
REFERENCE_GRASP_PATH = PROJECT_ROOT / "data" / "reorientation" / "grasps" / "hx5_d20_palm_up_v1.json"
GENERATED_GRASP_PATH = PROJECT_ROOT / "runs" / "reorientation" / "grasp_snapshot.json"
