# Validated grasp reference data

`hx5_d20_palm_up_v1.json` is the canonical *experimental input* for the
fixed-grasp Reorientation task. It must be an exact, validated copy of the
local Newton feasibility output, not a hand-written approximation.

This GitHub branch cannot include the JSON until it is promoted from the
machine where `runs/reorientation/grasp_snapshot.json` was generated.

From the repository root, after pulling the code change:

```bash
uv run python scripts/reorientation/promote_grasp_snapshot.py
git add data/reorientation/grasps/hx5_d20_palm_up_v1.json
git commit -m "Add validated HX5 palm-up grasp reference"
git push origin feature/reorientation-feasibility
```

Promotion preserves every byte, validates the snapshot schema and refuses
to overwrite a different canonical reference. The training config logs the
selected path and exact SHA-256 digest in `params/env.yaml`.

The exploratory feasibility script continues to write new candidate data
under `runs/reorientation/`. A new validated grasp should be published
under a new versioned filename rather than overwriting v1.
