"""Validated Reorientation reset state, exported by the Newton feasibility runner."""

from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT_SNAPSHOT_PATH = "runs/reorientation/grasp_snapshot.json"


def load_grasp_snapshot(path: str | Path) -> dict:
    """Reject incomplete or malformed snapshots instead of training from an unverified pose."""
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or not snapshot.get("valid", False):
        raise ValueError("Reorientation requires a validated version-1 grasp snapshot")
    for group, expected in (
        ("arm_joint_pos", {f"panda_joint{i}" for i in range(1, 8)}),
        ("finger_joint_pos", {f"finger_r_joint{i}" for i in range(1, 21)}),
        ("finger_target_pos", {f"finger_r_joint{i}" for i in range(1, 21)}),
    ):
        values = snapshot.get(group, {})
        if set(values) != expected or not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Incomplete or non-finite grasp snapshot: {group}")
    for name, size in (("cube_pos_w", 3), ("cube_quat_w", 4), ("axis_world", 3)):
        values = snapshot.get(name)
        if not isinstance(values, list) or len(values) != size or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid grasp snapshot: {name}")
    if abs(sum(v * v for v in snapshot["cube_quat_w"]) - 1.0) > 0.01:
        raise ValueError("Cube quaternion must be unit length (XYZW)")
    if abs(sum(v * v for v in snapshot["axis_world"]) - 1.0) > 0.01:
        raise ValueError("Palm rotation axis must be unit length")
    return snapshot
